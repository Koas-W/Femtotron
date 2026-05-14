"""diagnose_ac_mismatch.py
找出 forward 在两次连续调用之间 save 数为什么变化。
"""
from __future__ import annotations

import os
import traceback
from collections import OrderedDict

import torch
import torch.distributed as dist

from femtotron.parallel_context import ParallelContext
from femtotron.model.llama import build_llama_model
from transformers import AutoConfig


class SaveLogger:
    """记录每个 save_for_backward 调用的 shape/dtype + 用户代码位置。"""
    
    def __init__(self, name: str):
        self.name = name
        self.saves: list[dict] = []
        self._ctx = None
    
    def _pack(self, tensor: torch.Tensor):
        # 提取调用栈,过滤掉 torch 内部的帧,只留用户代码
        stack = traceback.extract_stack()[:-1]   # 排除 _pack 本身
        user_frames = [
            f for f in stack
            if "torch/" not in f.filename and "site-packages" not in f.filename
        ]
        # 留最多最后 3 个用户帧
        loc = user_frames[-3:] if user_frames else stack[-3:]
        loc_str = " <- ".join(f"{os.path.basename(f.filename)}:{f.lineno}" for f in loc)
        
        self.saves.append({
            "idx": len(self.saves),
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "loc": loc_str,
        })
        return tensor
    
    def _unpack(self, t):
        return t
    
    def __enter__(self):
        self._ctx = torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack)
        self._ctx.__enter__()
        return self
    
    def __exit__(self, *a):
        self._ctx.__exit__(*a)
    
    def summary(self) -> str:
        return f"[{self.name}] {len(self.saves)} saves"


def diff_saves(log_a: SaveLogger, log_b: SaveLogger) -> None:
    """打印两次 save 序列的逐项 diff。"""
    na, nb = len(log_a.saves), len(log_b.saves)
    print(f"\n=== Diff: {log_a.name}({na}) vs {log_b.name}({nb}) ===")
    
    # 用 (shape, dtype, loc) 作为指纹做对齐
    def fp(s): return (s["shape"], s["dtype"], s["loc"])
    
    fps_a = [fp(s) for s in log_a.saves]
    fps_b = [fp(s) for s in log_b.saves]
    
    # 找出 A 有 B 没有 / B 有 A 没有的(用 multiset diff)
    from collections import Counter
    ca, cb = Counter(fps_a), Counter(fps_b)
    
    only_a = ca - cb
    only_b = cb - ca
    
    if only_a:
        print(f"  在 {log_a.name} 出现但 {log_b.name} 没有(或更少):")
        for (shape, dtype, loc), cnt in only_a.items():
            print(f"    ({cnt}×) shape={shape} dtype={dtype}")
            print(f"            at: {loc}")
    
    if only_b:
        print(f"  在 {log_b.name} 出现但 {log_a.name} 没有:")
        for (shape, dtype, loc), cnt in only_b.items():
            print(f"    ({cnt}×) shape={shape} dtype={dtype}")
            print(f"            at: {loc}")
    
    if not only_a and not only_b:
        print(f"  完全一致(只是顺序可能不同)")


def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    parallel_ctx = ParallelContext(OrderedDict([("pp", 1), ("dp", dist.get_world_size()), ("tp", 1)]))
    
    # 同样的小模型
    model_config = AutoConfig.for_model(
        "llama",
        hidden_size=1024, intermediate_size=2048,
        num_attention_heads=16, num_key_value_heads=4,
        num_hidden_layers=8, max_position_embeddings=128,
        vocab_size=1024, rms_norm_eps=1e-5, hidden_act="silu",
        tie_word_embeddings=False,
    )
    
    torch.manual_seed(42)
    with torch.device("meta"):
        model = build_llama_model(model_config, parallel_ctx)
    model = model.to_empty(device=device)
    for p in model.parameters():
        if p.requires_grad:
            torch.nn.init.normal_(p, mean=0.0, std=0.02)
    model = model.bfloat16()
    
    # 准备输入
    torch.manual_seed(2000)
    batch = torch.randint(0, 1024, (8, 32), device=device)
    
    # ─── 跑 forward 两次,记录所有 save,然后 diff ───
    # 用 layer 0 作为单独的 callable(因为 AC 是 per-layer,我们也按 layer 隔离测)
    
    # 我们不能直接调 model.layers[0],因为需要 hidden_states 作为输入。
    # 简单做法:跑完整 model forward 两次,记录两次的所有 save。
    
    log1 = SaveLogger("forward #1")
    with log1:
        out1 = model(input_ids=batch, labels=batch)
        # 不调 backward,我们只关心 forward 的 save
    
    log2 = SaveLogger("forward #2")
    with log2:
        out2 = model(input_ids=batch, labels=batch)
    
    print(f"\n{log1.summary()}")
    print(f"{log2.summary()}")
    
    diff_saves(log1, log2)
    
    # 顺便:loss 是否一致(确认 forward 至少在数学上等价)
    print(f"\nLoss #1 = {out1['loss'].item():.6f}")
    print(f"Loss #2 = {out2['loss'].item():.6f}")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()