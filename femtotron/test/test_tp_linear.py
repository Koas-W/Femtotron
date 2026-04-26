"""
Femtotron TP 线性层测试
======================

用法（2卡验证）：
    torchrun --nproc_per_node=2 test_tp_linear.py

用法（4卡验证）：
    torchrun --nproc_per_node=4 test_tp_linear.py

测试策略：
    在 rank 0 上创建完整的 nn.Linear，用相同输入跑 forward + backward，
    得到 ground truth（输出、输入梯度、权重梯度）。
    然后用 TP 版本在多卡上跑同样的输入，比较结果是否一致。

    关键点：每个测试都要验证三件事：
    1. forward 输出一致
    2. backward 后输入的梯度一致
    3. backward 后权重的梯度一致（切分后的部分应该等于完整梯度的对应切片）
"""

import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
from collections import OrderedDict

# === 根据你的项目结构调整这个 import ===
from femtotron.parallel_context import ParallelContext
from femtotron.tensor_parallel.linear import ColumnParallelLinear, RowParallelLinear
from femtotron.tensor_parallel.embedding import VocabParallelEmbedding


# ============================================================
# 测试工具函数
# ============================================================

def init_distributed():
    """初始化分布式环境。"""
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    torch.cuda.set_device(local_rank)
    return local_rank


def log(msg, rank=None):
    """只在 rank 0 打印，或者指定 rank 打印。"""
    current_rank = dist.get_rank()
    if rank is None and current_rank == 0:
        print(msg)
    elif rank is not None and current_rank == rank:
        print(f"  [Rank {current_rank}] {msg}")


def assert_close(name, actual, expected, atol=1e-5, rtol=1e-5):
    """比较两个 tensor 是否一致，打印结果。"""
    if actual.shape != expected.shape:
        log(f"  ✗ {name}: shape 不匹配 {actual.shape} vs {expected.shape}")
        return False
    
    if torch.allclose(actual, expected, atol=atol, rtol=rtol):
        max_diff = (actual - expected).abs().max().item()
        log(f"  ✓ {name}: 一致 (max diff: {max_diff:.2e})")
        return True
    else:
        max_diff = (actual - expected).abs().max().item()
        mean_diff = (actual - expected).abs().mean().item()
        log(f"  ✗ {name}: 不一致 (max diff: {max_diff:.2e}, mean diff: {mean_diff:.2e})")
        return False


def broadcast_tensor(tensor, src=0):
    """从 src rank 广播 tensor 到所有 rank。"""
    if dist.get_rank() == src:
        shape = torch.tensor(tensor.shape, device="cuda")
    else:
        shape = torch.zeros(2, dtype=torch.long, device="cuda")  # 假设2维
    dist.broadcast(shape, src=src)
    
    if dist.get_rank() != src:
        tensor = torch.zeros(*shape.tolist(), device="cuda", dtype=tensor.dtype if dist.get_rank() == src else torch.float32)
    dist.broadcast(tensor, src=src)
    return tensor


def set_seed(seed=42):
    """所有 rank 设置相同的随机种子，保证输入一致。"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# ============================================================
# 测试 1: ColumnParallelLinear 的 forward 和 backward
# ============================================================

def test_column_parallel_linear(parallel_ctx):
    """
    验证 ColumnParallelLinear 的 forward 输出和 backward 梯度
    与单卡 nn.Linear 一致。
    """
    log("\n" + "=" * 60)
    log("测试 1: ColumnParallelLinear")
    log("=" * 60)
    
    tp_size = parallel_ctx.tp_size
    tp_rank = parallel_ctx.tp_rank
    
    in_features = 256
    out_features = 512
    batch_size = 4
    seq_len = 16
    
    set_seed(42)
    
    # --- Ground truth: 单卡 nn.Linear ---
    ref_linear = nn.Linear(in_features, out_features, bias=False).cuda()
    ref_weight = ref_linear.weight.data.clone()  # [out, in]
    
    x = torch.randn(batch_size, seq_len, in_features, device="cuda", requires_grad=True)
    ref_x = x.clone().detach().requires_grad_(True)
    
    ref_out = ref_linear(ref_x)         # [B, S, out]
    ref_out.sum().backward()
    ref_x_grad = ref_x.grad.clone()     # [B, S, in]
    ref_w_grad = ref_linear.weight.grad.clone()  # [out, in]
    
    # --- TP 版本: ColumnParallelLinear (gather_output=True) ---
    # 用 from_linear 从完整权重构造（测试用）
    col_linear = ColumnParallelLinear.from_linear(
        nn.Linear(in_features, out_features, bias=False).cuda(),
        parallel_ctx,
        gather_output=True,
    )
    # 手动设置权重为 ref_weight 的对应切片
    chunk = out_features // tp_size
    col_linear.weight.data.copy_(ref_weight[tp_rank * chunk : (tp_rank + 1) * chunk, :])
    
    tp_x = x.clone().detach().requires_grad_(True)
    tp_out = col_linear(tp_x)           # [B, S, out] (因为 gather_output=True)
    tp_out.sum().backward()
    tp_x_grad = tp_x.grad.clone()
    tp_w_grad = col_linear.weight.grad.clone()  # [chunk, in]
    
    # --- 验证 ---
    passed = True
    
    # forward 输出应该和 ref 完全一致
    passed &= assert_close("forward output", tp_out, ref_out)
    
    # 输入梯度应该一致
    passed &= assert_close("input grad", tp_x_grad, ref_x_grad)
    
    # 权重梯度：TP 版本的梯度应该等于完整梯度的对应切片
    ref_w_grad_chunk = ref_w_grad[tp_rank * chunk : (tp_rank + 1) * chunk, :]
    passed &= assert_close("weight grad", tp_w_grad, ref_w_grad_chunk)
    
    # --- 测试 gather_output=False 的情况 ---
    log("\n  --- gather_output=False ---")
    col_linear_no_gather = ColumnParallelLinear.from_linear(
        nn.Linear(in_features, out_features, bias=False).cuda(),
        parallel_ctx,
        gather_output=False,
    )
    col_linear_no_gather.weight.data.copy_(ref_weight[tp_rank * chunk : (tp_rank + 1) * chunk, :])
    
    tp_x2 = x.clone().detach().requires_grad_(True)
    tp_out2 = col_linear_no_gather(tp_x2)  # [B, S, out//tp]
    
    # 输出应该等于完整输出的对应切片
    ref_out_chunk = ref_out[:, :, tp_rank * chunk : (tp_rank + 1) * chunk].detach()
    passed &= assert_close("partial output", tp_out2, ref_out_chunk)
    
    log(f"\n  {'✓ 测试通过' if passed else '✗ 测试失败'}")
    return passed


# ============================================================
# 测试 2: RowParallelLinear 的 forward 和 backward
# ============================================================

def test_row_parallel_linear(parallel_ctx):
    """
    验证 RowParallelLinear 的 forward 输出和 backward 梯度
    与单卡 nn.Linear 一致。
    """
    log("\n" + "=" * 60)
    log("测试 2: RowParallelLinear")
    log("=" * 60)
    
    tp_size = parallel_ctx.tp_size
    tp_rank = parallel_ctx.tp_rank
    
    in_features = 512
    out_features = 256
    batch_size = 4
    seq_len = 16
    
    set_seed(42)
    
    # --- Ground truth ---
    ref_linear = nn.Linear(in_features, out_features, bias=False).cuda()
    ref_weight = ref_linear.weight.data.clone()  # [out, in]
    
    x = torch.randn(batch_size, seq_len, in_features, device="cuda", requires_grad=True)
    ref_x = x.clone().detach().requires_grad_(True)
    
    ref_out = ref_linear(ref_x)
    ref_out.sum().backward()
    ref_x_grad = ref_x.grad.clone()
    ref_w_grad = ref_linear.weight.grad.clone()
    
    # --- TP 版本: RowParallelLinear (scatter_input=False) ---
    # RowParallel 期望输入已经是切分的
    row_linear = RowParallelLinear.from_linear(
        nn.Linear(in_features, out_features, bias=False).cuda(),
        parallel_ctx,
        scatter_input=False,
    )
    chunk_in = in_features // tp_size
    row_linear.weight.data.copy_(ref_weight[:, tp_rank * chunk_in : (tp_rank + 1) * chunk_in])
    
    # 手动切分输入（模拟来自上游 ColumnParallel 的切分输出）
    x_partial = x[:, :, tp_rank * chunk_in : (tp_rank + 1) * chunk_in].clone().detach().requires_grad_(True)
    
    tp_out = row_linear(x_partial)       # [B, S, out] (经过 all-reduce，完整)
    tp_out.sum().backward()
    tp_x_partial_grad = x_partial.grad.clone()  # [B, S, in//tp]
    tp_w_grad = row_linear.weight.grad.clone()   # [out, in//tp]
    
    # --- 验证 ---
    passed = True
    
    # forward 输出应该一致
    passed &= assert_close("forward output", tp_out, ref_out)
    
    # 输入梯度：TP 版本的部分梯度应该等于完整梯度的对应切片
    ref_x_grad_chunk = ref_x_grad[:, :, tp_rank * chunk_in : (tp_rank + 1) * chunk_in]
    passed &= assert_close("input grad (partial)", tp_x_partial_grad, ref_x_grad_chunk)
    
    # 权重梯度：TP 版本应该等于完整梯度的对应切片
    ref_w_grad_chunk = ref_w_grad[:, tp_rank * chunk_in : (tp_rank + 1) * chunk_in]
    passed &= assert_close("weight grad", tp_w_grad, ref_w_grad_chunk)
    
    log(f"\n  {'✓ 测试通过' if passed else '✗ 测试失败'}")
    return passed


# ============================================================
# 测试 3: Column → Row 链路（最重要的测试）
# ============================================================

def test_column_row_chain(parallel_ctx):
    """
    验证 ColumnParallel → RowParallel 的端到端链路。
    
    这是 transformer 中 Attention 和 FFN 的实际使用模式：
    - Attention: Q/K/V (Column) → attn → O (Row)
    - FFN: gate/up (Column) → activation → down (Row)
    
    中间不需要 gather/scatter，Column 的切分输出直接喂给 Row。
    整个链路只有 Row 结尾的一次 all-reduce。
    """
    log("\n" + "=" * 60)
    log("测试 3: Column → Row 端到端链路")
    log("=" * 60)
    
    tp_size = parallel_ctx.tp_size
    tp_rank = parallel_ctx.tp_rank
    
    hidden = 256
    ffn_dim = 512
    batch_size = 4
    seq_len = 16
    
    set_seed(42)
    
    # --- Ground truth: 两个串联的 nn.Linear ---
    ref_up = nn.Linear(hidden, ffn_dim, bias=False).cuda()
    ref_down = nn.Linear(ffn_dim, hidden, bias=False).cuda()
    ref_up_weight = ref_up.weight.data.clone()
    ref_down_weight = ref_down.weight.data.clone()
    
    x = torch.randn(batch_size, seq_len, hidden, device="cuda")
    ref_x = x.clone().detach().requires_grad_(True)
    
    ref_mid = ref_up(ref_x)          # [B, S, ffn_dim]
    ref_mid = torch.relu(ref_mid)    # 加个非线性，更接近真实场景
    ref_out = ref_down(ref_mid)      # [B, S, hidden]
    ref_out.sum().backward()
    ref_x_grad = ref_x.grad.clone()
    ref_up_w_grad = ref_up.weight.grad.clone()
    ref_down_w_grad = ref_down.weight.grad.clone()
    
    # --- TP 版本: Column(gather=False) → ReLU → Row(scatter=False) ---
    chunk_ffn = ffn_dim // tp_size
    
    tp_up = ColumnParallelLinear.from_linear(
        nn.Linear(hidden, ffn_dim, bias=False).cuda(),
        parallel_ctx,
        gather_output=False,
    )
    tp_up.weight.data.copy_(ref_up_weight[tp_rank * chunk_ffn : (tp_rank + 1) * chunk_ffn, :])
    
    tp_down = RowParallelLinear.from_linear(
        nn.Linear(ffn_dim, hidden, bias=False).cuda(),
        parallel_ctx,
        scatter_input=False,
    )
    tp_down.weight.data.copy_(ref_down_weight[:, tp_rank * chunk_ffn : (tp_rank + 1) * chunk_ffn])
    
    tp_x = x.clone().detach().requires_grad_(True)
    
    tp_mid = tp_up(tp_x)             # [B, S, ffn_dim//tp] — 切分的
    tp_mid = torch.relu(tp_mid)      # 每个 rank 独立做 ReLU
    tp_out = tp_down(tp_mid)         # [B, S, hidden] — all-reduce 后完整
    tp_out.sum().backward()
    
    tp_x_grad = tp_x.grad.clone()
    tp_up_w_grad = tp_up.weight.grad.clone()
    tp_down_w_grad = tp_down.weight.grad.clone()
    
    # --- 验证 ---
    passed = True
    
    # forward 输出
    passed &= assert_close("chain forward output", tp_out, ref_out)
    
    # 输入梯度
    passed &= assert_close("chain input grad", tp_x_grad, ref_x_grad)
    
    # up (Column) 的权重梯度
    ref_up_chunk = ref_up_w_grad[tp_rank * chunk_ffn : (tp_rank + 1) * chunk_ffn, :]
    passed &= assert_close("up weight grad", tp_up_w_grad, ref_up_chunk)
    
    # down (Row) 的权重梯度
    ref_down_chunk = ref_down_w_grad[:, tp_rank * chunk_ffn : (tp_rank + 1) * chunk_ffn]
    passed &= assert_close("down weight grad", tp_down_w_grad, ref_down_chunk)
    
    log(f"\n  {'✓ 测试通过' if passed else '✗ 测试失败'}")
    return passed


# ============================================================
# 测试 4: VocabParallelEmbedding
# ============================================================

def test_vocab_parallel_embedding(parallel_ctx):
    """
    验证 VocabParallelEmbedding 的 forward 和 backward。
    """
    log("\n" + "=" * 60)
    log("测试 4: VocabParallelEmbedding")
    log("=" * 60)
    
    tp_size = parallel_ctx.tp_size
    tp_rank = parallel_ctx.tp_rank
    
    vocab_size = 1024  # 用小 vocab 方便测试
    hidden_size = 256
    batch_size = 4
    seq_len = 16
    
    set_seed(42)
    
    # --- Ground truth ---
    ref_embed = nn.Embedding(vocab_size, hidden_size).cuda()
    ref_weight = ref_embed.weight.data.clone()
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")
    
    ref_out = ref_embed(input_ids)       # [B, S, H]
    ref_out.sum().backward()
    ref_w_grad = ref_embed.weight.grad.clone()
    
    # --- TP 版本 ---
    tp_embed = VocabParallelEmbedding.from_embedding(ref_embed, parallel_ctx)
    
    tp_out = tp_embed(input_ids)
    tp_out.sum().backward()
    tp_w_grad = tp_embed.weight.grad.clone()
    
    # --- 验证 ---
    passed = True
    
    # forward 输出应该一致
    passed &= assert_close("embedding forward", tp_out, ref_out)
    
    # 权重梯度：TP 版本应该等于完整梯度的对应切片
    chunk = vocab_size // tp_size
    ref_w_grad_chunk = ref_w_grad[tp_rank * chunk : (tp_rank + 1) * chunk, :]
    passed &= assert_close("embedding weight grad", tp_w_grad, ref_w_grad_chunk)
    
    log(f"\n  {'✓ 测试通过' if passed else '✗ 测试失败'}")
    return passed


# ============================================================
# 测试 5: TP=1 退化测试
# ============================================================

def test_tp_one_degeneracy(parallel_ctx):
    """
    验证 TP=1 时行为和普通 nn.Linear 完全一致。
    这是一个重要的 sanity check——并行度为 1 时不应该有任何副作用。
    """
    log("\n" + "=" * 60)
    log("测试 5: TP=1 退化验证")
    log("=" * 60)
    
    if parallel_ctx.tp_size != 1:
        log("  跳过（当前 tp_size != 1）")
        return True
    
    in_features = 256
    out_features = 512
    batch_size = 4
    seq_len = 16
    
    set_seed(42)
    
    ref_linear = nn.Linear(in_features, out_features, bias=False).cuda()
    col_linear = ColumnParallelLinear.from_linear(ref_linear, parallel_ctx, gather_output=False)
    
    x = torch.randn(batch_size, seq_len, in_features, device="cuda", requires_grad=True)
    ref_x = x.clone().detach().requires_grad_(True)
    
    ref_out = ref_linear(ref_x)
    ref_out.sum().backward()
    
    tp_out = col_linear(x)
    tp_out.sum().backward()
    
    passed = True
    passed &= assert_close("tp=1 forward", tp_out, ref_out)
    passed &= assert_close("tp=1 input grad", x.grad, ref_x.grad)
    passed &= assert_close("tp=1 weight grad", col_linear.weight.grad, ref_linear.weight.grad)
    
    log(f"\n  {'✓ 测试通过' if passed else '✗ 测试失败'}")
    return passed


# ============================================================
# Main
# ============================================================

def main():
    local_rank = init_distributed()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    if rank == 0:
        print("=" * 60)
        print(f"Femtotron TP 线性层测试")
        print(f"World size: {world_size}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("=" * 60)
    
    # 创建 ParallelContext，所有卡都用于 TP
    parallel_ctx = ParallelContext(OrderedDict([
        ("tp", world_size),
    ]))
    
    if rank == 0:
        print(f"并行配置: TP={parallel_ctx.tp_size}")
    
    # 同步所有 rank
    dist.barrier()
    
    # 运行测试
    all_passed = True
    
    all_passed &= test_column_parallel_linear(parallel_ctx)
    dist.barrier()
    
    all_passed &= test_row_parallel_linear(parallel_ctx)
    dist.barrier()
    
    all_passed &= test_column_row_chain(parallel_ctx)
    dist.barrier()
    
    all_passed &= test_vocab_parallel_embedding(parallel_ctx)
    dist.barrier()
    
    # TP=1 退化测试需要单独的 context
    if world_size >= 2:
        # 也可以测试 TP=1, DP=world_size 的配置
        tp1_ctx = ParallelContext(OrderedDict([
            ("dp", world_size),
            ("tp", 1),
        ]))
        all_passed &= test_tp_one_degeneracy(tp1_ctx)
        dist.barrier()
    
    # 总结
    if rank == 0:
        print("\n" + "=" * 60)
        if all_passed:
            print("所有测试通过 ✓")
        else:
            print("存在失败的测试 ✗")
        print("=" * 60)
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()