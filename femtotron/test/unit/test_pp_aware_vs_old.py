"""
femtotron/test/unit/test_pp_aware_vs_old.py

M2.5 验证测试:LlamaForCausalLM(layer_range=None) vs LlamaForTraining 必须 bit-exact。
跑通这个 = 新类是旧类的严格超集,所有现有 LlamaForTraining 调用方可以无缝迁移。

跑法:
    torchrun --nproc_per_node=1 femtotron/test/unit/test_pp_aware_vs_old.py
"""

import os
from typing_extensions import OrderedDict

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import LlamaConfig

from femtotron.parallel_context import ParallelContext
from femtotron.model.llama import LlamaForTraining
from femtotron.model.llama_causal import LlamaForCausalLM


# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────

def init_distributed():
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(
        backend="nccl",
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    torch.cuda.set_device(local_rank)
    return local_rank


def log(msg):
    if dist.get_rank() == 0:
        print(msg, flush=True)


def make_test_config(tie: bool = False) -> LlamaConfig:
    """Toy config。默认 tie=False,避开未实现路径。"""
    return LlamaConfig(
        vocab_size=1024,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        hidden_act="silu",
        tie_word_embeddings=tie,
        attn_implementation="sdpa",
    )


def materialize_and_sync(
    old: nn.Module,
    new: nn.Module,
    device: torch.device,
    seed: int = 0,
) -> None:
    """两个 meta 模型 materialize 到 device + 同步参数。

    通过初始化 old + 把它的 state_dict 拷到 new,确保两者参数完全一致。
    """
    old.to_empty(device=device)
    new.to_empty(device=device)

    torch.manual_seed(seed)
    for _, param in old.named_parameters():
        nn.init.normal_(param, mean=0.0, std=0.02)
    # buffers(比如 rotary_emb 的 inv_freq):to_empty 之后是垃圾值,需要重新算
    # 简单做法:依赖 PartialModel 内部 rotary_emb 在 forward 时按 config 重建
    # 如果 LlamaPartialModel 有 register_buffer 路径需要手动 reset,这里要补
    
    # 把 old 的所有参数同步给 new(包括 buffers)
    new.load_state_dict(old.state_dict(), strict=True)


def extract_loss(out) -> torch.Tensor:
    """适配两种返回格式:dict(旧)/ tensor(新)。"""
    if isinstance(out, dict):
        return out["loss"]
    return out


# ────────────────────────────────────────────────────────────────
# Tests
# ────────────────────────────────────────────────────────────────

def test_param_names_match():
    """两个类的 param 命名集合应该完全一致,这样才能 state_dict 互换。"""
    log("Test 1: param names match between old and new")

    config = make_test_config()
    ctx = ParallelContext(
        OrderedDict([("pp", 1), ("dp", 1), ("tp", 1)])
    )

    with torch.device("meta"):
        old = LlamaForTraining(config, ctx)
        new = LlamaForCausalLM(config, ctx, layer_range=None)

    old_names = {n for n, _ in old.named_parameters()}
    new_names = {n for n, _ in new.named_parameters()}

    only_old = old_names - new_names
    only_new = new_names - old_names

    assert not only_old, f"params only in old: {only_old}"
    assert not only_new, f"params only in new: {only_new}"

    log(f"  ✓ {len(old_names)} param names match exactly")


def test_forward_loss_equivalence():
    """forward 算出的 loss 必须 bit-exact。"""
    log("Test 2: forward loss bit-exact")

    config = make_test_config()
    ctx = ParallelContext(
        OrderedDict([("pp", 1), ("dp", 1), ("tp", 1)])
    )
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    with torch.device("meta"):
        old = LlamaForTraining(config, ctx)
        new = LlamaForCausalLM(config, ctx, layer_range=None)

    materialize_and_sync(old, new, device)
    old.bfloat16().eval()
    new.bfloat16().eval()

    torch.manual_seed(2000)
    input_ids = torch.randint(0, config.vocab_size, (2, 32), device=device)
    labels = torch.randint(0, config.vocab_size, (2, 32), device=device)

    with torch.no_grad():
        out_old = old(input_ids, labels=labels)
        out_new = new(input_ids, labels=labels)

    loss_old = extract_loss(out_old)
    loss_new = extract_loss(out_new)

    diff = (loss_old - loss_new).abs().item()
    assert torch.equal(loss_old, loss_new), (
        f"loss diverge: old={loss_old.item():.6f}, new={loss_new.item():.6f}, "
        f"diff={diff:.2e}"
    )

    log(f"  ✓ loss bit-exact: {loss_old.item():.6f}")


def test_backward_grad_equivalence():
    """backward 后所有 param.grad 必须 bit-exact。"""
    log("Test 3: backward gradients bit-exact")

    config = make_test_config()
    ctx = ParallelContext(
        OrderedDict([("pp", 1), ("dp", 1), ("tp", 1)])
    )
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    with torch.device("meta"):
        old = LlamaForTraining(config, ctx)
        new = LlamaForCausalLM(config, ctx, layer_range=None)

    materialize_and_sync(old, new, device)
    old.bfloat16().train()
    new.bfloat16().train()

    torch.manual_seed(2000)
    input_ids = torch.randint(0, config.vocab_size, (2, 32), device=device)
    labels = torch.randint(0, config.vocab_size, (2, 32), device=device)

    # 旧侧
    out_old = old(input_ids, labels=labels)
    extract_loss(out_old).backward()

    # 新侧
    out_new = new(input_ids, labels=labels)
    extract_loss(out_new).backward()

    # 对比
    old_grads = {n: p.grad for n, p in old.named_parameters() if p.grad is not None}
    new_grads = {n: p.grad for n, p in new.named_parameters() if p.grad is not None}

    assert set(old_grads.keys()) == set(new_grads.keys()), (
        f"grad sets differ: "
        f"only-old={set(old_grads) - set(new_grads)}, "
        f"only-new={set(new_grads) - set(old_grads)}"
    )

    mismatches = []
    for name in old_grads:
        if not torch.equal(old_grads[name], new_grads[name]):
            diff = (old_grads[name] - new_grads[name]).abs().max().item()
            mismatches.append((name, diff))

    assert not mismatches, "grad mismatches:\n" + "\n".join(
        f"  {n}: max abs diff = {d:.2e}" for n, d in mismatches[:5]
    )

    log(f"  ✓ {len(old_grads)} grad tensors all bit-exact")


def test_tie_word_embeddings_rejected_by_new():
    """新类必须在 tie=True 时显式拒绝(NotImplementedError),
    旧类不动(原本支持就继续支持)。
    """
    log("Test 4: new class rejects tie_word_embeddings=True")

    config = make_test_config(tie=True)
    ctx = ParallelContext(
        OrderedDict([("pp", 1), ("dp", 1), ("tp", 1)])
    )

    raised = False
    try:
        with torch.device("meta"):
            LlamaForCausalLM(config, ctx, layer_range=None)
    except NotImplementedError as e:
        raised = True
        assert "tie" in str(e).lower(), \
            f"expected 'tie' in error message, got: {e}"

    assert raised, "expected NotImplementedError when tie_word_embeddings=True"

    log("  ✓ correctly rejected with NotImplementedError")


def test_pp_aware_partial_construction():
    """新类的 PP 能力:layer_range != None 时,只构造对应的 layer。

    这部分行为已经在 M1 的 test_partial_model_equivalence.py 验证过了,
    这里只做 smoke test 确认 wrapper 层正确转发。
    """
    log("Test 5: PP-aware partial construction (smoke)")

    config = make_test_config()
    ctx = ParallelContext(
        OrderedDict([("pp", 1), ("dp", 1), ("tp", 1)])
    )

    # 模拟 pp_size=2 的 first stage:只持有 layer 0-1,有 embed 没 lm_head
    with torch.device("meta"):
        first = LlamaForCausalLM(config, ctx, layer_range=range(0, 2))
    
    assert first.is_first is True
    assert first.is_last is False
    assert not hasattr(first, "lm_head") or first.lm_head is None, \
        "first stage should not have lm_head"
    assert hasattr(first.model, "embed_tokens"), \
        "first stage should have embed_tokens"

    # 模拟 pp_size=2 的 last stage:只持有 layer 2-3,有 lm_head 没 embed
    with torch.device("meta"):
        last = LlamaForCausalLM(config, ctx, layer_range=range(2, 4))
    
    assert last.is_first is False
    assert last.is_last is True
    assert hasattr(last, "lm_head"), "last stage should have lm_head"
    assert not hasattr(last.model, "embed_tokens") or \
           last.model.embed_tokens is None, \
        "last stage should not have embed_tokens"

    log("  ✓ partial construction has correct components")


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────

def main():
    init_distributed()
    try:
        test_param_names_match()
        test_forward_loss_equivalence()
        test_backward_grad_equivalence()
        test_tie_word_embeddings_rejected_by_new()
        test_pp_aware_partial_construction()
        log("\n[All M2.5 equivalence tests passed ✓]")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()