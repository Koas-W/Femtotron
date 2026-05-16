"""
femtotron/test/integration/test_pipeline_stage.py

PipelineStage 集成测试。Single rank,但用真 Llama model + cuda + autograd
验证 Stage 抽象正确性。

跑法:
    torchrun --nproc_per_node=1 femtotron/test/integration/test_pipeline_stage.py

测试策略:
    - 用不同 layer_range 构造 LlamaForCausalLM 模拟 first/mid/last/pp_size=1
    - 通过 Stage 跑 forward+backward
    - 跟直接 model.forward+loss.backward 对比 bit-exact
    - 如果不一致,Stage 的 autograd handling 有问题
"""

from collections import OrderedDict
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import LlamaConfig

from femtotron.parallel_context import ParallelContext
from femtotron.parallel.pipeline_parallel.stage import PipelineStage
from femtotron.model.llama_causal import LlamaForCausalLM


# ════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════

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


def make_config():
    return LlamaConfig(
        vocab_size=1024, hidden_size=256, intermediate_size=512,
        num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=4,
        max_position_embeddings=128, rms_norm_eps=1e-5,
        hidden_act="silu", tie_word_embeddings=False,
        attn_implementation="sdpa",
    )


def make_model(config, ctx, device, layer_range=None, seed=0):
    """Construct LlamaForCausalLM on meta,materialize,seed-init。"""
    with torch.device("meta"):
        model = LlamaForCausalLM(config, ctx, layer_range=layer_range)
    model.to_empty(device=device)
    
    # ★ to_empty 留下了 garbage 的 inv_freq buffer;按 HF 同样的公式重新计算
    _reset_rotary_inv_freq(model.model.rotary_emb, config, device)

    torch.manual_seed(seed)
    for _, p in model.named_parameters():
        nn.init.normal_(p, mean=0.0, std=0.02)
    return model.bfloat16()


def clone_grads(model):
    """Snapshot 当前所有 param.grad(deep copy)。"""
    return {
        n: p.grad.clone()
        for n, p in model.named_parameters()
        if p.grad is not None
    }

def compare_grads(
    grads_a: dict[str, torch.Tensor],
    grads_b: dict[str, torch.Tensor],
    tag: str,
    *,
    atol: float = 1e-9,
    rtol: float = 1.5e-3,
) -> None:
    """Compare grads with tensor-level relative tolerance.

    Per-element allclose 对混合量级 tensor 太严:近零元素的 1 ULP 噪音
    会被放大到 rel ~= ∞。改用 max-norm 度量:
    
        max_i |a[i] - b[i]| <= atol + rtol * max_j |b[j]|
    
    优点:
        - 容忍 ~rtol 个 ULPs at tensor 的 max magnitude
        - 真 bug(sign flip / scale / missing term)在 max 位置也会触发,catch 得到
        - 自动 scale 到任意大小模型,不需要按模型调 atol
        - 对 atomic 噪音 robust(近零元素不放大相对误差)
    """
    assert set(grads_a.keys()) == set(grads_b.keys()), \
        f"{tag}: param key sets differ"

    mismatches = []
    for name in grads_a:
        ga, gb = grads_a[name], grads_b[name]
        max_diff = (ga - gb).abs().max().item()
        max_ref = gb.abs().max().item()
        tol = atol + rtol * max_ref
        
        if max_diff > tol:
            # Diagnostic: how many bf16 ULPs of max_ref does the diff represent?
            ulp_at_max = max(max_ref * (2 ** -7), 1e-30)
            ulps = max_diff / ulp_at_max
            mismatches.append((name, max_diff, max_ref, ulps))

    if mismatches:
        details = "\n".join(
            f"    {n}: max_diff={d:.3e}, max_ref={r:.3e}, "
            f"= {u:.2f} ULPs of max_ref (tol allows ~{rtol/2**-7:.1f})"
            for n, d, r, u in mismatches[:10]
        )
        raise AssertionError(
            f"{tag}: tensor-level grad mismatch:\n{details}\n"
            f"Diffs > ~2 ULPs of tensor's max magnitude → likely real bug."
        )

def assert_grads_equal(grads_a, grads_b, tag=""):
    """严格 grad 对比:大部分 param bit-exact,只对 embed 放宽 ULP 噪声。
    
    Rationale:
        - 大部分 op 在 GPU 上是 deterministic 的(matmul / softmax / 等),
          它们的 grad 应该 bit-exact,任何 diff 都是 bug
        - Embedding backward 在 CUDA 上用 atomic-add(scatter_add_),
          run-to-run 顺序不固定,会有 O(N) 个 bf16 ULP 噪声
        - 我们用基于 bf16 ULP 数学的精确容忍:8 × max(|grad|) / 128
          这接近 atomic-add 重排的理论上界,但远小于任何真 bug 的量级
    """
    assert set(grads_a) == set(grads_b), \
        f"{tag}: param sets differ: " \
        f"only_a={set(grads_a) - set(grads_b)}, " \
        f"only_b={set(grads_b) - set(grads_a)}"
    
    bit_exact_count = 0
    ulp_tolerated_count = 0
    
    for name in grads_a:
        a, b = grads_a[name], grads_b[name]
        is_atomic_add = "embed_tokens" in name  # backward 用 scatter_add 的 param
        
        if not is_atomic_add:
            # 严格 bit-exact——任何 diff 都是 bug
            if not torch.equal(a, b):
                max_diff = (a - b).abs().max().item()
                raise AssertionError(
                    f"{tag}: param '{name}' should be bit-exact but diff "
                    f"max={max_diff:.2e}. This is NOT atomic-add noise; "
                    f"likely a real Stage abstraction bug."
                )
            bit_exact_count += 1
        else:
            # ULP-级容忍:基于 bf16 ULP × N 个 atomic-add 的理论上界
            # bf16 ULP @ 值 v ≈ v / 128
            # 容忍 8 个 ULP(留余地给 atomic-add 重排)
            max_mag = max(a.abs().max().item(), b.abs().max().item())
            if max_mag == 0:
                atol = 0.0  # 都是 0,严格相等
            else:
                atol = 8.0 * max_mag / 128.0
            
            if not torch.allclose(a, b, rtol=0.0, atol=atol):
                max_diff = (a - b).abs().max().item()
                raise AssertionError(
                    f"{tag}: param '{name}' diff {max_diff:.2e} exceeds "
                    f"atomic-add noise tolerance {atol:.2e} (8 bf16 ULPs @ "
                    f"max grad magnitude {max_mag:.2e}). "
                    f"This is larger than expected reordering noise; "
                    f"could be a real bug."
                )
            ulp_tolerated_count += 1
    
    return bit_exact_count, ulp_tolerated_count

def make_partial_model(config, ctx, device, layer_range=None, seed=0):
    with torch.device("meta"):
        model = LlamaForCausalLM(config, ctx, layer_range=layer_range)
    model.to_empty(device=device)
    
    # ★ to_empty 留下了 garbage 的 inv_freq buffer;按 HF 同样的公式重新计算
    _reset_rotary_inv_freq(model.model.rotary_emb, config, device)
    
    torch.manual_seed(seed)
    for _, p in model.named_parameters():
        nn.init.normal_(p, mean=0.0, std=0.02)
    return model.bfloat16()


def _reset_rotary_inv_freq(rotary_emb, config, device):
    """重新计算 HF LlamaRotaryEmbedding.inv_freq(persistent=False,不在 state_dict)。
    
    `meta → to_empty` 之后该 buffer 是 garbage,两个 model 各自不同 ⇒
    rotary cos/sin 不一致 ⇒ q/k embedding 不一致 ⇒ q_proj/k_proj 的 grad 不一致。
    """
    # HF default rope 公式(对 default rope_type)
    base = config.rope_parameters["rope_theta"] if hasattr(config, "rope_parameters") \
           else getattr(config, "rope_theta", 10000.0)
    dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(
            device=device, dtype=torch.float
        ) / dim)
    )
    rotary_emb.inv_freq.copy_(inv_freq.to(rotary_emb.inv_freq.dtype))
    if hasattr(rotary_emb, "original_inv_freq"):
        rotary_emb.original_inv_freq.copy_(inv_freq.to(rotary_emb.original_inv_freq.dtype))
# ════════════════════════════════════════════════════════════════
# Test 1: pp_size=1 equivalence
# ════════════════════════════════════════════════════════════════

def test_full_model_equivalence():
    """Stage with full model == direct model.forward + loss.backward bit-exact."""
    log("Test 1: Stage(full model) ≡ direct model")
    config = make_config()
    ctx = ParallelContext(OrderedDict([("pp", 1), ("dp", 1), ("tp", 1)]))
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    model_direct = make_model(config, ctx, device, layer_range=None, seed=0)
    model_stage = make_model(config, ctx, device, layer_range=None, seed=0)
    model_stage.load_state_dict(model_direct.state_dict())

    torch.manual_seed(42)
    input_ids = torch.randint(0, config.vocab_size, (2, 32), device=device)
    labels = torch.randint(0, config.vocab_size, (2, 32), device=device)

    # Path A: direct
    out_direct = model_direct(input_ids, labels=labels)
    loss_direct = out_direct["loss"] if isinstance(out_direct, dict) else out_direct
    loss_direct.backward()
    grads_direct = clone_grads(model_direct)

    # Path B: through Stage
    stage = PipelineStage(model_stage, ctx)
    stage.stage_input(0, input_ids)
    stage.stage_labels(0, labels)
    stage.forward(0)
    stage.backward(0)

    losses = stage.pop_all_losses()
    assert 0 in losses
    loss_stage = losses[0]

    assert torch.equal(loss_direct, loss_stage), \
        f"loss diverge: direct={loss_direct.item()}, stage={loss_stage.item()}"

    grads_stage = clone_grads(model_stage)
    compare_grads(grads_direct, grads_stage, tag="Test 1")

    stage.assert_clean()

    log(f"  ✓ loss bit-exact: {loss_direct.item():.6f}")
    log(f"  ✓ {len(grads_direct)} param grads bit-exact")


# ════════════════════════════════════════════════════════════════
# Test 2: gradient accumulation
# ════════════════════════════════════════════════════════════════

def test_grad_accumulation():
    """N 个 microbatch 通过 Stage 累加 == N 次直接 forward+backward 累加。"""
    log("Test 2: grad accumulation across microbatches")

    config = make_config()
    ctx = ParallelContext(OrderedDict([("pp", 1), ("dp", 1), ("tp", 1)]))
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    N = 4

    model_direct = make_model(config, ctx, device, layer_range=None, seed=0)
    model_stage = make_model(config, ctx, device, layer_range=None, seed=0)
    model_stage.load_state_dict(model_direct.state_dict())

    torch.manual_seed(100)
    all_inputs = [
        torch.randint(0, config.vocab_size, (2, 32), device=device) for _ in range(N)
    ]
    all_labels = [
        torch.randint(0, config.vocab_size, (2, 32), device=device) for _ in range(N)
    ]

    # Path A
    for x, y in zip(all_inputs, all_labels):
        o = model_direct(x, labels=y)
        loss = o["loss"] if isinstance(o, dict) else o
        loss.backward()
    grads_direct = clone_grads(model_direct)

    # Path B
    stage = PipelineStage(model_stage, ctx)
    for mb_id, (x, y) in enumerate(zip(all_inputs, all_labels)):
        stage.stage_input(mb_id, x)
        stage.stage_labels(mb_id, y)
        stage.forward(mb_id)
        stage.backward(mb_id)

    grads_stage = clone_grads(model_stage)
    compare_grads(grads_direct, grads_stage, tag="Test 2")

    stage.assert_clean()
    losses = stage.pop_all_losses()
    assert len(losses) == N

    log(f"  ✓ {len(grads_direct)} grads bit-exact across {N} microbatches")


# ════════════════════════════════════════════════════════════════
# Test 3: mid-stage grad flow
# ════════════════════════════════════════════════════════════════

def test_mid_stage_grad_flow():
    """Mid stage(layer_range=range(1,3)):input.grad 正确计算。"""
    log("Test 3: mid-stage requires_grad / input.grad flow")

    config = make_config()
    ctx = ParallelContext(OrderedDict([("pp", 1), ("dp", 1), ("tp", 1)]))
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    model_stage = make_model(config, ctx, device, layer_range=range(1, 3), seed=0)
    model_direct = make_model(config, ctx, device, layer_range=range(1, 3), seed=0)
    model_direct.load_state_dict(model_stage.state_dict())

    assert not model_stage.is_first and not model_stage.is_last

    torch.manual_seed(200)
    hidden_in = torch.randn(2, 32, config.hidden_size,
                            dtype=torch.bfloat16, device=device)
    grad_out = torch.randn(2, 32, config.hidden_size,
                           dtype=torch.bfloat16, device=device)

    # Path A: direct
    h = hidden_in.clone().detach().requires_grad_(True)
    out = model_direct(h)
    out.backward(grad_out)
    input_grad_direct = h.grad.clone()
    param_grads_direct = clone_grads(model_direct)

    # Path B: through Stage
    stage = PipelineStage(model_stage, ctx)
    stage.stage_input(0, hidden_in.clone().detach())
    stage.forward(0)

    out_stage = stage.get_output(0)
    assert out_stage.shape == (2, 32, config.hidden_size)
    assert out_stage.dtype == torch.bfloat16

    stage.stage_grad(0, grad_out)
    stage.backward(0)
    input_grad_stage = stage.get_input_grad(0)
    param_grads_stage = clone_grads(model_stage)

    assert torch.equal(input_grad_direct, input_grad_stage), \
        f"input.grad diverge: max diff={(input_grad_direct - input_grad_stage).abs().max().item():.2e}"
    compare_grads(param_grads_direct, param_grads_stage, tag="Test 3")

    stage.assert_clean()

    log(f"  ✓ input.grad bit-exact, shape {tuple(input_grad_stage.shape)}")
    log(f"  ✓ {len(param_grads_direct)} mid-stage param grads bit-exact")


# ════════════════════════════════════════════════════════════════
# Test 4: first stage (LongTensor input, no input_grad)
# ════════════════════════════════════════════════════════════════

def test_first_stage():
    """First stage(layer_range=range(0,2)):LongTensor 输入,无 input_grad 可取。"""
    log("Test 4: first stage handles LongTensor input")

    config = make_config()
    ctx = ParallelContext(OrderedDict([("pp", 1), ("dp", 1), ("tp", 1)]))
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    model = make_model(config, ctx, device, layer_range=range(0, 2), seed=0)
    assert model.is_first and not model.is_last

    stage = PipelineStage(model, ctx)

    torch.manual_seed(300)
    input_ids = torch.randint(0, config.vocab_size, (2, 32), device=device)
    grad_out = torch.randn(2, 32, config.hidden_size,
                           dtype=torch.bfloat16, device=device)

    stage.stage_input(0, input_ids)
    stage.forward(0)
    stage.stage_grad(0, grad_out)
    stage.backward(0)

    # 不能 get_input_grad
    raised = False
    try:
        stage.get_input_grad(0)
    except RuntimeError as e:
        raised = True
        assert "first stage" in str(e).lower()
    assert raised, "expected RuntimeError on get_input_grad for first stage"

    # 但 embed_tokens 应该有 grad
    embed_has_grad = any(
        p.grad is not None and "embed_tokens" in n
        for n, p in model.named_parameters()
    )
    assert embed_has_grad, "embed_tokens should have grad after backward"

    stage.assert_clean()
    log("  ✓ first stage: LongTensor handled, embed.grad set, no input_grad")


# ════════════════════════════════════════════════════════════════
# Test 5: loss_scale
# ════════════════════════════════════════════════════════════════

def test_loss_scale():
    """loss_scale=1/N × N 次 == loss_scale=1 × 1 次(数学等价)。"""
    log("Test 5: loss_scale for grad accumulation balancing")

    config = make_config()
    ctx = ParallelContext(OrderedDict([("pp", 1), ("dp", 1), ("tp", 1)]))
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    torch.manual_seed(400)
    input_ids = torch.randint(0, config.vocab_size, (2, 32), device=device)
    labels = torch.randint(0, config.vocab_size, (2, 32), device=device)

    # Reference: scale=1.0,跑 1 次
    model_a = make_model(config, ctx, device, layer_range=None, seed=0)
    stage_a = PipelineStage(model_a, ctx, loss_scale=1.0)
    stage_a.stage_input(0, input_ids)
    stage_a.stage_labels(0, labels)
    stage_a.forward(0)
    stage_a.backward(0)
    grads_a = clone_grads(model_a)

    # Subject: scale=0.5,跑 2 次同样数据
    model_b = make_model(config, ctx, device, layer_range=None, seed=0)
    model_b.load_state_dict(model_a.state_dict())
    for p in model_b.parameters():
        p.grad = None

    stage_b = PipelineStage(model_b, ctx, loss_scale=0.5)
    for mb_id in range(2):
        stage_b.stage_input(mb_id, input_ids)
        stage_b.stage_labels(mb_id, labels)
        stage_b.forward(mb_id)
        stage_b.backward(mb_id)
    grads_b = clone_grads(model_b)

    compare_grads(grads_a, grads_b, tag="Test 5 (loss_scale balance)")
    log("  ✓ scale=0.5 × 2 runs ≡ scale=1.0 × 1 run")


# ════════════════════════════════════════════════════════════════
# Test 6: error handling
# ════════════════════════════════════════════════════════════════

def test_error_handling():
    """所有 error path 触发明确的 RuntimeError。"""
    log("Test 6: error handling")

    config = make_config()
    ctx = ParallelContext(OrderedDict([("pp", 1), ("dp", 1), ("tp", 1)]))
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    model = make_model(config, ctx, device, layer_range=None, seed=0)
    stage = PipelineStage(model, ctx)

    input_ids = torch.randint(0, config.vocab_size, (2, 32), device=device)
    labels = torch.randint(0, config.vocab_size, (2, 32), device=device)

    def expect_error(fn, keyword):
        try:
            fn()
            raise AssertionError(f"expected error with keyword '{keyword}'")
        except (RuntimeError, TypeError) as e:
            assert keyword.lower() in str(e).lower(), \
                f"error message missing '{keyword}': {e}"

    # forward 没 input
    expect_error(lambda: stage.forward(0), "no input")

    # forward last stage 但没 labels
    stage.stage_input(0, input_ids)
    expect_error(lambda: stage.forward(0), "labels")

    # 重复 stage_input
    expect_error(lambda: stage.stage_input(0, input_ids), "already staged")

    # 重复 forward
    stage.stage_labels(0, labels)
    stage.forward(0)
    expect_error(lambda: stage.forward(0), "already exists")

    # last stage 上 stage_grad
    expect_error(lambda: stage.stage_grad(0, torch.zeros(1, device=device)),
                 "last stage")

    # 不存在 mb_id 的 get_output
    expect_error(lambda: stage.get_output(99), "no output")

    # first stage 上 get_input_grad
    stage.backward(0)
    expect_error(lambda: stage.get_input_grad(0), "first stage")

    # assert_clean 有 leftover
    stage.stage_input(99, input_ids)
    expect_error(lambda: stage.assert_clean(), "leftover")
    stage.reset()

    # non-first stage + non-leaf input
    mid = make_model(config, ctx, device, layer_range=range(1, 3), seed=0)
    mid_stage = PipelineStage(mid, ctx)
    non_leaf = torch.randn(2, 32, config.hidden_size,
                           dtype=torch.bfloat16, device=device,
                           requires_grad=True) * 2
    expect_error(lambda: mid_stage.stage_input(0, non_leaf), "leaf")

    # non-first stage + non-float input
    long_input = torch.randint(0, 100, (2, 32, config.hidden_size),
                                dtype=torch.long, device=device)
    expect_error(lambda: mid_stage.stage_input(0, long_input), "float")

    # protocol violation:model 没 is_first
    class BadModel(nn.Module):
        def forward(self, x, labels=None):
            return x
    expect_error(
        lambda: PipelineStage(BadModel(), ctx),
        "is_first",
    )

    log("  ✓ all error paths raise correctly with informative messages")


# ════════════════════════════════════════════════════════════════
# Test 7: reset + pop_all_losses
# ════════════════════════════════════════════════════════════════

def test_state_management():
    """reset() 全清,pop_all_losses() 正确收集。"""
    log("Test 7: reset and pop_all_losses")

    config = make_config()
    ctx = ParallelContext(OrderedDict([("pp", 1), ("dp", 1), ("tp", 1)]))
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    model = make_model(config, ctx, device, layer_range=None, seed=0)
    stage = PipelineStage(model, ctx)

    # 跑 3 个 microbatch
    for mb_id in range(3):
        torch.manual_seed(500 + mb_id)
        x = torch.randint(0, config.vocab_size, (2, 32), device=device)
        y = torch.randint(0, config.vocab_size, (2, 32), device=device)
        stage.stage_input(mb_id, x)
        stage.stage_labels(mb_id, y)
        stage.forward(mb_id)
        stage.backward(mb_id)

    # 收集 losses
    losses = stage.pop_all_losses()
    assert len(losses) == 3
    assert set(losses.keys()) == {0, 1, 2}
    for v in losses.values():
        assert v.ndim == 0  # scalar
        assert v.grad_fn is None  # detached

    # 二次调用 returns empty
    assert len(stage.pop_all_losses()) == 0

    # reset:确保 _loss_values 也清空
    stage.stage_input(99, torch.randint(0, config.vocab_size, (2, 32), device=device))
    stage.reset()
    stage.assert_clean()
    assert len(stage._loss_values) == 0

    log("  ✓ pop_all_losses + reset semantics correct")


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main():
    init_distributed()
    # torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        test_full_model_equivalence()
        test_grad_accumulation()
        test_mid_stage_grad_flow()
        test_first_stage()
        test_loss_scale()
        test_error_handling()
        test_state_management()
        log("\n[All PipelineStage tests passed ✓]")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()