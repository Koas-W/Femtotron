"""验证 LlamaPartialModel 在 pp_size=1 时和 HF LlamaModel 完全等价。

测试命令:
    torchrun --nproc_per_node=1 tests/test_partial_model_equivalence.py
"""
from __future__ import annotations

import os
from collections import OrderedDict

import torch
import torch.distributed as dist
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel as HFLlamaModel

from femtotron.parallel_context import ParallelContext
from femtotron.model.llama_partial_model import LlamaPartialModel


# ────────────────────────────────────────────────────────────
# 共用工具
# ────────────────────────────────────────────────────────────

def setup():
    """初始化分布式环境(单卡也要 init,因为 ParallelContext 需要)。"""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)


def teardown():
    if dist.is_initialized():
        dist.destroy_process_group()


def make_config() -> LlamaConfig:
    """一个小配置以测试正确性。"""
    return LlamaConfig(
        hidden_size=256,
        intermediate_size=512,
        num_attention_heads=8,
        num_key_value_heads=4,
        num_hidden_layers=4,
        max_position_embeddings=128,
        vocab_size=1024,
        rms_norm_eps=1e-5,
        hidden_act="silu",
        tie_word_embeddings=False,
        attn_implementation="sdpa",
    )


def make_parallel_ctx() -> ParallelContext:
    """单 rank 的 ParallelContext。"""
    return ParallelContext(
        OrderedDict([("pp", 1), ("dp", 1), ("tp", 1)])
    )


# ────────────────────────────────────────────────────────────
# 测试 1:Forward 等价性
# ────────────────────────────────────────────────────────────

def test_forward_equivalence() -> None:
    """同一份 weight + 同一份 input,两个模型 forward 应该 bit-exact。"""
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    config = make_config()
    parallel_ctx = make_parallel_ctx()
    
    # 构造 HF model(它的 init 用 default seed,init 顺序无关因为接下来要覆盖)
    hf_model = HFLlamaModel(config).to(device).bfloat16().eval()

    # 构造我们的 model
    our_model = LlamaPartialModel(config, parallel_ctx).to(device).bfloat16().eval()
    
    # 关键:同步 weights(我们的命名兼容 HF,直接 load_state_dict)
    missing, unexpected = our_model.load_state_dict(hf_model.state_dict(), strict=True)
    assert len(missing) == 0, f"加载 HF state_dict 缺失 keys: {missing}"
    assert len(unexpected) == 0, f"加载 HF state_dict 多余 keys: {unexpected}"
    print(f"  Loaded {len(hf_model.state_dict())} parameters from HF model")
    
    # 一份输入
    torch.manual_seed(2000)
    input_ids = torch.randint(0, config.vocab_size, (2, 32), device=device)
    
    # 两边 forward
    with torch.no_grad():
        hf_out = hf_model(input_ids, use_cache=False).last_hidden_state
        our_out = our_model(input_ids)
    
    # Shape 检查
    assert hf_out.shape == our_out.shape, (
        f"Shape mismatch: HF {hf_out.shape} vs our {our_out.shape}"
    )
    
    # 数值检查 — bf16 同 kernel 路径应该 bit-exact
    max_abs_diff = (hf_out - our_out).abs().max().item()
    print(f"  Forward max abs diff: {max_abs_diff:.6e}")
    
    assert max_abs_diff == 0.0, (
        f"Forward outputs differ by {max_abs_diff} (expected 0 for bit-exact)"
    )
    print("✓ Forward 完全一致(bit-exact)")


# ────────────────────────────────────────────────────────────
# 测试 2:Backward 等价性
# ────────────────────────────────────────────────────────────

def test_backward_equivalence() -> None:
    """同 weight + 同 input + 同 loss → 完全相同的 gradients。"""
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    config = make_config()
    parallel_ctx = make_parallel_ctx()
    
    hf_model = HFLlamaModel(config).to(device).bfloat16().train()
    our_model = LlamaPartialModel(config, parallel_ctx).to(device).bfloat16().train()
    our_model.load_state_dict(hf_model.state_dict(), strict=True)
    
    torch.manual_seed(2000)
    input_ids = torch.randint(0, config.vocab_size, (2, 32), device=device)
    
    # HF backward
    hf_out = hf_model(input_ids, use_cache=False).last_hidden_state
    hf_loss = hf_out.float().sum()
    hf_loss.backward()
    
    # Our backward
    our_out = our_model(input_ids)
    our_loss = our_out.float().sum()
    our_loss.backward()
    
    # Loss 一致性
    loss_diff = abs(hf_loss.item() - our_loss.item())
    print(f"  Loss diff: {loss_diff:.6e}")
    assert loss_diff == 0.0, f"Loss differs by {loss_diff}"
    
    # 逐参数比较 gradients
    max_grad_diff = 0.0
    worst_param = ""
    n_params = 0
    
    hf_params = dict(hf_model.named_parameters())
    our_params = dict(our_model.named_parameters())
    
    assert set(hf_params.keys()) == set(our_params.keys()), (
        f"Parameter names differ:\n"
        f"  HF only: {set(hf_params.keys()) - set(our_params.keys())}\n"
        f"  Our only: {set(our_params.keys()) - set(hf_params.keys())}"
    )
    
    for name in hf_params:
        p_hf = hf_params[name]
        p_our = our_params[name]
        
        if p_hf.grad is None:
            assert p_our.grad is None, f"{name}: HF grad None but our is not"
            continue
        
        assert p_our.grad is not None, f"{name}: HF grad is set but ours is None"
        
        diff = (p_hf.grad - p_our.grad).abs().max().item()
        if diff > max_grad_diff:
            max_grad_diff = diff
            worst_param = name
        n_params += 1
    
    print(f"  Compared {n_params} param grads")
    print(f"  Max grad diff: {max_grad_diff:.6e} (at: {worst_param})")
    
    assert max_grad_diff == 0.0, (
        f"Grads differ by {max_grad_diff} at parameter {worst_param} "
        f"(expected 0 for bit-exact)"
    )
    print("✓ Backward 完全一致(bit-exact)")


# ────────────────────────────────────────────────────────────
# 测试 3:参数命名兼容性
# ────────────────────────────────────────────────────────────

def test_parameter_naming_compat() -> None:
    """我们的参数名应该和 HF LlamaModel 完全一致。"""
    config = make_config()
    parallel_ctx = make_parallel_ctx()
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    
    hf_model = HFLlamaModel(config).to(device)
    our_model = LlamaPartialModel(config, parallel_ctx).to(device)
    
    hf_names = set(dict(hf_model.named_parameters()).keys())
    our_names = set(dict(our_model.named_parameters()).keys())
    
    only_hf = hf_names - our_names
    only_our = our_names - hf_names
    
    if only_hf:
        print(f"  Only in HF: {only_hf}")
    if only_our:
        print(f"  Only in our: {only_our}")
    
    assert hf_names == our_names, (
        f"Parameter name sets differ.\n"
        f"  Only in HF: {only_hf}\n"
        f"  Only in our: {only_our}"
    )
    
    print(f"  Both models have identical {len(hf_names)} parameter names")
    print("✓ Parameter naming 完全兼容")


# ────────────────────────────────────────────────────────────
# 测试 4:PP-aware 模式至少能构造起来(不验证 PP 数值,那是 M3 的事)
# ────────────────────────────────────────────────────────────

def test_pp_aware_construction() -> None:
    """验证 PP 切分模式下构造没问题,参数集合符合预期。"""
    config = make_config()
    parallel_ctx = make_parallel_ctx()
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    
    # First stage(layers 0-1,有 embed 无 norm)
    first = LlamaPartialModel(
        config, parallel_ctx,
        layer_range=range(0, 2),
        is_first=True, is_last=False,
    ).to(device)
    
    first_names = set(dict(first.named_parameters()).keys())
    
    assert any("embed_tokens" in n for n in first_names), "first stage 应有 embed_tokens"
    assert not any("norm.weight" == n for n in first_names), "first stage 不应有 final norm"
    assert any("layers.0." in n for n in first_names), "first stage 应有 layer 0"
    assert any("layers.1." in n for n in first_names), "first stage 应有 layer 1"
    assert not any("layers.2." in n for n in first_names), "first stage 不应有 layer 2"
    assert not any("layers.3." in n for n in first_names), "first stage 不应有 layer 3"
    
    # Last stage(layers 2-3,无 embed 有 norm)
    last = LlamaPartialModel(
        config, parallel_ctx,
        layer_range=range(2, 4),
        is_first=False, is_last=True,
    ).to(device)
    
    last_names = set(dict(last.named_parameters()).keys())
    
    assert not any("embed_tokens" in n for n in last_names), "last stage 不应有 embed_tokens"
    assert any("norm.weight" == n for n in last_names), "last stage 应有 final norm"
    assert not any("layers.0." in n for n in last_names), "last stage 不应有 layer 0"
    assert not any("layers.1." in n for n in last_names), "last stage 不应有 layer 1"
    assert any("layers.2." in n for n in last_names), "last stage 应有 layer 2"
    assert any("layers.3." in n for n in last_names), "last stage 应有 layer 3"
    
    print(f"  First stage(layers 0-1):{len(first_names)} params")
    print(f"  Last stage(layers 2-3):{len(last_names)} params")
    print("✓ PP-aware 构造正确")


# ────────────────────────────────────────────────────────────
# 测试 5:layer_range 越界 / 非法参数
# ────────────────────────────────────────────────────────────

def test_layer_range_validation() -> None:
    """layer_range 越界应该立即报错。"""
    config = make_config()
    parallel_ctx = make_parallel_ctx()
    
    # 越界(stop > num_hidden_layers)
    try:
        LlamaPartialModel(
            config, parallel_ctx,
            layer_range=range(0, config.num_hidden_layers + 1),
        )
        assert False, "应该抛 ValueError"
    except ValueError as e:
        assert "越界" in str(e) or "out of range" in str(e).lower()
    
    # 负数 start
    try:
        LlamaPartialModel(
            config, parallel_ctx,
            layer_range=range(-1, 2),
        )
        assert False, "应该抛 ValueError"
    except ValueError:
        pass
    
    # step != 1
    try:
        LlamaPartialModel(
            config, parallel_ctx,
            layer_range=range(0, 4, 2),
        )
        assert False, "应该抛 ValueError(step != 1)"
    except ValueError:
        pass
    
    print("✓ layer_range 校验正确")


# ────────────────────────────────────────────────────────────
# 主入口
# ────────────────────────────────────────────────────────────

def main() -> None:
    setup()
    
    if dist.get_rank() == 0:
        print("=" * 64)
        print("LlamaPartialModel ≡ HF LlamaModel(pp_size=1)等价性测试")
        print("=" * 64)
    
    print()
    print("Test 1: Forward equivalence")
    test_forward_equivalence()
    
    print()
    print("Test 2: Backward equivalence")
    test_backward_equivalence()
    
    print()
    print("Test 3: Parameter naming compatibility")
    test_parameter_naming_compat()
    
    print()
    print("Test 4: PP-aware construction")
    test_pp_aware_construction()
    
    print()
    print("Test 5: layer_range validation")
    test_layer_range_validation()
    
    print()
    print("=" * 64)
    print("✓ 全部通过")
    print("=" * 64)
    
    teardown()


if __name__ == "__main__":
    main()