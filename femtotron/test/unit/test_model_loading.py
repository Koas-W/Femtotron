"""
Femtotron 1.3 模型封装层测试
============================

用法：
    torchrun --nproc_per_node=1 test_model_loading.py --test all
    torchrun --nproc_per_node=2 test_model_loading.py --test all
    torchrun --nproc_per_node=4 test_model_loading.py --test all
"""

import os
import argparse
import tempfile
import shutil
import torch
import torch.nn as nn
import torch.distributed as dist
from collections import OrderedDict

from femtotron.parallel_context import ParallelContext
from femtotron.parallel.tensor_parallel.linear import ColumnParallelLinear, RowParallelLinear
from femtotron.parallel.tensor_parallel.embedding import VocabParallelEmbedding
from femtotron.model.parallel_plan import get_llama_parallel_plan, ParallelPlan, ParallelRule
from femtotron.model.parallelize_model import parallelize_model
from femtotron.model.model_loader import ModelLoader
from femtotron.model.llama import LlamaForTraining, build_llama_model


# ============================================================
# 工具函数
# ============================================================

def init_distributed():
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    torch.cuda.set_device(local_rank)
    return local_rank


def log(msg):
    if dist.get_rank() == 0:
        print(msg)


def assert_close(name, actual, expected, atol=1e-4, rtol=1e-4):
    if actual.shape != expected.shape:
        log(f"  ✗ {name}: shape 不匹配 actual={actual.shape} expected={expected.shape}")
        return False
    if torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol):
        max_diff = (actual.float() - expected.float()).abs().max().item()
        log(f"  ✓ {name} (max diff: {max_diff:.2e})")
        return True
    else:
        max_diff = (actual.float() - expected.float()).abs().max().item()
        mean_diff = (actual.float() - expected.float()).abs().mean().item()
        log(f"  ✗ {name} (max diff: {max_diff:.2e}, mean diff: {mean_diff:.2e})")
        return False


def get_tiny_llama_config():
    from transformers import AutoConfig
    config = AutoConfig.for_model(
        "llama",
        hidden_size=256,
        intermediate_size=512,
        num_attention_heads=8,
        num_key_value_heads=4,
        num_hidden_layers=2,
        max_position_embeddings=128,
        vocab_size=1024,
        rms_norm_eps=1e-5,
        hidden_act="silu",
        tie_word_embeddings=False,
    )
    return config


def get_device():
    return torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")


def save_ref_model_and_broadcast_path(ref_model=None):
    """rank 0 保存模型到临时目录，广播路径给所有 rank。"""
    tmpdir = None
    if dist.get_rank() == 0:
        tmpdir = tempfile.mkdtemp()
        ref_model.save_pretrained(tmpdir)
        path_bytes = tmpdir.encode()
        path_len = torch.tensor([len(path_bytes)], device="cuda")
    else:
        path_len = torch.tensor([0], device="cuda")

    dist.broadcast(path_len, src=0)

    if dist.get_rank() == 0:
        path_tensor = torch.tensor(list(path_bytes), dtype=torch.uint8, device="cuda")
    else:
        path_tensor = torch.zeros(path_len.item(), dtype=torch.uint8, device="cuda")
    dist.broadcast(path_tensor, src=0)

    checkpoint_path = bytes(path_tensor.cpu().tolist()).decode()
    return checkpoint_path, tmpdir


def cleanup_tmpdir(tmpdir):
    if dist.get_rank() == 0 and tmpdir:
        shutil.rmtree(tmpdir, ignore_errors=True)


def build_tp_model_from_ref(config, parallel_ctx, plan, device, ref_model=None):
    """从 ref_model 构建 TP 模型（保存 → 分发 → 加载）。"""
    checkpoint_path, tmpdir = save_ref_model_and_broadcast_path(ref_model)

    with torch.device("meta"):
        tp_model = build_llama_model(config, parallel_ctx)

    loader = ModelLoader(parallel_ctx)
    loader.load_and_distribute(tp_model, checkpoint_path, parallel_plan=plan, device=device)
    tp_model = tp_model.float()

    return tp_model, tmpdir


# ============================================================
# 测试 1: 层替换类型检查
# ============================================================

def test_layer_replacement(parallel_ctx, plan, device):
    log("\n" + "=" * 60)
    log("测试 1: 层替换类型检查")
    log("=" * 60)

    config = get_tiny_llama_config()

    with torch.device("meta"):
        model = build_llama_model(config, parallel_ctx)

    passed = True

    embed = model.model.embed_tokens
    is_correct = isinstance(embed, VocabParallelEmbedding)
    log(f"  {'✓' if is_correct else '✗'} embed_tokens → VocabParallelEmbedding")
    passed &= is_correct

    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        mlp = layer.mlp

        checks = [
            (attn.q_proj, ColumnParallelLinear, "q_proj"),
            (attn.k_proj, ColumnParallelLinear, "k_proj"),
            (attn.v_proj, ColumnParallelLinear, "v_proj"),
            (attn.o_proj, RowParallelLinear,    "o_proj"),
            (mlp.gate_proj, ColumnParallelLinear, "gate_proj"),
            (mlp.up_proj,   ColumnParallelLinear, "up_proj"),
            (mlp.down_proj, RowParallelLinear,    "down_proj"),
        ]

        for module, expected_type, name in checks:
            is_correct = isinstance(module, expected_type)
            if not is_correct:
                log(f"  ✗ layer {i} {name}: 期望 {expected_type.__name__}，"
                    f"实际 {type(module).__name__}")
            passed &= is_correct

        for norm_name in ["input_layernorm", "post_attention_layernorm"]:
            norm = getattr(layer, norm_name)
            is_not_replaced = not isinstance(norm, (ColumnParallelLinear, RowParallelLinear))
            if not is_not_replaced:
                log(f"  ✗ layer {i} {norm_name}: 不应该被替换")
            passed &= is_not_replaced

    if passed:
        log(f"  ✓ 所有层替换类型正确")
    log(f"\n  {'✓ 测试通过' if passed else '✗ 测试失败'}")
    return passed


# ============================================================
# 测试 2: 替换后 shape 检查
# ============================================================

def test_shapes(parallel_ctx, plan, device):
    log("\n" + "=" * 60)
    log("测试 2: 替换后 shape 检查")
    log("=" * 60)

    config = get_tiny_llama_config()
    tp_size = parallel_ctx.tp_size

    with torch.device("meta"):
        model = build_llama_model(config, parallel_ctx)

    passed = True
    H = config.hidden_size
    H_ff = config.intermediate_size
    n_heads = config.num_attention_heads
    n_kv_heads = config.num_key_value_heads
    head_dim = H // n_heads
    V = config.vocab_size

    # === 模型级别的参数（不在 layer 内）===
    
    # embed_tokens
    actual = tuple(model.model.embed_tokens.weight.shape)
    expected = (V // tp_size, H)
    is_correct = actual == expected
    log(f"  {'✓' if is_correct else '✗'} embed_tokens.weight: actual={actual} expected={expected}")
    passed &= is_correct

    # lm_head
    actual = tuple(model.lm_head.weight.shape)
    expected = (V // tp_size, H)
    is_correct = actual == expected
    log(f"  {'✓' if is_correct else '✗'} lm_head.weight: actual={actual} expected={expected}")
    passed &= is_correct

    # === Per-layer 参数 ===
    
    per_layer_expected = {
        "self_attn.q_proj.weight": (H // tp_size, H),
        "self_attn.k_proj.weight": (n_kv_heads * head_dim // tp_size, H),
        "self_attn.v_proj.weight": (n_kv_heads * head_dim // tp_size, H),
        "self_attn.o_proj.weight": (H, H // tp_size),
        "mlp.gate_proj.weight": (H_ff // tp_size, H),
        "mlp.up_proj.weight":   (H_ff // tp_size, H),
        "mlp.down_proj.weight": (H, H_ff // tp_size),
        "input_layernorm.weight": (H,),
        "post_attention_layernorm.weight": (H,),
    }

    layer = model.model.layers[0]

    for name_suffix, expected_shape in per_layer_expected.items():
        parts = name_suffix.split(".")
        module = layer
        for part in parts[:-1]:
            module = getattr(module, part)
        param = getattr(module, parts[-1])

        actual_shape = tuple(param.shape)
        is_correct = actual_shape == expected_shape
        log(f"  {'✓' if is_correct else '✗'} {name_suffix}: "
            f"actual={actual_shape} expected={expected_shape}")
        passed &= is_correct

    log(f"\n  {'✓ 测试通过' if passed else '✗ 测试失败'}")
    return passed


# ============================================================
# 测试 3: 权重加载正确性
# ============================================================

def test_weight_loading(parallel_ctx, plan, device):
    log("\n" + "=" * 60)
    log("测试 3: 权重加载正确性")
    log("=" * 60)

    config = get_tiny_llama_config()
    tp_size = parallel_ctx.tp_size
    tp_rank = parallel_ctx.tp_rank

    ref_state = None
    ref_model = None
    if dist.get_rank() == 0:
        from transformers import AutoModelForCausalLM
        ref_model = AutoModelForCausalLM.from_config(config).float()
        ref_state = {k: v.clone() for k, v in ref_model.state_dict().items()}

    tp_model, tmpdir = build_tp_model_from_ref(config, parallel_ctx, plan, device, ref_model)

    passed = True

    if dist.get_rank() == 0:
        layer0 = tp_model.model.layers[0]

        # ColumnParallel (q_proj): 沿 dim=0
        q_weight = layer0.self_attn.q_proj.weight.data.cpu()
        ref_q = ref_state["model.layers.0.self_attn.q_proj.weight"]
        chunk = ref_q.shape[0] // tp_size
        expected_q = ref_q[tp_rank * chunk : (tp_rank + 1) * chunk, :]
        passed &= assert_close("q_proj weight slice", q_weight, expected_q, atol=1e-6)

        # RowParallel (o_proj): 沿 dim=1
        o_weight = layer0.self_attn.o_proj.weight.data.cpu()
        ref_o = ref_state["model.layers.0.self_attn.o_proj.weight"]
        chunk_in = ref_o.shape[1] // tp_size
        expected_o = ref_o[:, tp_rank * chunk_in : (tp_rank + 1) * chunk_in]
        passed &= assert_close("o_proj weight slice", o_weight, expected_o, atol=1e-6)

        # VocabParallelEmbedding
        embed_weight = tp_model.model.embed_tokens.weight.data.cpu()
        ref_embed = ref_state["model.embed_tokens.weight"]
        chunk_v = ref_embed.shape[0] // tp_size
        expected_embed = ref_embed[tp_rank * chunk_v : (tp_rank + 1) * chunk_v, :]
        passed &= assert_close("embedding weight slice", embed_weight, expected_embed, atol=1e-6)

        # RMSNorm: 完整副本
        norm_weight = layer0.input_layernorm.weight.data.cpu()
        ref_norm = ref_state["model.layers.0.input_layernorm.weight"]
        passed &= assert_close("layernorm weight (full copy)", norm_weight, ref_norm, atol=1e-6)

        # gate_proj (Column)
        gate_weight = layer0.mlp.gate_proj.weight.data.cpu()
        ref_gate = ref_state["model.layers.0.mlp.gate_proj.weight"]
        chunk_ff = ref_gate.shape[0] // tp_size
        expected_gate = ref_gate[tp_rank * chunk_ff : (tp_rank + 1) * chunk_ff, :]
        passed &= assert_close("gate_proj weight slice", gate_weight, expected_gate, atol=1e-6)

        # down_proj (Row)
        down_weight = layer0.mlp.down_proj.weight.data.cpu()
        ref_down = ref_state["model.layers.0.mlp.down_proj.weight"]
        chunk_ff_in = ref_down.shape[1] // tp_size
        expected_down = ref_down[:, tp_rank * chunk_ff_in : (tp_rank + 1) * chunk_ff_in]
        passed &= assert_close("down_proj weight slice", down_weight, expected_down, atol=1e-6)

    passed_tensor = torch.tensor([1 if passed else 0], device="cuda")
    dist.broadcast(passed_tensor, src=0)
    passed = passed_tensor.item() == 1

    cleanup_tmpdir(tmpdir)
    log(f"\n  {'✓ 测试通过' if passed else '✗ 测试失败'}")
    return passed


# ============================================================
# 测试 4: Forward 输出一致性
# ============================================================

def test_forward_consistency(parallel_ctx, plan, device):
    log("\n" + "=" * 60)
    log("测试 4: Forward 输出一致性")
    log("=" * 60)

    config = get_tiny_llama_config()

    torch.manual_seed(42)
    input_ids = torch.randint(0, config.vocab_size, (2, 32), device=device)

    ref_model = None
    if dist.get_rank() == 0:
        from transformers import AutoModelForCausalLM
        ref_model = AutoModelForCausalLM.from_config(config).cuda().float()
        ref_model.eval()
        with torch.no_grad():
            ref_outputs = ref_model(input_ids=input_ids, labels=input_ids)
            ref_logits = ref_outputs.logits.clone()
            ref_loss = ref_outputs.loss.clone()
        log(f"  Reference logits shape: {ref_logits.shape}")
        log(f"  Reference loss: {ref_loss.item():.6f}")

    tp_model, tmpdir = build_tp_model_from_ref(config, parallel_ctx, plan, device, ref_model)
    tp_model.eval()

    with torch.no_grad():
        tp_outputs = tp_model(input_ids=input_ids, labels=input_ids)
        tp_logits = tp_outputs["logits"]
        tp_loss = tp_outputs["loss"]

    log(f"  TP logits shape: {tp_logits.shape}")
    log(f"  TP loss: {tp_loss.item():.6f}")

    if dist.get_rank() == 0:
        ref_logits_cuda = ref_logits.cuda()
        ref_loss_scalar = ref_loss.cuda()
    else:
        ref_logits_cuda = torch.zeros_like(tp_logits)
        ref_loss_scalar = torch.zeros(1, device="cuda")
    dist.broadcast(ref_logits_cuda, src=0)
    dist.broadcast(ref_loss_scalar, src=0)

    passed = True
    passed &= assert_close("forward logits", tp_logits, ref_logits_cuda)
    passed &= assert_close("forward loss",
                           tp_loss.view(1),
                           ref_loss_scalar.view(1))

    cleanup_tmpdir(tmpdir)
    log(f"\n  {'✓ 测试通过' if passed else '✗ 测试失败'}")
    return passed


# ============================================================
# 测试 5: Backward 梯度一致性
# ============================================================

def test_backward_consistency(parallel_ctx, plan, device):
    log("\n" + "=" * 60)
    log("测试 5: Backward 梯度一致性")
    log("=" * 60)

    config = get_tiny_llama_config()
    tp_size = parallel_ctx.tp_size
    tp_rank = parallel_ctx.tp_rank

    torch.manual_seed(42)
    input_ids = torch.randint(0, config.vocab_size, (2, 32), device=device)

    ref_grads = {}
    ref_model = None
    if dist.get_rank() == 0:
        from transformers import AutoModelForCausalLM
        ref_model = AutoModelForCausalLM.from_config(config).cuda().float()
        ref_model.train()
        ref_outputs = ref_model(input_ids=input_ids, labels=input_ids)
        ref_outputs.loss.backward()
        for name, param in ref_model.named_parameters():
            if param.grad is not None:
                ref_grads[name] = param.grad.clone()

    tp_model, tmpdir = build_tp_model_from_ref(config, parallel_ctx, plan, device, ref_model)
    tp_model.train()

    tp_outputs = tp_model(input_ids=input_ids, labels=input_ids)
    tp_outputs["loss"].backward()

    passed = True

    if dist.get_rank() == 0:
        layer0 = tp_model.model.layers[0]

        # q_proj (Column): 沿 dim=0
        tp_grad = layer0.self_attn.q_proj.weight.grad
        ref_grad = ref_grads["model.layers.0.self_attn.q_proj.weight"]
        chunk = ref_grad.shape[0] // tp_size
        expected = ref_grad[tp_rank * chunk : (tp_rank + 1) * chunk, :]
        passed &= assert_close("q_proj grad", tp_grad, expected)

        # o_proj (Row): 沿 dim=1
        tp_grad = layer0.self_attn.o_proj.weight.grad
        ref_grad = ref_grads["model.layers.0.self_attn.o_proj.weight"]
        chunk_in = ref_grad.shape[1] // tp_size
        expected = ref_grad[:, tp_rank * chunk_in : (tp_rank + 1) * chunk_in]
        passed &= assert_close("o_proj grad", tp_grad, expected)

        # gate_proj (Column)
        tp_grad = layer0.mlp.gate_proj.weight.grad
        ref_grad = ref_grads["model.layers.0.mlp.gate_proj.weight"]
        chunk_ff = ref_grad.shape[0] // tp_size
        expected = ref_grad[tp_rank * chunk_ff : (tp_rank + 1) * chunk_ff, :]
        passed &= assert_close("gate_proj grad", tp_grad, expected)

        # down_proj (Row)
        tp_grad = layer0.mlp.down_proj.weight.grad
        ref_grad = ref_grads["model.layers.0.mlp.down_proj.weight"]
        chunk_ff_in = ref_grad.shape[1] // tp_size
        expected = ref_grad[:, tp_rank * chunk_ff_in : (tp_rank + 1) * chunk_ff_in]
        passed &= assert_close("down_proj grad", tp_grad, expected)

        # RMSNorm: 完整梯度
        tp_grad = layer0.input_layernorm.weight.grad
        ref_grad = ref_grads["model.layers.0.input_layernorm.weight"]
        passed &= assert_close("layernorm grad", tp_grad, ref_grad)

        # embedding
        tp_grad = tp_model.model.embed_tokens.weight.grad
        ref_grad = ref_grads["model.embed_tokens.weight"]
        chunk_v = ref_grad.shape[0] // tp_size
        expected = ref_grad[tp_rank * chunk_v : (tp_rank + 1) * chunk_v, :]
        passed &= assert_close("embedding grad", tp_grad, expected)

    passed_tensor = torch.tensor([1 if passed else 0], device="cuda")
    dist.broadcast(passed_tensor, src=0)
    passed = passed_tensor.item() == 1

    cleanup_tmpdir(tmpdir)
    log(f"\n  {'✓ 测试通过' if passed else '✗ 测试失败'}")
    return passed


# ============================================================
# 测试 6: TP=1 退化验证
# ============================================================

def test_tp1_degeneracy(parallel_ctx, plan, device):
    log("\n" + "=" * 60)
    log("测试 6: TP=1 退化验证")
    log("=" * 60)

    if parallel_ctx.tp_size != 1:
        log("  跳过（当前 tp_size != 1，需要 nproc=1 运行）")
        return True

    config = get_tiny_llama_config()

    torch.manual_seed(42)
    input_ids = torch.randint(0, config.vocab_size, (2, 32), device=device)

    from transformers import AutoModelForCausalLM
    ref_model = AutoModelForCausalLM.from_config(config).to(device).float()
    ref_model.eval()
    with torch.no_grad():
        ref_out = ref_model(input_ids=input_ids, labels=input_ids)

    tp_model, tmpdir = build_tp_model_from_ref(config, parallel_ctx, plan, device, ref_model)
    tp_model.eval()

    with torch.no_grad():
        tp_out = tp_model(input_ids=input_ids, labels=input_ids)

    passed = True
    passed &= assert_close("tp=1 logits", tp_out["logits"], ref_out.logits)
    passed &= assert_close("tp=1 loss", tp_out["loss"].view(1), ref_out.loss.view(1))

    cleanup_tmpdir(tmpdir)
    log(f"\n  {'✓ 测试通过' if passed else '✗ 测试失败'}")
    return passed


# ============================================================
# 测试 7: 多步训练 loss 轨迹
# ============================================================

def test_multistep_training(parallel_ctx, plan, device):
    log("\n" + "=" * 60)
    log("测试 7: 多步训练 loss 轨迹")
    log("=" * 60)

    config = get_tiny_llama_config()

    ref_model = None
    if dist.get_rank() == 0:
        from transformers import AutoModelForCausalLM
        ref_model = AutoModelForCausalLM.from_config(config).float()

    tp_model, tmpdir = build_tp_model_from_ref(config, parallel_ctx, plan, device, ref_model)
    tp_model.train()

    optimizer = torch.optim.AdamW(tp_model.parameters(), lr=1e-3)

    losses = []
    num_steps = 10

    log(f"  训练 {num_steps} 步...")
    for step in range(num_steps):
        torch.manual_seed(step + 100)
        input_ids = torch.randint(0, config.vocab_size, (4, 32), device=device)

        outputs = tp_model(input_ids=input_ids, labels=input_ids)
        loss = outputs["loss"]

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_val = loss.item()
        losses.append(loss_val)
        log(f"    Step {step+1}: loss = {loss_val:.4f}")

    passed = True

    has_nan = any(not torch.isfinite(torch.tensor(l)) for l in losses)
    log(f"  {'✓' if not has_nan else '✗'} 无 NaN/Inf")
    passed &= not has_nan

    expected_init = torch.tensor(config.vocab_size, dtype=torch.float).log().item()
    init_ok = abs(losses[0] - expected_init) < 2.0
    log(f"  {'✓' if init_ok else '✗'} 初始 loss {losses[0]:.4f} ≈ ln({config.vocab_size}) = {expected_init:.4f}")
    passed &= init_ok

    avg_first_3 = sum(losses[:3]) / 3
    avg_last_3 = sum(losses[-3:]) / 3
    is_decreasing = avg_last_3 < avg_first_3
    log(f"  {'✓' if is_decreasing else '✗'} Loss 下降: {avg_first_3:.4f} → {avg_last_3:.4f}")
    passed &= is_decreasing

    cleanup_tmpdir(tmpdir)
    log(f"\n  {'✓ 测试通过' if passed else '✗ 测试失败'}")
    return passed


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="all",
                        choices=["all", "replace", "shape", "load", "forward",
                                 "backward", "tp1", "multistep"])
    args = parser.parse_args()

    local_rank = init_distributed()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = get_device()

    if rank == 0:
        print("=" * 60)
        print("Femtotron 1.3 模型封装层测试")
        print(f"World size: {world_size}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("=" * 60)

    parallel_ctx = ParallelContext(OrderedDict([("tp", world_size)]))
    plan = get_llama_parallel_plan()

    if rank == 0:
        print(f"并行配置: TP={parallel_ctx.tp_size}")

    dist.barrier()

    tests = {
        "replace":   test_layer_replacement,
        "shape":     test_shapes,
        "load":      test_weight_loading,
        "forward":   test_forward_consistency,
        "backward":  test_backward_consistency,
        "tp1":       test_tp1_degeneracy,
        "multistep": test_multistep_training,
    }

    all_passed = True

    if args.test == "all":
        for name, test_fn in tests.items():
            all_passed &= test_fn(parallel_ctx, plan, device)
            dist.barrier()
    else:
        all_passed = tests[args.test](parallel_ctx, plan, device)

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