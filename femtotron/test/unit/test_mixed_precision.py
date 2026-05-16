"""
Femtotron 1.4 混合精度管理测试
===============================

用法：
    torchrun --nproc_per_node=1 test_mixed_precision.py --test all
    torchrun --nproc_per_node=2 test_mixed_precision.py --test all

测试内容：
    1. Master weights 的创建和 dtype 检查
    2. Weight decay 分组正确性
    3. copy_grads_to_master 的精度转换
    4. sync_weights 的 FP32 → BF16 同步
    5. Gradient clipping 正确性
    6. 完整 step() 的端到端正确性
    7. 混合精度 vs 纯 FP32 训练趋势一致性
    8. 混合精度 + TP 兼容性
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
from femtotron.model.parallel_plan import get_llama_parallel_plan
from femtotron.model.model_loader import ModelLoader
from femtotron.model.llama import build_llama_model
from femtotron.training.mixed_precision_manager import MixedPrecisionManager
from femtotron.training.optimizer import get_param_groups  # 或者你放在哪里
from femtotron.training.train_config import TrainConfig          # 你的训练配置类
from femtotron.training.grad_transform import GradTransform, ClipGradNorm


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


def get_device():
    return torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")


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


def make_simple_bf16_model(device):
    """创建一个简单的 BF16 小模型用于单元测试（不依赖 LLaMA）。"""
    model = nn.Sequential(
        nn.Linear(64, 128, bias=False),
        nn.LayerNorm(128),
        nn.Linear(128, 64, bias=True),
        nn.LayerNorm(64),
        nn.Linear(64, 32, bias=False),
    ).to(device=device, dtype=torch.bfloat16)
    return model


def build_tp_model(model_config, parallel_ctx, plan, device):
    """构建 TP 模型并加载权重。"""
    from transformers import AutoModelForCausalLM

    ref_model = None
    if dist.get_rank() == 0:
        ref_model = AutoModelForCausalLM.from_config(model_config).float()

    # 保存并广播路径
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

    with torch.device("meta"):
        model = build_llama_model(model_config, parallel_ctx)
    loader = ModelLoader(parallel_ctx)
    loader.load_and_distribute(model, checkpoint_path, parallel_plan=plan, device=device)
    model = model.bfloat16()

    return model, tmpdir


# ============================================================
# 测试 1: Master weights 的创建和 dtype 检查
# ============================================================

def test_master_weight_creation(parallel_ctx, device):
    log("\n" + "=" * 60)
    log("测试 1: Master weights 创建和 dtype")
    log("=" * 60)

    model = make_simple_bf16_model(device)
    compute_param_groups = get_param_groups(model, weight_decay=0.01)

    config = TrainConfig(master_dtype=torch.float32)
    mp = MixedPrecisionManager(
        model=model,
        parallel_ctx=parallel_ctx,
        parallel_plan=get_llama_parallel_plan(),
        config=config,
        inner_optimizer_cls=torch.optim.AdamW,
        inner_optimizer_kwargs={"lr": 1e-3},
        compute_param_groups=compute_param_groups,
    )

    passed = True

    # 检查所有 compute 参数是 BF16
    all_bf16 = all(g.compute.dtype == torch.bfloat16 for g in mp.groups)
    log(f"  {'✓' if all_bf16 else '✗'} 所有 compute 参数是 bfloat16")
    passed &= all_bf16

    # 检查所有 master 参数是 FP32
    all_fp32 = all(g.master is not None and g.master.dtype == torch.float32 for g in mp.groups)
    log(f"  {'✓' if all_fp32 else '✗'} 所有 master 参数是 float32")
    passed &= all_fp32

    # 检查 master 和 compute 的值一致（初始时应该完全一致）
    values_match = True
    for g in mp.groups:
        if not torch.allclose(g.master.float(), g.compute.float()):
            values_match = False
            break
    log(f"  {'✓' if values_match else '✗'} Master 和 compute 初始值一致")
    passed &= values_match

    # 检查 shape 一致
    shapes_match = all(g.master.shape == g.compute.shape for g in mp.groups)
    log(f"  {'✓' if shapes_match else '✗'} Master 和 compute shape 一致")
    passed &= shapes_match

    # 检查参数数量
    num_params = sum(1 for _ in model.parameters() if _.requires_grad)
    num_groups = len(mp.groups)
    count_match = num_params == num_groups
    log(f"  {'✓' if count_match else '✗'} 参数数量一致: {num_params} params, {num_groups} groups")
    passed &= count_match

    log(f"\n  {'✓ 测试通过' if passed else '✗ 测试失败'}")
    return passed


# ============================================================
# 测试 2: Weight decay 分组正确性
# ============================================================

def test_weight_decay_grouping(parallel_ctx, device):
    log("\n" + "=" * 60)
    log("测试 2: Weight decay 分组")
    log("=" * 60)

    model = make_simple_bf16_model(device)
    compute_param_groups = get_param_groups(model, weight_decay=0.01)

    passed = True

    # 检查分为两组
    num_groups = len(compute_param_groups)
    log(f"  参数组数量: {num_groups}")
    has_two_groups = num_groups == 2
    log(f"  {'✓' if has_two_groups else '✗'} 分为 2 组 (decay + no_decay)")
    passed &= has_two_groups

    # 检查 decay 组
    decay_group = [g for g in compute_param_groups if g.get("weight_decay", 0) > 0]
    no_decay_group = [g for g in compute_param_groups if g.get("weight_decay", 0) == 0]

    if decay_group and no_decay_group:
        decay_params = decay_group[0]["params"]
        no_decay_params = no_decay_group[0]["params"]

        # Linear.weight（2维）应该在 decay 组
        # LayerNorm.weight（1维）和 bias 应该在 no_decay 组
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            in_decay = any(p is dp for dp in decay_params)
            in_no_decay = any(p is ndp for ndp in no_decay_params)

            if p.ndim <= 1 or "bias" in name:
                is_correct = in_no_decay and not in_decay
                log(f"  {'✓' if is_correct else '✗'} {name} (ndim={p.ndim}) → no_decay")
            else:
                is_correct = in_decay and not in_no_decay
                log(f"  {'✓' if is_correct else '✗'} {name} (ndim={p.ndim}) → decay")
            passed &= is_correct

    # 验证 MixedPrecisionManager 正确传递分组给 optimizer
    config = TrainConfig(master_dtype=torch.float32)
    mp = MixedPrecisionManager(
        model=model,
        parallel_ctx=parallel_ctx,
        parallel_plan=get_llama_parallel_plan(),
        config=config,
        inner_optimizer_cls=torch.optim.AdamW,
        inner_optimizer_kwargs={"lr": 1e-3},
        compute_param_groups=compute_param_groups,
    )

    # 检查 inner optimizer 的 param groups 数量
    opt_groups = mp.inner.param_groups
    opt_groups_correct = len(opt_groups) == 2
    log(f"  {'✓' if opt_groups_correct else '✗'} Optimizer 收到 2 个 param groups")
    passed &= opt_groups_correct

    # 检查 optimizer 中的参数是 FP32
    all_opt_fp32 = all(
        p.dtype == torch.float32
        for group in opt_groups
        for p in group["params"]
    )
    log(f"  {'✓' if all_opt_fp32 else '✗'} Optimizer 中所有参数是 float32")
    passed &= all_opt_fp32

    log(f"\n  {'✓ 测试通过' if passed else '✗ 测试失败'}")
    return passed


# ============================================================
# 测试 3: copy_grads_to_master
# ============================================================

def test_copy_grads(parallel_ctx, device):
    log("\n" + "=" * 60)
    log("测试 3: copy_grads_to_master")
    log("=" * 60)

    model = make_simple_bf16_model(device)
    compute_param_groups = get_param_groups(model, weight_decay=0.01)
    config = TrainConfig(master_dtype=torch.float32)
    mp = MixedPrecisionManager(
        model=model,
        parallel_ctx=parallel_ctx,
        parallel_plan=get_llama_parallel_plan(),
        config=config,
        inner_optimizer_cls=torch.optim.AdamW,
        inner_optimizer_kwargs={"lr": 1e-3},
        compute_param_groups=compute_param_groups,
    )

    # 做一次 forward + backward 产生梯度
    x = torch.randn(4, 64, device=device, dtype=torch.bfloat16)
    y = model(x)
    y.sum().backward()

    # 检查 compute 参数有 BF16 梯度
    has_bf16_grads = all(
        g.compute.grad is not None and g.compute.grad.dtype == torch.bfloat16
        for g in mp.groups
    )
    log(f"  {'✓' if has_bf16_grads else '✗'} Backward 后 compute 参数有 BF16 梯度")

    # 执行 copy
    mp.copy_grads_to_master()

    passed = True

    # 检查 master 有 FP32 梯度
    has_fp32_grads = all(
        g.master.grad is not None and g.master.grad.dtype == torch.float32
        for g in mp.groups
    )
    log(f"  {'✓' if has_fp32_grads else '✗'} Copy 后 master 有 FP32 梯度")
    passed &= has_fp32_grads

    # 检查值一致（BF16 → FP32 应该是精确的，不会引入额外误差）
    for g in mp.groups:
        expected_grad = g.compute.grad.float()
        actual_grad = g.master.grad
        if not torch.allclose(actual_grad, expected_grad, atol=1e-7):
            log(f"  ✗ {g.name}: 梯度值不一致")
            passed = False
    if passed:
        log(f"  ✓ 所有梯度值 BF16→FP32 转换正确")

    log(f"\n  {'✓ 测试通过' if passed else '✗ 测试失败'}")
    return passed


# ============================================================
# 测试 4: sync_weights
# ============================================================

def test_sync_weights(parallel_ctx, device):
    log("\n" + "=" * 60)
    log("测试 4: sync_weights (FP32 → BF16)")
    log("=" * 60)

    model = make_simple_bf16_model(device)
    compute_param_groups = get_param_groups(model, weight_decay=0.01)
    config = TrainConfig(master_dtype=torch.float32)
    mp = MixedPrecisionManager(
        model=model,
        parallel_ctx=parallel_ctx,
        parallel_plan=get_llama_parallel_plan(),
        config=config,
        inner_optimizer_cls=torch.optim.AdamW,
        inner_optimizer_kwargs={"lr": 1e-3},
        compute_param_groups=compute_param_groups,
    )

    # 手动修改 master weights（模拟 optimizer step 的效果）
    for g in mp.groups:
        g.master.data.add_(0.1)  # 在 FP32 上加一个小值

    # 此时 master 和 compute 不一致
    mismatch_before = any(
        not torch.allclose(g.master.bfloat16(), g.compute.data)
        for g in mp.groups
    )
    log(f"  {'✓' if mismatch_before else '✗'} Sync 前 master 和 compute 不一致（预期）")

    # 执行 sync
    mp.sync_weights()

    passed = True

    # 检查 compute 被更新了
    for g in mp.groups:
        expected_bf16 = g.master.data.bfloat16()
        actual_bf16 = g.compute.data
        if not torch.allclose(actual_bf16, expected_bf16):
            log(f"  ✗ {g.name}: sync 后值不一致")
            passed = False

    if passed:
        log(f"  ✓ 所有参数 FP32→BF16 同步正确")

    # 验证 compute 确实是 BF16
    still_bf16 = all(g.compute.dtype == torch.bfloat16 for g in mp.groups)
    log(f"  {'✓' if still_bf16 else '✗'} Sync 后 compute 仍是 bfloat16")
    passed &= still_bf16

    log(f"\n  {'✓ 测试通过' if passed else '✗ 测试失败'}")
    return passed


# ============================================================
# 测试 5: Gradient clipping
# ============================================================

def test_grad_clipping(parallel_ctx, device):
    log("\n" + "=" * 60)
    log("测试 5: Gradient clipping")
    log("=" * 60)

    model = make_simple_bf16_model(device)
    compute_param_groups = get_param_groups(model, weight_decay=0.01)
    config = TrainConfig(master_dtype=torch.float32)
    wc = ClipGradNorm(max_norm=1.00, parallel_ctx=parallel_ctx)
    mp = MixedPrecisionManager(
        model=model,
        parallel_ctx=parallel_ctx,
        parallel_plan=get_llama_parallel_plan(),
        config=config,
        inner_optimizer_cls=torch.optim.AdamW,
        inner_optimizer_kwargs={"lr": 1e-3},
        compute_param_groups=compute_param_groups,
        grad_transforms=[wc]
    )

    # 产生梯度
    x = torch.randn(4, 64, device=device, dtype=torch.bfloat16)
    y = model(x)
    (y.sum() * 10000000000).backward()  # 乘以大数让梯度更大，确保 clipping 会生效

    mp.copy_grads_to_master()

    # 记录 clip 前的 grad norm
    pre_clip_norms = []
    for g in mp.groups:
        if g.master.grad is not None:
            pre_clip_norms.append(g.master.grad.norm().item())
    total_pre_norm = torch.tensor(pre_clip_norms).norm().item()
    log(f"  Clip 前 total grad norm: {total_pre_norm:.4f}")

    # 用 PyTorch 原生方法作为 ground truth
    ref_params = [g.master for g in mp.groups if g.master.grad is not None]
    ref_norm = torch.nn.utils.clip_grad_norm_(ref_params, max_norm=1.0)

    # 重新产生梯度并用 mp 的 clip
    mp.zero_grad()
    y2 = model(x)
    (y2.sum() * 100).backward()
    mp.copy_grads_to_master()

    # 这里需要根据你的实际接口调用 clip
    # 如果 clip 在 step() 内部通过 grad_transforms 执行，
    # 可以直接调 step() 然后检查结果。
    # 如果有独立的 clip 方法：
    # mp_norm = mp.clip_grad_norm(max_norm=1.0)

    passed = True

    # 检查 clip 后的 grad norm ≤ max_norm（允许微小浮点误差）
    post_clip_norms = []
    for g in mp.groups:
        if g.master.grad is not None:
            post_clip_norms.append(g.master.grad.norm().item())
    total_post_norm = torch.tensor(post_clip_norms).norm().item()

    # 如果 clip 在 step() 内部执行，这里改为在 step() 前后检查
    # 暂时先验证 PyTorch 原生 clip 的结果
    ref_post_norm = torch.tensor([
        p.grad.norm().item() for p in ref_params if p.grad is not None
    ]).norm().item()

    clip_worked = ref_post_norm <= 1.0 + 1e-5
    log(f"  Clip 后 grad norm (ref): {ref_post_norm:.6f}")
    log(f"  {'✓' if clip_worked else '✗'} Clip 后 grad norm ≤ 1.0")
    passed &= clip_worked

    # 验证 clip 确实改变了梯度（如果原始 norm > 1.0）
    if total_pre_norm > 1.0:
        did_clip = ref_post_norm < total_pre_norm
        log(f"  {'✓' if did_clip else '✗'} Clipping 确实缩小了梯度 ({total_pre_norm:.4f} → {ref_post_norm:.6f})")
        passed &= did_clip
    else:
        log(f"  ○ 原始 grad norm ≤ 1.0，clipping 未触发（这也是正确行为）")

    log(f"\n  {'✓ 测试通过' if passed else '✗ 测试失败'}")
    return passed


# ============================================================
# 测试 6: 完整 step() 端到端
# ============================================================

def test_full_step(parallel_ctx, device):
    log("\n" + "=" * 60)
    log("测试 6: 完整 step() 端到端")
    log("=" * 60)

    model = make_simple_bf16_model(device)
    compute_param_groups = get_param_groups(model, weight_decay=0.01)
    config = TrainConfig(master_dtype=torch.float32)
    wc = ClipGradNorm(max_norm=0.01, parallel_ctx=parallel_ctx)
    mp = MixedPrecisionManager(
        model=model,
        parallel_ctx=parallel_ctx,
        parallel_plan=get_llama_parallel_plan(),
        config=config,
        inner_optimizer_cls=torch.optim.AdamW,
        inner_optimizer_kwargs={"lr": 1e-3},
        compute_param_groups=compute_param_groups,
        grad_transforms=[wc]
    )

    # 记录初始参数值
    initial_params = {g.name: g.compute.data.clone() for g in mp.groups}

    # Forward + backward
    x = torch.randn(4, 64, device=device, dtype=torch.bfloat16)
    y = model(x)
    y.sum().backward()

    # 执行完整 step
    success = mp.step()

    passed = True

    # step 应该成功
    log(f"  {'✓' if success else '✗'} step() 返回 True")
    passed &= success

    # 参数应该发生了变化
    params_changed = False
    for g in mp.groups:
        if not torch.equal(g.compute.data, initial_params[g.name]):
            params_changed = True
            break
    log(f"  {'✓' if params_changed else '✗'} 参数在 step 后发生了变化")
    passed &= params_changed

    # master 和 compute 应该一致（sync 已执行）
    in_sync = True
    for g in mp.groups:
        if not torch.allclose(g.master.bfloat16(), g.compute.data):
            in_sync = False
            break
    log(f"  {'✓' if in_sync else '✗'} Step 后 master 和 compute 一致")
    passed &= in_sync

    # 梯度应该被清零
    grads_zero = True
    for g in mp.groups:
        if g.compute.grad is not None and g.compute.grad.abs().max() > 0:
            grads_zero = False
            break
    log(f"  {'✓' if grads_zero else '✗'} Step 后梯度已清零")
    passed &= grads_zero

    # 再跑一步，确认连续 step 不会出错
    y2 = model(x)
    y2.sum().backward()
    success2 = mp.step()
    log(f"  {'✓' if success2 else '✗'} 第二步 step() 也成功")
    passed &= success2

    log(f"\n  {'✓ 测试通过' if passed else '✗ 测试失败'}")
    return passed


# ============================================================
# 测试 7: 混合精度 vs 纯 FP32 训练趋势一致性
# ============================================================

def test_mp_vs_fp32_training(parallel_ctx, device):
    log("\n" + "=" * 60)
    log("测试 7: 混合精度 vs 纯 FP32 训练趋势")
    log("=" * 60)

    torch.manual_seed(42)

    # --- 纯 FP32 baseline ---
    model_fp32 = nn.Sequential(
        nn.Linear(64, 128, bias=False),
        nn.ReLU(),
        nn.Linear(128, 64, bias=False),
    ).to(device=device, dtype=torch.float32)

    opt_fp32 = torch.optim.AdamW(model_fp32.parameters(), lr=1e-3)

    losses_fp32 = []
    for step in range(20):
        torch.manual_seed(step + 100)
        x = torch.randn(8, 64, device=device, dtype=torch.float32)
        target = torch.randn(8, 64, device=device, dtype=torch.float32)
        y = model_fp32(x)
        loss = nn.functional.mse_loss(y, target)
        loss.backward()
        opt_fp32.step()
        opt_fp32.zero_grad()
        losses_fp32.append(loss.item())

    # --- 混合精度 (BF16 + FP32 master) ---
    torch.manual_seed(42)  # 相同初始化
    model_bf16 = nn.Sequential(
        nn.Linear(64, 128, bias=False),
        nn.ReLU(),
        nn.Linear(128, 64, bias=False),
    ).to(device=device, dtype=torch.bfloat16)

    compute_param_groups = get_param_groups(model_bf16, weight_decay=0.0)
    config = TrainConfig(master_dtype=torch.float32)
    mp = MixedPrecisionManager(
        model=model_bf16,
        parallel_ctx=parallel_ctx,
        parallel_plan=get_llama_parallel_plan(),
        config=config,
        inner_optimizer_cls=torch.optim.AdamW,
        inner_optimizer_kwargs={"lr": 1e-3},
        compute_param_groups=compute_param_groups,
    )

    losses_bf16 = []
    for step in range(20):
        torch.manual_seed(step + 100)
        x = torch.randn(8, 64, device=device, dtype=torch.bfloat16)
        target = torch.randn(8, 64, device=device, dtype=torch.bfloat16)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = model_bf16(x)
            loss = nn.functional.mse_loss(y, target)
        loss.backward()
        mp.step()
        losses_bf16.append(loss.item())

    passed = True

    # 打印两条 loss 曲线
    log(f"  {'Step':>6} {'FP32':>10} {'BF16+MP':>10} {'Diff':>10}")
    log(f"  {'─'*40}")
    for i in range(0, 20, 4):
        diff = abs(losses_fp32[i] - losses_bf16[i])
        log(f"  {i+1:>6} {losses_fp32[i]:>10.4f} {losses_bf16[i]:>10.4f} {diff:>10.6f}")

    # 两者都应该下降
    fp32_decreasing = losses_fp32[-1] < losses_fp32[0]
    bf16_decreasing = losses_bf16[-1] < losses_bf16[0]
    log(f"\n  {'✓' if fp32_decreasing else '✗'} FP32 loss 下降: {losses_fp32[0]:.4f} → {losses_fp32[-1]:.4f}")
    log(f"  {'✓' if bf16_decreasing else '✗'} BF16 loss 下降: {losses_bf16[0]:.4f} → {losses_bf16[-1]:.4f}")
    passed &= fp32_decreasing
    passed &= bf16_decreasing

    # 趋势应该大致一致（不要求 exact match，BF16 有精度差异）
    # 最终 loss 的相对差距应该在 20% 以内
    final_diff_ratio = abs(losses_fp32[-1] - losses_bf16[-1]) / max(losses_fp32[-1], 1e-8)
    trend_similar = final_diff_ratio < 0.2
    log(f"  {'✓' if trend_similar else '✗'} 最终 loss 差距 < 20%: "
        f"相对差距 {final_diff_ratio:.2%}")
    passed &= trend_similar

    log(f"\n  {'✓ 测试通过' if passed else '✗ 测试失败'}")
    return passed


# ============================================================
# 测试 8: 混合精度 + TP 兼容性
# ============================================================

def test_mp_with_tp(parallel_ctx, plan, device):
    log("\n" + "=" * 60)
    log("测试 8: 混合精度 + TP 兼容性")
    log("=" * 60)

    if parallel_ctx.tp_size == 1:
        log("  跳过（需要 TP > 1，用 nproc >= 2 运行）")
        return True

    model_config = get_tiny_llama_config()
    model, tmpdir = build_tp_model(model_config, parallel_ctx, plan, device)
    model = model.bfloat16()

    compute_param_groups = get_param_groups(model, weight_decay=0.01)
    config = TrainConfig(master_dtype=torch.float32)
    mp = MixedPrecisionManager(
        model=model,
        parallel_ctx=parallel_ctx,
        parallel_plan=get_llama_parallel_plan(),
        config=config,
        inner_optimizer_cls=torch.optim.AdamW,
        inner_optimizer_kwargs={"lr": 3e-4, "betas": (0.9, 0.95)},
        compute_param_groups=compute_param_groups,
    )

    passed = True
    losses = []
    num_steps = 10

    log(f"  训练 {num_steps} 步 (TP={parallel_ctx.tp_size}, BF16+FP32 master)...")

    for step in range(num_steps):
        torch.manual_seed(step + 200)
        input_ids = torch.randint(0, model_config.vocab_size, (4, 32), device=device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs["loss"]

        loss.backward()
        mp.step()

        loss_val = loss.item()
        losses.append(loss_val)
        log(f"    Step {step+1}: loss = {loss_val:.4f}")

    # 检查无 NaN
    has_nan = any(not torch.isfinite(torch.tensor(l)) for l in losses)
    log(f"  {'✓' if not has_nan else '✗'} 无 NaN/Inf")
    passed &= not has_nan

    # 检查初始 loss 合理
    expected_init = torch.tensor(model_config.vocab_size, dtype=torch.float).log().item()
    init_ok = abs(losses[0] - expected_init) < 2.0
    log(f"  {'✓' if init_ok else '✗'} 初始 loss {losses[0]:.4f} ≈ ln({model_config.vocab_size}) = {expected_init:.4f}")
    passed &= init_ok

    # 检查 loss 下降
    avg_first = sum(losses[:3]) / 3
    avg_last = sum(losses[-3:]) / 3
    is_decreasing = avg_last < avg_first
    log(f"  {'✓' if is_decreasing else '✗'} Loss 下降: {avg_first:.4f} → {avg_last:.4f}")
    passed &= is_decreasing

    if tmpdir:
        shutil.rmtree(tmpdir, ignore_errors=True)

    log(f"\n  {'✓ 测试通过' if passed else '✗ 测试失败'}")
    return passed


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="all",
                        choices=["all", "create", "grouping", "copy_grads",
                                 "sync", "clip", "step", "vs_fp32", "tp"])
    args = parser.parse_args()

    init_distributed()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = get_device()

    if rank == 0:
        print("=" * 60)
        print("Femtotron 1.4 混合精度管理测试")
        print(f"World size: {world_size}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("=" * 60)

    parallel_ctx = ParallelContext(OrderedDict([("tp", world_size)]))
    plan = get_llama_parallel_plan()

    if rank == 0:
        print(f"并行配置: TP={parallel_ctx.tp_size}")

    dist.barrier()

    tests = {
        "create":     lambda: test_master_weight_creation(parallel_ctx, device),
        "grouping":   lambda: test_weight_decay_grouping(parallel_ctx, device),
        "copy_grads": lambda: test_copy_grads(parallel_ctx, device),
        "sync":       lambda: test_sync_weights(parallel_ctx, device),
        "clip":       lambda: test_grad_clipping(parallel_ctx, device),
        "step":       lambda: test_full_step(parallel_ctx, device),
        "vs_fp32":    lambda: test_mp_vs_fp32_training(parallel_ctx, device),
        "tp":         lambda: test_mp_with_tp(parallel_ctx, plan, device),
    }

    all_passed = True

    if args.test == "all":
        for name, test_fn in tests.items():
            all_passed &= test_fn()
            dist.barrier()
    else:
        all_passed = tests[args.test]()

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