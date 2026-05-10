"""
Femtotron 1.6 梯度累积一致性测试
==================================

验证核心：相同总 batch size 下，不同 (micro_batch_size, grad_accum_steps) 
组合的 loss 曲线一致。

用法：
    torchrun --nproc_per_node=2 tests/integration/test_gradient_accum.py

原理：
    配置 A: mbs=8, accum=1  → total=8，一次 forward 8 条
    配置 B: mbs=4, accum=2  → total=8，两次 forward 各 4 条，梯度累加
    配置 C: mbs=2, accum=4  → total=8，四次 forward 各 2 条，梯度累加

    三者的梯度在数学上完全等价（前提是 loss 正确除以 accum_steps），
    因此 optimizer step 后的参数更新一致，loss 曲线一致。
"""

import os
import sys
import torch
import torch.distributed as dist
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from femtotron.parallel_context import ParallelContext
from femtotron.model.parallel_plan import get_llama_parallel_plan
from femtotron.model.llama import build_llama_model
from femtotron.training.mixed_precision_manager import MixedPrecisionManager
from femtotron.training.optimizer import get_param_groups
from femtotron.training.train_config import TrainConfig
from femtotron.parallel.data_parallel.ddp import DataParallelGradSync
from transformers import AutoConfig


def init_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    return local_rank


def log(msg):
    if dist.get_rank() == 0:
        print(msg)


def get_tiny_config(vocab_size=1024):
    return AutoConfig.for_model(
        "llama",
        hidden_size=256,
        intermediate_size=512,
        num_attention_heads=8,
        num_key_value_heads=4,
        num_hidden_layers=2,
        max_position_embeddings=128,
        vocab_size=vocab_size,
        rms_norm_eps=1e-5,
        hidden_act="silu",
        tie_word_embeddings=False,
    )


def create_model_and_optimizer(model_config, parallel_ctx, device, seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    plan = get_llama_parallel_plan()

    with torch.device("meta"):
        model = build_llama_model(model_config, parallel_ctx)

    model = model.to_empty(device=device)
    for p in model.parameters():
        torch.nn.init.normal_(p, mean=0.0, std=0.02)
    model = model.bfloat16()

    train_config = TrainConfig(master_dtype=torch.float32, grad_clip=1.0)
    compute_param_groups = get_param_groups(model, weight_decay=0.01)

    mp_manager = MixedPrecisionManager(
        model=model,
        parallel_ctx=parallel_ctx,
        parallel_plan=plan,
        config=train_config,
        inner_optimizer_cls=torch.optim.AdamW,
        inner_optimizer_kwargs={"lr": 3e-4, "betas": (0.9, 0.95), "eps": 1e-8},
        compute_param_groups=compute_param_groups,
    )

    grad_sync = DataParallelGradSync(mp_manager.groups, parallel_ctx)

    return model, mp_manager, grad_sync


def run_n_steps_with_accum(model, mp_manager, grad_sync, parallel_ctx,
                           device, num_steps=15, micro_batch_size=8,
                           grad_accum_steps=1, seq_len=32, vocab_size=1024):
    """
    跑 N 个 optimizer step，每步累积 grad_accum_steps 个 micro batch。
    返回每步的 loss。
    
    关键：所有配置看到的数据必须完全一致。
    total_batch = micro_batch_size * grad_accum_steps * dp_size
    每步的全局数据量相同，只是切分方式不同。
    """
    model.train()
    dp_rank = parallel_ctx.dp_rank
    dp_size = parallel_ctx.dp_size
    total_batch_per_rank = micro_batch_size * grad_accum_steps

    losses = []

    for step in range(num_steps):
        # 生成本步全局一致的数据
        # total = total_batch_per_rank * dp_size 条
        torch.manual_seed(step + 2000)
        total_global = total_batch_per_rank * dp_size
        global_data = torch.randint(0, vocab_size, (total_global, seq_len), device=device)

        # 每个 DP rank 取自己的 slice
        rank_start = dp_rank * total_batch_per_rank
        rank_data = global_data[rank_start : rank_start + total_batch_per_rank]

        # 梯度累积循环
        step_loss = 0.0

        for micro_step in range(grad_accum_steps):
            # 取本 micro batch
            mb_start = micro_step * micro_batch_size
            mb_data = rank_data[mb_start : mb_start + micro_batch_size]

            is_last = (micro_step == grad_accum_steps - 1)
            sync_ctx = nullcontext() if is_last else grad_sync.no_sync()

            with sync_ctx:
                outputs = model(input_ids=mb_data, labels=mb_data)
                loss = outputs["loss"] / grad_accum_steps
                loss.backward()
                step_loss += loss.detach().float().item() * grad_accum_steps

        # DP 梯度同步
        grad_sync.sync_gradients()

        # Optimizer step
        mp_manager.step()

        # 记录平均 loss
        avg_loss = step_loss / grad_accum_steps
        losses.append(avg_loss)

    return losses


def test_gradient_accumulation(world_size, device):
    log("\n" + "=" * 60)
    log("梯度累积一致性测试")
    log(f"World size: {world_size}")
    log("=" * 60)

    model_config = get_tiny_config()
    num_steps = 15
    seq_len = 32

    # 总 batch per rank = mbs * accum = 8
    configs = [
        ("mbs=8, accum=1", 8, 1),
        ("mbs=4, accum=2", 4, 2),
        ("mbs=2, accum=4", 2, 4),
    ]

    # 确定并行配置
    tp_size = min(world_size, 4)  # 限制 TP 不超过 kv_heads
    while model_config.num_key_value_heads % tp_size != 0:
        tp_size -= 1
    dp_size = world_size // tp_size

    parallel_ctx = ParallelContext(OrderedDict([
        ("dp", dp_size),
        ("tp", tp_size),
    ]))

    log(f"并行配置: DP={dp_size}, TP={tp_size}")
    log(f"总 batch per rank per step: 8")
    log(f"总 batch 全局 per step: {8 * dp_size}")

    all_losses = {}

    for config_name, mbs, accum in configs:
        log(f"\n--- 配置: {config_name} ---")

        # 每种配置重新创建模型（相同 seed → 相同初始权重）
        model, mp_manager, grad_sync = create_model_and_optimizer(
            model_config, parallel_ctx, device, seed=42
        )

        losses = run_n_steps_with_accum(
            model, mp_manager, grad_sync, parallel_ctx,
            device, num_steps=num_steps,
            micro_batch_size=mbs, grad_accum_steps=accum,
            seq_len=seq_len, vocab_size=model_config.vocab_size,
        )

        if dist.get_rank() == 0:
            all_losses[config_name] = losses
            for i in range(0, num_steps, 3):
                log(f"  Step {i+1:>3}: loss = {losses[i]:.6f}")

        del model, mp_manager, grad_sync
        torch.cuda.empty_cache()
        dist.barrier()

    # ========== 对比 ==========
    passed = True

    if dist.get_rank() == 0:
        log(f"\n{'=' * 60}")
        log("Loss 对比")
        log(f"{'=' * 60}")

        config_names = list(all_losses.keys())
        baseline_name = config_names[0]
        baseline_losses = all_losses[baseline_name]

        for other_name in config_names[1:]:
            other_losses = all_losses[other_name]

            log(f"\n  {baseline_name} vs {other_name}:")
            log(f"  {'Step':>6} {'Baseline':>12} {'Other':>12} {'Diff':>12}")
            log(f"  {'─' * 46}")

            max_diff = 0.0
            for i in range(num_steps):
                diff = abs(baseline_losses[i] - other_losses[i])
                max_diff = max(max_diff, diff)
                if i % 3 == 0:
                    log(f"  {i+1:>6} {baseline_losses[i]:>12.6f} {other_losses[i]:>12.6f} {diff:>12.8f}")

            # BF16 下梯度累加有浮点误差，允许 0.02 的差距
            threshold = 0.02
            is_consistent = max_diff < threshold
            log(f"\n  Max diff: {max_diff:.8f}")
            log(f"  {'✓' if is_consistent else '✗'} 一致性 (threshold={threshold})")

            if not is_consistent:
                passed = False

        # 额外检查：所有配置的 loss 都在下降
        log(f"\n--- Loss 下降检查 ---")
        for name, losses in all_losses.items():
            first_3 = sum(losses[:3]) / 3
            last_3 = sum(losses[-3:]) / 3
            decreasing = last_3 < first_3
            log(f"  {'✓' if decreasing else '✗'} {name}: {first_3:.4f} → {last_3:.4f}")
            if not decreasing:
                passed = False

    # 同步结果
    passed_tensor = torch.tensor([1 if passed else 0], device="cuda")
    dist.broadcast(passed_tensor, src=0)
    passed = passed_tensor.item() == 1

    log(f"\n{'=' * 60}")
    log(f"{'✓ 梯度累积一致性测试通过' if passed else '✗ 梯度累积一致性测试失败'}")
    log(f"{'=' * 60}")
    return passed


def main():
    local_rank = init_distributed()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    if dist.get_rank() == 0:
        print("=" * 60)
        print("Femtotron Integration Test: 梯度累积一致性")
        print(f"World size: {world_size}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("=" * 60)

    passed = test_gradient_accumulation(world_size, device)

    dist.destroy_process_group()
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()