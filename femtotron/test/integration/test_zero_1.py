import torch
from torch import nn, Tensor
from torch import distributed as dist
from torch.distributed import ProcessGroup
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from pathlib import Path
from collections import defaultdict
from safetensors import safe_open
from typing import cast, Protocol
from dataclasses import dataclass

from femtotron.parallel_context import ParallelContext
from femtotron.training.param_group import ParamGroup
from femtotron.sharding.sharding_spec import ShardingSpec
from femtotron.sharding.sharding_strategy import ShardingStrategy
from femtotron.sharding.no_shard import NoShardStrategy
from femtotron.sharding.zero1 import ZeRO1Strategy

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
from femtotron.parallel.data_parallel.gradient_synchronizer import create_grad_synchronizer
from femtotron.sharding.factory import create_sharding_strategy, ZeROConfig
from femtotron.sharding.no_shard import NoShardStrategy
from femtotron.sharding.zero1 import ZeRO1Strategy
from transformers import AutoConfig


def init_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
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


def create_model_and_optimizer(model_config, parallel_ctx, device, zero_stage=0, seed=42):
    """构造模型 + mp_manager + grad_sync。zero_stage=0 走 NoShard，=1 走 ZeRO-1。"""
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

    # ←—— 关键差异 1:构造 sharding strategy
    zero_config = ZeROConfig(stage=zero_stage)
    sharding_strategy = create_sharding_strategy(parallel_ctx, zero_config)

    mp_manager = MixedPrecisionManager(
        model=model,
        parallel_ctx=parallel_ctx,
        parallel_plan=plan,
        config=train_config,
        inner_optimizer_cls=torch.optim.AdamW,
        inner_optimizer_kwargs={"lr": 3e-4, "betas": (0.9, 0.95), "eps": 1e-8},
        compute_param_groups=compute_param_groups,
        sharding_strategy=sharding_strategy,
    )

    # ←—— 关键差异 2:工厂函数根据 strategy 决定 grad_sync
    grad_sync = create_grad_synchronizer(
        mp_manager.groups, parallel_ctx, sharding_strategy
    )

    return model, mp_manager, grad_sync, sharding_strategy


def run_n_steps(model, mp_manager, grad_sync, parallel_ctx,
                device, num_steps=10, micro_batch_size=8,
                seq_len=32, vocab_size=1024):
    """跑 N 步,返回每步 loss 和最终峰值显存。
    
    不做 grad_accum,简化对比。
    """
    model.train()
    dp_rank = parallel_ctx.dp_rank
    dp_size = parallel_ctx.dp_size

    losses = []
    torch.cuda.reset_peak_memory_stats()

    for step in range(num_steps):
        # 每步用相同 seed 生成全局数据,各 rank 取自己 slice
        # 保证 baseline 和 zero1 看到完全一样的数据
        torch.manual_seed(step + 2000)
        total_global = micro_batch_size * dp_size
        global_data = torch.randint(0, vocab_size, (total_global, seq_len), device=device)

        rank_start = dp_rank * micro_batch_size
        rank_data = global_data[rank_start : rank_start + micro_batch_size]

        outputs = model(input_ids=rank_data, labels=rank_data)
        loss = outputs["loss"]
        loss.backward()

        grad_sync.sync_gradients()
        mp_manager.step()

        losses.append(loss.detach().float().item())

    peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    return losses, peak_mem_gb


def verify_weights_consistent_across_dp(model, parallel_ctx):
    """ZeRO-1 在 gather_weights 后,所有 dp rank 的 compute weight 必须一致。"""
    if parallel_ctx.dp_group is None or parallel_ctx.dp_size == 1:
        return True, 0.0

    max_diff = 0.0
    for name, p in model.named_parameters():
        gathered = [torch.zeros_like(p) for _ in range(parallel_ctx.dp_size)]
        dist.all_gather(gathered, p, group=parallel_ctx.dp_group)
        for g in gathered[1:]:
            diff = (g - gathered[0]).abs().max().item()
            max_diff = max(max_diff, diff)
    return max_diff < 1e-5, max_diff


def inspect_master_shapes(mp_manager, label):
    """打印前几个参数的 master shape,直观确认是否分片。"""
    log(f"\n  [{label}] master 形状采样:")
    for i, g in enumerate(mp_manager.groups[:3]):
        compute_shape = tuple(g.compute.shape)
        master_shape = tuple(g.master.shape) if g.master is not None else None
        sharded = "(sharded)" if g.is_master_sharded else "(replicated)"
        log(f"    {g.name}: compute={compute_shape}, master={master_shape} {sharded}")


def test_zero1_correctness(world_size, device):
    log("\n" + "=" * 60)
    log("ZeRO-1 正确性测试")
    log(f"World size: {world_size}")
    log("=" * 60)

    model_config = get_tiny_config()
    num_steps = 10
    micro_batch_size = 8
    seq_len = 32

    # 并行配置:dp_size 至少 2 才能看出 ZeRO-1 效果
    tp_size = min(world_size, 4)
    while model_config.num_key_value_heads % tp_size != 0:
        tp_size -= 1
    dp_size = world_size // tp_size

    if dp_size < 2:
        log(f"⚠️  dp_size={dp_size},ZeRO-1 在 dp=1 下退化为 NoShard,无法验证。")
        log(f"   需要 world_size 让 dp_size >= 2(当前 tp_size={tp_size})。")
        return False

    parallel_ctx = ParallelContext(OrderedDict([
        ("dp", dp_size),
        ("tp", tp_size),
    ]))

    log(f"并行配置: DP={dp_size}, TP={tp_size}")

    # ============ Baseline:zero_stage=0 ============
    log("\n--- 跑 baseline (zero_stage=0) ---")
    model_b, mp_b, sync_b, strat_b = create_model_and_optimizer(
        model_config, parallel_ctx, device, zero_stage=0, seed=42
    )
    inspect_master_shapes(mp_b, "baseline")

    losses_b, mem_b = run_n_steps(
        model_b, mp_b, sync_b, parallel_ctx, device,
        num_steps=num_steps, micro_batch_size=micro_batch_size,
        seq_len=seq_len, vocab_size=model_config.vocab_size,
    )

    log(f"\n  Baseline strategy: {type(strat_b).__name__}")
    log(f"  Baseline grad_sync: {type(sync_b).__name__}")
    log(f"  Baseline 峰值显存: {mem_b:.3f} GB")

    del model_b, mp_b, sync_b, strat_b
    torch.cuda.empty_cache()
    dist.barrier()

    # ============ ZeRO-1:zero_stage=1 ============
    log("\n--- 跑 ZeRO-1 (zero_stage=1) ---")
    model_z, mp_z, sync_z, strat_z = create_model_and_optimizer(
        model_config, parallel_ctx, device, zero_stage=1, seed=42
    )
    inspect_master_shapes(mp_z, "zero1")

    losses_z, mem_z = run_n_steps(
        model_z, mp_z, sync_z, parallel_ctx, device,
        num_steps=num_steps, micro_batch_size=micro_batch_size,
        seq_len=seq_len, vocab_size=model_config.vocab_size,
    )

    log(f"\n  ZeRO-1 strategy: {type(strat_z).__name__}")
    log(f"  ZeRO-1 grad_sync: {type(sync_z).__name__}")
    log(f"  ZeRO-1 峰值显存: {mem_z:.3f} GB")

    # 在最后一个 step 后检查 weight 一致性
    weights_ok, weight_diff = verify_weights_consistent_across_dp(model_z, parallel_ctx)

    del model_z, mp_z, sync_z, strat_z
    torch.cuda.empty_cache()
    dist.barrier()

    # ============ 对比 ============
    passed = True

    if dist.get_rank() == 0:
        log(f"\n{'=' * 60}")
        log("结果对比")
        log(f"{'=' * 60}")

        # 1. Loss 一致性
        log(f"\n[1] Loss 一致性 (baseline vs zero1)")
        log(f"  {'Step':>6} {'Baseline':>12} {'ZeRO-1':>12} {'Diff':>12}")
        log(f"  {'─' * 46}")
        max_loss_diff = 0.0
        for i in range(num_steps):
            diff = abs(losses_b[i] - losses_z[i])
            max_loss_diff = max(max_loss_diff, diff)
            if i % 2 == 0 or i == num_steps - 1:
                log(f"  {i+1:>6} {losses_b[i]:>12.6f} {losses_z[i]:>12.6f} {diff:>12.8f}")

        # 第一步 loss 应该完全相同(weight 还没被分片更新过,forward 完全一致)
        first_step_diff = abs(losses_b[0] - losses_z[0])
        first_step_ok = first_step_diff < 1e-4
        log(f"\n  第一步 diff: {first_step_diff:.8f} {'✓' if first_step_ok else '✗'} (期望 < 1e-4)")

        # 后续步骤允许小误差(reduce_scatter vs all_reduce 浮点累加顺序差异)
        loss_threshold = 0.02
        loss_ok = max_loss_diff < loss_threshold
        log(f"  最大 diff:    {max_loss_diff:.8f} {'✓' if loss_ok else '✗'} (threshold {loss_threshold})")

        if not (first_step_ok and loss_ok):
            passed = False

        # 2. 显存对比
        log(f"\n[2] 显存对比")
        mem_savings = (mem_b - mem_z) / mem_b * 100 if mem_b > 0 else 0
        log(f"  Baseline: {mem_b:.3f} GB")
        log(f"  ZeRO-1:   {mem_z:.3f} GB")
        log(f"  节省:     {mem_savings:.1f}%")

        # ZeRO-1 应该明显省显存。tiny model 上节省比例不会很大(激活占大头),
        # 但至少应该比 baseline 小一点。
        mem_ok = mem_z < mem_b
        log(f"  {'✓' if mem_ok else '✗'} ZeRO-1 < Baseline")
        if not mem_ok:
            passed = False
            log(f"  ⚠️  显存没省下来,说明 master 可能没真的分片")

        # 3. Weight 一致性
        log(f"\n[3] DP rank 间 weight 一致性 (ZeRO-1 gather_weights 后)")
        log(f"  最大 diff: {weight_diff:.8f}")
        log(f"  {'✓' if weights_ok else '✗'} 所有 dp rank weight 一致 (< 1e-5)")
        if not weights_ok:
            passed = False

        # 4. Loss 下降
        log(f"\n[4] Loss 下降检查")
        for name, losses in [("baseline", losses_b), ("zero1", losses_z)]:
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
    log(f"{'✓ ZeRO-1 正确性测试通过' if passed else '✗ ZeRO-1 正确性测试失败'}")
    log(f"{'=' * 60}")
    return passed


def main():
    local_rank = init_distributed()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    if dist.get_rank() == 0:
        print("=" * 60)
        print("Femtotron Integration Test: ZeRO-1 正确性")
        print(f"World size: {world_size}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("=" * 60)

    passed = test_zero1_correctness(world_size, device)

    dist.destroy_process_group()
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()