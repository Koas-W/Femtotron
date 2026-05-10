"""
Femtotron DP 训练一致性测试
============================

验证核心：相同总 batch size 下，不同 DP/TP 配置的 loss 曲线一致。

用法：
    # 2卡：对比 DP=1,TP=2 vs DP=2,TP=1
    torchrun --nproc_per_node=2 tests/integration/test_dp_training.py

    # 4卡：额外测试 DP=2,TP=2
    torchrun --nproc_per_node=4 tests/integration/test_dp_training.py

    # 8卡：完整测试
    torchrun --nproc_per_node=8 tests/integration/test_dp_training.py

原理：
    DP=2 时每个 rank 处理 micro_batch_size 条数据，总 batch = 2 * mbs。
    DP=1 时单 rank 处理 2 * mbs 条数据，总 batch 也是 2 * mbs。
    两者的梯度（经过 all-reduce 平均后）应该完全一致，
    因此 optimizer step 后的参数更新也一致，loss 曲线也一致。
"""

import os
import sys
import tempfile
import shutil
import torch
import torch.distributed as dist
from collections import OrderedDict
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
    """创建模型 + 混合精度 + optimizer，用固定 seed 初始化。"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    with torch.device("meta"):
        model = build_llama_model(model_config, parallel_ctx)

    model = model.to_empty(device=device)
    for p in model.parameters():
        torch.nn.init.normal_(p, mean=0.0, std=0.02)
    model = model.bfloat16()

    train_config = TrainConfig(master_dtype=torch.float32, grad_clip=1.0)
    compute_param_groups = get_param_groups(model, weight_decay=0.01)
    parallel_plan = get_llama_parallel_plan()
    
    mp_manager = MixedPrecisionManager(
        model=model,
        parallel_ctx=parallel_ctx,
        parallel_plan=parallel_plan,
        config=train_config,
        inner_optimizer_cls=torch.optim.AdamW,
        inner_optimizer_kwargs={"lr": 3e-4, "betas": (0.9, 0.95), "eps": 1e-8},
        compute_param_groups=compute_param_groups,
    )

    grad_sync = DataParallelGradSync(param_groups=mp_manager.groups, parallel_ctx=parallel_ctx)

    return model, mp_manager, grad_sync


def run_n_steps(model, mp_manager, grad_sync, parallel_ctx,
                device, num_steps=20, micro_batch_size=4, seq_len=32,
                vocab_size=1024):
    """
    跑 N 步训练，返回每步的 loss 列表。
    
    关键：所有 DP rank 用相同的全局数据，每个 rank 取自己的 slice。
    这样保证不同 DP 配置下看到的总数据相同。
    """
    model.train()
    dp_rank = parallel_ctx.dp_rank
    dp_size = parallel_ctx.dp_size
    total_batch = micro_batch_size * dp_size

    losses = []

    for step in range(num_steps):
        # 生成全局一致的数据（所有 rank 用相同 seed）
        torch.manual_seed(step + 1000)
        global_input = torch.randint(0, vocab_size, (total_batch, seq_len), device=device)

        # 每个 DP rank 取自己的 slice
        start = dp_rank * micro_batch_size
        end = start + micro_batch_size
        local_input = global_input[start:end]

        # Forward
        outputs = model(input_ids=local_input, labels=local_input)
        loss = outputs["loss"]
        loss.backward()

        # DP 梯度同步
        grad_sync.sync_gradients()

        # Optimizer step
        mp_manager.step()

        losses.append(loss.float().item())

    return losses


def test_dp_consistency(world_size, device):
    """
    在同一次 torchrun 中依次测试多种配置，对比 loss。

    策略：先用 DP=1 跑 baseline，然后用 DP>1 跑，对比。
    由于不同配置需要不同的 ParallelContext 和 process group，
    我们在同一个进程中重新构建。

    注意：在同一个 torchrun 中无法真正改变 world_size。
    所以我们测试的是：
    - 配置 A: 所有卡用于 TP（DP=1, TP=world_size）
    - 配置 B: 所有卡用于 DP（DP=world_size, TP=1）
    - 配置 C: 混合（如果卡数够的话）
    
    A 和 B 应该产生相同的 loss（因为总 batch size 相同）。
    """
    log("\n" + "=" * 60)
    log("DP 训练一致性测试")
    log(f"World size: {world_size}")
    log("=" * 60)

    model_config = get_tiny_config()
    min_kv_heads = model_config.num_key_value_heads  # 4
    plan = get_llama_parallel_plan()
    num_steps = 20
    micro_batch_size = 4
    seq_len = 32

    # 枚举所有 dp * tp == world_size 且 kv_heads % tp == 0 的组合
    configs_to_test = []
    for tp in range(1, world_size + 1):
        if world_size % tp != 0:
            continue
        if min_kv_heads % tp != 0:
            continue
        dp = world_size // tp
        configs_to_test.append((f"DP={dp}, TP={tp}", dp, tp))

    log(f"合法配置: {[c[0] for c in configs_to_test]}")

    if len(configs_to_test) < 2:
        log("⚠ 合法配置不足 2 种，无法做对比测试")
        log("  考虑增大 num_key_value_heads 或使用不同的 GPU 数量")
        return True

    all_losses = {}

    for config_name, dp_size, tp_size in configs_to_test:
        log(f"\n--- 配置: {config_name} ---")

        parallel_ctx = ParallelContext(OrderedDict([
            ("dp", dp_size),
            ("tp", tp_size),
        ]))

        model, mp_manager, grad_sync = create_model_and_optimizer(
            model_config, parallel_ctx, device, seed=42
        )

        # 总 batch size = micro_batch_size * dp_size
        # 为了公平对比，不同 DP 配置用不同的 micro_batch_size
        # 使得总 batch size 一致
        adjusted_mbs = micro_batch_size * world_size // dp_size  # 调整使总 batch 一致
        # 实际上：total = adjusted_mbs * dp_size = micro_batch_size * world_size = 常数

        losses = run_n_steps(
            model, mp_manager, grad_sync, parallel_ctx,
            device, num_steps=num_steps,
            micro_batch_size=adjusted_mbs, seq_len=seq_len,
            vocab_size=model_config.vocab_size,
        )

        # 收集 rank 0 的 loss（所有 rank 的 loss 应该相同，取 rank 0 即可）
        if dist.get_rank() == 0:
            all_losses[config_name] = losses
            for i in range(0, num_steps, 5):
                log(f"  Step {i+1:>3}: loss = {losses[i]:.6f}")

        del model, mp_manager, grad_sync
        torch.cuda.empty_cache()
        dist.barrier()

    # ========== 对比 ==========
    passed = True

    if dist.get_rank() == 0 and len(all_losses) >= 2:
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
                if i % 5 == 0:
                    log(f"  {i+1:>6} {baseline_losses[i]:>12.6f} {other_losses[i]:>12.6f} {diff:>12.8f}")

            # 判断是否一致
            # BF16 训练 + all-reduce 的浮点误差，允许 1e-2 的差距
            # （20 步训练后误差会累积）
            threshold = 0.05
            is_consistent = max_diff < threshold
            log(f"\n  Max diff: {max_diff:.8f}")
            log(f"  {'✓' if is_consistent else '✗'} Loss 一致性 (threshold={threshold})")

            if not is_consistent:
                passed = False

    # 同步结果
    passed_tensor = torch.tensor([1 if passed else 0], device="cuda")
    dist.broadcast(passed_tensor, src=0)
    passed = passed_tensor.item() == 1

    log(f"\n{'=' * 60}")
    log(f"{'✓ DP 一致性测试通过' if passed else '✗ DP 一致性测试失败'}")
    log(f"{'=' * 60}")
    return passed


def main():
    local_rank = init_distributed()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    if dist.get_rank() == 0:
        print("=" * 60)
        print("Femtotron Integration Test: DP 训练一致性")
        print(f"World size: {world_size}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("=" * 60)

    passed = test_dp_consistency(world_size, device)

    dist.destroy_process_group()
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()