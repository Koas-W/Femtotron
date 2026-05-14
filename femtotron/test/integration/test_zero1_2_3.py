import os
import sys
import torch
import torch.distributed as dist
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
import gc

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

def get_larger_config(vocab_size=1024):
    return AutoConfig.for_model(
        "llama",
        hidden_size=1024,           # ↑ from 256
        intermediate_size=4096,     # ↑ from 512
        num_attention_heads=16,
        num_key_value_heads=8,
        num_hidden_layers=8,        # ↑ from 2
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
    if zero_stage == 3:
        from femtotron.sharding.wrap_policy import llama_wrap_policy
        zero_config = ZeROConfig(stage=3, wrap_policy=llama_wrap_policy)
    else:
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


def clean_slate(label: str = ""):
    """彻底清理 GPU,准备下一个测试。"""
    gc.collect()                              # 触发 cyclic GC,处理循环引用
    torch.cuda.empty_cache()                   # 把 cached 内存还给 driver
    torch.cuda.synchronize()                   # 确保所有 kernel 完成
    torch.cuda.reset_peak_memory_stats()       # 重置峰值计数器
    
    allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
    if dist.get_rank() == 0:
        print(f"  [clean_slate {label}] active after cleanup: {allocated_mb:.1f} MB")
    
    # Sanity check:期望接近 0
    if allocated_mb > 100:
        print(f"  [WARN] {allocated_mb:.1f} MB 没释放干净,测量可能被污染")
    
    return allocated_mb


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

    # 每步用相同 seed 生成全局数据,各 rank 取自己 slice
    # 保证 baseline 和 zero1 看到完全一样的数据
    torch.manual_seed(2000)
    total_global = micro_batch_size * dp_size
    global_data = torch.randint(0, vocab_size, (total_global, seq_len), device=device)

    for step in range(num_steps):

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
    log(f"\n  [{label}] master shape sample result:")
    for i, g in enumerate(mp_manager.groups[:3]):
        compute_shape = tuple(g.compute.shape)
        master_shape = tuple(g.master.shape) if g.master is not None else None
        sharded = "(sharded)" if g.is_master_sharded else "(replicated)"
        log(f"    {g.name}: compute={compute_shape}, master={master_shape} {sharded}")

def verify_master_sharding(mp_manager, dp_size):
    """验证 ZeRO-1 下 master 确实被分片了。"""
    for g in mp_manager.groups:
        if g.is_master_sharded:
            master_numel = g.master.numel()
            compute_numel = g.compute.numel()
            expected_numel = (compute_numel + dp_size - 1) // dp_size
            if master_numel != expected_numel:
                return False, f"{g.name}: master {master_numel} != expected {expected_numel}"
        else:
            # 非分片参数的 master 应该和 compute 大小一致
            if g.master is not None and g.master.numel() != g.compute.numel():
                return False, f"{g.name}: non-sharded but sizes differ"
    return True, "all parameter sharded correctly"

def test_zero_correctness(world_size, device):
    passed = True
    log("\n" + "=" * 60)
    log("ZeRO-1 正确性测试")
    log(f"World size: {world_size}")
    log("=" * 60)

    model_config = get_larger_config()
    num_steps = 10
    micro_batch_size = 8
    seq_len = 32

    # 并行配置:dp_size 至少 2 才能看出 ZeRO-1 效果
    max_tp = world_size // 2  # 预留至少 2 给 DP
    tp_size = min(max_tp, 4)
    while tp_size > 1 and model_config.num_key_value_heads % tp_size != 0:
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
    zero_stage = 0
    clean_slate(f"before stage{zero_stage}")
    log("\n--- baseline (zero_stage=0) ---")
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
    log(f"  Baseline peak memory: {mem_b:.3f} GB")

    del model_b, mp_b, sync_b, strat_b
    torch.cuda.empty_cache()
    dist.barrier()

    # ============ ZeRO-1:zero_stage=1 ============
    zero_stage = 1
    clean_slate(f"before stage{zero_stage}")
    log("\n--- ZeRO-1 (zero_stage=1) ---")
    model_z1, mp_z1, sync_z1, strat_z1 = create_model_and_optimizer(
        model_config, parallel_ctx, device, zero_stage=1, seed=42
    )
    inspect_master_shapes(mp_z1, "zero1")

    # master 分片 shape 验证
    shard_ok_z1, shard_msg_z1 = verify_master_sharding(mp_z1, dp_size)
    log(f"  Master sharding verification: {'✓' if shard_ok_z1 else '✗'} {shard_msg_z1}")
    if not shard_ok_z1:
        passed = False   # 需要把 passed 提前初始化，或者存起来后面汇总

    losses_z1, mem_z1 = run_n_steps(
        model_z1, mp_z1, sync_z1, parallel_ctx, device,
        num_steps=num_steps, micro_batch_size=micro_batch_size,
        seq_len=seq_len, vocab_size=model_config.vocab_size,
    )

    log(f"\n  ZeRO-1 strategy: {type(strat_z1).__name__}")
    log(f"  ZeRO-1 grad_sync: {type(sync_z1).__name__}")
    log(f"  ZeRO-1 peak memory: {mem_z1:.3f} GB")

    # 在最后一个 step 后检查 weight 一致性
    weights_ok_z1, weight_diff_z1 = verify_weights_consistent_across_dp(model_z1, parallel_ctx)
    
    mp_z1.cleanup()
    del model_z1, mp_z1, sync_z1, strat_z1
    torch.cuda.empty_cache()
    gc.collect()
    dist.barrier()

    # ============ ZeRO-2:zero_stage=2 ============
    zero_stage = 2
    clean_slate(f"before stage{zero_stage}")
    log("\n--- ZeRO-2 (zero_stage=2) ---")
    model_z2, mp_z2, sync_z2, strat_z2 = create_model_and_optimizer(
        model_config, parallel_ctx, device, zero_stage=2, seed=42
    )
    inspect_master_shapes(mp_z2, "zero2")
    strat_z2.prepare_for_backward(mp_z2.groups)

    # master 分片 shape 验证
    shard_ok_z2, shard_msg_z2 = verify_master_sharding(mp_z2, dp_size)
    log(f"  Master sharding verification: {'✓' if shard_ok_z2 else '✗'} {shard_msg_z2}")
    if not shard_ok_z2:
        passed = False   # 需要把 passed 提前初始化，或者存起来后面汇总

    losses_z2, mem_z2 = run_n_steps(
        model_z2, mp_z2, sync_z2, parallel_ctx, device,
        num_steps=num_steps, micro_batch_size=micro_batch_size,
        seq_len=seq_len, vocab_size=model_config.vocab_size,
    )

    log(f"\n  ZeRO-2 strategy: {type(strat_z2).__name__}")
    log(f"  ZeRO-2 grad_sync: {type(sync_z2).__name__}")
    log(f"  ZeRO-2 peak memory: {mem_z2:.3f} GB")

    # 在最后一个 step 后检查 weight 一致性
    weights_ok_z2, weight_diff_z2 = verify_weights_consistent_across_dp(model_z2, parallel_ctx)

    # report_cuda_tensors("after ZeRO-2 run, before cleanup", min_mb=1.0)
    mp_z2.cleanup()
    # report_cuda_tensors("after manual cleanup, before del", min_mb=1.0)
    # # 1. 找一个 leaked MLP 参数
    # trace_leak((2048, 1024), torch.bfloat16, max_depth=4)
    # if dist.get_rank() == 0:
    #     print()

    # # 2. 找一个 leaked master shard
    # trace_leak((1048576,), torch.float32, max_depth=4)
    # if dist.get_rank() == 0:
    #     print()
    
    del model_z2, mp_z2, sync_z2, strat_z2
    gc.collect()
    torch.cuda.empty_cache()
    # # 在 cleanup 之后调
    # if dist.get_rank() == 0:
    #     find_strategy_holders()
    #     count_alive_instances()
    #     trace_instance_referrers('MixedPrecisionManager')
    #     trace_instance_referrers('GradAccumulator')
    #     trace_instance_referrers('LlamaModel')
    dist.barrier()
    # report_cuda_tensors("after del+gc+empty_cache", min_mb=1.0)

    # ============ ZeRO-3:zero_stage=3 ============
    zero_stage = 3
    clean_slate(f"before stage{zero_stage}")
    log("\n--- ZeRO-3 (zero_stage=3) ---")
    model_z3, mp_z3, sync_z3, strat_z3 = create_model_and_optimizer(
        model_config, parallel_ctx, device, zero_stage=3, seed=42,
    )
    log(f"  Strategy: {type(strat_z3).__name__}")
    log(f"  num clusters: {len(strat_z3.clusters)}")
    log(f"  Cluster summary:")
    for c in strat_z3.clusters:
        log(f"    {c}")

    # 不需要手动调 prepare_for_backward——hook 在 make_clusters 时已注册
    losses_z3, mem_z3 = run_n_steps(
        model_z3, mp_z3, sync_z3, parallel_ctx, device,
        num_steps=num_steps, micro_batch_size=micro_batch_size,
        seq_len=seq_len, vocab_size=model_config.vocab_size,
    )

    log(f"  ZeRO-3 peak memory: {mem_z3:.3f} GB")
    
    mp_z3.cleanup()
    del model_z3, mp_z3, sync_z3, strat_z3
    torch.cuda.empty_cache()
    gc.collect()
    dist.barrier()

    # ============ 对比 ============

    if dist.get_rank() == 0:
        log(f"\n{'=' * 60}")
        log("结果对比")
        log(f"{'=' * 60}")

        # 1. Loss 一致性(三方对比)
        log(f"\n[1] Loss 一致性 (baseline vs ZeRO-1 vs ZeRO-2 vs ZeRO-3)")
        log(f"  {'Step':>6} {'Baseline':>12} {'ZeRO-1':>12} {'ZeRO-2':>12} {'ZeRO-3':>12} {'D1':>10} {'D2':>10} {'D3':>10}")
        log(f"  {'─' * 76}")
        max_diff_z1 = 0.0
        max_diff_z2 = 0.0
        max_diff_z3 = 0.0
        for i in range(num_steps):
            d1 = abs(losses_b[i] - losses_z1[i])
            d2 = abs(losses_b[i] - losses_z2[i])
            d3 = abs(losses_b[i] - losses_z3[i])
            max_diff_z1 = max(max_diff_z1, d1)
            max_diff_z2 = max(max_diff_z2, d2)
            max_diff_z3 = max(max_diff_z3, d3)
            if i % 2 == 0 or i == num_steps - 1:
                log(
                    f"  {i+1:>6} {losses_b[i]:>12.6f} {losses_z1[i]:>12.6f} {losses_z2[i]:>12.6f} {losses_z3[i]:>12.6f} "
                    f"{d1:>10.6f} {d2:>10.6f} {d3:>10.6f}"
                )

        # 第一步 loss 应该完全相同(weight 还没被分片更新过)
        first_diff_z1 = abs(losses_b[0] - losses_z1[0])
        first_diff_z2 = abs(losses_b[0] - losses_z2[0])
        first_diff_z3 = abs(losses_b[0] - losses_z3[0])
        first_z1_ok = first_diff_z1 < 1e-4
        first_z2_ok = first_diff_z2 < 1e-4
        first_z3_ok = first_diff_z3 < 1e-4
        log(f"\n  第一步 diff ZeRO-1: {first_diff_z1:.8f} {'✓' if first_z1_ok else '✗'} (期望 < 1e-4)")
        log(f"  第一步 diff ZeRO-2: {first_diff_z2:.8f} {'✓' if first_z2_ok else '✗'} (期望 < 1e-4)")
        log(f"  第一步 diff ZeRO-3: {first_diff_z3:.8f} {'✓' if first_z3_ok else '✗'} (期望 < 1e-4)")

        # 后续步骤允许小误差(reduce_scatter vs all_reduce 浮点累加顺序差异)
        loss_threshold = 0.02
        loss_z1_ok = max_diff_z1 < loss_threshold
        loss_z2_ok = max_diff_z2 < loss_threshold
        loss_z3_ok = max_diff_z3 < loss_threshold
        log(f"  最大 diff ZeRO-1:   {max_diff_z1:.8f} {'✓' if loss_z1_ok else '✗'} (threshold {loss_threshold})")
        log(f"  最大 diff ZeRO-2:   {max_diff_z2:.8f} {'✓' if loss_z2_ok else '✗'} (threshold {loss_threshold})")
        log(f"  最大 diff ZeRO-3:   {max_diff_z3:.8f} {'✓' if loss_z3_ok else '✗'} (threshold {loss_threshold})")

        if not (first_z1_ok and first_z2_ok and first_z3_ok and loss_z1_ok and loss_z2_ok and loss_z3_ok):
            passed = False

        # 2. 显存对比(三方对比 + 节省层次)
        log(f"\n[2] 显存对比")
        log(f"  Baseline: {mem_b:.3f} GB")
        log(f"  ZeRO-1:   {mem_z1:.3f} GB  (节省 {(mem_b - mem_z1) / mem_b * 100:>5.1f}% vs baseline)")
        log(f"  ZeRO-2:   {mem_z2:.3f} GB  (节省 {(mem_b - mem_z2) / mem_b * 100:>5.1f}% vs baseline,"
            f" {(mem_z1 - mem_z2) / mem_z1 * 100:>5.1f}% vs ZeRO-1)")
        log(f"  ZeRO-3:   {mem_z3:.3f} GB  (节省 {(mem_b - mem_z3) / mem_b * 100:>5.1f}% vs baseline,"
            f" {(mem_z1 - mem_z3) / mem_z1 * 100:>5.1f}% vs ZeRO-1)")

        # 期望:ZeRO-1 < Baseline(省 optimizer state),ZeRO-2 < ZeRO-1(额外省 grad)
        mem_z1_ok = mem_z1 < mem_b
        mem_z2_ok = mem_z2 < mem_z1
        log(f"  {'✓' if mem_z1_ok else '✗'} ZeRO-1 < Baseline  (optimizer state 应被分片)")
        log(f"  {'✓' if mem_z2_ok else '✗'} ZeRO-2 < ZeRO-1    (grad 应被额外分片)")
        if not mem_z1_ok:
            log(f"  ⚠️  ZeRO-1 没省显存,master 可能没真的分片")
            passed = False
        if not mem_z2_ok:
            log(f"  ⚠️  ZeRO-2 没省下额外显存,hook 可能没正确释放 compute.grad")
            passed = False

        # 3. Weight 一致性(ZeRO-1 和 ZeRO-2 都要检查)
        log(f"\n[3] DP rank 间 weight 一致性 (gather_weights 后)")
        log(f"  ZeRO-1 最大 diff: {weight_diff_z1:.8f}  {'✓' if weights_ok_z1 else '✗'}")
        log(f"  ZeRO-2 最大 diff: {weight_diff_z2:.8f}  {'✓' if weights_ok_z2 else '✗'}")
        if not (weights_ok_z1 and weights_ok_z2):
            passed = False

        # 4. Loss 下降检查(三方都要下降)
        log(f"\n[4] Loss 下降检查")
        for name, losses in [("baseline", losses_b), ("zero1", losses_z1), ("zero2", losses_z2)]:
            first_3 = sum(losses[:3]) / 3
            last_3 = sum(losses[-3:]) / 3
            decreasing = last_3 < first_3
            log(f"  {'✓' if decreasing else '✗'} {name:>8}: {first_3:.4f} → {last_3:.4f}")
            if not decreasing:
                passed = False
        
        # 5. Master 分片验证
        log(f"\n[5] Master 分片 shape 验证")
        log(f"  {'✓' if shard_ok_z1 else '✗'} {shard_msg_z1} (ZeRO-1)")
        log(f"  {'✓' if shard_ok_z2 else '✗'} {shard_msg_z2} (ZeRO-2)")
        if not (shard_ok_z1 and shard_ok_z2):
            passed = False

    # 同步结果
    passed_tensor = torch.tensor([1 if passed else 0], device="cuda")
    dist.broadcast(passed_tensor, src=0)
    passed = passed_tensor.item() == 1

    log(f"\n{'=' * 60}")
    log(f"{'✓ ZeRO-1 正确性测试通过' if passed else '✗ ZeRO-1 正确性测试失败'}")
    log(f"{'=' * 60}")
    return passed

def test_zero_with_grad_accum(world_size, device, zero_stage):
    """ZeRO + 梯度累积兼容性冒烟测试。"""
    log("\n" + "=" * 60)
    log(f"ZeRO-{zero_stage} + 梯度累积兼容性测试")
    log("=" * 60)

    model_config = get_tiny_config()

    max_tp = world_size // 2
    tp_size = min(max_tp, 4)
    while tp_size > 1 and model_config.num_key_value_heads % tp_size != 0:
        tp_size -= 1
    dp_size = world_size // tp_size

    if dp_size < 2:
        log("  跳过（dp_size < 2）")
        return True

    parallel_ctx = ParallelContext(OrderedDict([("dp", dp_size), ("tp", tp_size)]))

    model, mp_manager, grad_sync, strat = create_model_and_optimizer(
        model_config, parallel_ctx, device, zero_stage=zero_stage, seed=42
    )
    model.train()

    num_steps = 5
    accum_steps = 2
    mbs = 4
    seq_len = 32
    vocab_size = model_config.vocab_size

    losses = []
    
    torch.manual_seed(5000)
    data = torch.randint(0, vocab_size, (mbs, seq_len), device=device)

    for step in range(num_steps):
        step_loss = 0.0

        for micro_step in range(accum_steps):
            # torch.manual_seed(step * accum_steps + micro_step + 5000)
            # data = torch.randint(0, vocab_size, (mbs, seq_len), device=device)

            is_last = (micro_step == accum_steps - 1)
            sync_ctx = nullcontext() if is_last else grad_sync.no_sync()

            with sync_ctx:
                outputs = model(input_ids=data, labels=data)
                loss = outputs["loss"] / accum_steps
                loss.backward()
                step_loss += loss.detach().float().item() * accum_steps

        grad_sync.sync_gradients()
        mp_manager.step()

        avg_loss = step_loss / accum_steps
        losses.append(avg_loss)
        log(f"  Step {step+1}: loss = {avg_loss:.6f}")

    passed = True

    # 无 NaN
    has_nan = any(not torch.isfinite(torch.tensor(l)) for l in losses)
    log(f"  {'✓' if not has_nan else '✗'} 无 NaN/Inf")
    passed &= not has_nan

    # Loss 合理
    expected = torch.tensor(vocab_size, dtype=torch.float).log().item()
    init_ok = abs(losses[0] - expected) < 2.0
    log(f"  {'✓' if init_ok else '✗'} 初始 loss {losses[0]:.4f} ≈ ln({vocab_size}) = {expected:.4f}")
    passed &= init_ok

    # Weight 一致性
    weights_ok, weight_diff = verify_weights_consistent_across_dp(model, parallel_ctx)
    log(f"  {'✓' if weights_ok else '✗'} DP rank 间 weight 一致 (diff={weight_diff:.8f})")
    passed &= weights_ok

    del model, mp_manager, grad_sync, strat
    torch.cuda.empty_cache()

    # 同步
    passed_tensor = torch.tensor([1 if passed else 0], device="cuda")
    dist.broadcast(passed_tensor, src=0)
    passed = passed_tensor.item() == 1

    log(f"  {'✓ ZeRO-' + str(zero_stage) + ' + grad_accum 通过' if passed else '✗ 测试失败'}")
    return passed


def main():
    local_rank = init_distributed()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    if dist.get_rank() == 0:
        print("=" * 60)
        print("Femtotron Integration Test: ZeRO 正确性")
        print(f"World size: {world_size}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("=" * 60)

    passed = True
    passed &= test_zero_correctness(world_size, device)
    dist.barrier()

    for stage in [1, 2, 3]:
        passed &= test_zero_with_grad_accum(world_size, device, zero_stage=stage)
        dist.barrier()

    dist.destroy_process_group()
    sys.exit(0 if passed else 1)

import gc
import torch

def list_cuda_storages(min_mb=1.0):
    """枚举所有 Python 可见的 CUDA storage,按 storage 去重(避免 view 重复)。"""
    seen = {}  # storage_ptr -> {'size_mb', 'tensors': [...]}
    
    for obj in gc.get_objects():
        try:
            if not torch.is_tensor(obj):
                continue
            if not obj.is_cuda:
                continue
            try:
                storage = obj.untyped_storage()
                ptr = storage.data_ptr()
                nbytes = storage.nbytes()
            except Exception:
                continue
            
            if ptr not in seen:
                seen[ptr] = {
                    'size_mb': nbytes / (1024**2),
                    'tensors': [],
                }
            seen[ptr]['tensors'].append({
                'shape': tuple(obj.shape),
                'dtype': str(obj.dtype).replace('torch.', ''),
                'is_leaf': obj.is_leaf,
                'requires_grad': obj.requires_grad,
                'is_param': isinstance(obj, torch.nn.Parameter),
            })
        except Exception:
            continue
    
    big = [(info['size_mb'], info['tensors']) for info in seen.values() if info['size_mb'] >= min_mb]
    big.sort(key=lambda x: x[0], reverse=True)
    return big


def report_cuda_tensors(label, min_mb=1.0):
    if dist.get_rank() != 0:
        return
    storages = list_cuda_storages(min_mb=min_mb)
    total = sum(s for s, _ in storages)
    untracked = torch.cuda.memory_allocated() / 1024**2 - total
    
    print(f"\n=== {label} ===")
    print(f"Tracked storages ≥ {min_mb} MB: {len(storages)} 个, 合计 {total:.1f} MB")
    print(f"Allocator active total: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
    print(f"Untracked (cache/internal/small): {untracked:.1f} MB")
    print()
    
    for size_mb, tensors in storages[:30]:
        print(f"  [{size_mb:7.2f} MB] {len(tensors)} 个 view")
        for t in tensors[:2]:
            tag = "Parameter" if t['is_param'] else ("leaf" if t['is_leaf'] else "view")
            print(f"             shape={t['shape']} {t['dtype']} {tag} grad={t['requires_grad']}")
        if len(tensors) > 2:
            print(f"             ... +{len(tensors)-2} 个其他 view")
def trace_leak(shape, dtype, max_depth=4, max_per_level=3):
    """找一个匹配 shape/dtype 的 leaked tensor,沿引用链回溯。"""
    import sys
    
    # 1. 在 gc 里找匹配的 tensor
    target = None
    for obj in gc.get_objects():
        try:
            if (torch.is_tensor(obj) and obj.is_cuda
                and tuple(obj.shape) == shape
                and obj.dtype == dtype):
                target = obj
                break
        except Exception:
            continue
    
    if target is None:
        print(f"没找到 shape={shape} dtype={dtype}")
        return
    
    is_param = isinstance(target, torch.nn.Parameter)
    print(f"\nTracing: shape={shape} dtype={dtype} is_param={is_param}\n")
    
    visited = {id(target)}
    
    def walk(obj, depth, indent):
        if depth >= max_depth:
            return
        
        # 过掉我们自己 frame 的 locals
        my_locals = sys._getframe(0).f_locals
        outer_locals = sys._getframe(1).f_locals
        
        refs = []
        for r in gc.get_referrers(obj):
            if r is my_locals or r is outer_locals:
                continue
            if id(r) in visited:
                continue
            refs.append(r)
        
        for ref in refs[:max_per_level]:
            visited.add(id(ref))
            cls = type(ref)
            desc = f"{cls.__module__}.{cls.__qualname__}"
            
            extra = ""
            if isinstance(ref, dict):
                keys = list(ref.keys())[:6]
                keys_str = [repr(k)[:40] for k in keys]
                extra = f"  dict[{len(ref)}] keys={keys_str}"
            elif isinstance(ref, (list, tuple)):
                extra = f"  len={len(ref)}"
            elif hasattr(ref, '__dict__'):
                attrs = list(vars(ref).keys())[:5]
                extra = f"  attrs={attrs}"
            
            print(f"{indent}└─ {desc}{extra}")
            walk(ref, depth + 1, indent + "   ")
        
        if len(refs) > max_per_level:
            print(f"{indent}   ... +{len(refs) - max_per_level} 个其他 referrer 未显示")
    
    walk(target, 0, "")
    target = None
def count_alive_instances():
    """统计感兴趣的类型还活着多少个。"""
    types_of_interest = {
        # 模型类
        'ColumnParallelLinear', 'RowParallelLinear', 'VocabParallelEmbedding',
        'LlamaDecoderLayer', 'LlamaMLP', 'LlamaAttention', 'LlamaForCausalLM',
        'LlamaRMSNorm', 'LlamaModel',
        # femtotron 训练框架
        'MixedPrecisionManager', 'ParamGroup', 'GradAccumulator',
        'ZeRO1Strategy', 'ZeRO2Strategy', 'ZeRO3Strategy',
        'ParamGroupCluster', 'NoShardStrategy',
        # PyTorch
        'AdamW', 'Adam',
    }
    
    counts = {}
    for obj in gc.get_objects():
        try:
            t = type(obj).__name__
            if t in types_of_interest:
                counts[t] = counts.get(t, 0) + 1
        except Exception:
            continue
    
    if dist.get_rank() != 0:
        return
    print("\n=== 活着的对象计数 ===")
    if not counts:
        print("  ✓ 没有相关对象残留")
        return
    for t in sorted(counts):
        print(f"  {t}: {counts[t]}")
        
def trace_instance_referrers(type_name, max_depth=2, max_per_level=5):
    """找到这个类型的实例,沿引用链回溯。"""
    import sys, types
    
    target = None
    for obj in gc.get_objects():
        try:
            if type(obj).__name__ == type_name:
                target = obj
                break
        except Exception:
            continue
    
    if target is None:
        print(f"No {type_name} alive")
        return
    
    if dist.get_rank() != 0:
        return
    
    print(f"\n=== Who holds {type_name}? ===")
    
    def describe(obj):
        cls = type(obj)
        name = f"{cls.__module__}.{cls.__qualname__}"
        if isinstance(obj, dict):
            keys = list(obj.keys())[:6]
            return f"{name} keys={[repr(k)[:40] for k in keys]}"
        if isinstance(obj, (list, tuple)):
            return f"{name}[{len(obj)}]"
        if isinstance(obj, types.FrameType):
            return f"FRAME {obj.f_code.co_filename}:{obj.f_lineno} in {obj.f_code.co_name}"
        if hasattr(obj, '__dict__'):
            attrs = list(vars(obj).keys())[:5]
            return f"{name} attrs={attrs}"
        return name
    
    visited = {id(target)}
    
    def walk(obj, depth, indent):
        if depth >= max_depth:
            return
        my_locals = sys._getframe(0).f_locals
        outer_locals = sys._getframe(1).f_locals if sys._getframe(1) else None
        
        refs = []
        for r in gc.get_referrers(obj):
            if r is my_locals or r is outer_locals:
                continue
            if id(r) in visited:
                continue
            refs.append(r)
        
        for r in refs[:max_per_level]:
            visited.add(id(r))
            print(f"{indent}← {describe(r)}")
            walk(r, depth + 1, indent + "  ")
        if len(refs) > max_per_level:
            print(f"{indent}  ... +{len(refs)-max_per_level} 其他 referrer")
    
    walk(target, 0, "")
    target = None

def find_strategy_holders():
    """找到活着的 ZeRO2Strategy,dump 它的状态 + 谁拽着它。"""
    import sys
    
    strat = None
    for obj in gc.get_objects():
        if type(obj).__name__ == 'ZeRO2Strategy':
            strat = obj
            break
    
    if strat is None or dist.get_rank() != 0:
        return
    
    # ─── strategy 内部状态(确认 cleanup 真的执行了) ───
    print("\n=== ZeRO2Strategy 内部状态 ===")
    for k, v in vars(strat).items():
        size = f" len={len(v)}" if hasattr(v, '__len__') else ""
        val = ""
        if isinstance(v, dict) and len(v) > 0:
            val = f" first_key={list(v.keys())[0]!r}"
        elif isinstance(v, (list, tuple)) and len(v) > 0:
            val = f" first_elem_type={type(v[0]).__name__}"
        print(f"  .{k}: {type(v).__name__}{size}{val}")
    
    # ─── 谁直接持有 strategy ───
    print("\n=== 谁持有 ZeRO2Strategy ===")
    my_locals = sys._getframe(0).f_locals
    refs = [r for r in gc.get_referrers(strat) if r is not my_locals]
    
    print(f"Total referrers: {len(refs)}")
    for i, r in enumerate(refs):
        t = type(r)
        desc = f"{t.__module__}.{t.__qualname__}"
        
        if t.__name__ == 'cell':       # 闭包变量!最可能是这个
            try:
                contents = r.cell_contents
                desc += f" → cell_contents={type(contents).__name__}"
                if isinstance(contents, type(strat)):
                    desc += " (refers back to strategy)"
            except ValueError:
                desc += " (empty cell)"
        elif isinstance(r, dict):
            keys = list(r.keys())[:6]
            desc += f"  dict[{len(r)}] keys={[repr(k)[:30] for k in keys]}"
        elif isinstance(r, (list, tuple)):
            desc += f"[{len(r)}]"
        elif t.__name__ == 'function':
            desc += f"  qualname={r.__qualname__}"
        elif hasattr(r, '__dict__'):
            attrs = list(vars(r).keys())[:5]
            desc += f"  attrs={attrs}"
        print(f"  [{i}] ← {desc}")
        
        # 再上一层:谁持有 referrer
        sub_refs = gc.get_referrers(r)
        sub_refs = [sr for sr in sub_refs if sr is not my_locals]
        # 也过掉我们已知的(strategy 自己等)
        sub_refs = [sr for sr in sub_refs if sr is not strat]
        for sr in sub_refs[:3]:
            sub_t = type(sr)
            print(f"        ← {sub_t.__module__}.{sub_t.__qualname__}")

if __name__ == "__main__":
    main()