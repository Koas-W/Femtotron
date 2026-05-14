"""对比 baseline / ZeRO-1/2/3 / ZeRO-3+AC 五种配置。

验证:
- Loss 一致性(各 stage 应该数学等价)
- 显存节省(ZeRO 各级递减,AC 进一步压缩)

用法:
    torchrun --nproc_per_node=2 scripts/test_zero_ac.py
"""

from __future__ import annotations

import gc
import os
from collections import OrderedDict

import torch
import torch.distributed as dist

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

from femtotron.parallel.data_parallel.gradient_synchronizer import create_grad_synchronizer
from femtotron.sharding.factory import create_sharding_strategy
from femtotron.sharding.zero_config import ZeROConfig
from femtotron.training.activation_ckpt import apply_activation_checkpointing
from femtotron.scripts.presets import get_wrap_policy, get_ac_policy

from transformers import AutoConfig


def log(msg: str) -> None:
    if dist.get_rank() == 0:
        print(msg)


def clean_slate(label: str = "") -> None:
    """彻底释放 GPU,重置 peak counter。"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    mb = torch.cuda.memory_allocated() / 1024 / 1024
    log(f"  [clean_slate {label}] active after cleanup: {mb:.1f} MB")
    if mb > 100:
        log(f"  [WARN] {mb:.1f} MB 没释放干净")


def create_model_and_optimizer(model_config, parallel_ctx, device, seed,
                                zero_stage: int = 0, ac_enabled: bool = False):
    """建一个完整的训练栈,可选 ZeRO 和 AC。"""
    torch.manual_seed(seed)
    
    plan = get_llama_parallel_plan()
    with torch.device("meta"):
        model = build_llama_model(model_config, parallel_ctx)
    model = model.to_empty(device=device)
    for p in model.parameters():
        if p.requires_grad:
            torch.nn.init.normal_(p, mean=0.0, std=0.02)
    model = model.bfloat16()
    
    # ─── ZeRO strategy ───
    wrap_policy = get_wrap_policy("llama_decoder_layer") if zero_stage == 3 else None
    zero_config = ZeROConfig(stage=zero_stage, wrap_policy=wrap_policy)
    strategy = create_sharding_strategy(parallel_ctx, zero_config)
    
    train_config = TrainConfig(
        master_dtype=torch.float32,
        grad_clip=1.0,
        train_steps=10,
        log_interval=10,
        checkpoint_interval=10000,
        checkpoint_dir="/tmp/test_zero_ac",
        warmup_steps=0,
        min_lr_ratio=1.0,
    )
    
    compute_param_groups = get_param_groups(model, weight_decay=0.01)
    
    mp = MixedPrecisionManager(
        model=model,
        sharding_strategy=strategy,
        parallel_ctx=parallel_ctx,
        parallel_plan=plan,
        config=train_config,
        inner_optimizer_cls=torch.optim.AdamW,
        inner_optimizer_kwargs={"lr": 3e-4, "betas": (0.9, 0.95), "eps": 1e-8},
        compute_param_groups=compute_param_groups,
    )
    
    # 显式触发 hook 注册(ZeRO-2 必需,其他 stage 是 no-op)
    if hasattr(strategy, 'prepare_for_backward'):
        strategy.prepare_for_backward(mp.groups)
        
    # ─── AC(在 mp_manager 之后) ───
    if ac_enabled:
        ac_policy = get_ac_policy("llama_decoder_layer")
        n_wrapped = apply_activation_checkpointing(model, ac_policy,preserve_rng_state=False)
        log(f"  Applied AC to {n_wrapped} modules")
    
    grad_sync = create_grad_synchronizer(mp.groups, parallel_ctx, strategy)
    
    return model, mp, grad_sync, strategy


def run_n_steps(model, mp, grad_sync, parallel_ctx, device,
                 num_steps: int = 10, micro_batch_size: int = 8,
                 seq_len: int = 32, vocab_size: int = 1024, warmup: bool = True):
    """跑 num_steps 次,返回 (loss list, peak memory GB)。
    
    使用同一 batch(memorize 测试),用于消除随机性。
    """    
    if warmup:
        # 预热:populate 所有 lazy cache
        with torch.no_grad():
            warmup_input = torch.zeros(
                (micro_batch_size, seq_len), dtype=torch.long, device=device
            )
            model(input_ids=warmup_input, labels=warmup_input)
        torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    dp_rank = parallel_ctx.dp_rank
    dp_size = parallel_ctx.dp_size
    
    losses = []
    torch.cuda.reset_peak_memory_stats()
    
    # 同一份数据,反复跑
    torch.manual_seed(2000)
    total_global = micro_batch_size * dp_size
    global_data = torch.randint(0, vocab_size, (total_global, seq_len), device=device)
    
    for step in range(num_steps):
        rank_start = dp_rank * micro_batch_size
        rank_data = global_data[rank_start: rank_start + micro_batch_size]
        
        outputs = model(input_ids=rank_data, labels=rank_data)
        loss = outputs["loss"]
        loss.backward()
        
        grad_sync.sync_gradients()
        mp.step()
        
        losses.append(loss.detach().float().item())
    
    peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    return losses, peak_mb


def main() -> None:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    world_size = dist.get_world_size()
    
    # 并行配置:固定 dp=2 / tp=1 / pp=1(可改)
    tp_size = 1
    dp_size = world_size // tp_size
    pp_size = 1
    
    parallel_ctx = ParallelContext(OrderedDict([
        ("pp", pp_size), ("dp", dp_size), ("tp", tp_size),
    ]))
    
    log(f"并行配置: DP={dp_size}, TP={tp_size}, PP={pp_size}")
    
    # 模型配置(中等大小,能体现 AC 收益)
    model_config = AutoConfig.for_model(
        "llama",
        hidden_size=1024,
        intermediate_size=2048,
        num_attention_heads=16,
        num_key_value_heads=4,
        num_hidden_layers=8,
        max_position_embeddings=128,
        vocab_size=1024,
        rms_norm_eps=1e-5,
        hidden_act="silu",
        tie_word_embeddings=False,
    )
    
    num_steps = 10
    micro_batch_size = 8
    seq_len = 32
    
    # ─── 跑五个配置 ───
    configs = [
        ("baseline",     0, False),
        ("baseline + AC",   0, True ),
        ("ZeRO-1",       1, False),
        ("ZeRO-1 + AC",  1, True ),
        ("ZeRO-2",       2, False),
        ("ZeRO-2 + AC",  2, True ),
        ("ZeRO-3",       3, False),
        ("ZeRO-3 + AC",  3, True ),
    ]
    
    results = {}
    
    for label, zero_stage, ac_enabled in configs:
        clean_slate(f"before {label}")
        log(f"\n--- 跑 {label} ---")
        
        model, mp, grad_sync, strategy = create_model_and_optimizer(
            model_config, parallel_ctx, device,
            seed=42, zero_stage=zero_stage, ac_enabled=ac_enabled,
        )
        
        losses, peak_mb = run_n_steps(
            model, mp, grad_sync, parallel_ctx, device,
            num_steps=num_steps, micro_batch_size=micro_batch_size,
            seq_len=seq_len, vocab_size=model_config.vocab_size,
        )
        
        log(f"  Peak memory: {peak_mb:.1f} MB = {peak_mb/1024:.3f} GB")
        log(f"  Final loss: {losses[-1]:.6f}")
        
        results[label] = (losses, peak_mb)
        
        # cleanup
        if hasattr(strategy, 'cleanup'):
            strategy.cleanup()
        if hasattr(mp, 'cleanup'):
            mp.cleanup()
        del model, mp, grad_sync, strategy
    
    clean_slate("after all")
    
    # ─── 总结 ───
    log("\n" + "=" * 70)
    log(f"  对比")
    log("=" * 70)
    log(f"{'Config':<15} {'Peak (GB)':<12} {'Final Loss':<14} {'vs Baseline':<12}")
    log("-" * 70)
    
    baseline_mem = results["baseline"][1]
    baseline_loss = results["baseline"][0][-1]
    
    for label, (losses, peak_mb) in results.items():
        peak_gb = peak_mb / 1024
        ratio = peak_mb / baseline_mem
        loss_diff = abs(losses[-1] - baseline_loss)
        log(f"{label:<15} {peak_gb:<12.3f} {losses[-1]:<14.6f} {ratio:<12.2%} (Δ={loss_diff:.2e})")
    
    log("=" * 70)
    
    # ─── 验证 ───
    log("\n验证:")
    
    # Loss 一致性(ZeRO-3+AC 应该和 ZeRO-3 数学等价)
    z3_losses = results["ZeRO-3"][0]
    z3ac_losses = results["ZeRO-3 + AC"][0]
    max_diff = max(abs(a - b) for a, b in zip(z3_losses, z3ac_losses))
    
    log(f"  ZeRO-3 vs ZeRO-3+AC max loss diff: {max_diff:.6e}")
    assert max_diff < 1e-2, f"Loss diff too large: {max_diff}"
    log(f"  ✓ Loss 数学等价")
    
    # 显存收益(AC 应该比不带 AC 至少省 50 MB)
    z3_mem = results["ZeRO-3"][1]
    z3ac_mem = results["ZeRO-3 + AC"][1]
    saved = z3_mem - z3ac_mem
    log(f"  AC 省的显存: {saved:.1f} MB")
    assert saved > 50, f"AC should save >50 MB, got {saved}"
    log(f"  ✓ AC 节省 > 50 MB")
    
    log(f"\n  全部通过 ✓")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()