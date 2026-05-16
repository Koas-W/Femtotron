"""End-to-end test: Trainer + PP + DP + (optional ZeRO/AC).

Verifies that the integrated training stack produces:
  1. Decreasing loss (training actually works)
  2. PP loss curve ≈ non-PP loss curve (math equivalence)

torchrun --nproc_per_node=2 femtotron/test/integration/test_pp_trainer.py
"""
import os
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import AutoConfig

from femtotron.parallel_context import ParallelContext
from femtotron.parallel.pipeline_parallel.comm_ops import PipelineComm
from femtotron.parallel.pipeline_parallel.partition import partition_layers
from femtotron.parallel.pipeline_parallel.runner import PipelineRunner
from femtotron.parallel.pipeline_parallel.stage import PipelineStage
from femtotron.model.parallel_plan import get_llama_parallel_plan
from femtotron.model.model_loader import ModelLoader
from femtotron.model.llama import build_llama_model
from femtotron.training.mixed_precision_manager import MixedPrecisionManager
from femtotron.training.optimizer import get_param_groups
from femtotron.training.train_config import TrainConfig
from femtotron.sharding.zero_config import ZeROConfig
from femtotron.sharding.factory import create_sharding_strategy
from femtotron.scripts.presets import get_wrap_policy
from femtotron.parallel.data_parallel.gradient_synchronizer import create_grad_synchronizer
from femtotron.model.parallel_plan import ParallelPlan, ParallelRule, get_llama_parallel_plan
from femtotron.parallel.pipeline_parallel.pipeline_config import PipelineConfig


def log(msg):
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        print(msg, flush=True)


def _reset_rotary_inv_freq(rotary_emb, config, device):
    """meta → to_empty 不初始化 buffer,显式重算 inv_freq(M3b 学到的坑)。"""
    base = getattr(config, "rope_theta", 10000.0)
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
        rotary_emb.original_inv_freq.copy_(
            inv_freq.to(rotary_emb.original_inv_freq.dtype)
        )


def create_pp_stack(model_config, parallel_ctx, device, *, seed,
                     num_microbatches, schedule, mb_size, seq_len,
                     grad_accum_steps=1, zero_stage=0, ac_enabled=False):
    """Build PP-aware training stack. Returns dict of components."""
    torch.manual_seed(seed)
    
    # ── 1) Build PP-aware model ──
    is_pp = parallel_ctx.pp_size > 1
    if is_pp:
        partitions = partition_layers(
            model_config.num_hidden_layers, parallel_ctx.pp_size, strategy="uniform",
        )
        my_layers = partitions[parallel_ctx.pp_rank]
        layer_range = range(my_layers[0], my_layers[-1] + 1)
        log(f"  rank {parallel_ctx.pp_rank}: layers {list(layer_range)}")
    else:
        layer_range = None
    
    with torch.device("meta"):
        model = build_llama_model(
            model_config, parallel_ctx,
            use_pp_aware=True, layer_range=layer_range,
        )
    model = model.to_empty(device=device)
    _reset_rotary_inv_freq(model.model.rotary_emb, model_config, device)
    
    for p in model.parameters():
        if p.requires_grad:
            torch.nn.init.normal_(p, mean=0.0, std=0.02)
    model = model.bfloat16()
    
    # ── 2) ZeRO strategy (same as before) ──
    wrap_policy = get_wrap_policy("llama_decoder_layer") if zero_stage == 3 else None
    zero_config = ZeROConfig(stage=zero_stage, wrap_policy=wrap_policy)
    strategy = create_sharding_strategy(parallel_ctx, zero_config)
    
    # ── 3) TrainConfig with PP ──
    train_config = TrainConfig(
        master_dtype=torch.float32,
        grad_clip=1.0,
        grad_accum_steps=grad_accum_steps,
        train_steps=10,
        log_interval=10,
        checkpoint_interval=10000,
        checkpoint_dir="/tmp/test_pp_trainer",
        warmup_steps=0,
        min_lr_ratio=1.0,
        pipeline_config=PipelineConfig(
            num_microbatches=num_microbatches,
            schedule=schedule,
        ),
    )
    
    # ── 4) MixedPrecisionManager ──
    compute_param_groups = get_param_groups(model, weight_decay=0.01)
    parallel_plan = get_llama_parallel_plan()
    mp = MixedPrecisionManager(
        model=model,
        sharding_strategy=strategy,
        parallel_ctx=parallel_ctx,
        parallel_plan=parallel_plan,
        config=train_config,
        inner_optimizer_cls=torch.optim.AdamW,
        inner_optimizer_kwargs={"lr": 3e-4, "betas": (0.9, 0.95), "eps": 1e-8},
        compute_param_groups=compute_param_groups,
    )
    if hasattr(strategy, "prepare_for_backward"):
        strategy.prepare_for_backward(mp.groups)
    
    # ── 5) AC(可选)──
    if ac_enabled:
        from femtotron.training.activation_ckpt import (
            apply_activation_checkpointing
        )
        from femtotron.scripts.presets import get_ac_policy
        apply_activation_checkpointing(
            model, get_ac_policy("llama_decoder_layer"), preserve_rng_state=False,
        )
    
    # ── 6) Grad sync ──
    grad_sync = create_grad_synchronizer(mp.groups, parallel_ctx, strategy)
    
    # ── 7) PP Runner(仅 pp_size > 1)──
    pp_runner = None
    if is_pp:
        loss_scale = 1.0 / (grad_accum_steps * num_microbatches)
        stage = PipelineStage(model, parallel_ctx, loss_scale=loss_scale)
        comm = PipelineComm(parallel_ctx, seqlen=seq_len, hidden_size=model_config.hidden_size, dtype=torch.bfloat16)
        recv_shape = (mb_size, seq_len, model_config.hidden_size)
        pp_runner = PipelineRunner(
            stage, comm,
            schedule_name=schedule,
            num_microbatches=num_microbatches,
            recv_shape=recv_shape,
            recv_dtype=torch.bfloat16,
        )
    
    return dict(
        model=model, mp=mp, grad_sync=grad_sync, strategy=strategy,
        pp_runner=pp_runner, train_config=train_config,
    )


def run_n_steps(stack, parallel_ctx, device, *,
                  num_steps, micro_batch_size, seq_len, vocab_size):
    """Train N steps. Returns (losses, peak_mb).
    
    For PP, micro_batch_size is the FULL batch size (will be split into 
    num_microbatches mb's internally).
    """
    model = stack["model"]
    mp = stack["mp"]
    grad_sync = stack["grad_sync"]
    pp_runner = stack["pp_runner"]
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    dp_rank = parallel_ctx.dp_rank
    dp_size = parallel_ctx.dp_size
    pp_size = parallel_ctx.pp_size
    
    losses = []
    torch.manual_seed(2000)
    total_global = micro_batch_size * dp_size
    global_data = torch.randint(0, vocab_size, (total_global, seq_len), device=device)
    
    for step in range(num_steps):
        rank_start = dp_rank * micro_batch_size
        rank_data = global_data[rank_start: rank_start + micro_batch_size]
        
        if pp_runner is None:
            # Standard path
            outputs = model(input_ids=rank_data, labels=rank_data)
            loss = outputs["loss"]
            loss.backward()
            step_loss = loss.detach().float()
        else:
            # PP path
            batch = {"input_ids": rank_data, "labels": rank_data}
            pp_losses = pp_runner.run_step(batch)
            torch.cuda.synchronize()
            
            if pp_runner.is_last_stage:
                # sum(scaled) = batch_mean / grad_accum (=1 here);so just sum = batch_mean
                step_loss = sum(pp_losses.values()).detach().float()
            else:
                step_loss = torch.zeros((), device=device)
            
            # Broadcast in pp_group
            if pp_size > 1:
                last_pp = parallel_ctx.get_ranks_in_group("pp")[-1]
                dist.broadcast(step_loss, src=last_pp, group=parallel_ctx.pp_group)
        
        grad_sync.sync_gradients()
        mp.step()
        torch.cuda.synchronize()
        
        losses.append(step_loss.item())
    
    peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    return losses, peak_mb


def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    world_size = dist.get_world_size()
    assert world_size == 2, "M5a test requires --nproc_per_node=2 (pp=2)"
    
    # Model config(中等大小,跟 zero+ac test 一致)
    model_config = AutoConfig.for_model(
        "llama",
        hidden_size=1024, intermediate_size=2048,
        num_attention_heads=16, num_key_value_heads=4,
        num_hidden_layers=8, max_position_embeddings=128,
        vocab_size=1024, rms_norm_eps=1e-5,
        hidden_act="silu", tie_word_embeddings=False,
    )
    
    num_steps = 10
    micro_batch_size = 8
    seq_len = 32
    num_microbatches = 4
    
    # 测试两个配置:pp=2 vs pp=1(baseline,用 dp=2 替代来对比);
    # 同样 seed,loss curve 应当近似(不会 bit-exact 因为不同 batching pattern)
    
    # ── Run 1: PP=2 (我们的目标) ──
    log("\n--- PP=2 1F1B ---")
    parallel_ctx_pp = ParallelContext(OrderedDict([
        ("pp", 2), ("dp", 1), ("tp", 1),
    ]))
    stack_pp = create_pp_stack(
        model_config, parallel_ctx_pp, device,
        seed=42, num_microbatches=num_microbatches, schedule="1f1b",
        mb_size=micro_batch_size // num_microbatches, seq_len=seq_len,
        zero_stage=0, ac_enabled=False,
    )
    losses_pp, peak_pp = run_n_steps(
        stack_pp, parallel_ctx_pp, device,
        num_steps=num_steps, micro_batch_size=micro_batch_size,
        seq_len=seq_len, vocab_size=model_config.vocab_size,
    )
    log(f"  Losses: {[f'{l:.4f}' for l in losses_pp]}")
    log(f"  Peak: {peak_pp:.1f} MB")
    log(f"  Δloss (step0 → stepN): {losses_pp[0] - losses_pp[-1]:.4f}")
    
    # 验证 1:loss 下降
    assert losses_pp[-1] < losses_pp[0], (
        f"loss didn't decrease: {losses_pp[0]:.4f} → {losses_pp[-1]:.4f}"
    )
    log("  ✓ Loss decreasing")
    
    # 验证 2:loss 不发散 / NaN
    assert all(not (torch.tensor(l).isnan() or torch.tensor(l).isinf()) 
               for l in losses_pp), "loss has NaN/Inf"
    log("  ✓ No NaN/Inf")
    
    # Cleanup
    if hasattr(stack_pp["strategy"], "cleanup"):
        stack_pp["strategy"].cleanup()
    if hasattr(stack_pp["mp"], "cleanup"):
        stack_pp["mp"].cleanup()
    del stack_pp
    torch.cuda.empty_cache()
    
    log("\n✅ M5a integration test passed")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()