"""Integration test for PipelineRunner: pp_size=2 GPipe numerical equivalence.

torchrun --nproc_per_node=2 femtotron/test/integration/test_pipeline_runner.py

Validates that the full pipeline stack (Stage + Comm + Runner + Schedule)
produces gradients numerically equivalent to a single-rank full-model baseline,
within ~2 bf16 ULPs (tensor-level relative tolerance).

Setup:
    - 4-layer Llama, split as rank0=[0,1] | rank1=[2,3]
    - Both ranks build the SAME full baseline (seed 42) for comparison
    - PP partial weights are copied from baseline to ensure bit-exact start
    - GPipe schedule with N=4 microbatches
    - Each rank verifies its slice of grads against baseline's slice
"""
from __future__ import annotations

from collections import OrderedDict
import os
import signal
import sys

import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import LlamaConfig
from uri_template import partial

from femtotron.parallel_context import ParallelContext
from femtotron.parallel.pipeline_parallel.comm_ops import PipelineComm
from femtotron.parallel.pipeline_parallel.microbatch import split_microbatches
from femtotron.parallel.pipeline_parallel.runner import PipelineRunner
from femtotron.parallel.pipeline_parallel.schedule import gpipe_schedule
from femtotron.parallel.pipeline_parallel.stage import PipelineStage
from femtotron.model.llama_causal import LlamaForCausalLM
from femtotron.parallel.pipeline_parallel.schedule import gpipe_schedule, one_f_one_b_schedule
from femtotron.parallel.pipeline_parallel.partition import partition_layers


# ────────────────────── helpers ──────────────────────

def init_distributed():
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(
        backend="nccl",
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    torch.cuda.set_device(local_rank)
    return local_rank

def log(msg: str) -> None:
    """Print prefixed by rank to keep multi-rank logs readable."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        print(f"[rank {rank}] {msg}", flush=True)


def make_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=1024,
        hidden_size=1024,
        intermediate_size=2048,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=128,
        tie_word_embeddings=False,
        attn_implementation="sdpa",
    )


def _reset_rotary_inv_freq(rotary_emb, config, device):
    """Recompute inv_freq after `to_empty`. (M3b debugging finding.)
    See test_pipeline_stage.py for the full story."""
    base = (config.rope_parameters["rope_theta"]
            if hasattr(config, "rope_parameters")
            else getattr(config, "rope_theta", 10000.0))
    dim = (getattr(config, "head_dim", None)
           or (config.hidden_size // config.num_attention_heads))
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(
            device=device, dtype=torch.float
        ) / dim)
    )
    rotary_emb.inv_freq.copy_(inv_freq.to(rotary_emb.inv_freq.dtype))
    if hasattr(rotary_emb, "original_inv_freq"):
        rotary_emb.original_inv_freq.copy_(inv_freq.to(rotary_emb.original_inv_freq.dtype))


def make_model(config, ctx, device, *, layer_range, seed):
    """Meta-construct → to_empty → reset rotary → init params → bf16."""
    with torch.device("meta"):
        model = LlamaForCausalLM(config, ctx, layer_range=layer_range)
    model.to_empty(device=device)
    _reset_rotary_inv_freq(model.model.rotary_emb, config, device)
    torch.manual_seed(seed)
    for _, p in model.named_parameters():
        nn.init.normal_(p, mean=0.0, std=0.02)
    return model.bfloat16()


def sync_partial_from_full(partial, full) -> None:
    """Copy params from full's state_dict to partial.

    partial's param names are a subset of full's (same naming convention).
    After this call, partial.params ≡ full.params on their intersection.
    Buffers (rotary inv_freq) already match because both went through
    _reset_rotary_inv_freq with the same config.
    """
    full_sd = full.state_dict()
    partial_sd_keys = set(partial.state_dict().keys())
    new_sd = {}
    for k in partial_sd_keys:
        if k not in full_sd:
            raise KeyError(
                f"partial param '{k}' not in full state_dict. "
                f"Available: {sorted(full_sd.keys())[:5]}..."
            )
        new_sd[k] = full_sd[k].clone()
    missing, unexpected = partial.load_state_dict(new_sd, strict=True)
    assert not missing and not unexpected, \
        f"sync mismatch — missing={missing}, unexpected={unexpected}"

    # Sanity: every partial param should now equal baseline's
    full_named = dict(full.named_parameters())
    for name, p in partial.named_parameters():
        if not torch.equal(p, full_named[name]):
            raise RuntimeError(f"Sync failed for {name}")


def compare_grads(grads_a, grads_b, tag, *, atol=1e-8, rtol=1.5e-2):
    """Tensor-level relative-tolerance grad compare (see M3b notes)."""
    assert set(grads_a.keys()) == set(grads_b.keys()), \
        f"{tag}: param key sets differ ({set(grads_a)^set(grads_b)})"
    mismatches = []
    for name in grads_a:
        ga, gb = grads_a[name], grads_b[name]
        max_diff = (ga - gb).abs().max().item()
        max_ref = gb.abs().max().item()
        tol = atol + rtol * max_ref
        if max_diff > tol:
            ulp_at_max = max(max_ref * (2 ** -7), 1e-30)
            ulps = max_diff / ulp_at_max
            mismatches.append((name, max_diff, max_ref, ulps))
    if mismatches:
        details = "\n".join(
            f"    {n}: max_diff={d:.3e}, max_ref={r:.3e}, = {u:.2f} ULPs"
            for n, d, r, u in mismatches[:10]
        )
        raise AssertionError(
            f"{tag}: grad mismatch:\n{details}\n"
            f"(> ~2 ULPs of max ref → likely real bug)"
        )


def clone_grads(model) -> dict[str, torch.Tensor]:
    return {n: p.grad.detach().clone()
            for n, p in model.named_parameters()
            if p.grad is not None}


# ────────────────────── tests ──────────────────────

def test_pp_schedule_equivalence(
    pp_size: int,
    num_microbatches: int,
    schedule_name: str,  # "gpipe" or "1f1b"
):
    """Compare PP via Runner vs single-rank baseline.

    Same framework as M3c, but parameterized:
        - pp_size: 2, 3, ... (must match torchrun world_size)
        - num_microbatches: any N >= 1
        - schedule_name: "gpipe" or "1f1b"
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Skip if world_size doesn't match (so one file handles --nproc=2 and =3)
    if world_size != pp_size:
        if rank == 0:
            log(f"  [SKIP] requires world_size={pp_size}, got {world_size}")
        dist.barrier()
        return

    log(f"Test: pp_size={pp_size}, N={num_microbatches}, schedule={schedule_name}")
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    config = make_config()
    num_layers = config.num_hidden_layers

    ctx = ParallelContext(
        OrderedDict([("pp", pp_size), ("dp", 1), ("tp", 1)])
    )

    # ── 1) Build full baseline (identical on all ranks via seed=42) ──
    baseline = make_model(
        config, ctx, device,
        layer_range=range(0, num_layers), seed=42,
    )
    assert baseline.is_first and baseline.is_last

    # ── 2) Build PP partial for this rank ──
    # partition_layers returns layer index lists per rank
    partitions = partition_layers(num_layers, pp_size, strategy="uniform")
    my_layers = partitions[rank]
    layer_range = range(my_layers[0], my_layers[-1] + 1)
    log(f"  rank {rank}: layers {list(layer_range)}")

    partial = make_model(
        config, ctx, device,
        layer_range=layer_range, seed=999,  # gets overwritten by sync
    )
    sync_partial_from_full(partial, baseline)

    # ── 3) Generate same data on all ranks ──
    torch.manual_seed(123)
    mb_size = 2  # >1 to keep activations substantive
    global_batch_size = num_microbatches * mb_size
    seqlen = 1024
    input_ids = torch.randint(
        0, config.vocab_size, (global_batch_size, seqlen),
        dtype=torch.long, device=device,
    )
    labels = torch.randint(
        0, config.vocab_size, (global_batch_size, seqlen),
        dtype=torch.long, device=device,
    )

    # ── 4) Run PP path with selected schedule ──
    loss_scale = 1.0 / num_microbatches
    stage = PipelineStage(partial, ctx, loss_scale=loss_scale)
    comm = PipelineComm(ctx, seqlen=seqlen, hidden_size=config.hidden_size, dtype=torch.bfloat16)
    mb_size = global_batch_size // num_microbatches
    recv_shape = (mb_size, seqlen, config.hidden_size)
    runner = PipelineRunner(
        stage, comm,
        schedule_name=schedule_name,
        num_microbatches=num_microbatches,
        recv_shape=recv_shape,
        recv_dtype=torch.bfloat16,
    )
    log(f"  schedule has {len(runner.actions)} actions")


    torch.cuda.reset_peak_memory_stats()
    # run_step 处理 microbatch split,trainer 和 test 共用同一个接口
    batch = {"input_ids": input_ids, "labels": labels}
    pp_losses = runner.run_step(batch)
    torch.cuda.synchronize()  # M3c lesson: flush NCCL stream residual
    gpipe_peak = torch.cuda.max_memory_allocated() / 1024**2
    log(f"  Peak memory={gpipe_peak:.1f}MB")
    pp_grads = clone_grads(partial)
    log(f"  PP path done ({len(pp_grads)} param grads)")

    # ── 5) Run baseline ──
    baseline_out = baseline(input_ids, labels=labels)
    baseline_loss = baseline_out["loss"]          # ← 解 dict
    baseline_loss.backward()
    baseline_grads = clone_grads(baseline)

    # ── 6) Compare loss (last stage owns it) ──
    if partial.is_last:
        sorted_losses = [pp_losses[i].detach() for i in range(num_microbatches)]
        pp_total_loss = torch.stack(sorted_losses).sum()
        loss_diff = (pp_total_loss - baseline_loss).abs().item()
        log(f"  PP total loss = {pp_total_loss.item():.6f}, "
            f"baseline = {baseline_loss.item():.6f}, diff = {loss_diff:.3e}")
        assert loss_diff < 1e-3, \
            f"PP loss {pp_total_loss.item()} != baseline {baseline_loss.item()}"
        log(f"  ✓ Loss matches")

    # ── 7) Compare grads (each rank's slice) ──
    baseline_slice = {k: v for k, v in baseline_grads.items() if k in pp_grads}
    assert set(baseline_slice.keys()) == set(pp_grads.keys()), \
        f"key mismatch: extras in pp={set(pp_grads) - set(baseline_slice)}"
    compare_grads(
        pp_grads, baseline_slice,
        tag=f"rank{rank} pp{pp_size} N={num_microbatches} {schedule_name}",
    )
    log(f"  ✓ {len(pp_grads)} param grads match baseline slice")

    dist.barrier()

# ────────────────────── main ──────────────────────

def _timeout_handler(signum, frame):
    raise RuntimeError("Test timed out — likely a deadlock in P2P comm")


def main():
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(120)
    try:
        init_distributed()
        torch.backends.cuda.matmul.allow_tf32 = False  # be strict about precision
        torch.backends.cudnn.allow_tf32 = False

        tests = [            # ── PP=2: existing GPipe (regression) + new 1F1B ──
            ("PP=2 GPipe N=4", lambda: test_pp_schedule_equivalence(2, 4, "gpipe")),
            ("PP=2 GPipe N=1", lambda: test_pp_schedule_equivalence(2, 1, "gpipe")),
            ("PP=2 GPipe N=2", lambda: test_pp_schedule_equivalence(2, 2, "gpipe")),
            ("PP=2 1F1B  N=4", lambda: test_pp_schedule_equivalence(2, 4, "1f1b")),
            ("PP=2 1F1B  N=2", lambda: test_pp_schedule_equivalence(2, 2, "1f1b")),

            # ── PP=3: first time exercising mid stage ──
            ("PP=3 GPipe N=3", lambda: test_pp_schedule_equivalence(3, 3, "gpipe")),
            ("PP=3 GPipe N=6", lambda: test_pp_schedule_equivalence(3, 6, "gpipe")),
            ("PP=3 1F1B  N=3", lambda: test_pp_schedule_equivalence(3, 3, "1f1b")),
            ("PP=3 1F1B  N=6", lambda: test_pp_schedule_equivalence(3, 6, "1f1b")),
        ]
        for desc, fn in tests:
            log(f"\n=== {desc} ===")
            fn()

        if dist.get_rank() == 0:
            print(f"\n✅ All {len(tests)} PP2 integration tests passed\n", flush=True)
    finally:
        signal.alarm(0)
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()