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
    print(f"[rank {rank}] {msg}", flush=True)


def make_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=1024,
        hidden_size=256,
        intermediate_size=512,
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

def test_pp2_gpipe_equivalence(num_microbatches: int):
    """Compare PP path (Runner + GPipe + Stage + Comm) vs single-rank baseline.

    Args:
        num_microbatches: N. Global batch is split into N micro-batches.
    """
    log(f"Test: PP2 GPipe equivalence, N={num_microbatches}")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2, f"requires --nproc_per_node=2, got {world_size}"
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    config = make_config()
    num_layers = config.num_hidden_layers
    assert num_layers % 2 == 0
    half = num_layers // 2

    ctx = ParallelContext(OrderedDict([("pp", 2), ("dp", 1), ("tp", 1)]))

    # ── 1) Build full baseline (identical on both ranks via shared seed) ──
    # layer_range=range(0, num_layers) → is_first=True AND is_last=True
    # (full standalone model; runs locally without inter-rank comm)
    baseline = make_model(
        config, ctx, device,
        layer_range=range(0, num_layers), seed=42,
    )
    assert baseline.is_first and baseline.is_last

    # ── 2) Build PP partial for this rank ──
    if rank == 0:
        layer_range = range(0, half)         # is_first=True, is_last=False
    else:
        layer_range = range(half, num_layers)  # is_first=False, is_last=True
    partial = make_model(
        config, ctx, device,
        layer_range=layer_range, seed=999,  # seed doesn't matter, gets overwritten
    )

    # ── 3) Sync partial's weights from baseline ──
    sync_partial_from_full(partial, baseline)

    # ── 4) Generate same data on both ranks ──
    torch.manual_seed(123)  # same seed across ranks → same data
    global_batch_size = num_microbatches  # 1 sample per microbatch for clarity
    seqlen = 32
    input_ids = torch.randint(
        0, config.vocab_size, (global_batch_size, seqlen),
        dtype=torch.long, device=device,
    )
    labels = torch.randint(
        0, config.vocab_size, (global_batch_size, seqlen),
        dtype=torch.long, device=device,
    )

    # ── 5) Run PP path ──
    loss_scale = 1.0 / num_microbatches  # so PP grad ≡ baseline grad
    stage = PipelineStage(partial, ctx, loss_scale=loss_scale)
    comm = PipelineComm(ctx, seqlen=seqlen, hidden_size=config.hidden_size, dtype=torch.bfloat16)
    runner = PipelineRunner(stage, comm)

    actions = gpipe_schedule(
        num_microbatches=num_microbatches,
        is_first=partial.is_first,
        is_last=partial.is_last,
    )
    log(f"  schedule has {len(actions)} actions")

    mb_size = global_batch_size // num_microbatches
    recv_shape = (mb_size, seqlen, config.hidden_size)

    inputs_dict = None
    labels_dict = None
    if partial.is_first:
        input_mbs = split_microbatches(input_ids, num_microbatches)
        inputs_dict = {i: mb for i, mb in enumerate(input_mbs)}
    if partial.is_last:
        label_mbs = split_microbatches(labels, num_microbatches)
        labels_dict = {i: mb for i, mb in enumerate(label_mbs)}

    pp_losses = runner.run(
        actions,
        recv_shape=recv_shape,
        recv_dtype=torch.bfloat16,
        microbatch_inputs=inputs_dict,
        microbatch_labels=labels_dict,
    )
    torch.cuda.synchronize()   # ← 清掉 NCCL stream 的 residual
    pp_grads = clone_grads(partial)
    log(f"  PP path done. {len(pp_grads)} params have grads")

    # ── 6) Run baseline (full model, single forward+backward, local) ──
    baseline_loss = baseline(input_ids, labels=labels)
    baseline_loss.backward()
    baseline_grads = clone_grads(baseline)
    log(f"  baseline loss = {baseline_loss.item():.6f}")

    # ── 7) Compare loss (only last stage owns it) ──
    if partial.is_last:
        assert len(pp_losses) == num_microbatches
        sorted_losses = [pp_losses[i].detach() for i in range(num_microbatches)]
        # ↓ pop_all_losses 存的是 scaled loss (= raw × 1/N),所以 sum 才等价于 baseline mean loss
        pp_total_loss = torch.stack(sorted_losses).sum()
        torch.cuda.synchronize()
        loss_diff = (pp_total_loss - baseline_loss).abs().item()
        log(f"  PP total loss = {pp_total_loss.item():.6f}, "
            f"baseline = {baseline_loss.item():.6f}, diff = {loss_diff:.3e}")
        assert loss_diff < 1e-3, \
            f"PP total loss {pp_total_loss.item()} != baseline {baseline_loss.item()}"
        log(f"  ✓ Loss matches within tolerance")

    # ── 8) Compare grads slice on this rank ──
    baseline_slice = {k: v for k, v in baseline_grads.items() if k in pp_grads}
    assert set(baseline_slice.keys()) == set(pp_grads.keys()), \
        f"key mismatch: pp has {set(pp_grads) - set(baseline_slice)}, " \
        f"baseline slice has {set(baseline_slice) - set(pp_grads)}"
    compare_grads(pp_grads, baseline_slice, tag=f"rank {rank}, N={num_microbatches}")
    log(f"  ✓ All {len(pp_grads)} param grads match baseline slice")

    dist.barrier()  # don't run next test until both ranks done


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

        tests = [
            ("N=4 microbatches (typical)", lambda: test_pp2_gpipe_equivalence(4)),
            ("N=1 microbatch (edge case)", lambda: test_pp2_gpipe_equivalence(1)),
            ("N=2 microbatches",           lambda: test_pp2_gpipe_equivalence(2)),
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