"""Pipeline parallelism schedules.

A schedule is a pure-data function: given the topology (num_microbatches,
is_first, is_last), it returns a list of PPActions to execute. The Runner
then dispatches these actions against a Stage and Comm.

Why pure data:
    - Single rank can introspect / print the schedule for debugging
    - Easy to unit test (no GPU, no comm, no model)
    - Easy to swap schedule algorithms (GPipe → 1F1B → Interleaved) without
      touching Runner / Stage / Comm
"""
from __future__ import annotations

from .action import (
    PPAction,
    Forward, Backward,
    RecvForward, SendForward,
    RecvBackward, SendBackward,
    SendForwardRecvBackward, SendBackwardRecvForward,
)
from .microbatch import split_microbatches
from .stage import PipelineStage
from .comm_ops import PipelineComm


def gpipe_schedule(
    num_microbatches: int,
    is_first: bool,
    is_last: bool,
) -> list[PPAction]:
    """GPipe (Huang et al. 2019): all microbatches finish forward,
    then all do backward in reverse order.

    Args:
        num_microbatches: Number of microbatches in this batch. >= 1.
        is_first: True if this is the first pipeline stage.
        is_last: True if this is the last pipeline stage.

    Returns:
        List of PPActions to execute on THIS rank.

    Memory cost: O(num_microbatches × activation_size) — all forward outputs
        retained until backward. This is GPipe's main drawback vs 1F1B.

    Bubble: 2 × (pp_size - 1) / num_microbatches relative to total work.
        Larger num_microbatches → smaller bubble, but more memory.

    pp_size=1 (degenerate): both is_first and is_last are True. No comm
        actions; the schedule reduces to plain [F, F, ..., B, B, ...].
    """
    if num_microbatches < 1:
        raise ValueError(f"num_microbatches must be >= 1, got {num_microbatches}")

    actions: list[PPAction] = []

    # ── Forward phase: mb 0, 1, ..., N-1 ──
    for mb in range(num_microbatches):
        if not is_first:
            actions.append(RecvForward(mb))   # recv hidden from prev stage
        actions.append(Forward(mb))         # forward through model
        if not is_last:
            actions.append(SendForward(mb))    # send hidden to next stage

    # ── Backward phase: mb N-1, N-2, ..., 0 ──
    # Reversed order matches LIFO of the autograd graph each mb's activations
    # are released after its backward, allowing early memory reuse.
    for mb in reversed(range(num_microbatches)):
        if not is_last:
            actions.append(RecvBackward(mb))   # recv grad_output from next stage
        actions.append(Backward(mb))         # backward, accumulate param grads
        if not is_first:
            actions.append(SendBackward(mb))    # send grad_input to prev stage

    return actions


def one_f_one_b_schedule(
    num_microbatches: int,
    pp_size: int,
    pp_rank: int,
) -> list[PPAction]:
    """1F1B schedule (Megatron-style).

    Three phases for stage at `pp_rank` (0-indexed), P=pp_size, N=num_microbatches:
        1. Warm-up:   min(P - 1 - rank, N) pure forwards — fill the pipeline
        2. Steady:    N - warmup interleaved 1F1B iterations — peak utilization
        3. Cool-down: warmup pure backwards — drain the pipeline

    Memory cost: O(pp_size × activation_size) — far less than GPipe's
        O(num_microbatches × activation_size). This is the production-relevant win.

    Bubble: 2 × (pp_size - 1) / num_microbatches — same as GPipe;
        1F1B doesn't change bubble, only peak memory.

    Combined ops used in steady state to prevent deadlock:
        - SendForwardRecvBackward (SFRB): non-last stages, atomically sends
          F output forward and receives B grad backward
        - SendBackwardRecvForward (SBRF): non-first stages, atomically sends
          B grad backward and receives next F input forward

    Edge cases:
        - pp_size=1 (degenerate): no warmup/cooldown, no comm,
          just [F(0), B(0), F(1), B(1), ...] — interleaved per mb
        - num_microbatches < pp_size: warmup capped at N, num_steady can be 0
          (degenerates to GPipe-like since no F/B overlap possible)
    """
    if num_microbatches < 1:
        raise ValueError(f"num_microbatches must be >= 1, got {num_microbatches}")
    if pp_size < 1:
        raise ValueError(f"pp_size must be >= 1, got {pp_size}")
    if pp_rank < 0 or pp_rank >= pp_size:
        raise ValueError(f"pp_rank must be in [0, {pp_size}), got {pp_rank}")

    is_first = (pp_rank == 0)
    is_last = (pp_rank == pp_size - 1)

    num_warmup = min(pp_size - 1 - pp_rank, num_microbatches)
    num_steady = num_microbatches - num_warmup
    num_cooldown = num_warmup  # by symmetry

    actions: list[PPAction] = []

    # ── Phase 1: Warm-up forwards (fill the pipeline) ──
    for j in range(num_warmup):
        if not is_first:
            actions.append(RecvForward(mb_id=j))
        actions.append(Forward(mb_id=j))
        if not is_last:
            actions.append(SendForward(mb_id=j))

    # ── Phase 2: Steady-state 1F1B ──
    for k in range(num_steady):
        fwd_mb = num_warmup + k
        bwd_mb = k
        is_first_steady = (k == 0)
        is_last_steady = (k == num_steady - 1)

        # F(fwd_mb): need its input
        #   - first_steady & not is_first: warmup did RF(0..warmup-1),
        #     so explicit RF(fwd_mb) here
        #   - subsequent: input arrived via prev iter's SBRF
        #   - is_first: input from caller dict, no recv ever needed
        if is_first_steady and not is_first:
            actions.append(RecvForward(mb_id=fwd_mb))
        actions.append(Forward(mb_id=fwd_mb))

        # Send F output forward + recv B grad backward (combined to avoid deadlock)
        if not is_last:
            actions.append(SendForwardRecvBackward(fwd_mb=fwd_mb, bwd_mb=bwd_mb))
        # else: last stage — no SF, no RB (loss provides grad locally)

        actions.append(Backward(mb_id=bwd_mb))

        # Send B grad backward + recv next F input (combined; or plain SB at end)
        if not is_first:
            if is_last_steady:
                # No more F to recv (cooldown only does B's)
                actions.append(SendBackward(mb_id=bwd_mb))
            else:
                actions.append(SendBackwardRecvForward(
                    bwd_mb=bwd_mb, fwd_mb=fwd_mb + 1,
                ))
        # else: first stage — no SB, no need for RF (input always from caller)

    # ── Phase 3: Cool-down backwards (drain the pipeline) ──
    for j in range(num_cooldown):
        bwd_mb = num_steady + j
        if not is_last:
            actions.append(RecvBackward(mb_id=bwd_mb))
        actions.append(Backward(mb_id=bwd_mb))
        if not is_first:
            actions.append(SendBackward(mb_id=bwd_mb))

    return actions