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

from .action import PPAction, Forward, Backward, RecvForward, SendForward, RecvBackward, SendBackward, SendForwardRecvBackward
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