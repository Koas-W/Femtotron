"""Pipeline parallelism runner: stateless action dispatcher.

The Runner takes a schedule (list of PPActions) and executes each action
against a Stage (per-mb tensor state) and a Comm (P2P transport).

Design principle: no per-mb state in the Runner. All state lives in the
Stage's per-mb dicts. The Runner is pure dispatch.
"""
from __future__ import annotations

import torch

from .action import (
    PPAction,
    Forward, Backward,
    RecvForward, SendForward,
    RecvBackward, SendBackward,
    SendForwardRecvBackward, SendBackwardRecvForward,
)
from .stage import PipelineStage
from .comm_ops import PipelineComm
from femtotron.parallel.pipeline_parallel import stage


class PipelineRunner:
    """Execute an action stream against a Stage and Comm.

    Not responsible for:
        - Microbatch splitting (use `split_microbatches` separately)
        - Schedule generation (use `gpipe_schedule` etc. separately)
        - Grad sync across DP ranks (Trainer handles via GradSynchronizer)
        - Optimizer step (Trainer handles)
        - Loss aggregation across microbatches (caller handles)
    """

    def __init__(self, stage: PipelineStage, comm: PipelineComm):
        self.stage = stage
        self.comm = comm
        self._device = next(stage.model.parameters()).device

    def run(
        self,
        actions: list[PPAction],
        *,
        recv_shape: tuple[int, ...],
        recv_dtype: torch.dtype,
        microbatch_inputs: dict[int, torch.Tensor] | None = None,
        microbatch_labels: dict[int, torch.Tensor] | None = None,
    ) -> dict[int, torch.Tensor]:
        """Execute the action stream.

        Args:
            actions: From a schedule fn, e.g. `gpipe_schedule(...)`.
            recv_shape: Shape of tensors received via RecvForward / RecvBackward.
                Per-call (not at construction) so dynamic seqlen works.
            recv_dtype: Dtype of received tensors.
            microbatch_inputs: For FIRST stage only, dict mb_id → input.
                Required if stage is_first; ignored otherwise.
            microbatch_labels: For LAST stage only, dict mb_id → labels.
                If None on last stage, model returns logits, no loss recorded.

        Returns:
            dict[mb_id, loss_scalar] for last stage; empty dict otherwise.
            Detached, no autograd graph attached. Caller aggregates (mean / sum)
            as needed for reporting.

        After run() returns:
            - All param grads are accumulated in self.stage.model.parameters()
            - Stage state is clean (asserted) — safe to call run() again
            - No optimizer.step() or grad sync has happened
        """
        if self.stage.is_first and microbatch_inputs is None:
            raise ValueError(
                "First stage requires microbatch_inputs dict; got None"
            )

        for action in actions:
            self._dispatch(
                action, microbatch_inputs, microbatch_labels,
                recv_shape, recv_dtype,
            )

        losses = self.stage.pop_all_losses()
        self.stage.assert_clean()
        return losses

    # ────────────────────── private ──────────────────────

    def _dispatch(
        self,
        action: PPAction,
        inputs: dict[int, torch.Tensor] | None,
        labels: dict[int, torch.Tensor] | None,
        recv_shape: tuple[int, ...],
        recv_dtype: torch.dtype,
    ) -> None:
        # Compute actions(不变)
        if isinstance(action, Forward):
            self._do_forward(action.mb_id, inputs, labels)
        elif isinstance(action, Backward):
            self.stage.backward(action.mb_id)
        
        # ── Single-direction comm:caller 自己分配 buf ──
        elif isinstance(action, RecvForward):
            buf = torch.empty(recv_shape, dtype=recv_dtype, device=self._device)
            self.comm.recv_forward(out=buf)
            self.stage.stage_input(action.mb_id, buf)
        elif isinstance(action, SendForward):
            out = self.stage.get_output(action.mb_id)
            self.comm.send_forward(out)
        elif isinstance(action, RecvBackward):
            buf = torch.empty(recv_shape, dtype=recv_dtype, device=self._device)
            self.comm.recv_backward(out=buf)
            self.stage.stage_grad(action.mb_id, buf)
        elif isinstance(action, SendBackward):
            g = self.stage.get_input_grad(action.mb_id)
            self.comm.send_backward(g)
        
        # ── Combined comm(1F1B 用,GPipe 不会触发) ──
        elif isinstance(action, SendForwardRecvBackward):
            act = self.stage.get_output(action.fwd_mb)
            recv_buf = torch.empty(recv_shape, dtype=recv_dtype, device=self._device)
            self.comm.send_forward_recv_backward(act, out=recv_buf)
            self.stage.stage_grad(action.bwd_mb, recv_buf)

        elif isinstance(action, SendBackwardRecvForward):
            grad = self.stage.get_input_grad(action.bwd_mb)
            recv_buf = torch.empty(recv_shape, dtype=recv_dtype, device=self._device)
            self.comm.send_backward_recv_forward(grad, out=recv_buf)
            self.stage.stage_input(action.fwd_mb, recv_buf)
        else:
            raise NotImplementedError(f"Unsupported action type: {type(action).__name__}")

    def _do_forward(
        self,
        mb_id: int,
        inputs: dict[int, torch.Tensor] | None,
        labels: dict[int, torch.Tensor] | None,
    ) -> None:
        # First stage: input from caller dict (not from comm)
        if self.stage.is_first:
            assert inputs is not None  # checked at run() entry
            if mb_id not in inputs:
                raise RuntimeError(
                    f"First stage missing input for microbatch {mb_id}; "
                    f"available: {sorted(inputs.keys())}"
                )
            self.stage.stage_input(mb_id, inputs[mb_id])
        # Last stage: labels from caller dict (if doing training)
        if self.stage.is_last and labels is not None and mb_id in labels:
            self.stage.stage_labels(mb_id, labels[mb_id])
        # Forward through model — reads _inputs, writes _outputs (and
        # _loss_values if last stage with labels).
        self.stage.forward(mb_id)