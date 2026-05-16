"""
femtotron/parallel/pipeline/comm.py

PipelineComm: P2P communication primitives for pipeline parallelism.

所有通信通过 torch.distributed.batch_isend_irecv,保证相邻 stage 之间不死锁。

边界 stage(第一/最后)对应方向的操作是 no-op:
    - 第一 stage: recv_forward() → None, send_backward(g) → no-op
    - 最后 stage: recv_backward() → None, send_forward(t) → no-op

Combined ops (send_X_recv_Y) 在一个 batch_isend_irecv 里原子完成。
这是 1F1B 避免死锁的关键:中间 stage 同时要"发 fwd 给下游 + 收 bwd 从下游",
必须 atomic 否则跟对面对跑死锁。

职责边界:
    - Comm 只管搬数据
    - 不管 autograd(returned tensor 没有 grad_fn,Stage 自行 .requires_grad_)
    - 不管 microbatch ID(那是 Runner/Stage 的事)
    - 不管何时调用(那是 Schedule 的事)
"""

from __future__ import annotations
from typing import Optional, Sequence

import torch
import torch.distributed as dist
from torch.distributed import P2POp

from femtotron.parallel_context import ParallelContext


class PipelineComm:
    """PP 相邻 stage 之间的 P2P 通信原语。

    Args:
        parallel_ctx: 提供 pp_group / pp_prev_rank / pp_next_rank / local_rank。
        microbatch_size: 单个 microbatch 的 batch dim。
        seqlen: 序列长度(静态)。
        hidden_size: hidden 维度(模型层间 activation 的最后一维)。
        dtype: activation/grad 传输 dtype。从 MixedPrecisionManager.compute_dtype
            传入(通常 bf16);**不要硬编码**。

    Note on buffer allocation:
        recv 默认 fresh allocate(CUDA caching allocator 下 ~1-5us,可忽略)。
        Caller 也可以传 `out=` 自带 buffer 实现 pool 复用——但 1F1B 下 activation
        要 cache 给后续 backward,naive 复用 single buffer 会被覆盖,**必须 clone**,
        而 clone 比 fresh alloc 更贵(memcpy 64MB ~20us vs alloc ~5us),所以
        默认走 fresh alloc 反而最优。Stage 层之后若做 pool,用 out= 即可。
    """

    def __init__(
        self,
        parallel_ctx: ParallelContext,
        microbatch_size: int,
        seqlen: int,
        hidden_size: int,
        dtype: torch.dtype,
    ) -> None:
        self.ctx = parallel_ctx
        self._act_shape = (microbatch_size, seqlen, hidden_size)
        self._dtype = dtype
        # device 从 parallel_ctx 推出
        self._device = torch.device(f"cuda:{self.ctx.world_rank}")

    # ────────────────────────────────────────────────────────────────
    # Public properties
    # ────────────────────────────────────────────────────────────────

    @property
    def is_first_stage(self) -> bool:
        return self.ctx.pp_prev_rank is None

    @property
    def is_last_stage(self) -> bool:
        return self.ctx.pp_next_rank is None

    @property
    def act_shape(self) -> tuple[int, int, int]:
        return self._act_shape

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    # ────────────────────────────────────────────────────────────────
    # Single-direction primitives
    # ────────────────────────────────────────────────────────────────

    def send_forward(self, act: torch.Tensor) -> None:
        """Send activation to next stage. Last stage 上是 no-op。"""
        if self.is_last_stage:
            return
        self._batch_p2p([(dist.isend, act, self.ctx.pp_next_rank)])

    def recv_forward(self, out: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """Recv activation from prev stage. First stage 上返回 None。

        Args:
            out: 可选 buffer 复用。None 则 fresh allocate。
        """
        if self.is_first_stage:
            return None
        buf = self._alloc_or_check(out)
        self._batch_p2p([(dist.irecv, buf, self.ctx.pp_prev_rank)])
        return buf

    def send_backward(self, grad: torch.Tensor) -> None:
        """Send input-grad to prev stage. First stage 上是 no-op。"""
        if self.is_first_stage:
            return
        self._batch_p2p([(dist.isend, grad, self.ctx.pp_prev_rank)])

    def recv_backward(self, out: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """Recv output-grad from next stage. Last stage 上返回 None。"""
        if self.is_last_stage:
            return None
        buf = self._alloc_or_check(out)
        self._batch_p2p([(dist.irecv, buf, self.ctx.pp_next_rank)])
        return buf

    # ────────────────────────────────────────────────────────────────
    # Combined primitives (atomicity = 反死锁的核心)
    # ────────────────────────────────────────────────────────────────

    def send_forward_recv_backward(
        self,
        act: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Atomic: 发 fwd 给下游 + 收 bwd 从下游。

        1F1B 中间 stage 的核心原语。Last stage 上等价于 no-op 返回 None。

        死锁防御:这两个 op 在一个 batch_isend_irecv 里,NCCL 内部协调发起,
        不会因为对端没准备 recv 就 block 在 send 上。
        """
        if self.is_last_stage:
            return None
        grad_buf = self._alloc_or_check(out)
        self._batch_p2p([
            (dist.isend, act, self.ctx.pp_next_rank),
            (dist.irecv, grad_buf, self.ctx.pp_next_rank),
        ])
        return grad_buf

    def send_backward_recv_forward(
        self,
        grad: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Atomic: 发 bwd 给上游 + 收 fwd 从上游。

        1F1B 中间 stage 的另一个核心原语。First stage 上等价于 no-op 返回 None。
        """
        if self.is_first_stage:
            return None
        act_buf = self._alloc_or_check(out)
        self._batch_p2p([
            (dist.isend, grad, self.ctx.pp_prev_rank),
            (dist.irecv, act_buf, self.ctx.pp_prev_rank),
        ])
        return act_buf

    # ────────────────────────────────────────────────────────────────
    # Internal helpers
    # ────────────────────────────────────────────────────────────────

    def _alloc_or_check(self, out: Optional[torch.Tensor]) -> torch.Tensor:
        """如果 out 是 None 就 fresh allocate;否则校验 shape/dtype/device 后直接用。"""
        if out is None:
            return torch.empty(self._act_shape, dtype=self._dtype, device=self._device)
        # Caller 复用 buffer——校验一致性,避免静默错位
        if tuple(out.shape) != self._act_shape:
            raise ValueError(
                f"out buffer shape {tuple(out.shape)} != expected {self._act_shape}"
            )
        if out.dtype != self._dtype:
            raise ValueError(
                f"out buffer dtype {out.dtype} != expected {self._dtype}"
            )
        if out.device != self._device:
            raise ValueError(
                f"out buffer device {out.device} != expected {self._device}"
            )
        return out

    def _batch_p2p(self, op_specs: Sequence[tuple]) -> None:
        """把 (op_fn, tensor, peer) 列表批量提交到 batch_isend_irecv 并 wait 完。

        所有 op 共享同一个 pp_group。
        """
        ops = [
            P2POp(op_fn, tensor, peer, self.ctx.pp_group)
            for op_fn, tensor, peer in op_specs
        ]
        # batch_isend_irecv 接受空列表会报错,所以调用方进入这里之前已经保证非空
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()