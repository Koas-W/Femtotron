"""
femtotron/parallel/pipeline/action.py

PP 调度的 action 数据载体。

Schedule 生成 list[PPAction],Runner 按列表 dispatch。Action 是不可变 dataclass
(frozen=True),无逻辑、无 tensor、纯数据——这样 schedule 可以预生成、序列化、
单元测试、跨进程对比。

所有 action 都是 PPAction 的子类。Runner 通过 isinstance() 分派。
增加新 action 类型(如未来 zero-bubble 的 BackwardInputGrad / BackwardWeightGrad)
只需要加新子类 + Runner 加一个分支,不动其它代码。

Compact __repr__ 设计:打印 action 列表时形如 [F(0), SF(0), F(1), SFRB(1,0), ...],
方便调度调试和测试断言。
"""

from __future__ import annotations
from dataclasses import dataclass


# ────────────────────────────────────────────────────────────────
# Base
# ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PPAction:
    """所有 PP action 的基类。"""


# ────────────────────────────────────────────────────────────────
# 计算 actions
# ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Forward(PPAction):
    """跑指定 microbatch 的 forward。结果暂存在 stage._outputs[mb_id]。"""
    mb_id: int

    def __repr__(self) -> str:
        return f"F({self.mb_id})"


@dataclass(frozen=True)
class Backward(PPAction):
    """跑指定 microbatch 的 backward。
    
    Last stage 上 grad_output 由 stage 内部从 loss 得到;
    其它 stage 上 grad_output 必须在此之前由 RecvBackward / SendForwardRecvBackward 收到。
    """
    mb_id: int

    def __repr__(self) -> str:
        return f"B({self.mb_id})"


# ────────────────────────────────────────────────────────────────
# 单向通信 actions
# ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SendForward(PPAction):
    """把 mb_id 的 forward output 发给下游 stage。Last stage 上是 no-op。"""
    mb_id: int

    def __repr__(self) -> str:
        return f"SF({self.mb_id})"


@dataclass(frozen=True)
class RecvForward(PPAction):
    """从上游 stage 收 mb_id 的 forward 输入。First stage 上是 no-op(直接用 dataloader)。"""
    mb_id: int

    def __repr__(self) -> str:
        return f"RF({self.mb_id})"


@dataclass(frozen=True)
class SendBackward(PPAction):
    """把 mb_id 的 input grad 发给上游 stage。First stage 上是 no-op。"""
    mb_id: int

    def __repr__(self) -> str:
        return f"SB({self.mb_id})"


@dataclass(frozen=True)
class RecvBackward(PPAction):
    """从下游 stage 收 mb_id 的 output grad。Last stage 上是 no-op(loss 是起点)。"""
    mb_id: int

    def __repr__(self) -> str:
        return f"RB({self.mb_id})"


# ────────────────────────────────────────────────────────────────
# Combined 通信 actions(1F1B 防死锁必备)
# ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SendForwardRecvBackward(PPAction):
    """原子操作:把 fwd_mb 的 forward output 发下游 + 收 bwd_mb 的 output grad。
    
    1F1B 中间/前段 stage 用,Last stage 上退化成 no-op。
    """
    fwd_mb: int
    bwd_mb: int

    def __repr__(self) -> str:
        return f"SFRB({self.fwd_mb},{self.bwd_mb})"


@dataclass(frozen=True)
class SendBackwardRecvForward(PPAction):
    """原子操作:把 bwd_mb 的 input grad 发上游 + 收 fwd_mb 的 forward 输入。
    
    1F1B 中间/后段 stage 用,First stage 上退化成 no-op。
    """
    bwd_mb: int
    fwd_mb: int

    def __repr__(self) -> str:
        return f"SBRF({self.bwd_mb},{self.fwd_mb})"