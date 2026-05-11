import torch
from torch import nn, Tensor
from torch import distributed as dist
from torch.distributed import ProcessGroup
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from pathlib import Path
from collections import defaultdict
from safetensors import safe_open
from typing import cast, Protocol

from femtotron.sharding.sharding_spec import ShardingSpec
from femtotron.model.parallel_plan import ParallelPlan
from femtotron.training.param_group import ParamGroup

class ShardingStrategy(Protocol):
    """决定 ParamHandle 的分片方式和同步行为。"""
    
    def make_master(
        self,
        compute: Parameter,
        master_dtype: torch.dtype,
    ) -> tuple[Tensor, ShardingSpec | None]:
        """从 compute 创建 master，返回 (master_tensor, spec)。"""
        ...
    
    def prepare_for_backward(self, groups: list[ParamGroup]) -> None:
        """backward 之前调用,注册 hook、分配 grad shard buffer 等。
        
        NoShard / ZeRO-1: 不做任何事
        ZeRO-2: 给每个 param 注册 post-accumulate-grad hook
        """
        ...
    
    def reduce_grads(
        self,
        compute_grads: list[Tensor | None],     # ZeRO-2 下基本都是 None(hook 清掉了)
        targets: list[Tensor],
        target_specs: list[ShardingSpec | None],
    ) -> list[Tensor]:
        """backward 完成后调用。
        
        NoShard:  cast 到 master dtype（通信由 grad_sync 做）
        ZeRO-1:   reduce_scatter + cast
        ZeRO-2:   只从 grad shard buffer 拿出来（hook 已经 reduce_scatter 过了）
        """
        ...

    def reduce_grads(
        self,
        compute_grads: list[Tensor],     # 各 rank 完整的 bf16 grad
        targets: list[Tensor],            # ParamGroup.optimized_param
        target_specs: list[ShardingSpec | None],
    ) -> list[Tensor]:
        """把完整 grad 同步成"每个 rank 看到自己 master 的那部分 grad"。
        
        - 无 ZeRO：all-reduce，返回完整 grad
        - ZeRO-1：reduce-scatter，返回本 rank 那段 grad
        """
        ...
    
    def gather_weights(
        self,
        groups: list[ParamGroup],
    ) -> None:
        """step 之后让所有 rank 的 compute 一致。
        
        - 无 ZeRO：master cast 到 compute（每 rank 自己做，没有通信）
        - ZeRO-1：all-gather master 到 compute
        """
        ...

    def grads_are_dp_sharded(self) -> bool:
        """grad 是否在 DP 维度上被分片。
        
        - 无 ZeRO / ZeRO-0：False（grad 是完整的）
        - ZeRO-1/2/3：True（grad 经过 reduce_scatter，每 rank 只有一段）
        """
        ...

    def post_step(self) -> None: ...    # 对zero1是no-op，但 zero2 需要在 step 之后清理 grad shard buffer