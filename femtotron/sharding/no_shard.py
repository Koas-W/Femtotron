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
from femtotron.training.param_group import ParamGroup

class NoShardStrategy:
    """不做任何分片。等价于普通 DDP + mixed precision。"""
    
    def __init__(self, dp_group: ProcessGroup | None):
        self.dp_group = dp_group
    
    def make_master(self, compute: nn.Parameter, master_dtype: torch.dtype):
        master = compute.detach().clone().to(master_dtype)
        master.requires_grad_(True)
        return master, None    # spec=None 表示不分片
    
    def reduce_grads(self,
        compute_grads: list[Tensor],     # 各 rank 完整的 bf16 grad
        targets: list[Tensor],            # ParamHandle.optimized_param
        target_specs: list[ShardingSpec | None],
        ):
        # if self.dp_group is None or dist.get_world_size(self.dp_group) == 1:
        #     # 无需同步
        #     return [g.to(t.dtype) for g, t in zip(compute_grads, targets)]
        
        # # 标准 all-reduce
        # for g in compute_grads:
        #     dist.all_reduce(g, op=dist.ReduceOp.AVG, group=self.dp_group)
        
        # cast 到 master dtype
        # 通信由独立的 grad_sync 组件处理；strategy 只做 cast
        return [g.to(t.dtype) for g, t in zip(compute_grads, targets)]
    
    def gather_weights(self,
        groups: list[ParamGroup],
        ):
        for g in groups:
            g.sync_master_to_compute()

    def grads_are_dp_sharded(self) -> bool:
        return False