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
from femtotron.sharding.param_group_cluster import ParamGroupCluster

class ZeRO1Strategy:
    """ZeRO-1：仅分片 master / optimizer state。"""
    
    def __init__(self, dp_group: ProcessGroup):
        self.dp_group = dp_group
        self.dp_rank = dist.get_rank(dp_group)
        self.dp_size = dist.get_world_size(dp_group)
    
    def make_master(self, compute: nn.Parameter, master_dtype: torch.dtype):
        spec = ShardingSpec.from_full(compute, self.dp_rank, self.dp_size)
        
        # 把完整 compute flatten + pad，取自己那段作为 master
        full_flat = compute.detach().flatten().to(master_dtype)
        if spec.pad_size > 0:
            padding = torch.zeros(spec.pad_size, dtype=master_dtype, device=full_flat.device)
            full_flat = torch.cat([full_flat, padding])
        
        master_shard = full_flat[spec.shard_start : spec.shard_end].clone()
        master_shard.requires_grad_(True)
        return master_shard, spec
    
    def prepare_for_backward(self, groups: list[ParamGroup]) -> None:
        pass
    
    def reduce_grads(self,
        compute_grads: list[Tensor],     # 各 rank 完整的 bf16 grad
        targets: list[Tensor],            # ParamGroup.optimized_param
        target_specs: list[ShardingSpec | None],
        ):
        """对每个 grad 做 reduce-scatter。
        
        compute_grad 是 [N1, N2, ...]，flatten + pad 后 reduce-scatter 得到本 rank 1/dp_size。
        """
        result = []
        for grad, target, spec in zip(compute_grads, targets, target_specs):
            if spec is None or spec.world_size == 1:
                # 这个 param 没分片（边界情况），走 all-reduce
                dist.all_reduce(grad, op=dist.ReduceOp.AVG, group=self.dp_group)
                result.append(grad.to(target.dtype))
                continue
            
            # flatten + pad
            flat = grad.flatten()
            if spec.pad_size > 0:
                padding = torch.zeros(spec.pad_size, dtype=flat.dtype, device=flat.device)
                flat = torch.cat([flat, padding])
            
            # reduce-scatter
            shard = torch.empty(
                spec.shard_size,
                dtype=flat.dtype,
                device=flat.device,
            )
            dist.reduce_scatter_tensor(shard, flat, op=dist.ReduceOp.AVG, group=self.dp_group)
            
            # cast 到 master dtype
            result.append(shard.to(target.dtype))
        return result
    
    def gather_weights(self,
        groups: list[ParamGroup],
        ):
        """对每个 param 做 all-gather，把 master 收集成完整 compute。"""
        for g in groups:
            if g.master is None or g.master_spec is None:
                continue
            spec = g.master_spec
            
            # all-gather
            gathered = torch.empty(
                spec.flat_size,
                dtype=g.master.dtype,
                device=g.master.device,
            )
            dist.all_gather_into_tensor(gathered, g.master, group=self.dp_group)
            
            # 去 padding + reshape + cast
            unpadded = gathered[: spec.flat_size - spec.pad_size]
            full = unpadded.view(spec.full_shape).to(g.compute.dtype)
            with torch.no_grad():
                g.compute.copy_(full)
    
    def grads_are_dp_sharded(self) -> bool:
        return True
    
    def post_step(self) -> None:
        pass
    
    def make_clusters(
        self,
        model: nn.Module,
        groups: list[ParamGroup],
        master_dtype: torch.dtype | None,
        ) -> list["ParamGroupCluster"]:
        return []