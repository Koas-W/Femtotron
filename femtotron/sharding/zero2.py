import torch
from torch import nn, Tensor
from torch import distributed as dist
from torch.distributed import ProcessGroup, group
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from pathlib import Path
from collections import defaultdict
from safetensors import safe_open
from typing import cast, Protocol
from contextlib import contextmanager

from femtotron.sharding.sharding_spec import ShardingSpec
from femtotron.training.param_group import ParamGroup

class ZeRO2Strategy:
    def __init__(self, dp_group: ProcessGroup):
        self.dp_group = dp_group
        self.dp_rank = dist.get_rank(dp_group)
        self.dp_size = dist.get_world_size(dp_group)
        
        self._hooks_registered = False
        self._sync_enabled = True
        self._grad_shards: dict[str, Tensor] = {}
        self._group_by_param_id: dict[int, ParamGroup] = {}
        self.groups_ref: list[ParamGroup] = []
    
    # ─── ShardingStrategy 接口 ───
    
    def make_master(self, compute: Parameter, master_dtype: torch.dtype):
        # 和 ZeRO-1 完全一样
        spec = ShardingSpec.from_full(compute, self.dp_rank, self.dp_size)
        full_flat = compute.detach().flatten().to(master_dtype)
        if spec.pad_size > 0:
            padding = torch.zeros(spec.pad_size, dtype=master_dtype, device=full_flat.device)
            full_flat = torch.cat([full_flat, padding])
        master_shard = full_flat[spec.shard_start : spec.shard_end].clone()
        master_shard.requires_grad_(True)
        return master_shard, spec
    
    def prepare_for_backward(self, groups: list[ParamGroup]) -> None:
        """trainer 在第一个 backward 之前调用。"""
        if self._hooks_registered:
            return
        self.groups_ref = groups
        for g in groups:
            self._group_by_param_id[id(g.compute)] = g
            self._register_hook(g)
        self._hooks_registered = True
    
    def _register_hook(self, group: ParamGroup):
        spec = group.master_spec
        
        def hook(param):
            if not self._sync_enabled:
                return    # no_sync 期间放过
            if param.grad is None:
                return
            
            flat = param.grad.flatten()
            if spec.pad_size > 0:
                padding = torch.zeros(spec.pad_size, dtype=flat.dtype, device=flat.device)
                flat = torch.cat([flat, padding])
            
            shard = torch.empty(spec.shard_size, dtype=flat.dtype, device=flat.device)
            dist.reduce_scatter_tensor(
                shard, flat, op=dist.ReduceOp.AVG, group=self.dp_group
            )
            
            self._grad_shards[group.name] = shard
            param.grad = None    # 释放 compute.grad,这是 ZeRO-2 省显存的关键
        
        group.compute.register_post_accumulate_grad_hook(hook)
    
    def reduce_grads(
        self,
        compute_grads: list[Tensor | None],     # ZeRO-2 下基本都是 None(hook 清掉了)
        targets: list[Tensor],
        target_specs: list[ShardingSpec | None],
    ) -> list[Tensor]:
        # compute_grads 参数在 ZeRO-2 下基本都是 None（被 hook 清空了)
        # 从 _grad_shards 拿
        result = []
        for group, target in zip(self.groups_ref, targets):
            shard = self._grad_shards.get(group.name)
            if shard is None:
                shard = torch.zeros_like(target)
            if shard.dtype != target.dtype:
                shard = shard.to(target.dtype)
            result.append(shard)
        return result
    
    def gather_weights(self, groups: list[ParamGroup]):
        # 和 ZeRO-1 完全一样
        for g in groups:
            if g.master is None or g.master_spec is None:
                continue
            spec = g.master_spec
            gathered = torch.empty(spec.flat_size, dtype=g.master.dtype, device=g.master.device)
            dist.all_gather_into_tensor(gathered, g.master, group=self.dp_group)
            unpadded = gathered[: spec.flat_size - spec.pad_size]
            full = unpadded.view(spec.full_shape).to(g.compute.dtype)
            with torch.no_grad():
                g.compute.copy_(full)
    
    def post_step(self):
        self._grad_shards.clear()
    
    def grads_are_dp_sharded(self) -> bool:
        return True
    
    # ─────────────────────────────────
    # ─── GradientSynchronizer 接口 ───
    # ─────────────────────────────────
    
    @contextmanager
    def no_sync(self):
        prev = self._sync_enabled
        self._sync_enabled = False
        try:
            yield
        finally:
            self._sync_enabled = prev
    
    def sync_gradients(self):
        pass    # 通信由 hook 完成
    
    def state_dict(self):
        return {}
    
    def load_state_dict(self, sd: dict):
        pass