import os, sys
import torch
from torch import nn, Tensor
from torch import distributed as dist
import torch.nn.functional as F
import json
from pathlib import Path
from collections import defaultdict
from safetensors import safe_open
from typing import cast, Protocol
from contextlib import contextmanager

from femtotron.parallel_context import ParallelContext
from femtotron.training.param_group import ParamGroup

class DataParallelGradSync:
    """朴素的 DP 梯度同步：backward 完成后一次性 all-reduce。
    
    不重叠通信和计算，但实现简单、易调试。
    可作为后期优化的 baseline。
    """
    
    def __init__(
        self,
        param_groups: list[ParamGroup],
        parallel_ctx: ParallelContext,
    ):
        dp_group = parallel_ctx.dp_group
        self.param_groups = param_groups
        self.dp_group = dp_group
        self._sync_disabled = False
    
    @contextmanager
    def no_sync(self):
        """grad accumulation 中间 step 用，跳过同步。"""
        prev = self._sync_disabled
        self._sync_disabled = True
        try:
            yield
        finally:
            self._sync_disabled = prev
    
    def sync_gradients(self) -> None:
        if self._sync_disabled:
            return
        
        # 收集所有有 grad 的 tensor
        grads = []
        for h in self.param_groups:
            g = h.compute.grad
            if g is None:
                continue
            grads.append(g)
        
        if not grads:
            return
        
        # 简单实现：每个 grad 单独 all-reduce
        # 性能优化版本会 flatten 成一个 bucket 一次 reduce
        for g in grads:
            dist.all_reduce(g, op=dist.ReduceOp.AVG, group=self.dp_group)
    
    def state_dict(self) -> dict:
        return {}   # 无状态
    
    def load_state_dict(self, sd: dict) -> None:
        pass
