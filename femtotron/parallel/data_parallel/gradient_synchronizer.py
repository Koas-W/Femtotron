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
from contextlib import nullcontext, contextmanager, AbstractContextManager

from femtotron.parallel_context import ParallelContext
from femtotron.training.param_group import ParamGroup
from femtotron.model.shard_loader import ShardLoader, ReplicateLoader, DimShardLoader, _LOADER_REGISTRY
from femtotron.parallel.data_parallel.ddp import DataParallelGradSync

class GradientSynchronizer(Protocol):
    """
    同步 DP rank 间梯度的抽象。
    
    最简单的 DDP 实现：backward 完成后，在 DP group 内
    对所有参数的梯度做 all-reduce（取平均）。
    """
    
    def sync_gradients(self) -> None:
        """
        在 DP group 内 all-reduce 所有参数的梯度，除以 dp_size 取平均。
        
        调用时机：backward 完成后、optimizer step 之前。
        
        实现：
        遍历 model 的所有 requires_grad=True 的参数，
        对每个 param.grad 做 dist.all_reduce(grad, group=dp_group)，
        然后 grad /= dp_size。
        
        优化空间：
        - 把多个小 grad 打包成一个大 tensor 再 all-reduce（减少通信次数）
        - 和 backward 计算重叠（用 hook 在每个参数的梯度算完后立即发起异步 all-reduce）
        这些优化留给 1.8 ZeRO-2 和 2.4 通信 overlap。
        """
        ...

    def no_sync(self) -> AbstractContextManager[None]:
        """
        返回一个 context manager，在其中 backward 不触发梯度同步。
        用于 gradient accumulation 的中间步骤。
        """
        ...

    def state_dict(self) -> dict: ...
    def load_state_dict(self, sd: dict) -> None: ...

class NoOpGradSync:
    def sync_gradients(self): pass
    
    def no_sync(self) -> AbstractContextManager[None]:
        return nullcontext()
    def state_dict(self): return {}
    def load_state_dict(self, sd): pass

def create_grad_synchronizer(
    param_groups: list[ParamGroup],
    parallel_ctx: ParallelContext,
) -> GradientSynchronizer:
    if parallel_ctx.dp_group is None or parallel_ctx.dp_size == 1:
        return NoOpGradSync()
    return DataParallelGradSync(param_groups, parallel_ctx)