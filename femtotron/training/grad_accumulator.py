import torch
from torch import nn, Tensor
from torch import distributed as dist
from torch.distributed import ProcessGroup
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from safetensors import safe_open
from typing import cast

from femtotron.parallel_context import ParallelContext
from femtotron.training.train_config import TrainConfig
from femtotron.training.param_group import ParamGroup

class GradAccumulator:
    """管理一个 ParamGroup 的梯度累加和精度转换。"""
    
    def __init__(self, group: ParamGroup, train_config: TrainConfig, parallel_ctx: ParallelContext):
        self.group = group
        self.train_config = train_config
        self.parallel_ctx = parallel_ctx
        self.acc_buffer: Tensor | None = None  # 延迟分配
    
    def accumulate(self) -> None:
        """把 compute.grad 累加到内部 buffer，并清零 compute.grad。
        
        在每个 micro-batch backward 后调用。
        """
        g = self.group.compute.grad
        if g is None:
            return
        if self.train_config.grad_acc_dtype is None:
            # 不单独维护 buffer，原地累
            return
        if self.acc_buffer is None:
            self.acc_buffer = torch.zeros_like(g, dtype=self.train_config.grad_acc_dtype)
        self.acc_buffer.add_(g.to(self.train_config.grad_acc_dtype))
        g.zero_()
    
    def finalize(self) -> Tensor | None:
        """完成所有累加，返回最终给 optimizer 的梯度。"""
        if self.acc_buffer is not None:
            grad = self.acc_buffer
        else:
            grad = self.group.compute.grad
        
        if grad is None:
            return None
        
        # DP all-reduce
        if self.group.is_replicated_across_dp and self.parallel_ctx.dp_group is not None:
            # 通信精度
            comm_dtype = self.train_config.reduce_dtype or grad.dtype
            if grad.dtype != comm_dtype:
                grad = grad.to(comm_dtype)
            dist.all_reduce(grad, group=self.parallel_ctx.dp_group)
            grad.div_(dist.get_world_size(self.parallel_ctx.dp_group))   # 取平均
        
        # cast 到 master 期望的 dtype
        if self.group.master is not None and grad.dtype != self.group.master.dtype:
            grad = grad.to(self.group.master.dtype)
        
        return grad
    
    def reset(self) -> None:
        """optimizer step 后调用，清掉累加 buffer。"""
        if self.acc_buffer is not None:
            self.acc_buffer.zero_()
        if self.group.compute.grad is not None:
            self.group.compute.grad.zero_()

    def state_dict(self) -> dict:
        return {
            "acc_buffer": self.acc_buffer.detach().cpu() if self.acc_buffer is not None else None,
        }
    
    def load_state_dict(self, sd: dict) -> None:
        if sd["acc_buffer"] is not None:
            if self.acc_buffer is None:
                self.acc_buffer = sd["acc_buffer"].to(...)
            else:
                self.acc_buffer.copy_(sd["acc_buffer"].to(self.acc_buffer.device))
        else:
            self.acc_buffer = None