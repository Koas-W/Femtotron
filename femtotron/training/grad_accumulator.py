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
        """
        返回累加好的本地 grad（不做通信、不做 dtype 转换）。
        
        通信和到 master_dtype 的 cast 由 ShardingStrategy 处理。
        
        Returns:
            本 rank 持有的完整（compute_param 形状的）grad，dtype 是 grad 原生的（通常 bf16）。
            若没有 grad（参数没参与本次 forward）返回 None。
        """
        if self.acc_buffer is not None:
            return self.acc_buffer
        return self.group.compute.grad
    
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