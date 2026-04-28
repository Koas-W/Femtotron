import torch
from torch import nn, Tensor
from torch import distributed as dist
from typing import Protocol, Callable

from femtotron.parallel_context import ParallelContext
from femtotron.training.param_group import ParamGroup

class GradTransform(Protocol):
    def __call__(self, param_groups: list[ParamGroup], grads: list[Tensor]) -> Tensor: ...
    """原地修改 grads。"""

class ClipGradNorm:
    """
    分布式感知的 gradient norm clipping。
    
    调用时机：copy_grads_to_fp32() 之后，optimizer.step() 之前。
    
    流程：
    1. 计算本 rank 上所有 FP32 master weight 梯度的 L2 norm 的平方和
        local_norm_sq = sum(grad.norm() ** 2 for grad in all_grads)
    
    2. 如果有 dp_group（多 DP rank），all_reduce local_norm_sq 得到 global_norm_sq
        注意：在 TP 下，同一个 TP group 内的 rank 持有不同参数的梯度，
        它们的 grad norm 加起来才是那一层的完整 grad norm。
        - 如果只有 TP 没有 DP：不需要 all-reduce（每个 rank 的 grad norm
            只对应自己持有的参数切片，clip 也只 clip 自己的）
    
    3. global_norm = sqrt(global_norm_sq)
    
    4. 如果 global_norm > max_norm：
        clip_coeff = max_norm / global_norm
        对所有梯度: grad *= clip_coeff
    
    5. 返回 global_norm（用于 logging）
    """
    def __init__(self, max_norm: float, parallel_ctx: ParallelContext):
        self.max_norm = max_norm
        self.parallel_ctx = parallel_ctx
    
    def __call__(self, param_groups: list[ParamGroup], grads: list[Tensor]) -> Tensor:
        # 计算每个 grad 的局部 norm²
        sharded_sq_sum = torch.zeros((), device=grads[0].device, dtype=torch.float32)
        replicated_sq_sum = torch.zeros((), device=grads[0].device, dtype=torch.float32)
        
        for group, g in zip(param_groups, grads):
            local_sq = g.float().pow(2).sum()
            if group.is_tp_sharded:
                sharded_sq_sum += local_sq
            else:
                replicated_sq_sum += local_sq
        
        # 只对 sharded 部分跨 TP all-reduce
        if self.parallel_ctx.tp_group is not None and dist.get_world_size(self.parallel_ctx.tp_group) > 1:
            dist.all_reduce(sharded_sq_sum, group=self.parallel_ctx.tp_group)
        
        total_sq = sharded_sq_sum + replicated_sq_sum
        total_norm = total_sq.sqrt()
        coef = (self.max_norm / (total_norm + 1e-6)).clamp(max=1.0)        
        for g in grads:
            g.mul_(coef)

        return total_norm

# class UnscaleGrads:
#     def __init__(self, scaler: LossScaler):
#         self.scaler = scaler
    
#     def __call__(self, grads):
#         for g in grads:
#             g.div_(self.scaler.scale)