import torch
from torch import nn, Tensor
import math

def create_lr_schedule(optimizer: torch.optim.Optimizer,
                       warmup_steps: int,
                       total_steps: int,
                       min_lr_ratio: float = 0.1) -> torch.optim.lr_scheduler.LambdaLR:
    """创建 linear warmup + cosine decay 的学习率调度。
    
    行为：
        - step ∈ [0, warmup_steps)：lr 从 0 线性增长到 base_lr
        - step ∈ [warmup_steps, total_steps)：从 base_lr cosine 衰减到 base_lr * min_lr_ratio
        - step ≥ total_steps：保持 base_lr * min_lr_ratio
    
    这是 LLaMA / GPT 预训练的标准配置。
    """

    assert 0 <= warmup_steps <= total_steps, (
        f"warmup_steps ({warmup_steps}) must be in [0, total_steps={total_steps}]"
    )
    assert 0.0 <= min_lr_ratio <= 1.0, (
        f"min_lr_ratio ({min_lr_ratio}) must be in [0, 1]"
    )
    
    decay_steps = max(1, total_steps - warmup_steps)   # 防止除零
    
    def lr_lambda(current_step: int) -> float:
        current_step = current_step + 1
        # Warmup 阶段：线性增长
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        
        # Decay 阶段：cosine 衰减
        progress = (current_step - warmup_steps) / decay_steps
        progress = min(progress, 1.0)   # 超过 total_steps 时停在终点
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        # cosine ∈ [0, 1]，1 表示开始（base_lr），0 表示终点（min_lr_ratio * base_lr）
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)