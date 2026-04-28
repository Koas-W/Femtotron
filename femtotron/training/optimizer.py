import torch
from torch import nn, Tensor
import math

def get_param_groups(model: nn.Module, weight_decay: float = 0.01) -> list[dict]:
    """
    模型级别的参数分组。知道哪些参数要 weight decay。
    返回的 params 是模型的 BF16 参数。
    
    返回两个组：
    - 需要 weight decay 的参数：所有非 bias、非 LayerNorm/RMSNorm 的权重
    - 不需要 weight decay 的参数：bias、LayerNorm weight、RMSNorm weight

    分组规则：
    - decay 组：所有 weight 矩阵（Linear.weight 等）
    - no_decay 组：bias、RMSNorm/LayerNorm 的 weight
    
    判断依据：参数名包含 "bias" 或参数是一维的（norm weight 都是 [H]）

    返回格式：
    [
        {"params": [fp32_w1, fp32_w2, ...], "weight_decay": 0.01},
        {"params": [fp32_b1, fp32_norm1, ...], "weight_decay": 0.0},
    ]
    """
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or "bias" in name or "embed" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

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