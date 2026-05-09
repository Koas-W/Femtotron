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
