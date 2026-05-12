from collections.abc import Callable
from dataclasses import dataclass
import torch
from torch import nn, Tensor
from torch.distributed import ProcessGroup

@dataclass
class ZeROConfig:
    stage: int = 0   # 0 = no shard, 1 = ZeRO-1, 2 = ZeRO-2, 3 = ZeRO-3

    wrap_policy: Callable[[nn.Module], bool] | None = None
    """ZeRO-3 专用:判定 FSDP unit 的函数。stage<3 时忽略。"""