from collections.abc import Callable
from dataclasses import dataclass
import torch
from torch import nn, Tensor
from torch.distributed import ProcessGroup

@dataclass
class PipelineConfig:
    num_microbatches: int = 1
    schedule: str = "1f1b"   # "gpipe" or "1f1b"