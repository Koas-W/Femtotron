import torch
from torch import nn, Tensor
from torch import distributed as dist
from typing import cast, Protocol
from dataclasses import dataclass

@dataclass
class ShardingSpec:
    """描述一个 tensor 在 dp 维度的分片情况。"""
    full_shape: torch.Size       # 未分片时的形状
    flat_size: int                # 展平后的元素数（含 padding）
    shard_size: int               # 每个 rank 持有多少元素
    rank: int                      # 本 rank
    world_size: int                # dp_size
    pad_size: int = 0             # 为了整除而 pad 的元素数
    
    @property
    def shard_start(self) -> int:
        return self.rank * self.shard_size
    
    @property
    def shard_end(self) -> int:
        return self.shard_start + self.shard_size
    
    @classmethod
    def from_full(cls, full_tensor: Tensor, rank: int, world_size: int) -> "ShardingSpec":
        flat_size = full_tensor.numel()
        # 向上取整 pad 到能整除
        shard_size = (flat_size + world_size - 1) // world_size
        padded_size = shard_size * world_size
        return cls(
            full_shape=full_tensor.shape,
            flat_size=padded_size,
            shard_size=shard_size,
            rank=rank,
            world_size=world_size,
            pad_size=padded_size - flat_size,
        )
    
    @classmethod
    def no_shard(cls, full_tensor: Tensor) -> "ShardingSpec":
        """不分片时的 spec：world_size=1，每个 rank 都是完整的。"""
        return cls(
            full_shape=full_tensor.shape,
            flat_size=full_tensor.numel(),
            shard_size=full_tensor.numel(),
            rank=0,
            world_size=1,
        )