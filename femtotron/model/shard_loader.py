from typing import Protocol, Callable
from torch import Tensor
from .parallel_plan import ParallelRule

class ShardLoader(Protocol):
    def load(self, handle, rank: int, world_size: int) -> Tensor: ...


class ReplicateLoader:
    """完整加载，所有 rank 拿到一样的副本。"""
    def load(self, handle, rank, world_size):
        return handle[:]


class DimShardLoader:
    """沿固定维度切分。"""
    def __init__(self, dim: int):
        self.dim = dim

    def load(self, handle, rank, world_size):
        shape = handle.get_shape()
        size = shape[self.dim]
        assert size % world_size == 0, f"dim {self.dim} size {size} not divisible by {world_size}"
        chunk = size // world_size
        slices: list[slice] = [slice(None)] * len(shape)
        slices[self.dim] = slice(rank * chunk, (rank + 1) * chunk)
        return handle[tuple(slices)]
    

##################### 注册工厂 #######################
LoaderFactory = Callable[[ParallelRule, str], ShardLoader]
# 第二个参数是 param 名后缀，比如 ".weight" / ".bias"，用于区分

_LOADER_REGISTRY: dict[str, LoaderFactory] = {}

def register_loader(kind: str):
    def deco(fn: LoaderFactory) -> LoaderFactory:
        _LOADER_REGISTRY[kind] = fn
        return fn
    return deco


@register_loader("column")
def _column_loader(rule: ParallelRule, suffix: str) -> ShardLoader:
    # column parallel: weight 切 dim 0；bias 也切 dim 0
    return DimShardLoader(dim=0)


@register_loader("row")
def _row_loader(rule: ParallelRule, suffix: str) -> ShardLoader:
    # row parallel: weight 切 dim 1；bias 不切（每个 rank 加完整 bias，最后 all-reduce 时会重复加，所以需要其他处理）
    if suffix == ".bias":
        return ReplicateLoader()
    return DimShardLoader(dim=1)


@register_loader("vocab_embed")
def _vocab_loader(rule: ParallelRule, suffix: str) -> ShardLoader:
    return DimShardLoader(dim=0)


@register_loader("replicate")
def _replicate_loader(rule: ParallelRule, suffix: str) -> ShardLoader:
    return ReplicateLoader()
