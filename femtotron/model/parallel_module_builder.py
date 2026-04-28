import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
from typing import Protocol

from femtotron.parallel_context import ParallelContext
from femtotron.model.parallel_plan import ParallelPlan, ParallelRule
from femtotron.parallel.tensor_parallel.linear import ColumnParallelLinear, RowParallelLinear
from femtotron.parallel.tensor_parallel.embedding import VocabParallelEmbedding

class ParallelBuilder(Protocol):
    def __call__(
        self,
        old: nn.Module,
        parallel_ctx: ParallelContext,
        rule: ParallelRule,
    ) -> nn.Module: ...

_BUILDER_REGISTRY: dict[tuple[str, type[nn.Module]], ParallelBuilder] = {}

def register_builder(kind: str, source_type: type[nn.Module]):
    def deco(fn: ParallelBuilder) -> ParallelBuilder:
        _BUILDER_REGISTRY[(kind, source_type)] = fn
        return fn
    return deco

@register_builder("column", nn.Linear)
def _build_column_linear(old, parallel_ctx, rule):
    assert isinstance(old, nn.Linear)
    return ColumnParallelLinear.from_linear(old, parallel_ctx, rule)


@register_builder("row", nn.Linear)
def _build_row_linear(old, parallel_ctx, rule):
    assert isinstance(old, nn.Linear)
    return RowParallelLinear.from_linear(old, parallel_ctx, rule)


@register_builder("vocab_embed", nn.Embedding)
def _build_vocab_embedding(old, parallel_ctx, rule):
    assert isinstance(old, nn.Embedding)
    return VocabParallelEmbedding.from_embedding(old, parallel_ctx, rule)