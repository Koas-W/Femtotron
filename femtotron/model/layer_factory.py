import torch
from torch import Tensor, dtype
import torch.distributed as dist
from torch.distributed import ReduceOp, ProcessGroup
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Protocol, Callable
from torch.nn import RMSNorm

from femtotron.model.parallel_plan import ParallelRule
from femtotron.parallel.tensor_parallel.linear import ColumnParallelLinear, RowParallelLinear
from femtotron.parallel.tensor_parallel.embedding import VocabParallelEmbedding

class LayerFactory(Protocol):
    def make_embedding(self, config, parallel_ctx, *, dtype, device) -> nn.Module: ...
    # def make_decoder_layer(self, config, parallel_ctx, *, layer_idx, dtype, device) -> nn.Module: ...
    def make_norm(self, config, *, dtype, device) -> nn.Module: ...
    def make_lm_head(self, config, parallel_ctx, *, tied_weight, dtype, device) -> nn.Module: ...

class DefaultLayerFactory:
    def make_embedding(self, config, parallel_ctx, *, dtype, device):
        return VocabParallelEmbedding(
            config.vocab_size, config.hidden_size,
            parallel_ctx,
            dtype=dtype, device=device,
        )

    # def make_decoder_layer(self, config, parallel_ctx, *, layer_idx, dtype, device):
    #     return LlamaDecoderLayer(config, parallel_ctx, layer_idx=layer_idx, dtype=dtype, device=device)

    def make_norm(self, config, *, dtype, device):
        return RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, device=device)

    def make_lm_head(self, config, parallel_ctx, *, tied_weight, dtype, device):
        head = ColumnParallelLinear(
            config.hidden_size, config.vocab_size,
            parallel_ctx, gather_output=True, bias=False,
            dtype=dtype, device=device,
        )
        if tied_weight is not None:
            head.weight = tied_weight
        return head