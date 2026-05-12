import torch
from torch import nn, Tensor
from torch import distributed as dist
from torch.distributed import ProcessGroup
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from pathlib import Path
from collections import defaultdict
from safetensors import safe_open
from typing import cast, Protocol
from dataclasses import dataclass

from femtotron.sharding.sharding_spec import ShardingSpec
from femtotron.parallel_context import ParallelContext
from femtotron.sharding.sharding_strategy import ShardingStrategy
from femtotron.sharding.no_shard import NoShardStrategy
from femtotron.sharding.zero1 import ZeRO1Strategy
from femtotron.sharding.zero2 import ZeRO2Strategy
from femtotron.sharding.zero3 import ZeRO3Strategy
from femtotron.sharding.zero_config import ZeROConfig

def create_sharding_strategy(
    parallel_ctx: ParallelContext,
    config: ZeROConfig,
) -> ShardingStrategy:
    if config.stage == 0:
        return NoShardStrategy(parallel_ctx.dp_group)
    elif config.stage == 1:
        assert parallel_ctx.dp_group is not None
        return ZeRO1Strategy(parallel_ctx.dp_group)
    elif config.stage == 2:
        assert parallel_ctx.dp_group is not None
        return ZeRO2Strategy(parallel_ctx.dp_group)
    elif config.stage == 3:
        if config.wrap_policy is None:
            raise ValueError(
                "ZeRO-3 requires wrap_policy. Set config.wrap_policy to a callable "
                "that returns True for modules to wrap as FSDP units "
                "(typically each transformer block)."
            )
        return ZeRO3Strategy(parallel_ctx.dp_group, config.wrap_policy)
    raise ValueError(config.stage)
