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

@dataclass
class ZeROConfig:
    stage: int = 0   # 0 = no shard, 1 = ZeRO-1, 2 = ZeRO-2, 3 = ZeRO-3

def create_sharding_strategy(
    parallel_ctx: ParallelContext,
    config: ZeROConfig,
) -> ShardingStrategy:
    if config.stage == 0:
        return NoShardStrategy(parallel_ctx.dp_group)
    elif config.stage == 1:
        assert parallel_ctx.dp_group is not None
        return ZeRO1Strategy(parallel_ctx.dp_group)
    # elif config.stage == 2:
    #     return ZeRO2Strategy(...)    # 未实现
    # elif config.stage == 3:
    #     return ZeRO3Strategy(...)    # 未实现
    raise ValueError(config.stage)
