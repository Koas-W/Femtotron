import os, sys
import torch
from torch import distributed as dist
from torch.utils.data import Sampler

from femtotron.parallel_context import ParallelContext

class DistributedSampler(Sampler[int]):
    """
    分布式 sampler，按 DP rank 分片。
    
    保证：
    - 所有 rank 算出相同的全局 shuffle 顺序（基于 seed + epoch）
    - 各 rank 取自己的分片，不重叠
    - 支持 resume（通过 state_dict 跳过已消费部分）
    """
    def __init__(
        self,
        dataset_size: int,
        parallel_ctx: ParallelContext,
        *,
        seed: int = 42, # 这个seed应该对于所有dp rank完全相同
        shuffle: bool = True,  # 是否打乱原本的data顺序
        drop_last: bool = True,  # 对于一个epoch，如果不能整除，即dataset_size % dp_size != 0，
                                 # 应该扔掉还是用前面的重复填充
    ):
        dp_rank = parallel_ctx.dp_rank
        dp_size = parallel_ctx.dp_size
        assert 0 <= dp_rank < dp_size
        assert dataset_size > 0
        
        self.dataset_size = dataset_size
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.seed = seed
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        self.epoch = 0
        self._start_offset = 0   # resume 时跳过的样本数
    
    def set_epoch(self, epoch: int) -> None:
        """每个 epoch 开始时调用，影响 shuffle 种子且重置 resume 偏移。"""
        self.epoch = epoch
        self._start_offset = 0
    
    def _compute_indices(self) -> list[int]:
        """计算本 rank 在当前 epoch 应该看的所有 index（不含 resume 跳过）。"""
        # 1. 全局顺序（需要保证所有 rank 算出一样的）
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # 需要传入一个generator，持有其句柄，保证不会被外部代码污染
            indices = torch.randperm(self.dataset_size, generator=g).tolist()
        else:
            indices = list(range(self.dataset_size))
        
        # 2. 处理整除问题
        if self.drop_last:
            total = (self.dataset_size // self.dp_size) * self.dp_size
            indices = indices[:total]
        else:
            target_total = ((self.dataset_size + self.dp_size - 1) // self.dp_size) * self.dp_size
            pad_count = target_total - self.dataset_size
            indices = indices + indices[:pad_count]
        
        # 3. 切自己分片
        per_rank = len(indices) // self.dp_size
        start = self.dp_rank * per_rank
        end = start + per_rank
        return indices[start:end]
    
    def __iter__(self):
        indices = self._compute_indices()
        for idx in indices[self._start_offset:]:
            yield idx
    
    def __len__(self) -> int:
        if self.drop_last:
            full = self.dataset_size // self.dp_size
        else:
            full = (self.dataset_size + self.dp_size - 1) // self.dp_size
        return max(0, full - self._start_offset)
    
    def state_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "start_offset": self._start_offset,
            "seed": self.seed,
            "dp_size": self.dp_size,
            "dataset_size": self.dataset_size,
        }
    
    def load_state_dict(self, sd: dict) -> None:
        # 一致性检查（防止 resume 到了不兼容的配置）
        assert sd["seed"] == self.seed, "seed mismatch"
        assert sd["dp_size"] == self.dp_size, (
            f"dp_size changed: {sd['dp_size']} -> {self.dp_size}; "
            "cannot resume across different parallelism configs"
        )
        assert sd["dataset_size"] == self.dataset_size, "dataset changed"
        
        self.epoch = sd["epoch"]
        self._start_offset = sd["start_offset"]
    
    def advance(self, n: int) -> None:
        """供外部调用：标记"已消费 n 个样本"。
        
        DistributedDataLoader 在每次成功 yield 后调用。
        """
        self._start_offset += n