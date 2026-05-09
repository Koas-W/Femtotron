import os, sys
import torch
from torch import nn, Tensor
from torch import distributed as dist
import torch.nn.functional as F
import json
from pathlib import Path
from collections import defaultdict
from safetensors import safe_open
from typing import cast, Callable

from femtotron.parallel_context import ParallelContext
from femtotron.data.data_source import PackedDataset
from femtotron.data.distributed_sampler import DistributedSampler
from femtotron.data.collator import Collator

class DistributedDataLoader:
    """
    分布式训练的数据加载器。
    
    职责：
    - 协调 dataset、sampler、collator 三个组件
    - 包装 PyTorch DataLoader，借用其 IO 优化
    - 提供统一的 state_dict/load_state_dict 用于 resume
    - 自动管理 sampler 状态推进
    
    数据流：sampler 产生 idx → dataset[idx] → collator → batch
    
    关于 TP/PP 一致性：
        TP group 内必须所有 rank 看到相同 batch。这通过 sampler 用 dp_rank 
        和 dp_size 分片实现——同一个 TP group 内的 rank 有相同的 dp_rank。

    本类负责组装和封装迭代器接口，具体策略交给注入的组件：
        - sampler 决定每步给哪些 index（含 DP 分片、shuffle、checkpoint）
        - packer 决定如何把样本打包成定长序列（含 padding/concat）
        - 多进程预取由 PyTorch DataLoader 处理
    
    关于 epoch：
        大模型预训练常常不"完整跑完一个 epoch"。本类不强制 epoch 语义，
        sampler 决定"什么时候 reshuffle"以及"是否会回头"。
    """
    
    def __init__(self, 
                 dataset: PackedDataset,           # 继承torch风格的dataset
                 parallel_ctx: ParallelContext,
                 micro_batch_size: int,  # 每个 DP rank 的 batch size
                 collator: Callable | None = None,
                 sampler: DistributedSampler | None = None,
                 *,
                 seed: int = 42,
                 num_workers: int = 2,
                 prefetch_factor: int = 2,
                 pin_memory: bool = True,
                 drop_last: bool = True,
                 ):
        """
        初始化时做的事：
        1. 创建 DistributedSampler，用 dp_rank 和 dp_size 做分片
           （不是 world_rank 和 world_size！）
        2. 创建 PyTorch DataLoader
        
        数据格式：
        - 输入是已经 tokenized 并 packed 的 tensor [num_samples, seq_len]
        - 输出每个 batch 是 {"input_ids": [mbs, seq_len], "labels": [mbs, seq_len]}
        - labels 就是 input_ids 的 clone（next-token prediction）
        """
        self.parallel_ctx = parallel_ctx
        self.micro_batch_size = micro_batch_size
        self.dataset = dataset
        self.collator = collator
        self.num_workers = num_workers
        self.seed = seed

        # Sampler：默认按 DP rank 分片
        if sampler is None:
            sampler = DistributedSampler(
                dataset_size=len(dataset),
                parallel_ctx=parallel_ctx,
                seed=seed,
                shuffle=True,
                drop_last=drop_last,
            )
        self.sampler = sampler
        
        # 不应该使用默认collator，这是因为不同类型任务可能差异很大
        assert collator is not None
        self.collator = collator
        
        
        # 内层 PyTorch DataLoader 处理 IO
        self._inner = torch.utils.data.DataLoader(
            dataset,
            batch_size=micro_batch_size,
            sampler=sampler,
            collate_fn=collator,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=num_workers > 0,
        )
    
    def __iter__(self):
        """返回 batch 迭代器。"""
        for batch in self._inner:
            # 在 yield 之前推进 sampler 状态
            # 这样如果迭代被打断（异常、break），下次 resume 跳过的位置也对
            self.sampler.advance(self.micro_batch_size)
            yield batch
    
    def __len__(self):
        """返回本 epoch 的 batch 数。"""
        return len(self._inner)
    
    def set_epoch(self, epoch: int):
        """设置 epoch 数，用于 DistributedSampler 的 shuffle。"""
        self.sampler.set_epoch(epoch)
    
    def state_dict(self) -> dict:
        return {
            "sampler": self.sampler.state_dict(),
            "seed": self.seed,
            "micro_batch_size": self.micro_batch_size,
        }
    
    def load_state_dict(self, sd: dict) -> None:
        # 配置一致性检查
        assert sd["seed"] == self.seed, (
            f"seed mismatch on resume: ckpt={sd['seed']}, current={self.seed}"
        )
        assert sd["micro_batch_size"] == self.micro_batch_size, (
            f"micro_batch_size mismatch: ckpt={sd['micro_batch_size']}, "
            f"current={self.micro_batch_size}"
        )
        self.sampler.load_state_dict(sd["sampler"])

    @property
    def tokens_per_step(self) -> int:
        """每步（所有 DP rank 合计）处理的 token 数。
        = micro_batch_size * seq_len * dp_size
        """
        return 0
