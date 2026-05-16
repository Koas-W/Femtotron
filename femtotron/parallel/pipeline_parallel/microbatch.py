"""
femtotron/parallel/pipeline/microbatch.py

把一个 global batch 切成 num_microbatches 个 microbatch。

设计极简:不做 padding、不做 imbalanced split、不做 cross-rank coordination。
要求 batch.shape[0] 必须能被 num_microbatches 整除——这是训练脚本的约定。

返回 list[Tensor],每个是 batch 的一个 view(torch.split 不拷贝)。
"""

from __future__ import annotations
import torch


def split_microbatches(
    batch: torch.Tensor,
    num_microbatches: int,
) -> list[torch.Tensor]:
    """沿 dim 0 把 batch 等分成 num_microbatches 份。
    
    Args:
        batch: 形如 (B, ...) 的 tensor
        num_microbatches: 分多少份;B 必须被它整除
    
    Returns:
        长度 num_microbatches 的列表,每个是 (B/num_microbatches, ...) 的 view
    
    Raises:
        ValueError: B 不能被 num_microbatches 整除
    """
    if num_microbatches <= 0:
        raise ValueError(f"num_microbatches must be >= 1, got {num_microbatches}")
    
    bsz = batch.shape[0]
    if bsz % num_microbatches != 0:
        raise ValueError(
            f"batch_size {bsz} not divisible by num_microbatches {num_microbatches}"
        )
    
    mb_size = bsz // num_microbatches
    return list(batch.split(mb_size, dim=0))