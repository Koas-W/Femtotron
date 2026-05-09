import os, sys
import torch
from torch import nn, Tensor
from torch import distributed as dist
import torch.nn.functional as F
import json
from pathlib import Path
from collections import defaultdict
from safetensors import safe_open
from typing import cast, Protocol


class Collator(Protocol):
    def __call__(self, samples: list) -> dict[str, Tensor]: ...

def simple_pretrain_collator(
    samples: list[Tensor],
) -> dict[str, Tensor]:
    """
    预训练 collator：假设输入已经是定长（dataset 层完成 packing）。
    
    输入：每个 sample 是长度相同的 token list 或 tensor
    输出：
        input_ids: [batch, seq_len], int64
        labels:    [batch, seq_len], int64（= input_ids.clone()，模型内部做 shift）
    """
    if isinstance(samples[0], Tensor):
        input_ids = torch.stack(samples).long()
    else:
        input_ids = torch.tensor(samples, dtype=torch.long)
    
    return {
        "input_ids": input_ids,
        "labels": input_ids.clone(),
    }


class PadSftCollator:
    """SFT collator：把变长样本 pad 到 batch 内最长（或固定 seq_len），
    并 mask 掉 prompt 和 padding 部分的 loss。
    """
    
    def __init__(
        self,
        pad_token_id: int,
        seq_len: int | None = None,   # None = pad 到 batch 内最长
        ignore_index: int = -100,
    ):
        self.pad_token_id = pad_token_id
        self.seq_len = seq_len
        self.ignore_index = ignore_index
    
    def __call__(self, samples: list[dict]) -> dict[str, Tensor]:
        # 每个 sample: {"input_ids": [...], "labels": [...]}
        # labels 已经在 dataset 层把 prompt 部分置为 ignore_index
        target_len = self.seq_len or max(len(s["input_ids"]) for s in samples)
        
        input_ids = []
        labels = []
        attention_mask = []
        for s in samples:
            ids = list(s["input_ids"])
            lbl = list(s["labels"])
            pad_n = target_len - len(ids)
            assert pad_n >= 0, f"sample length {len(ids)} > target_len {target_len}"
            
            input_ids.append(ids + [self.pad_token_id] * pad_n)
            labels.append(lbl + [self.ignore_index] * pad_n)
            attention_mask.append([1] * len(ids) + [0] * pad_n)
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }