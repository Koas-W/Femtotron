"""
femtotron/parallel/pipeline/partition.py

把 num_layers 个 transformer layer 分配到 pp_size 个 stage。

返回 pp_size 个 range,每个是该 rank 持有的**全局 layer idx**(不是 local 编号)。
Llama 这边把 range 喂给 LlamaPartialModel(layer_range=...) 就能构造对应的 partial。

设计要点:
- 默认 uniform 策略:余数尽量靠前分(早 stage 多 1 层)
- 留 manual 策略接口:未来想做 "last stage 少分几层避开 lm_head 不平衡" 直接用
- 不假设 num_layers >= pp_size 时一定能分;严格校验
"""

from __future__ import annotations
from typing import Sequence


def partition_layers(
    num_layers: int,
    pp_size: int,
    strategy: str = "uniform",
    **kwargs,
) -> list[range]:
    """把 layer 分配到 pp_size 个 stage。
    
    Args:
        num_layers: transformer 总层数
        pp_size: pipeline 并行度
        strategy: "uniform"(均分,余数靠前)或 "manual"(显式给 layer_counts)
        **kwargs: strategy="manual" 时需要 layer_counts=list[int]
    
    Returns:
        list of pp_size ranges。stage i 持有 ranges[i] 里的所有 layer。
        
    Raises:
        ValueError: 参数不合法
        NotImplementedError: 未知 strategy
    """
    if pp_size < 1:
        raise ValueError(f"pp_size must be >= 1, got {pp_size}")
    if num_layers < pp_size:
        raise ValueError(
            f"num_layers={num_layers} < pp_size={pp_size}; "
            f"need at least one layer per stage"
        )
    
    if strategy == "uniform":
        return _partition_uniform(num_layers, pp_size)
    
    if strategy == "manual":
        layer_counts = kwargs.get("layer_counts")
        if layer_counts is None:
            raise ValueError("strategy='manual' requires layer_counts kwarg")
        return _partition_manual(num_layers, pp_size, layer_counts)
    
    raise NotImplementedError(
        f"Unknown partition strategy: {strategy!r}. "
        f"Supported: 'uniform', 'manual'."
    )


# ────────────────────────────────────────────────────────────────
# Strategies
# ────────────────────────────────────────────────────────────────

def _partition_uniform(num_layers: int, pp_size: int) -> list[range]:
    """均分,余数靠前分(早 stage 多 1 层)。
    
    例子:num_layers=10, pp_size=4 → [range(0,3), range(3,6), range(6,8), range(8,10)]
                                     #     3 层       3 层       2 层       2 层
    """
    base = num_layers // pp_size
    remainder = num_layers % pp_size
    
    ranges = []
    start = 0
    for i in range(pp_size):
        count = base + (1 if i < remainder else 0)
        ranges.append(range(start, start + count))
        start += count
    
    assert start == num_layers, "internal: layer accounting error"
    return ranges


def _partition_manual(
    num_layers: int,
    pp_size: int,
    layer_counts: Sequence[int],
) -> list[range]:
    """按 layer_counts 显式分配。
    
    例子:layer_counts=[9, 9, 8, 6] → 给 lm_head-aware 切分用,
         让 last stage 少分几层抵消 lm_head 的额外负载。
    """
    if len(layer_counts) != pp_size:
        raise ValueError(
            f"layer_counts has {len(layer_counts)} entries, expected pp_size={pp_size}"
        )
    if any(c < 1 for c in layer_counts):
        raise ValueError(
            f"each stage must have >= 1 layer, got layer_counts={layer_counts}"
        )
    if sum(layer_counts) != num_layers:
        raise ValueError(
            f"sum(layer_counts)={sum(layer_counts)} != num_layers={num_layers}"
        )
    
    ranges = []
    start = 0
    for count in layer_counts:
        ranges.append(range(start, start + count))
        start += count
    
    return ranges