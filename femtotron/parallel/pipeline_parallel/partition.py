# femtotron/parallel/pipeline/partition.py

def partition_layers(num_layers: int, pp_size: int) -> list[range]:
    """把 num_layers 个 layer 均匀分给 pp_size 个 stage。
    
    Returns:
        list of range,第 i 个 range 是 stage i 持有的 layer indices(全局编号)。
    
    Examples:
        partition_layers(8, 2) → [range(0, 4), range(4, 8)]
        partition_layers(7, 3) → [range(0, 3), range(3, 5), range(5, 7)]
                                  (3, 2, 2 — 越前面层数越多)
        partition_layers(8, 1) → [range(0, 8)]
    """
    assert num_layers >= pp_size, (
        f"num_layers={num_layers} < pp_size={pp_size},无法切分"
    )
    
    base = num_layers // pp_size
    extra = num_layers % pp_size
    
    # 多余的 layer 分给前 `extra` 个 stage(每个多一个)
    ranges = []
    start = 0
    for i in range(pp_size):
        size = base + (1 if i < extra else 0)
        ranges.append(range(start, start + size))
        start += size
    
    return ranges