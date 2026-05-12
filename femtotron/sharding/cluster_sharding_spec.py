"""Cluster 内部的参数布局描述。

与 ShardingSpec 的关系:
- ShardingSpec:描述"一个 tensor 在 dp_group 内被切成 dp_size 份后的本 rank 视角"
  用于 ZeRO-1/2 这种"单参数独立 reduce_scatter"的场景。
- ClusterShardingSpec:描述"一个 param 在 cluster 的 flat 表示中的位置"
  用于 ZeRO-3 这种"unit 内多参数打包通信、分片"的场景。

两者各管不同的 sharding 维度——前者是"参数自身的分片",后者是"参数在 unit flat 中的定位"。
不复用结构,因为概念不重叠。
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ClusterShardingSpec:
    """描述一个参数在 ParamGroupCluster 的 flat 表示中的位置。
    
    一个 cluster 内有多个参数,所有参数被 flatten 后 concat 成一个 flat tensor,
    然后按 dp_size 切片。每个参数有两套坐标:
    
    1. 全局坐标 (global_*):在完整(未切分)flat tensor 中的位置
       用于 unshard 时把 full_buffer 切回各 Parameter.data 的 view
    
    2. 本 rank 局部坐标 (local_*):在本 rank 持有的 master shard 中的位置
       用于创建 optimizer 看到的 per-param view
    
    参数和本 rank 的关系有三种:
    - 完全在本 rank shard 范围内
    - 完全在本 rank shard 范围外
    - 跨边界(部分在本 rank、部分在邻居)
    
    has_local_view 区分前两者,local_offset / local_numel 描述跨边界的具体情况。
    """
    
    # ─── 全局坐标(所有 rank 一致) ───
    global_offset: int
    """在完整 flat tensor 中的起始位置(以元素为单位)"""
    
    numel: int
    """参数总元素数"""
    
    original_shape: torch.Size
    """原始 Parameter 形状,unshard 时用 .view() 还原"""
    
    original_dtype: torch.dtype
    """原始 Parameter dtype(通常 bf16),区别于 master 的 fp32"""
    
    # ─── 本 rank 局部坐标(每 rank 不同) ───
    has_local_view: bool
    """本 rank 是否持有该参数的任何元素。
    
    False 时 local_offset / local_numel 无意义(置 0)。
    """
    
    local_offset: int = 0
    """在本 rank master shard 中的起始位置。
    
    只在 has_local_view=True 时有效。
    """
    
    local_numel: int = 0
    """本 rank 持有的元素数(可能小于 numel,跨边界时)。
    
    只在 has_local_view=True 时有效。
    """
    
    @property
    def global_end(self) -> int:
        """便利属性:全局结束位置(exclusive)。"""
        return self.global_offset + self.numel
    
    @property
    def local_end(self) -> int:
        """便利属性:本 rank 结束位置(exclusive)。
        
        仅在 has_local_view=True 时有意义。
        """
        return self.local_offset + self.local_numel
    
    @property
    def is_fully_local(self) -> bool:
        """该参数的所有元素都在本 rank 上(没跨边界)。"""
        return self.has_local_view and self.local_numel == self.numel
    
    @property
    def is_partially_local(self) -> bool:
        """该参数跨 rank 边界,本 rank 只持有一部分。"""
        return self.has_local_view and self.local_numel < self.numel


def compute_cluster_layout(
    param_numels: list[int],
    param_shapes: list[torch.Size],
    param_dtypes: list[torch.dtype],
    dp_rank: int,
    dp_size: int,
) -> tuple[list[ClusterShardingSpec], int, int]:
    """为一组参数计算它们在 cluster flat 中的布局。
    
    Args:
        param_numels:每个参数的元素数(按 cluster 内顺序)
        param_shapes:每个参数的原始形状
        param_dtypes:每个参数的 dtype
        dp_rank:本 rank 在 dp_group 中的 rank
        dp_size:dp_group 的 world size
    
    Returns:
        (layouts, shard_size, pad_size):
        - layouts:每个参数的 ClusterShardingSpec
        - shard_size:本 rank 的 master shard 大小(向上取整到 dp_size 的倍数后除以 dp_size)
        - pad_size:padding 元素数(flat 末尾)
    """
    assert len(param_numels) == len(param_shapes) == len(param_dtypes)
    
    total_numel = sum(param_numels)
    # 向上取整到 dp_size 的倍数
    padded_size = ((total_numel + dp_size - 1) // dp_size) * dp_size
    shard_size = padded_size // dp_size
    pad_size = padded_size - total_numel
    
    # 本 rank 在 padded flat 中持有的范围 [local_start, local_end)
    local_start = dp_rank * shard_size
    local_end = local_start + shard_size
    
    layouts: list[ClusterShardingSpec] = []
    global_offset = 0
    
    for numel, shape, dtype in zip(param_numels, param_shapes, param_dtypes):
        param_start = global_offset
        param_end = global_offset + numel
        
        # 计算本 rank 持有这个参数的范围(交集)
        intersect_start = max(param_start, local_start)
        intersect_end = min(param_end, local_end)
        
        if intersect_start < intersect_end:
            has_local = True
            local_offset = intersect_start - local_start
            local_numel = intersect_end - intersect_start
        else:
            has_local = False
            local_offset = 0
            local_numel = 0
        
        layouts.append(ClusterShardingSpec(
            global_offset=global_offset,
            numel=numel,
            original_shape=shape,
            original_dtype=dtype,
            has_local_view=has_local,
            local_offset=local_offset,
            local_numel=local_numel,
        ))
        global_offset += numel
    
    return layouts, shard_size, pad_size