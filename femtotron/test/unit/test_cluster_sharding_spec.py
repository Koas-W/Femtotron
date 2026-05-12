# tests/unit/test_cluster_sharding_spec.py

import torch
from femtotron.sharding.cluster_sharding_spec import compute_cluster_layout, ClusterShardingSpec


def test_simple_aligned():
    """3 个参数,总 numel 刚好被 dp_size 整除,无 padding。"""
    # numels: [4, 4, 4], dp_size=3 → shard_size=4, pad=0
    layouts, shard_size, pad_size = compute_cluster_layout(
        param_numels=[4, 4, 4],
        param_shapes=[torch.Size([2, 2]), torch.Size([2, 2]), torch.Size([4])],
        param_dtypes=[torch.bfloat16] * 3,
        dp_rank=0,
        dp_size=3,
    )
    
    assert shard_size == 4
    assert pad_size == 0
    
    # Rank 0 持有 [0:4] = 完整的 param 0
    assert layouts[0].has_local_view and layouts[0].is_fully_local
    assert layouts[0].local_offset == 0 and layouts[0].local_numel == 4
    
    # Param 1 在 [4:8],rank 0 不持有
    assert not layouts[1].has_local_view
    
    # Param 2 在 [8:12],rank 0 不持有
    assert not layouts[2].has_local_view


def test_padding_needed():
    """总 numel=9, dp_size=4 → padded=12, shard_size=3, pad=3。"""
    layouts, shard_size, pad_size = compute_cluster_layout(
        param_numels=[5, 2, 2],
        param_shapes=[torch.Size([5]), torch.Size([2]), torch.Size([2])],
        param_dtypes=[torch.bfloat16] * 3,
        dp_rank=0,
        dp_size=4,
    )
    
    assert shard_size == 3
    assert pad_size == 3
    
    # 全局布局:[a0 a1 a2 a3 a4 b0 b1 c0 c1 p p p]
    # Rank 0 持有 [0:3] = a0 a1 a2(param A 的前 3 个)
    assert layouts[0].has_local_view and layouts[0].is_partially_local
    assert layouts[0].local_offset == 0 and layouts[0].local_numel == 3


def test_cross_boundary():
    """参数跨 rank 边界的复杂情况。
    
    numels: [5, 2, 2], dp_size=3 → padded=9, shard_size=3
    Global flat: [a0 a1 a2 | a3 a4 b0 | b1 c0 c1]
                  rank 0    rank 1    rank 2
    
    Param A 跨 rank 0 和 rank 1。
    Param B 跨 rank 1 和 rank 2。
    """
    # Rank 0
    layouts_r0, _, _ = compute_cluster_layout(
        param_numels=[5, 2, 2],
        param_shapes=[torch.Size([5]), torch.Size([2]), torch.Size([2])],
        param_dtypes=[torch.bfloat16] * 3,
        dp_rank=0,
        dp_size=3,
    )
    # Rank 0 持有 A 的 [0:3]
    assert layouts_r0[0].local_offset == 0 and layouts_r0[0].local_numel == 3
    assert not layouts_r0[1].has_local_view    # B 不在 rank 0
    assert not layouts_r0[2].has_local_view    # C 不在 rank 0
    
    # Rank 1
    layouts_r1, _, _ = compute_cluster_layout(
        param_numels=[5, 2, 2],
        param_shapes=[torch.Size([5]), torch.Size([2]), torch.Size([2])],
        param_dtypes=[torch.bfloat16] * 3,
        dp_rank=1,
        dp_size=3,
    )
    # Rank 1 持有 A 的 [3:5](2 个元素)+ B 的 [0:1](1 个元素)
    assert layouts_r1[0].is_partially_local
    assert layouts_r1[0].local_offset == 0 and layouts_r1[0].local_numel == 2
    assert layouts_r1[1].is_partially_local
    assert layouts_r1[1].local_offset == 2 and layouts_r1[1].local_numel == 1
    assert not layouts_r1[2].has_local_view    # C 不在 rank 1
    
    # Rank 2
    layouts_r2, _, _ = compute_cluster_layout(
        param_numels=[5, 2, 2],
        param_shapes=[torch.Size([5]), torch.Size([2]), torch.Size([2])],
        param_dtypes=[torch.bfloat16] * 3,
        dp_rank=2,
        dp_size=3,
    )
    # Rank 2 持有 B 的 [1:2](1 个元素)+ C 的 [0:2](2 个元素)
    assert not layouts_r2[0].has_local_view    # A 不在 rank 2
    assert layouts_r2[1].is_partially_local
    assert layouts_r2[1].local_offset == 0 and layouts_r2[1].local_numel == 1
    assert layouts_r2[2].is_fully_local
    assert layouts_r2[2].local_offset == 1 and layouts_r2[2].local_numel == 2


def test_single_param_smaller_than_shard():
    """单参数远小于 shard_size 的情况。"""
    layouts, shard_size, pad_size = compute_cluster_layout(
        param_numels=[10],
        param_shapes=[torch.Size([10])],
        param_dtypes=[torch.bfloat16],
        dp_rank=1,
        dp_size=4,
    )
    
    # padded=12, shard_size=3
    # Rank 1 持有 [3:6](全部 3 个元素都来自 param 0)
    assert layouts[0].local_offset == 0 and layouts[0].local_numel == 3

if __name__ == "__main__":
    test_simple_aligned()
    test_padding_needed()
    test_cross_boundary()
    test_single_param_smaller_than_shard()
    print("All tests passed!")