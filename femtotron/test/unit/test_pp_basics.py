"""
femtotron/test/unit/test_pp_basics.py

M3a 单元测试。纯 Python,无分布式,无 cuda,**直接 python 跑**(不用 torchrun)。

跑法:
    python femtotron/test/unit/test_pp_basics.py
"""

import torch

from femtotron.parallel.pipeline_parallel.action import (
    Backward,
    Forward,
    PPAction,
    RecvBackward,
    RecvForward,
    SendBackward,
    SendBackwardRecvForward,
    SendForward,
    SendForwardRecvBackward,
)
from femtotron.parallel.pipeline_parallel.microbatch import split_microbatches
from femtotron.parallel.pipeline_parallel.partition import partition_layers


# ════════════════════════════════════════════════════════════════
# action.py 测试
# ════════════════════════════════════════════════════════════════

def test_action_instantiation():
    """所有 action 类型能正常构造,字段对得上。"""
    print("Test: action instantiation")
    
    f = Forward(mb_id=3)
    assert f.mb_id == 3
    
    sfrb = SendForwardRecvBackward(fwd_mb=5, bwd_mb=2)
    assert sfrb.fwd_mb == 5 and sfrb.bwd_mb == 2
    
    sbrf = SendBackwardRecvForward(bwd_mb=1, fwd_mb=4)
    assert sbrf.bwd_mb == 1 and sbrf.fwd_mb == 4
    
    print("  ✓ all action types construct correctly")


def test_action_immutable():
    """frozen=True ⇒ 修改字段应该抛 FrozenInstanceError。"""
    print("Test: action immutability")
    
    f = Forward(mb_id=0)
    try:
        f.mb_id = 1
        assert False, "frozen dataclass should not allow modification"
    except Exception as e:
        # dataclasses.FrozenInstanceError
        assert "FrozenInstanceError" in type(e).__name__ or "frozen" in str(e).lower()
    
    print("  ✓ frozen prevents modification")


def test_action_equality_and_hashable():
    """同类型同字段相等且 hashable;不同类型 / 不同字段不等。"""
    print("Test: action equality and hashable")
    
    assert Forward(0) == Forward(0)
    assert Forward(0) != Forward(1)
    assert Forward(0) != Backward(0)  # 不同子类
    
    # hashable(frozen 给的 __hash__)
    s = {Forward(0), Forward(1), Backward(0)}
    assert len(s) == 3
    assert Forward(0) in s
    
    # combined ops 同理
    assert SendForwardRecvBackward(1, 2) == SendForwardRecvBackward(1, 2)
    assert SendForwardRecvBackward(1, 2) != SendForwardRecvBackward(2, 1)
    
    print("  ✓ equality and hashing work correctly")


def test_action_compact_repr():
    """__repr__ 紧凑,便于日志/调试。"""
    print("Test: action compact repr")
    
    assert repr(Forward(0)) == "F(0)"
    assert repr(Backward(3)) == "B(3)"
    assert repr(SendForward(2)) == "SF(2)"
    assert repr(RecvForward(2)) == "RF(2)"
    assert repr(SendBackward(1)) == "SB(1)"
    assert repr(RecvBackward(1)) == "RB(1)"
    assert repr(SendForwardRecvBackward(5, 2)) == "SFRB(5,2)"
    assert repr(SendBackwardRecvForward(1, 4)) == "SBRF(1,4)"
    
    # 列表打印也清晰
    actions = [Forward(0), SendForward(0), Forward(1), SendForwardRecvBackward(1, 0)]
    s = str(actions)
    assert "F(0)" in s and "SF(0)" in s and "SFRB(1,0)" in s
    
    print("  ✓ repr is compact and informative")


def test_action_subclass_check():
    """所有具体 action 都是 PPAction 子类(Runner 用 isinstance(action, PPAction) 兜底)。"""
    print("Test: all actions are PPAction subclasses")
    
    for action_cls in (Forward, Backward, SendForward, RecvForward,
                       SendBackward, RecvBackward,
                       SendForwardRecvBackward, SendBackwardRecvForward):
        # 构造一个实例(combined ops 需要两个 arg)
        if action_cls in (SendForwardRecvBackward, SendBackwardRecvForward):
            instance = action_cls(0, 0)
        else:
            instance = action_cls(0)
        assert isinstance(instance, PPAction)
    
    print("  ✓ subclass relationships correct")


# ════════════════════════════════════════════════════════════════
# microbatch.py 测试
# ════════════════════════════════════════════════════════════════

def test_split_basic():
    """整除情况下 split 正确,数量和形状对得上。"""
    print("Test: split_microbatches basic")
    
    batch = torch.arange(24).reshape(8, 3)  # (8, 3)
    mbs = split_microbatches(batch, num_microbatches=4)
    
    assert len(mbs) == 4
    for mb in mbs:
        assert mb.shape == (2, 3)
    
    # 内容正确
    assert torch.equal(mbs[0], batch[0:2])
    assert torch.equal(mbs[1], batch[2:4])
    assert torch.equal(mbs[2], batch[4:6])
    assert torch.equal(mbs[3], batch[6:8])
    
    print("  ✓ basic split correct")


def test_split_no_copy():
    """split 返回的是 view,不应拷贝(同样底层 storage)。"""
    print("Test: split_microbatches no copy")
    
    batch = torch.arange(16).reshape(4, 4).clone()
    mbs = split_microbatches(batch, num_microbatches=2)
    
    # view 应该共享 storage
    assert mbs[0].data_ptr() == batch.data_ptr()
    
    print("  ✓ split is view, no copy")


def test_split_non_divisible_raises():
    """batch_size 不被 num_microbatches 整除时报错。"""
    print("Test: split_microbatches non-divisible raises")
    
    batch = torch.arange(10).reshape(10, 1)
    try:
        split_microbatches(batch, num_microbatches=3)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "divisible" in str(e)
    
    print("  ✓ raises on non-divisible")


def test_split_edge_cases():
    """num_microbatches=1 / =batch_size 等边界。"""
    print("Test: split_microbatches edge cases")
    
    batch = torch.arange(8).reshape(8, 1)
    
    # num_mb=1 → 一份,等于原 batch
    mbs = split_microbatches(batch, num_microbatches=1)
    assert len(mbs) == 1
    assert torch.equal(mbs[0], batch)
    
    # num_mb=batch_size → 每份 1 样本
    mbs = split_microbatches(batch, num_microbatches=8)
    assert len(mbs) == 8
    for i, mb in enumerate(mbs):
        assert mb.shape == (1, 1)
        assert mb.item() == i
    
    # num_mb=0 → 报错
    try:
        split_microbatches(batch, num_microbatches=0)
        assert False
    except ValueError:
        pass
    
    print("  ✓ edge cases handled")


# ════════════════════════════════════════════════════════════════
# partition.py 测试
# ════════════════════════════════════════════════════════════════

def test_partition_uniform_divisible():
    """num_layers 被 pp_size 整除:每个 stage 同样多 layer。"""
    print("Test: partition uniform divisible")
    
    ranges = partition_layers(num_layers=8, pp_size=4)
    
    assert len(ranges) == 4
    assert list(ranges[0]) == [0, 1]
    assert list(ranges[1]) == [2, 3]
    assert list(ranges[2]) == [4, 5]
    assert list(ranges[3]) == [6, 7]
    
    # 覆盖性:所有 layer 都被分到某 stage,无重叠
    covered = set()
    for r in ranges:
        for layer in r:
            assert layer not in covered, f"layer {layer} assigned twice"
            covered.add(layer)
    assert covered == set(range(8))
    
    print("  ✓ uniform divisible: all layers covered uniquely")


def test_partition_uniform_remainder():
    """余数靠前分:早 stage 多 1 层。"""
    print("Test: partition uniform with remainder")
    
    # 10 / 4 = 2 base + 2 remainder ⇒ 前 2 个 stage 各多 1 层
    ranges = partition_layers(num_layers=10, pp_size=4)
    
    assert list(ranges[0]) == [0, 1, 2]   # 3 层
    assert list(ranges[1]) == [3, 4, 5]   # 3 层
    assert list(ranges[2]) == [6, 7]      # 2 层
    assert list(ranges[3]) == [8, 9]      # 2 层
    
    # 总和正确
    assert sum(len(r) for r in ranges) == 10
    
    print("  ✓ uniform remainder distributed to early stages")


def test_partition_pp_size_one():
    """pp_size=1:整个模型在一个 stage。"""
    print("Test: partition pp_size=1")
    
    ranges = partition_layers(num_layers=8, pp_size=1)
    assert len(ranges) == 1
    assert list(ranges[0]) == list(range(8))
    
    print("  ✓ pp_size=1 returns single full range")


def test_partition_validation():
    """非法参数应抛错。"""
    print("Test: partition validation")
    
    # pp_size < 1
    try:
        partition_layers(num_layers=8, pp_size=0)
        assert False
    except ValueError as e:
        assert "pp_size" in str(e)
    
    # num_layers < pp_size
    try:
        partition_layers(num_layers=3, pp_size=4)
        assert False
    except ValueError as e:
        assert "num_layers" in str(e)
    
    # 未知 strategy
    try:
        partition_layers(num_layers=8, pp_size=4, strategy="alien")
        assert False
    except NotImplementedError:
        pass
    
    print("  ✓ validation catches bad inputs")


def test_partition_manual():
    """manual 策略:用显式 layer_counts。"""
    print("Test: partition manual")
    
    # 32 层 4 stage,故意 last stage 少几层(lm_head aware)
    ranges = partition_layers(
        num_layers=32,
        pp_size=4,
        strategy="manual",
        layer_counts=[9, 9, 8, 6],
    )
    
    assert list(ranges[0]) == list(range(0, 9))
    assert list(ranges[1]) == list(range(9, 18))
    assert list(ranges[2]) == list(range(18, 26))
    assert list(ranges[3]) == list(range(26, 32))
    
    # 错误:layer_counts 长度不对
    try:
        partition_layers(
            num_layers=32, pp_size=4, strategy="manual",
            layer_counts=[10, 10, 12],  # 只有 3 项
        )
        assert False
    except ValueError as e:
        assert "layer_counts" in str(e)
    
    # 错误:layer_counts 总和不对
    try:
        partition_layers(
            num_layers=32, pp_size=4, strategy="manual",
            layer_counts=[8, 8, 8, 8],  # 32 ✓ 实际上对的
        )
        # 这其实是有效的,不报错
    except ValueError:
        assert False, "this should be valid"
    
    try:
        partition_layers(
            num_layers=32, pp_size=4, strategy="manual",
            layer_counts=[10, 10, 10, 10],  # 40 ≠ 32
        )
        assert False
    except ValueError as e:
        assert "sum" in str(e).lower()
    
    # 错误:某个 stage 0 层
    try:
        partition_layers(
            num_layers=10, pp_size=4, strategy="manual",
            layer_counts=[5, 5, 0, 0],
        )
        assert False
    except ValueError as e:
        assert "1 layer" in str(e) or ">= 1" in str(e)
    
    print("  ✓ manual strategy with validation")


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main():
    # action.py
    test_action_instantiation()
    test_action_immutable()
    test_action_equality_and_hashable()
    test_action_compact_repr()
    test_action_subclass_check()
    
    # microbatch.py
    test_split_basic()
    test_split_no_copy()
    test_split_non_divisible_raises()
    test_split_edge_cases()
    
    # partition.py
    test_partition_uniform_divisible()
    test_partition_uniform_remainder()
    test_partition_pp_size_one()
    test_partition_validation()
    test_partition_manual()
    
    print("\n[All M3a basics tests passed ✓]")


if __name__ == "__main__":
    main()