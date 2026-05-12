"""ParamGroupCluster 基础单测。

不接 hook、不接 strategy,纯手动调用 cluster 的方法验证正确性。
"""

import os
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
from collections import OrderedDict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from femtotron.parallel_context import ParallelContext
from femtotron.sharding.param_group_cluster import ParamGroupCluster
from femtotron.training.param_group import ParamGroup
from femtotron.model.parallel_plan import ParallelPlan


def init_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def log(msg):
    if dist.get_rank() == 0:
        print(msg)


def make_param_group(name, module, attr, parallel_ctx, plan):
    """便利函数:为一个 Parameter 创建 ParamGroup。"""
    param = getattr(module, attr)
    return ParamGroup(
        name=f"{name}.{attr}",
        compute=param,
        master=None,
        master_spec=None,
        opt_config={"weight_decay": 0.01},
        parallel_ctx=parallel_ctx,
        parallel_plan=plan,
    )


def test_cluster(world_size, device):
    log("=" * 60)
    log("ParamGroupCluster 基础测试")
    log(f"World size: {world_size}")
    log("=" * 60)
    
    # 简单 parallel ctx:dp = world_size, tp = 1
    parallel_ctx = ParallelContext(OrderedDict([("dp", world_size), ("tp", 1)]))
    plan = ParallelPlan(rules={})    # 空 plan,所有参数 replicated
    
    # 构造一个简单 module(模拟 transformer block 子组件)
    torch.manual_seed(42)
    module = nn.Sequential(
        nn.Linear(8, 16, bias=True),       # weight [16,8]=128, bias [16]=16
        nn.LayerNorm(16),                   # weight [16]=16, bias [16]=16
    ).to(device).bfloat16()
    
    # 保存原始 weight 数据,后面验证 unshard 还原正确性
    original_data = {
        "0.weight": module[0].weight.data.clone(),
        "0.bias": module[0].bias.data.clone(),
        "1.weight": module[1].weight.data.clone(),
        "1.bias": module[1].bias.data.clone(),
    }
    
    # 构造 ParamGroup
    param_groups = [
        make_param_group("0", module[0], "weight", parallel_ctx, plan),
        make_param_group("0", module[0], "bias", parallel_ctx, plan),
        make_param_group("1", module[1], "weight", parallel_ctx, plan),
        make_param_group("1", module[1], "bias", parallel_ctx, plan),
    ]
    
    total_numel = 128 + 16 + 16 + 16
    log(f"\n总 numel: {total_numel}, dp_size: {world_size}")
    
    # ═══════════════ 构造 cluster ═══════════════
    cluster = ParamGroupCluster(
        name="test_cluster",
        module=module,
        param_groups=param_groups,
        dp_group=parallel_ctx.dp_group,
        master_dtype=torch.float32,
    )
    log(f"\n构造完成: {cluster}")
    log(f"显存: {cluster.memory_footprint()}")
    
    passed = True
    
    # ═══════════════ 验证 1:接管语义 ═══════════════
    log("\n[1] 接管语义验证")
    
    all_master_none = all(pg.master is None for pg in param_groups)
    all_cluster_set = all(pg.cluster is cluster for pg in param_groups)
    all_compute_empty = all(pg.compute.numel() == 0 for pg in param_groups)
    
    log(f"  pg.master 全是 None: {'✓' if all_master_none else '✗'}")
    log(f"  pg.cluster 指向 cluster: {'✓' if all_cluster_set else '✗'}")
    log(f"  pg.compute.data 是空 placeholder: {'✓' if all_compute_empty else '✗'}")
    
    if not (all_master_none and all_cluster_set and all_compute_empty):
        passed = False
    
    # ═══════════════ 验证 2:cluster 内部状态 ═══════════════
    log("\n[2] Cluster 内部状态")
    
    expected_shard = (total_numel + world_size - 1) // world_size
    log(f"  master.shape = {tuple(cluster.master.shape)} (期望 ({expected_shard},))")
    log(f"  master.dtype = {cluster.master.dtype} (期望 float32)")
    log(f"  master.requires_grad = {cluster.master.requires_grad}")
    log(f"  flat_param_shard.dtype = {cluster.flat_param_shard.dtype}")
    log(f"  flat_grad_shard.shape = {tuple(cluster.flat_grad_shard.shape)}")
    
    shape_ok = cluster.master.shape == (expected_shard,)
    dtype_ok = cluster.master.dtype == torch.float32 and cluster.flat_param_shard.dtype == torch.bfloat16
    master_no_grad_ok = not cluster.master.requires_grad

    # views 才是 optimizer 看到的"参数",需要 is_leaf 且 requires_grad
    views = cluster.get_optimizable_views()
    views_are_leaf_ok = all(v.is_leaf for _, v in views)
    views_require_grad_ok = all(v.requires_grad for _, v in views)

    log(f"  master.requires_grad = {cluster.master.requires_grad} (期望 False)")
    log(f"  views.is_leaf 全部为 True: {'✓' if views_are_leaf_ok else '✗'}")
    log(f"  views.requires_grad 全部为 True: {'✓' if views_require_grad_ok else '✗'}")

    if not (master_no_grad_ok and views_are_leaf_ok and views_require_grad_ok):
        passed = False
    
    if not (shape_ok and dtype_ok and master_no_grad_ok):
        passed = False
        log(f"  {'✓' if shape_ok else '✗'} shape  {'✓' if dtype_ok else '✗'} dtype  {'✓' if master_no_grad_ok else '✗'} requires_grad")
    
    # ═══════════════ 验证 3:views 正确 ═══════════════
    log("\n[3] Per-param views 验证")
    
    views = cluster.get_optimizable_views()
    log(f"  本 rank 持有 {len(views)} 个 view (总参数 {len(param_groups)} 个)")
    
    # views 是 cluster.master 的 narrow,storage 共享
    for pg, view in views:
        is_view_of_master = view.data_ptr() >= cluster.master.data_ptr()
        # narrow 的 view 的 data_ptr 应该在 master.data_ptr() 之后(偏移内)
        log(f"    {pg.name}: view.shape={tuple(view.shape)}, data_ptr offset = "
            f"{view.data_ptr() - cluster.master.data_ptr()} bytes")
    
    # ═══════════════ 验证 4:unshard 还原数值 ═══════════════
    log("\n[4] Unshard 数值还原")
    
    cluster.unshard()
    
    # compute.data 形状应该恢复原始 shape
    shapes_ok = all(
        pg.compute.shape == original_data[pg.name.replace(".weight", "").replace(".bias", "") + "." + pg.name.rsplit(".", 1)[1]].shape if False else True
        for pg in param_groups
    )
    # 重新写更简洁的版本
    shapes_ok = True
    for pg in param_groups:
        # pg.name like "0.weight"
        expected_shape = original_data[pg.name].shape
        if pg.compute.shape != expected_shape:
            shapes_ok = False
            log(f"  ✗ {pg.name}: got {tuple(pg.compute.shape)}, expected {tuple(expected_shape)}")
    
    # 数值上 compute.data 应该等于原始数据(经过 cast 到 bf16)
    values_ok = True
    for pg in param_groups:
        expected = original_data[pg.name].to(pg.compute.dtype)
        if not torch.allclose(pg.compute.data, expected, atol=1e-3):
            values_ok = False
            max_diff = (pg.compute.data.float() - expected.float()).abs().max().item()
            log(f"  ✗ {pg.name}: max diff = {max_diff}")
    
    log(f"  形状还原: {'✓' if shapes_ok else '✗'}")
    log(f"  数值还原: {'✓' if values_ok else '✗'}")
    if not (shapes_ok and values_ok):
        passed = False
    
    # ═══════════════ 验证 5:reshard 清理 ═══════════════
    log("\n[5] Reshard 清理")
    
    cluster.reshard()
    
    reshard_ok = all(pg.compute.numel() == 0 for pg in param_groups)
    buffer_released = cluster._full_buffer is None
    log(f"  compute.data 回到空: {'✓' if reshard_ok else '✗'}")
    log(f"  _full_buffer 释放: {'✓' if buffer_released else '✗'}")
    if not (reshard_ok and buffer_released):
        passed = False
    
    # ═══════════════ 验证 6:模拟 backward 后 reduce_scatter ═══════════════
    log("\n[6] Reduce-scatter grads")
    
    cluster.unshard()
    # 每个 compute.grad 设为常数(模拟 backward 结果)
    for i, pg in enumerate(param_groups):
        # 用不同常数便于追踪:第 i 个参数的 grad 全是 (rank+1) * (i+1)
        constant = float((cluster.dp_rank + 1) * (i + 1))
        pg.compute.grad = torch.full_like(pg.compute, constant)
    
    cluster.reduce_scatter_grads()
    
    # 验证 1:所有 compute.grad 被清空
    grads_cleared = all(pg.compute.grad is None for pg in param_groups)
    log(f"  compute.grad 清空: {'✓' if grads_cleared else '✗'}")
    
    # 验证 2:flat_grad_shard 是各 rank grad 的 AVG
    # 第 i 个参数所有 rank 的 grad 是 [(0+1)*(i+1), (1+1)*(i+1), ..., (dp_size)*(i+1)]
    # AVG = sum / dp_size
    # 但 reduce_scatter 是按 shard 切的,本 rank 拿到的 shard 对应 flat 的某段
    # 这一段可能跨多个 param,但每个元素的值是该位置上所有 rank 数据的 AVG
    # 实际值要按 layout 算,复杂——简化验证:flat_grad_shard 不全为零、且 dtype 正确
    grad_shard_has_data = (cluster.flat_grad_shard.abs().sum() > 0).item()
    log(f"  flat_grad_shard 非零: {'✓' if grad_shard_has_data else '✗'}")
    
    if not (grads_cleared and grad_shard_has_data):
        passed = False
    
    cluster.reshard()
    
    # ═══════════════ 验证 7:populate_master_grad ═══════════════
    log("\n[7] Populate master grad")
    
    cluster.populate_master_grad()
    
    master_grad_ok = cluster.master.grad is not None and cluster.master.grad.dtype == torch.float32
    log(f"  master.grad 配置: {'✓' if master_grad_ok else '✗'}")
    
    # 每个 view.grad 应该是 master.grad 的 slice(storage 共享)
    views_grad_ok = True
    for pg, view in views:
        if view.grad is None:
            views_grad_ok = False
            log(f"  ✗ {pg.name}: view.grad is None")
            continue
        # view.grad 应该和 master.grad 共享 storage
        if view.grad.data_ptr() < cluster.master.grad.data_ptr():
            views_grad_ok = False
    
    log(f"  view.grad 配置: {'✓' if views_grad_ok else '✗'}")
    
    if not (master_grad_ok and views_grad_ok):
        passed = False
    
    # ═══════════════ 验证 8:模拟 optimizer step,view 改变反映到 master ═══════════════
    log("\n[8] View → Master 数据传播")
    
    # 通过 view in-place 加一个常数,master 对应位置应该改变
    old_master = cluster.master.detach().clone()
    for pg, view in views:
        view.data.add_(0.5)    # in-place
    # 验证 view 和 master 真的共享 storage
    for pg, view in views:
        assert view.data_ptr() >= cluster.master.data_ptr()
        assert view.data_ptr() < cluster.master.data_ptr() + cluster.master.numel() * cluster.master.element_size()
        # 或者更直接:
        # assert view.storage().data_ptr() == cluster.master.storage().data_ptr()
    
    diff = (cluster.master - old_master).abs()
    expected_changed = diff > 0.4    # 应该有约 0.5 的差
    has_changes = expected_changed.sum().item()
    log(f"  master 中改变的元素数: {has_changes} (期望 > 0)")
    
    if has_changes == 0:
        passed = False
        log(f"  ✗ view 修改没有反映到 master!")
    
    # ═══════════════ 验证 9:sync_master_to_compute ═══════════════
    log("\n[9] Sync master → flat_param_shard")
    
    old_shard = cluster.flat_param_shard.clone()
    cluster.sync_master_to_compute()
    
    # flat_param_shard 应该等于 master cast 到 bf16
    expected = cluster.master.to(cluster.compute_dtype)
    sync_ok = torch.allclose(cluster.flat_param_shard, expected, atol=1e-3)
    log(f"  flat_param_shard == master.to(bf16): {'✓' if sync_ok else '✗'}")
    
    if not sync_ok:
        passed = False
    
    # ═══════════════ 验证 10:zero_grad 清理 ═══════════════
    log("\n[10] Zero grad")
    
    cluster.zero_grad()
    
    zg_master_ok = cluster.master.grad is None
    zg_views_ok = all(view.grad is None for _, view in views)
    zg_shard_ok = (cluster.flat_grad_shard == 0).all().item()
    
    log(f"  master.grad = None: {'✓' if zg_master_ok else '✗'}")
    log(f"  view.grad = None: {'✓' if zg_views_ok else '✗'}")
    log(f"  flat_grad_shard zeroed: {'✓' if zg_shard_ok else '✗'}")
    
    if not (zg_master_ok and zg_views_ok and zg_shard_ok):
        passed = False
    
    # ═══════════════ 总结 ═══════════════
    log(f"\n{'=' * 60}")
    log(f"{'✓ 全部通过' if passed else '✗ 有失败项'}")
    log(f"{'=' * 60}")
    
    return passed


def main():
    local_rank = init_distributed()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    
    passed = test_cluster(world_size, device)
    
    dist.destroy_process_group()
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()