"""ParamGroupCluster:ZeRO-3 的单元级优化抽象。

一个 cluster 对应一个 FSDP unit(通常一个 transformer block),管理:
- 通信:一次 all_gather/reduce_scatter 处理 unit 内所有参数
- 优化:为每个 ParamGroup 创建 view 到 cluster master,
  保留 per-param 的 opt_config(weight_decay 等)语义

设计约束:
- cluster 构造时接管 ParamGroup.master(掏空,设 cluster 反向引用)
- compute.data 替换为空 placeholder——分片状态下不应被访问
- 所有 cluster 内部状态(flat shard、master、buffer)由 cluster 独占管理
- view 是 master 的 narrow,storage 共享,optimizer 通过 view 间接更新 master
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.distributed import ProcessGroup

from femtotron.sharding.cluster_sharding_spec import (
    ClusterShardingSpec,
    compute_cluster_layout,
)


class ParamGroupCluster:
    """一个 FSDP unit 内多个 ParamGroup 的打包优化单元。
    
    生命周期:
    1. __init__:构造时计算 layout、收集初始数据到 flat、创建 master 和 views、
                 接管 ParamGroup.master、清空 compute.data
    2. unshard():forward/backward 前 hook 调用,all_gather flat_param_shard 到 full buffer,
                  把每个 Parameter.data 指向 full buffer 的对应 slice
    3. reshard():forward/backward 后 hook 调用,释放 full buffer,compute.data 回到 placeholder
    4. reduce_scatter_grads():backward 完成后调用,收集 grad → reduce_scatter → flat_grad_shard
    5. populate_master_grad():step 时调用,flat_grad_shard → master.grad(fp32),配置 view.grad
    6. sync_master_to_compute():step 完成后调用,master → flat_param_shard
    7. zero_grad():清理 grad 状态
    """
    
    def __init__(
        self,
        name: str,
        module: nn.Module,
        param_groups: list,    # list[ParamGroup],避免循环导入
        dp_group: ProcessGroup,
        master_dtype: torch.dtype,
    ) -> None:
        """构造 cluster,接管 param_groups 中的 master,把 compute 替换为 placeholder。
        
        Args:
            name:cluster 标识(通常是 unit module 的 qualified name)
            module:对应的 FSDP unit module(hook 会装在它上面)
            param_groups:被打包的 ParamGroup 列表(顺序决定 flat 中的布局)
            dp_group:用于 all_gather / reduce_scatter 的 process group
            master_dtype:master 的 dtype(必须非空,由 strategy 保证)
        
        前置条件:
        - 所有 param_group.compute.data 必须有意义的初始数据(已加载权重)
        - 所有 param_group.cluster 必须是 None(未被其他 cluster 接管)
        - 所有 param_group.compute 在同一 device 上
        """
        self.name = name
        self.module = module
        self.param_groups = param_groups
        self.dp_group = dp_group
        self.dp_rank = dist.get_rank(dp_group)
        self.dp_size = dist.get_world_size(dp_group)
        self.master_dtype = master_dtype
        
        # ─── 前置检查 ───
        self._validate_param_groups()
        
        # ─── 确定 device 和 compute dtype(从第一个 ParamGroup 取) ───
        first = param_groups[0].compute
        self.device = first.device
        self.compute_dtype = first.dtype
        
        # ─── 1. 计算 layout ───
        self.layouts: list[ClusterShardingSpec]
        self.shard_size: int
        self.pad_size: int
        self.layouts, self.shard_size, self.pad_size = compute_cluster_layout(
            param_numels=[pg.compute.numel() for pg in param_groups],
            param_shapes=[pg.compute.shape for pg in param_groups],
            param_dtypes=[pg.compute.dtype for pg in param_groups],
            dp_rank=self.dp_rank,
            dp_size=self.dp_size,
        )
        self.total_numel = sum(l.numel for l in self.layouts)
        self.padded_size = self.shard_size * self.dp_size
        
        # ─── 2. 收集所有 compute.data 到完整 flat,切出本 rank shard ───
        self.flat_param_shard: Tensor = self._initialize_flat_param_shard()
        
        # ─── 3. 创建 master(fp32 拷贝)+ per-param view ───
        self.master: Tensor = self._initialize_master()
        self.param_views: dict[str, Tensor] = self._create_param_views()
        
        # ─── 4. 预分配 flat_grad_shard(可延后,但预分配简单) ───
        self.flat_grad_shard: Tensor = torch.zeros(
            self.shard_size,
            dtype=self.compute_dtype,
            device=self.device,
        )
        
        # ─── 5. unshard 期间的 buffer 占位 ───
        self._full_buffer: Tensor | None = None
        
        # ─── 6. 共享 opt_config(校验所有 ParamGroup 一致) ───
        self.opt_config: dict = self._derive_opt_config()
        
        # ─── 7. 接管 master 所有权:清空 ParamGroup.master 字段,
        #       设反向引用,把 compute.data 替换为空 placeholder ───
        self._take_over_param_groups()
    
    # ═════════════════════════════════════════════════════════════
    # 构造期辅助方法
    # ═════════════════════════════════════════════════════════════
    
    def _validate_param_groups(self) -> None:
        """前置条件检查。"""
        assert len(self.param_groups) > 0, f"cluster {self.name}: empty param_groups"
        
        # 所有 ParamGroup 未被接管
        for pg in self.param_groups:
            assert pg.cluster is None, (
                f"cluster {self.name}: ParamGroup {pg.name} already claimed by another cluster"
            )
        
        # 所有参数同 device
        devices = {pg.compute.device for pg in self.param_groups}
        assert len(devices) == 1, (
            f"cluster {self.name}: ParamGroups on different devices: {devices}"
        )
    
    def _initialize_flat_param_shard(self) -> Tensor:
        """从所有 compute.data 构造完整 flat(padded),切出本 rank 那段。"""
        # 完整 flat(padded 大小)
        full = torch.empty(self.padded_size, dtype=self.compute_dtype, device=self.device)
        
        # 把每个 param 的当前 data copy 进去
        for pg, layout in zip(self.param_groups, self.layouts):
            full[layout.global_offset : layout.global_end].copy_(
                pg.compute.data.flatten()
            )
        
        # padding 部分填 0(reduce_scatter 时需要,否则 AVG 引入垃圾)
        if self.pad_size > 0:
            full[self.total_numel:].zero_()
        
        # 切出本 rank 那段
        start = self.dp_rank * self.shard_size
        return full[start : start + self.shard_size].clone()
    
    def _initialize_master(self) -> Tensor:
        """fp32 master 是 flat_param_shard 的 fp32 拷贝。
        
        master 是 inner_optimizer 看到的"逻辑参数"(虽然通过 view 间接访问)。
        requires_grad=True,但 grad 由我们手动 assign,不由 autograd 填充。
        """
        master = self.flat_param_shard.to(self.master_dtype)
        # master.requires_grad_(True)
        return master
    
    def _create_param_views(self) -> dict[str, Tensor]:
        """为每个 has_local_view 的 ParamGroup 创建 master 上的 view。
        
        view 是 master 的 narrow,共享 storage:
        - optimizer 读 view.data / view.grad → 实际访问 master 对应 slice
        - optimizer 写 view.data(in-place add_、addcmul_ 等)→ 直接更新 master
        
        没 local view 的 ParamGroup(参数完全在其他 rank)不进 param_views。
        """
        views: dict[str, Tensor] = {}
        for pg, layout in zip(self.param_groups, self.layouts):
            if layout.has_local_view:
                view = (
                    self.master.narrow(0, layout.local_offset, layout.local_numel)
                    .detach()
                    .requires_grad_(True)
                )
                views[pg.name] = view
        return views
    
    def _derive_opt_config(self) -> dict:
        """从 ParamGroup 们 derive 共享的 opt_config。
        
        Cluster 内所有 ParamGroup 必须有相同的 opt_config——否则一个 cluster 内
        部分参数用 wd=0、部分用 wd=0.01,我们的"per-param view + 按 opt_config 聚合"
        机制仍然能处理(不同 view 进不同 optimizer.param_groups),所以技术上不强制。
        
        但 cluster.opt_config 这个属性提供给 mp_manager 做整体描述(如果有需要),
        我们记录第一个 ParamGroup 的 opt_config 作为代表。如果存在不一致,警告。
        """
        first_cfg = self.param_groups[0].opt_config
        for pg in self.param_groups[1:]:
            if pg.opt_config != first_cfg:
                # 不同 opt_config 在 cluster 内 OK——每个 view 按自己的 opt_config
                # 分到对应 optimizer.param_groups。这里仅记录第一个作为 cluster
                # 的"代表 opt_config"。
                pass
        return dict(first_cfg)
    
    def _take_over_param_groups(self) -> None:
        """接管 ParamGroup 的 master 所有权。
        
        - pg.master 置 None:表明它不再独立优化
        - pg.master_spec 置 None:它不被 ZeRO-1/2 风格分片
        - pg.cluster 设为 self:反向引用,便于调试
        - pg.compute.data 替换为空 placeholder:分片状态下不应被访问,
          访问空 tensor 会立刻 fail,而不是 silently 用错值
        """
        for pg in self.param_groups:
            pg.master = None
            pg.master_spec = None
            pg.cluster = self
            
            # 把 Parameter.data 替换为空 placeholder
            # 保留 dtype 和 device 以便外部代码访问 .dtype / .device 不报错
            pg.compute.data = torch.empty(
                0, dtype=pg.compute.dtype, device=pg.compute.device,
            )
    
    # ═════════════════════════════════════════════════════════════
    # Forward / Backward 生命周期(由 hook 触发)
    # ═════════════════════════════════════════════════════════════
    
    def unshard(self) -> None:
        """all_gather flat_param_shard 成完整 buffer,
        把每个 Parameter.data 指向 buffer 的对应 slice。
        
        幂等:重复调用直接返回(已 unshard 状态)。
        """
        if self._full_buffer is not None:
            return   # 已经是 unsharded 状态,no-op
        
        # 分配 full buffer(padded 大小)
        self._full_buffer = torch.empty(
            self.padded_size,
            dtype=self.compute_dtype,
            device=self.device,
        )
        
        # all_gather:各 rank 的 flat_param_shard 拼成完整 buffer
        dist.all_gather_into_tensor(
            self._full_buffer,
            self.flat_param_shard,
            group=self.dp_group,
        )
        
        # 把每个 Parameter.data 指向 buffer 的对应 slice
        # buffer 末尾的 padding 部分不会被任何 view 引用(没 layout 指向它)
        for pg, layout in zip(self.param_groups, self.layouts):
            slice_ = self._full_buffer[layout.global_offset : layout.global_end]
            pg.compute.data = slice_.view(layout.original_shape)
    
    def reshard(self) -> None:
        """释放 full buffer,Parameter.data 回到空 placeholder。
        
        幂等:重复调用直接返回(已 reshard 状态)。
        """
        if self._full_buffer is None:
            return   # 已经是 sharded 状态,no-op
        
        for pg in self.param_groups:
            pg.compute.data = torch.empty(
                0, dtype=pg.compute.dtype, device=pg.compute.device,
            )
        
        self._full_buffer = None
    
    def reduce_scatter_grads(self) -> None:
        """backward 完成后:收集所有 param.grad,拼成 flat,reduce_scatter 到本 rank shard。
        
        前置条件:
        - 处于 unshard 状态(compute.grad 有意义)
        - 每个 compute.grad 形状 == compute.shape(== layout.original_shape)
        
        操作:
        1. 准备完整 flat_grad(padded 大小)
        2. 把每个 compute.grad flatten 后 copy 进对应位置
        3. padding 部分填 0
        4. reduce_scatter 到 flat_grad_shard
        5. 清空每个 compute.grad(释放完整大小的 grad 显存)
        """
        # 1. 完整 flat_grad
        flat_grad = torch.empty(
            self.padded_size,
            dtype=self.compute_dtype,
            device=self.device,
        )
        
        # 2. 收集各 param.grad
        for pg, layout in zip(self.param_groups, self.layouts):
            if pg.compute.grad is not None:
                flat_grad[layout.global_offset : layout.global_end].copy_(
                    pg.compute.grad.flatten()
                )
            else:
                # 该参数没参与 backward,填 0
                flat_grad[layout.global_offset : layout.global_end].zero_()
        
        # 3. padding 部分置 0
        if self.pad_size > 0:
            flat_grad[self.total_numel:].zero_()
        
        # 4. reduce_scatter:本 rank 得到对应 shard 的平均
        dist.reduce_scatter_tensor(
            self.flat_grad_shard,
            flat_grad,
            op=dist.ReduceOp.AVG,
            group=self.dp_group,
        )
        
        # 5. 释放各 compute.grad
        for pg in self.param_groups:
            pg.compute.grad = None
    
    # ═════════════════════════════════════════════════════════════
    # Step 期接口(被 MixedPrecisionManager 调用)
    # ═════════════════════════════════════════════════════════════
    
    def populate_master_grad(self) -> None:
        """把 flat_grad_shard(bf16)cast 到 master.grad(fp32),配置 view.grad。
        
        关键操作:每个 view.grad 是 master.grad 的对应 slice——
        共享 storage,optimizer 通过 view.grad 读 grad、通过 view.data 写 master.data。
        """
        # 1. master.grad = flat_grad_shard cast 到 fp32
        self.master.grad = self.flat_grad_shard.to(self.master_dtype)
        
        # 2. 每个 view.grad 是 master.grad 的对应 slice
        for pg, layout in zip(self.param_groups, self.layouts):
            if layout.has_local_view:
                view = self.param_views[pg.name]
                view.grad = self.master.grad.narrow(
                    0, layout.local_offset, layout.local_numel,
                )
    
    def sync_master_to_compute(self) -> None:
        """Step 完成后:master(fp32)→ flat_param_shard(bf16)。
        
        注意:这里只更新 flat_param_shard,不做 all_gather。
        下次 forward 时由 unshard hook 触发 all_gather 出完整 buffer。
        """
        with torch.no_grad():
            self.flat_param_shard.copy_(self.master.to(self.compute_dtype))
    
    def zero_grad(self) -> None:
        """清理 grad 状态。
        
        清:
        - master.grad:置 None(下次 populate 时重新分配)
        - param_views 的 grad:置 None(views 的 grad 是 master.grad 的 slice,
          master.grad 一旦置 None,views 持有的 slice 引用 stale,需主动清)
        - flat_grad_shard:置 0(下次 reduce_scatter 会覆盖,这里 zero 是防御性)
        """
        self.master.grad = None
        for view in self.param_views.values():
            view.grad = None
        self.flat_grad_shard.zero_()
    
    # ═════════════════════════════════════════════════════════════
    # 暴露给 MixedPrecisionManager 的查询接口
    # ═════════════════════════════════════════════════════════════
    
    def get_optimizable_views(self) -> list[tuple]:
        """返回本 rank 上 (ParamGroup, view_tensor) 对。
        
        用于 MixedPrecisionManager 构造 inner_optimizer.param_groups。
        每个 view 继承所属 ParamGroup.opt_config——weight_decay 等参数按 view 应用。
        
        Returns:
            list[(ParamGroup, Tensor)]:跳过 has_local_view=False 的 ParamGroup。
        """
        return [
            (pg, self.param_views[pg.name])
            for pg, layout in zip(self.param_groups, self.layouts)
            if layout.has_local_view
        ]
    
    # ═════════════════════════════════════════════════════════════
    # 调试 / 诊断接口
    # ═════════════════════════════════════════════════════════════
    
    def __repr__(self) -> str:
        return (
            f"ParamGroupCluster(name={self.name!r}, "
            f"params={len(self.param_groups)}, "
            f"total_numel={self.total_numel}, "
            f"shard_size={self.shard_size}, "
            f"pad_size={self.pad_size}, "
            f"dp_rank={self.dp_rank}/{self.dp_size})"
        )
    
    def memory_footprint(self) -> dict[str, int]:
        """诊断:cluster 占用的显存字节数(各组件)。"""
        sizes = {
            "flat_param_shard": self.flat_param_shard.numel() * self.flat_param_shard.element_size(),
            "master": self.master.numel() * self.master.element_size(),
            "flat_grad_shard": self.flat_grad_shard.numel() * self.flat_grad_shard.element_size(),
        }
        if self.master.grad is not None:
            sizes["master_grad"] = self.master.grad.numel() * self.master.grad.element_size()
        if self._full_buffer is not None:
            sizes["full_buffer"] = self._full_buffer.numel() * self._full_buffer.element_size()
        return sizes