"""ZeRO-3 strategy:weight + grad + optimizer state 全部分片。

实现方式:
- 通过 wrap_policy 识别 FSDP unit(通常每个 transformer block 一个)
- 为每个 unit 构造 ParamGroupCluster,接管其内部 ParamGroup 的 master
- 在 unit module 上注册 forward/backward 4 个 hook,自动管理 unshard/reshard

同时实现 GradientSynchronizer 接口:no_sync 由本类管理,
因为 hook 中的 reduce_scatter 需要知道是否在 no_sync 状态。
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.distributed import ProcessGroup

from femtotron.sharding.param_group_cluster import ParamGroupCluster
from femtotron.training.param_group import ParamGroup


class ZeRO3Strategy:
    """ZeRO-3:全分片 + 即时 unshard/reshard。
    
    与 Pipeline Parallel 组合时:
    - 数学正确(已验证 bit-exact 与 ZeRO-0)
    - 但 peak memory 比理论值偏高 ~5-10%:autograd 在 1F1B 的多个 in-flight 
      microbatch 间持有 unsharded param view,使 _full_buffer storage 
      无法立即释放
    - 生产场景推荐配合 activation checkpointing(AC)使用,AC 通过 
      backward 时重算 forward 消除了这种 view 持有,实测 PP + ZeRO-3 + AC 
      内存反低于 PP + ZeRO-2
    - 注意:DeepSpeed 直接禁用了 PP + ZeRO-2/3 组合;本框架支持但不优化到极致
    """
    
    def __init__(
        self,
        dp_group: ProcessGroup,
        wrap_policy: Callable[[nn.Module], bool],
    ) -> None:
        """
        Args:
            dp_group:用于 cluster 内部 all_gather / reduce_scatter 的 process group
            wrap_policy:判定函数,True 表示该 module 作为一个 FSDP unit
        """
        self.dp_group = dp_group
        self.dp_rank = dist.get_rank(dp_group)
        self.dp_size = dist.get_world_size(dp_group)
        self.wrap_policy = wrap_policy
        
        # 所有 cluster 的列表(make_clusters 时填充)
        self.clusters: list[ParamGroupCluster] = []
        # gc 相关状态
        self._hook_handles: list = []
        
        # GradientSynchronizer 状态
        self._sync_enabled = True
    
    # ═══════════════════════════════════════════════
    # ShardingStrategy 接口
    # ═══════════════════════════════════════════════
    
    def make_master(
        self,
        compute: nn.Parameter,
        master_dtype: torch.dtype,
    ) -> tuple[Tensor, None]:
        """为"游离"参数(未在任何 cluster 内)创建非分片 master。
        
        典型情况:LM head 或 embedding 没被 wrap_policy 选中时,它们走这条路径,
        相当于在 ZeRO-3 模式下退化为"该参数不分片"。
        """
        master = compute.detach().clone().to(master_dtype)
        master.requires_grad_(True)
        return master, None    # spec=None 表示不分片
    
    def make_clusters(
        self,
        model: nn.Module,
        groups: list[ParamGroup],
        master_dtype: torch.dtype | None,
        ) -> list["ParamGroupCluster"]:
        """识别 unit、构造 cluster、安装 hook。
        
        master_dtype 为 None 时抛错(ZeRO-3 必须有 master)。
        """
        if master_dtype is None:
            raise ValueError(
                "ZeRO-3 requires master_dtype to be non-None. "
                "Without an fp32 master, there is no optimizer state to shard."
            )

        clusters = self._build_clusters(model, groups, master_dtype)
        
        # 新增:记录 cluster 外的 ParamGroup
        clustered_pg_ids = set()
        for cluster in clusters:
            for pg in cluster.param_groups:
                clustered_pg_ids.add(id(pg))
        
        return clusters
    
    def _build_clusters(
        self,
        model: nn.Module,
        groups: list[ParamGroup],
        master_dtype: torch.dtype,
    ) -> list[ParamGroupCluster]:
        # 1. 通过 wrap_policy 识别 unit
        units = [m for m in model.modules() if self.wrap_policy(m)]
        
        if not units:
            raise RuntimeError(
                "ZeRO-3 wrap_policy didn't match any module. "
                "Check that the policy targets exist in the model."
            )
        
        # 2. 检查 unit 不嵌套(避免 hook 触发顺序混乱)
        self._verify_no_nesting(units)
        
        # 3. Parameter id 到 ParamGroup 的映射
        param_id_to_pg = {id(pg.compute): pg for pg in groups}
        
        # 4. 为每个 unit 构造 cluster
        for unit in units:
            unit_pgs = self._collect_unit_param_groups(unit, param_id_to_pg)
            if not unit_pgs:
                continue    # 跳过空 unit
            
            cluster = ParamGroupCluster(
                name=self._unit_qualified_name(unit, model),
                module=unit,
                param_groups=unit_pgs,
                dp_group=self.dp_group,
                master_dtype=master_dtype,
            )
            self.clusters.append(cluster)
            self._install_hooks(cluster)
        
        return self.clusters
    
    def _collect_unit_param_groups(
        self,
        unit: nn.Module,
        param_id_to_pg: dict,
    ) -> list:
        """收集 unit 内所有 ParamGroup,按 Parameter id 去重。"""
        unit_pgs = []
        seen_ids = set()
        
        for p in unit.parameters(recurse=True):
            if id(p) in seen_ids:
                continue    # tied weight 去重
            seen_ids.add(id(p))
            
            if id(p) in param_id_to_pg:
                pg = param_id_to_pg[id(p)]
                if pg.cluster is not None:
                    raise RuntimeError(
                        f"Parameter {pg.name} already claimed by cluster {pg.cluster.name}. "
                        f"Check wrap_policy for overlapping units."
                    )
                unit_pgs.append(pg)
        
        return unit_pgs
    
    def _verify_no_nesting(self, units: list[nn.Module]) -> None:
        """禁止嵌套 wrap——一个 unit 是另一个 unit 的祖先。"""
        unit_ids = {id(u) for u in units}
        for unit in units:
            for submod in unit.modules():
                if submod is unit:
                    continue
                if id(submod) in unit_ids:
                    raise RuntimeError(
                        f"Nested wrap detected: {type(submod).__name__} is inside "
                        f"{type(unit).__name__}. wrap_policy must produce non-nested units."
                    )
    
    def _unit_qualified_name(self, unit: nn.Module, model: nn.Module) -> str:
        """获取 unit 在 model 中的限定名(如 'model.layers.0')。"""
        for name, m in model.named_modules():
            if m is unit:
                return name or "<root>"
        return f"<unnamed:{type(unit).__name__}>"
    
    def _install_hooks(self, cluster: ParamGroupCluster) -> None:
        """给 cluster 的 module 装 4 个 hook。
        
        - forward_pre / forward_post:管理 forward 期间 unshard/reshard
        - backward_pre / backward_post:管理 backward 期间 unshard/reshard
        - backward_post 还负责 reduce_scatter_grads(gated by _sync_enabled)
        
        闭包捕获 cluster:lambda 中 cluster 是 free variable,
        循环里每次给不同的 cluster 注册不同的 hook。
        """
        handles = [
            cluster.module.register_forward_pre_hook(
                lambda module, args, c=cluster: c.unshard()
            ),
            cluster.module.register_forward_hook(
                lambda module, args, output, c=cluster: c.reshard()
            ),
            cluster.module.register_full_backward_pre_hook(
                lambda module, grad_output, c=cluster: c.unshard()
            ),
            cluster.module.register_full_backward_hook(
                lambda module, grad_input, grad_output, c=cluster:
                    self._after_unit_backward(c)
            ),
        ]
        self._hook_handles.extend(handles)
    
    def _after_unit_backward(self, cluster: ParamGroupCluster) -> None:
        """backward 完成后:可能 reduce_scatter,然后必定 reshard。
        
        在 no_sync 周期中跳过 reduce_scatter,让 compute.grad 在 unsharded 状态下
        跨 micro-step 累加。reshard 仍然做,以释放 full_buffer。
        """
        if self._sync_enabled:
            cluster.reduce_scatter_grads()
        cluster.reshard()
    
    def prepare_for_backward(self, groups: list[ParamGroup]) -> None:
        pass

    def reduce_grads(
        self,
        compute_grads: list,
        targets: list,
        target_specs: list,
    ) -> list[Tensor]:
        """对游离参数(未在 cluster 内)的 grad 处理:
        - 跨 DP all-reduce (AVG):游离参数的 master 是非分片的,
        用 all-reduce 而不是 reduce-scatter
        - cast 到 master dtype
        
        Cluster 内参数不走这条路径(被 cluster.populate_master_grad 接管)。
        """
        result = []
        for grad, target in zip(compute_grads, targets):
            if grad is None:
                result.append(torch.zeros_like(target))
                continue
            
            # 跨 DP 做平均(对称 ZeRO-1 在 reduce_grads 中做 reduce_scatter 的位置)
            if self.dp_size > 1:
                dist.all_reduce(
                    grad, op=dist.ReduceOp.AVG, group=self.dp_group
                )
            
            # cast 到 master dtype
            if grad.dtype != target.dtype:
                grad = grad.to(target.dtype)
            
            result.append(grad)
        return result
    
    def gather_weights(self, groups: list) -> None:
        """对游离参数的 master → compute 同步。Cluster 内的由 cluster 自己处理。"""
        for g in groups:
            if g.master is not None:
                with torch.no_grad():
                    g.compute.copy_(g.master)
    
    def post_step(self) -> None:
        """step 完成后清理。Cluster 的 zero_grad 由 mp_manager 调,这里 no-op。"""
        pass
    
    def grads_are_dp_sharded(self) -> bool:
        return True
    
    # ═══════════════════════════════════════════════
    # GradientSynchronizer 接口
    # ═══════════════════════════════════════════════
    
    @contextmanager
    def no_sync(self) -> Iterator[None]:
        """gradient accumulation 期间禁用 hook 的 reduce_scatter。"""
        prev = self._sync_enabled
        self._sync_enabled = False
        try:
            yield
        finally:
            self._sync_enabled = prev
    
    def sync_gradients(self) -> None:
        """Cluster 内参数:由 backward hook 中的 reduce_scatter 处理。
        Standalone 参数:由 strategy.reduce_grads 在 step 内做 all-reduce。
        
        这里 no-op,对称 ZeRO-2 的设计。
        """
        pass
    
    def state_dict(self) -> dict:
        return {}
    
    def load_state_dict(self, sd: dict) -> None:
        pass
    
    # ═══════════════════════════════════════════════
    # 调试 / 诊断
    # ═══════════════════════════════════════════════
    
    def force_reshard_all(self) -> None:
        """异常恢复:强制所有 cluster reshard,释放 full_buffer。
        
        trainer 的 exception handler 可以调这个,防止异常后 unshard 状态泄漏导致 OOM。
        """
        for cluster in self.clusters:
            try:
                cluster.reshard()
            except Exception:
                pass    # 防御性,不让 cleanup 异常掩盖原始异常
    
    def summary(self) -> dict:
        """诊断信息。"""
        return {
            "num_clusters": len(self.clusters),
            "total_sharded_params": sum(c.total_numel for c in self.clusters),
            "memory_per_cluster": [c.memory_footprint() for c in self.clusters],
        }
    
    def cleanup(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()
        # cluster 内的 tensor 都释放(防止 cluster 自己有循环引用)
        for c in self.clusters:
            c.master = None # type: ignore[attr-defined]
            c.flat_param_shard = None # type: ignore[attr-defined]
            c.flat_grad_shard = None # type: ignore[attr-defined]
            if hasattr(c, 'param_views'):
                c.param_views.clear()
        self.clusters.clear()