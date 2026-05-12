import torch
from torch import nn, Tensor
from torch import distributed as dist
from torch.distributed import ProcessGroup
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from safetensors import safe_open
from typing import cast

from femtotron.parallel_context import ParallelContext
from femtotron.model.parallel_plan import ParallelPlan
from femtotron.training.train_config import TrainConfig
from femtotron.training.param_group import ParamGroup
from femtotron.sharding.param_group_cluster import ParamGroupCluster
from femtotron.training.grad_accumulator import GradAccumulator
from femtotron.training.grad_transform import GradTransform
from femtotron.sharding.sharding_strategy import ShardingStrategy
from femtotron.sharding.no_shard import NoShardStrategy
from femtotron.sharding.sharding_spec import ShardingSpec

class MixedPrecisionManager:
    """
    管理 BF16 训练参数和 FP32 master weights 之间的生命周期。
    
    职责：
    1. 为模型的每个 BF16 参数创建对应的 FP32 副本（master weight）
    2. 提供 FP32 参数组给 optimizer
    3. optimizer step 之后将 FP32 master weights 同步回 BF16 模型参数
    4. 提供分布式感知的 gradient clipping

    """
    
    def __init__(self, model: nn.Module, 
                 parallel_ctx: ParallelContext, 
                 parallel_plan: ParallelPlan, 
                 config: TrainConfig,
                 inner_optimizer_cls: type,
                 inner_optimizer_kwargs: dict,
                 compute_param_groups: list[dict], # 用于区分weight_decay的param groups
                 grad_transforms: list[GradTransform] | None = None,
                 sharding_strategy: ShardingStrategy | None = None,    # 默认 None
                 ):
        """
        参数：
            model: 参数已经是 BF16 的模型
                   （在 1.3 的 ModelLoader 中可以加载时就转 BF16，
                    或者在这里显式 model.bfloat16()）
            parallel_ctx: 并行化上下文，包括 dp_group
            dp_group: DP 通信组，用于 gradient clipping 时
                      跨 DP rank 计算 global grad norm。
                      如果为 None，表示不做分布式 grad norm（单 DP rank）。
            inner_optimizer_cls: 使用的优化器类型，如 Adam, AdamW
            inner_optimizer_kwargs: 使用的优化器的相应参数
            grad_transforms: 对梯度进行后处理的变换规则，如 clip，
        
        行为：
        1. 遍历 model 的所有 requires_grad=True 的参数
        2. 对每个 BF16 参数创建一个 FP32 的 clone 作为 master weight
        3. 建立 BF16 param → FP32 master weight 的双向映射
        4. 构建 param_groups（用于传给 optimizer），
           其中每个 group 的 "params" 指向 FP32 master weights
        
        关键设计：
        - FP32 master weights 是独立的 tensor，不是 nn.Parameter
          （不挂在 model 上，不参与 forward）
        - BF16 参数的 .grad 在 backward 后会被填充（BF16 梯度）
        - optimizer step 前需要把 BF16 梯度拷贝到 FP32 master weight 的 .grad 上
        """
        self.parallel_ctx = parallel_ctx
        # 默认走 NoShardStrategy，等价于 ZeRO 关闭
        if sharding_strategy is None:
            sharding_strategy = NoShardStrategy(parallel_ctx.dp_group)
        self.strategy = sharding_strategy

        # 先从 compute_param_groups 建立映射：param id → 该 group 的配置（除了 params 以外的所有 key）
        param_to_group_meta = {}
        for compute_group in compute_param_groups:
            meta = {k: v for k, v in compute_group.items() if k != "params"}
            # meta 大概是 {"weight_decay": 0.01} 或 {"weight_decay": 0.0}
            for p in compute_group["params"]:
                param_to_group_meta[id(p)] = meta
        
        # 为每个参数建 ParamGroup（决定有没有 master）
        self.groups: list[ParamGroup] = []
        params = model.named_parameters()
        for name, p in params:
            # master = self._make_master(p, config) if config.master_dtype else None
            if config.master_dtype:
                master, spec = self.strategy.make_master(p, config.master_dtype)
            else:
                master, spec = None, None
            opt_config = param_to_group_meta.get(id(p), {})
            self.groups.append(ParamGroup(name=name, 
                                          compute=p, 
                                          master=master, 
                                          master_spec=spec,
                                          opt_config=opt_config, 
                                          parallel_ctx=parallel_ctx,
                                          parallel_plan=parallel_plan))
        
        # 按 opt_config 聚合成 optimizer param groups
        from collections import defaultdict
        config_to_params = defaultdict(list)
        for g in self.groups:
            # 用 frozenset 做 key，相同配置的参数归为一组
            key = frozenset(g.opt_config.items())
            param = g.master if g.master is not None else g.compute
            config_to_params[key].append(param)
        
        # 每个 group 配一个 GradAccumulator
        self.grad_accs: list[GradAccumulator] = [
            GradAccumulator(g, config, parallel_ctx) for g in self.groups
        ]
        
        # # 内部 optimizer 看的是 master（如果有），否则看 compute
        # opt_param_groups = []
        # for key, params in config_to_params.items():
        #     group_dict = {"params": params}
        #     group_dict.update(dict(key))  # 还原 weight_decay 等配置
        #     opt_param_groups.append(group_dict)
        
        # 让 strategy 决定是否构造 cluster
        # NoShard / Z1 / Z2: 返回 []
        # Z3: 返回真正的 cluster 列表,并把对应 group 的 master 掏空
        self.clusters: list[ParamGroupCluster] = self.strategy.make_clusters(
            model=model,
            groups=self.groups,
            master_dtype=config.master_dtype,
        )
        
        # 收集 inner_optimizer 看到的"逻辑参数"
        # - standalone group: (g.master, g.opt_config)
        # - 被 cluster 接管的 group: (cluster 上的 view, g.opt_config)
        #   ← Step 1 此分支为空(no clusters)
        # 注意:opt_config 在 cluster 内部决定每个 view 的 wd 等,
        # 实现优雅地保留了 per-param 优化语义(weight_decay 等)
        opt_targets: list[tuple[ParamGroup, Tensor]] = []
        for g in self.groups:
                if g.has_own_master and g.master is not None:
                    opt_targets.append((g, g.master))
                elif g.cluster is not None:
                    view = g.cluster.param_views.get(g.name)
                    if view is not None:
                        opt_targets.append((g, view))
                    # else:本 rank 没持有该 param 的 view,跳过
        self.opt_targets = opt_targets
        
        # 按 opt_config 聚合(weight_decay 自然分组)
        config_to_tensors = defaultdict(list)
        for pg, tensor in opt_targets:
            key = frozenset(pg.opt_config.items())
            config_to_tensors[key].append(tensor)
        
        opt_param_groups = [
            {"params": tensors, **dict(key)}
            for key, tensors in config_to_tensors.items()
        ]
        self.inner = inner_optimizer_cls(opt_param_groups, **inner_optimizer_kwargs)
        
        self.config = config
        self.grad_transforms = grad_transforms or []
        # self.loss_scaler = loss_scaler
    
    @staticmethod
    def _make_master(p: nn.Parameter, config: TrainConfig) -> Tensor:
        m = p.detach().clone().to(config.master_dtype)
        m.requires_grad_(True)
        return m

    def accumulate_grads(self) -> None:
        """每次 micro-batch backward 后调用。"""
        for ga in self.grad_accs:
            ga.accumulate()
    
    def copy_grads_to_master(self):
        """
        收集所有参数的 grad，通过 strategy 同步并 cast 到 master，装到 master.grad，应用 transforms。
        
        调用时机：backward 完成后，optimizer.step() 之前。
        
        具体操作：
        对每对 (bf16_param, fp32_master):
            if bf16_param.grad is not None:
                fp32_master.grad = bf16_param.grad.float()
        NoShard / Z1 / Z2 路径(self.clusters 为空):行为和原来完全一致。
        ZeRO-3 路径:standalone phase 处理游离 param(若有),cluster phase 处理被接管的。
                
        （可选）Returns:
            装好的 grad 列表（用于 transform 后的 logging,如 grad_norm 等)。
        """
        # ============ Phase 1: standalone groups via strategy.reduce_grads ============
        # 只处理 has_own_master 的 group。Cluster 接管的 group(master 是 None)跳过——
        # 它们的 grad 在 hook 触发的 reduce_scatter_grads 里已经填到 flat_grad_shard。
        
        # 一次遍历构造三个对齐列表,每个都是 narrowed 类型
        standalone_groups: list[ParamGroup] = []
        standalone_masters: list[Tensor] = []
        standalone_specs: list[ShardingSpec | None] = []
        standalone_local_grads: list[Tensor | None] = []

        for i, pg in enumerate(self.groups):
            master = pg.master
            if master is None:
                continue
            standalone_groups.append(pg)
            standalone_masters.append(master)
            standalone_specs.append(pg.master_spec)
            standalone_local_grads.append(self.grad_accs[i].finalize())

        if standalone_groups:
            # zeros_like 兜底
            filled: list[Tensor] = [
                g if g is not None else torch.zeros_like(group.compute)
                for g, group in zip(standalone_local_grads, standalone_groups)
            ]
            
            synced = self.strategy.reduce_grads(
                compute_grads=filled,
                targets=standalone_masters,    # list[Tensor] ✓
                target_specs=standalone_specs,
            )
            for pg, grad in zip(standalone_groups, synced):
                pg.assign_grad(grad)
        
        # ============ Phase 2: clusters - 各自把 flat_grad_shard → master.grad ============
        # NoShard / Z1 / Z2 下 self.clusters 为空,这个循环是 no-op。
        for cluster in self.clusters:
            cluster.populate_master_grad()
        
        # ============ Phase 3: grad transforms over all master.grads ============
        # transforms 接收 (logical_param_groups, logical_grads) 形式
        # 注意:cluster 的 view.grad 是 master.grad 的 slice,在 transform 看来就是"该参数的 grad"
        self.grad_transform()
        # return synced_grads
    
    def grad_transform(self) -> list[Tensor]:
        """对所有 master grad(包括 cluster view 的 grad)应用变换链。
        
        transforms 内部按 ParamGroup 的 is_tp_sharded、opt_config 等元数据决定如何处理。
        """
        # 构造对齐的 (group, grad) 列表
        # 用 opt_targets:它已经包含正确的 (ParamGroup, target_tensor) 对应关系
        # target_tensor.grad 就是这个 ParamGroup 的逻辑 grad(不管来自 master 还是 view)
        aligned_groups: list[ParamGroup] = []
        aligned_grads: list[Tensor] = []
        
        for pg, target in self.opt_targets:
            if target.grad is not None:
                aligned_groups.append(pg)
                aligned_grads.append(target.grad)
        
        # 应用每个 transform
        results = []
        for transform in self.grad_transforms:
            results.append(transform(aligned_groups, aligned_grads))
        
        return results
    
    def sync_weights(self):
        """
        将 FP32 master weights 同步回模型的 BF16 参数。
        
        调用时机：optimizer.step() 之后。
        
        具体操作：
        对每对 (bf16_param, fp32_master):
            bf16_param.data.copy_(fp32_master.data)
            # copy_ 会自动做 FP32 → BF16 的 cast
        
        为什么需要这一步：
        optimizer 在 FP32 master weights 上做了更新（param -= lr * grad），
        但下一步 forward 用的是模型的 BF16 参数，需要同步。
        """
        self.strategy.gather_weights(self.groups)
        for cluster in self.clusters:
            cluster.sync_master_to_compute()
    
    def zero_grad(self):
        """
        清零所有梯度（BF16 参数和 FP32 master weights 的梯度全部清零）。

        调用时机：optimizer.step() + sync_weights() 之后。
        """
        for ga in self.grad_accs:
            ga.reset()
        # for g in self.groups:
        #     target = g.master if g.master is not None else g.compute
        #     target.grad = None

        # 用 ParamGroup.zero_grad() 替代原来散落的 target.grad = None
        # 行为等价(都是清 master.grad 或 compute.grad)
        for g in self.groups:
            g.zero_grad()
        
        # 新增:cluster 清自己的状态(master.grad、view.grad、flat_grad_shard)
        # NoShard / Z1 / Z2 下 clusters 为空,no-op。
        for cluster in self.clusters:
            cluster.zero_grad()

        # 让 strategy 清理自己的状态
        self.strategy.post_step()
    
    def step(self) -> bool:
        """完成一个 optimizer step。返回是否成功（fp16 下可能因溢出跳过）。"""
        # 1. 收集 + 同步 + transform，装到 master.grad
        self.copy_grads_to_master()
        
        # 2. inner optimizer 在 master 上做更新
        self.inner.step()
        
        # 3. master → compute（NoShardStrategy 是本地 cast；ZeRO1 是 all_gather）
        self.sync_weights()
        
        # 4. 清理 grads（compute 和 master 两边的）
        self.zero_grad()
        return True
    
    def state_dict(self) -> dict:
        """序列化训练状态。
        
        包含：
        - 所有参数的 FP32 master weights（按 name 索引）
        - inner optimizer 的 state（Adam 的 m/v 等）
        - grad accumulator 的 buffer 状态
        - 配置元信息（用于 load 时一致性检查）
        
        不包含：
        - compute params（在 model.state_dict() 里，trainer 单独保存）
        - grad_transforms（无状态）
        """
        return {
            "version": 1,   # checkpoint 格式版本，后续兼容用
            
            # FP32 master weights，按参数 name 索引（不依赖顺序）
            "master_weights": {
                g.name: g.master.detach().cpu()
                for g in self.groups
                if g.master is not None
            },
            
            # 新增：每个 master 的 sharding spec
            "master_specs": {
                g.name: g.master_spec for g in self.groups if g.master_spec is not None
            },
        
            # 内部 optimizer 状态（Adam m/v、step 计数等）
            "inner_optimizer": self.inner.state_dict(),
            
            # 每个 ParamGroup 对应的 grad accumulator buffer
            "grad_accumulators": {
                g.name: ga.state_dict()
                for g, ga in zip(self.groups, self.grad_accs)
            },
            
            # 配置一致性检查
            "config": {
                "master_dtype": str(self.config.master_dtype) if self.config.master_dtype else None,
                "param_dtype": str(self.config.param_dtype),
                "num_groups": len(self.groups),
                "param_names": [g.name for g in self.groups],
                            
                # 新增：分布式拓扑信息
                "dp_size": self.parallel_ctx.dp_size,
                "tp_size": self.parallel_ctx.tp_size,
                "sharding_strategy_kind": type(self.strategy).__name__,    # "NoShardStrategy" / "ZeRO1Strategy"
            },
        }

    def load_state_dict(self, sd: dict) -> None:
        """从 state_dict 恢复训练状态。
        
        必须在 trainer 加载完 model.state_dict 之后调用——
        否则 master 会从未初始化的 compute 重建，覆盖 ckpt 内容。
        """
        # 1. 版本和配置一致性
        version = sd.get("version", 1)
        assert version <= 2, f"unsupported checkpoint version: {sd.get('version')}"
        
        ckpt_cfg = sd["config"]
        assert ckpt_cfg["num_groups"] == len(self.groups), (
            f"param group count mismatch: ckpt has {ckpt_cfg['num_groups']}, "
            f"current has {len(self.groups)}"
        )
        expected_dtype = str(self.config.master_dtype) if self.config.master_dtype else None
        assert ckpt_cfg["master_dtype"] == expected_dtype, (
            f"master_dtype mismatch: ckpt={ckpt_cfg['master_dtype']}, current={expected_dtype}"
        )
        
        # 新增的拓扑一致性检查
        if version >= 2:
            ckpt_dp = ckpt_cfg.get("dp_size")
            ckpt_strategy = ckpt_cfg.get("sharding_strategy_kind")
            current_strategy = type(self.strategy).__name__
            
            # ZeRO 配置必须一致——shard 切法不同，无法直接 resume
            assert ckpt_strategy == current_strategy, (
                f"sharding strategy mismatch: ckpt={ckpt_strategy}, current={current_strategy}; "
                f"cannot resume across different ZeRO stages without reshard"
            )
            
            # ZeRO 启用时 dp_size 必须一致
            if current_strategy != "NoShardStrategy":
                assert ckpt_dp == self.parallel_ctx.dp_size, (
                    f"dp_size mismatch under {current_strategy}: ckpt={ckpt_dp}, "
                    f"current={self.parallel_ctx.dp_size}; "
                    f"resharding across dp_size requires DCP-style checkpoint"
                )

        current_names = [g.name for g in self.groups]
        if ckpt_cfg["param_names"] != current_names:
            # 给个有用的诊断信息
            only_in_ckpt = set(ckpt_cfg["param_names"]) - set(current_names)
            only_in_current = set(current_names) - set(ckpt_cfg["param_names"])
            raise RuntimeError(
                f"param names mismatch.\n"
                f"  Only in ckpt: {sorted(only_in_ckpt)[:5]}{'...' if len(only_in_ckpt) > 5 else ''}\n"
                f"  Only in current: {sorted(only_in_current)[:5]}{'...' if len(only_in_current) > 5 else ''}"
            )
        
        # 2. 恢复 FP32 master weights
        # 恢复 master weights（每个都是本 rank 的 shard，shape 和当前 master 应该一致）
        master_weights = sd["master_weights"]
        for g in self.groups:
            if g.master is None:
                continue
            if g.name not in master_weights:
                raise RuntimeError(f"missing master weight for {g.name} in checkpoint")
            ckpt_w = master_weights[g.name]
            if ckpt_w.shape != g.master.shape:
                raise RuntimeError(
                    f"shape mismatch for {g.name}: "
                    f"ckpt={tuple(ckpt_w.shape)}, current={tuple(g.master.shape)}; "
                    f"likely caused by changed sharding configuration"
                )
            with torch.no_grad():
                g.master.copy_(ckpt_w.to(g.master.device))
        
        # 验证 spec 一致性（v2+）
        if version >= 2 and "master_specs" in sd:
            for g in self.groups:
                ckpt_spec = sd["master_specs"].get(g.name)
                if ckpt_spec is not None and g.master_spec is not None:
                    # 关键字段要一致
                    assert ckpt_spec.full_shape == g.master_spec.full_shape, (
                        f"full_shape mismatch for {g.name}"
                    )
                    assert ckpt_spec.world_size == g.master_spec.world_size, (
                        f"world_size mismatch for {g.name}"
                    )
                    assert ckpt_spec.rank == g.master_spec.rank, (
                        f"rank mismatch for {g.name} — likely loading wrong shard file"
                    )
        
        # 3. 恢复 inner optimizer 状态
        self.inner.load_state_dict(sd["inner_optimizer"])
        
        # 4. 恢复 grad accumulator buffer
        grad_acc_states = sd["grad_accumulators"]
        for g, ga in zip(self.groups, self.grad_accs):
            if g.name in grad_acc_states:
                ga.load_state_dict(grad_acc_states[g.name])