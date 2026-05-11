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
from femtotron.training.grad_accumulator import GradAccumulator
from femtotron.training.grad_transform import GradTransform
from femtotron.sharding.sharding_strategy import ShardingStrategy
from femtotron.sharding.no_shard import NoShardStrategy

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
        
        # 内部 optimizer 看的是 master（如果有），否则看 compute
        opt_param_groups = []
        for key, params in config_to_params.items():
            group_dict = {"params": params}
            group_dict.update(dict(key))  # 还原 weight_decay 等配置
            opt_param_groups.append(group_dict)
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
                
        （可选）Returns:
            装好的 grad 列表（用于 transform 后的 logging,如 grad_norm 等)。
        """
        # 1. 各 ParamGroup 调 finalize 拿本地 grad（bf16, 完整形状）
        local_grads: list[Tensor | None] = []
        for ga in self.grad_accs:
            local_grads.append(ga.finalize())
        
        # 2. 处理 None grad（缺梯度的参数装零张量，保证 inner.step 能正常跑）
        targets = [g.optimized_param for g in self.groups]
        specs = [g.master_spec for g in self.groups]
        
        filled_local_grads = [
            g if g is not None else torch.zeros_like(t)
            for g, t in zip(local_grads, targets)
        ]
        # 注意：这里的 zeros_like(t) 在 ZeRO 下 shape 是分片大小——但 strategy.reduce_grads
        # 期望接收 *完整* compute_param 形状的 grad，不是分片大小。所以要按 compute 形状填零：
        filled_local_grads = [
            g if g is not None else torch.zeros_like(group.compute)
            for g, group in zip(local_grads, self.groups)
        ]
        
        # 3. 通过 strategy 做通信 + cast 到 master_dtype
        #    - NoShardStrategy: cast（all-reduce 由独立的 grad_sync 组件做）
        #    - ZeRO1Strategy:   reduce_scatter + cast
        synced_grads = self.strategy.reduce_grads(
            compute_grads=filled_local_grads,
            targets=targets,
            target_specs=specs,
        )
        
        # 4. 装到 optimized_param.grad 上
        for group, grad in zip(self.groups, synced_grads):
            group.assign_grad(grad)
        
        # 5. 应用 grad transforms（clip 等）
        self.grad_transform(grads=synced_grads)
        # return synced_grads
    
    def grad_transform(self, grads: list[Tensor]) -> list[torch.Tensor]:
        """
        对梯度的变换链，其中包括的最主要的是
        分布式感知的 gradient norm clipping。
        """
        # 2. 应用 grad transforms（unscale、clip 等）
        r: list[Tensor] = []
        for transform in self.grad_transforms:
            r.append(transform(self.groups, grads))
        
        return r
    
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
    
    def zero_grad(self):
        """
        清零所有梯度（BF16 参数和 FP32 master weights 的梯度全部清零）。

        调用时机：optimizer.step() + sync_weights() 之后。
        """
        for ga in self.grad_accs:
            ga.reset()
        for g in self.groups:
            target = g.master if g.master is not None else g.compute
            target.grad = None

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