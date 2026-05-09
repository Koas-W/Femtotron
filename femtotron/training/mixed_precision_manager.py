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
            master = self._make_master(p, config) if config.master_dtype else None
            opt_config = param_to_group_meta.get(id(p), {})
            self.groups.append(ParamGroup(name=name, 
                                          compute=p, 
                                          master=master, 
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
        将模型 BF16 参数的梯度拷贝到对应的 FP32 master weight 上。
        同时也会进行相应的grad_transform的梯度链操作。
        
        调用时机：backward 完成后，optimizer.step() 之前。
        
        具体操作：
        对每对 (bf16_param, fp32_master):
            if bf16_param.grad is not None:
                fp32_master.grad = bf16_param.grad.float()
            
        为什么需要这一步：
        backward 产生的梯度挂在 BF16 参数上（因为 forward 用的是 BF16 参数），
        但 optimizer 操作的是 FP32 master weights。
        需要把梯度"搬"过去，同时提升精度到 FP32。
        """
        # 1. 收集 finalized grads
        grads: list[Tensor] = []
        for ga, group in zip(self.grad_accs, self.groups):
            grad = ga.finalize()
            if grad is None:
                grad = torch.zeros_like(group.optimized_param)
            group.optimized_param.grad = grad
            grads.append(grad)

        self.grad_transform(grads=grads)
    
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
        for g in self.groups:
            g.sync_master_to_compute()
    
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

    
    def step(self) -> bool:
        """完成一个 optimizer step。返回是否成功（fp16 下可能因溢出跳过）。"""
        # 1. 收集 finalized grads
        # 2. 应用 grad transforms（unscale、clip 等）
        self.copy_grads_to_master()
        
        # 4. 真正 step
        self.inner.step()
        
        # 5. master → compute 同步
        self.sync_weights()
        
        # 6. 清理
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
            },
        }

    def load_state_dict(self, sd: dict) -> None:
        """从 state_dict 恢复训练状态。
        
        必须在 trainer 加载完 model.state_dict 之后调用——
        否则 master 会从未初始化的 compute 重建，覆盖 ckpt 内容。
        """
        # 1. 版本和配置一致性
        assert sd.get("version") == 1, f"unsupported checkpoint version: {sd.get('version')}"
        
        ckpt_cfg = sd["config"]
        assert ckpt_cfg["num_groups"] == len(self.groups), (
            f"param group count mismatch: ckpt has {ckpt_cfg['num_groups']}, "
            f"current has {len(self.groups)}"
        )
        expected_dtype = str(self.config.master_dtype) if self.config.master_dtype else None
        assert ckpt_cfg["master_dtype"] == expected_dtype, (
            f"master_dtype mismatch: ckpt={ckpt_cfg['master_dtype']}, current={expected_dtype}"
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
                    f"ckpt={tuple(ckpt_w.shape)}, current={tuple(g.master.shape)}"
                )
            with torch.no_grad():
                g.master.copy_(ckpt_w.to(g.master.device))
        
        # 3. 恢复 inner optimizer 状态
        self.inner.load_state_dict(sd["inner_optimizer"])
        
        # 4. 恢复 grad accumulator buffer
        grad_acc_states = sd["grad_accumulators"]
        for g, ga in zip(self.groups, self.grad_accs):
            if g.name in grad_acc_states:
                ga.load_state_dict(grad_acc_states[g.name])