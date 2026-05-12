import torch
from torch import nn, Tensor, dtype
import torch.distributed as dist
from torch.distributed import ProcessGroup
from dataclasses import field

from femtotron.training.train_config import TrainConfig
from femtotron.model.parallel_plan import ParallelPlan
from femtotron.parallel_context import ParallelContext
from femtotron.sharding.sharding_spec import ShardingSpec
from femtotron.training.param_group_cluster import ParamGroupCluster

class ParamGroup:
    """一个参数的多份物理存储。"""
    name: str                       # 调试用
    compute: nn.Parameter           # forward/backward 用，dtype = param_dtype
    master: Tensor | None           # optimizer 用，dtype = master_dtype；None 表示无 master
    master_spec: ShardingSpec | None  # master 的分片信息
    opt_config: dict = field(default_factory=dict)  # {"weight_decay": 0.01} 等
    
    # zero-3 使用的字段，被哪个cluster接管
    cluster: "ParamGroupCluster | None" = field(default=None, repr=False)

    # 这些字段靠ParallelPlan和ParallelContext
    is_tp_sharded: bool                    # 这个参数在 TP 维度上是否分片
    tp_shard_dim: int | None               # 沿哪个 dim 切分（用于反推全局形状）
    is_replicated_across_dp: bool = True   # DP 副本是否需要保持一致（基本永远 True）
    
    def __init__(self, name: str, 
                 compute: nn.Parameter, 
                 master: Tensor | None, 
                 opt_config: dict,
                 parallel_ctx: ParallelContext,
                 parallel_plan: ParallelPlan,
                 master_spec: ShardingSpec | None = None
                 ) -> None:
        self.name = name
        self.compute = compute
        self.master = master
        self.master_spec = master_spec
        self.opt_config = opt_config
        
        # 从 plan 查这个参数的并行规则
        # name 格式如 "model.layers.0.self_attn.q_proj.weight"
        # 需要去掉末尾的 ".weight" / ".bias" 得到 module 名来匹配 plan
        module_name = name.rsplit(".", 1)[0] if "." in name else name
        rule = parallel_plan.get_rule(module_name)
        
        tp_size = parallel_ctx.tp_size
        
        if rule is None or rule.parallel_type == "replicate" or tp_size <= 1:
            self.is_tp_sharded = False
            self.tp_shard_dim = None
        elif rule.parallel_type == "column":
            # weight shape [out, in]，沿 out (dim=0) 切分
            self.is_tp_sharded = True
            self.tp_shard_dim = 0
        elif rule.parallel_type == "row":
            # weight shape [out, in]，沿 in (dim=1) 切分
            self.is_tp_sharded = True
            self.tp_shard_dim = 1
        elif rule.parallel_type == "vocab_embed":
            # weight shape [vocab, hidden]，沿 vocab (dim=0) 切分
            self.is_tp_sharded = True
            self.tp_shard_dim = 0
        else:
            self.is_tp_sharded = False
            self.tp_shard_dim = None
        
        # bias 不切分（即使所属 module 是 TP 切分的）
        # ColumnParallel 的 bias 跟着 out 切分 → 实际上是切分的
        # RowParallel 的 bias 不切分 → 每个 rank 完整副本
        # 但 LLaMA 没有 bias，这里做个防御性处理
        if name.endswith(".bias") and rule is not None and rule.parallel_type == "row":
            self.is_tp_sharded = False
            self.tp_shard_dim = None
        
        self.is_replicated_across_dp = True
        
    def sync_master_to_compute(self) -> None:
        """master 更新后，把数值同步到 compute（cast + copy）。"""
        if self.master is None:
            return
        with torch.no_grad():
            self.compute.copy_(self.master)
    
    def init_master_from_compute(self) -> None:
        """从加载好的 compute 初始化 master。"""
        if self.master is None:
            return
        with torch.no_grad():
            self.master.copy_(self.compute)
    
    @property
    def optimized_param(self) -> Tensor:
        """inner optimizer 实际看到、更新的 tensor。"""
        return self.master if self.master is not None else self.compute
    
    @property
    def is_master_sharded(self) -> bool:
        return self.master_spec is not None and self.master_spec.world_size > 1
    
    # 状态查询 properties
    @property
    def has_own_master(self) -> bool:
        return self.master is not None
    
    @property
    def is_clustered(self) -> bool:
        return self.cluster is not None
    
    def assign_grad(self, grad: Tensor | None) -> None:
        """grad 必须和 optimized_param 形状一致——
        ZeRO-1 下这意味着 grad 已经是分片的（reduce_scatter 后）。"""
        if grad is None:
            self.optimized_param.grad = None
            return
        target = self.optimized_param
        assert grad.shape == target.shape, f"shape mismatch: {grad.shape} vs {target.shape}"
        target.grad = grad
    
    def gather_master_to_compute(self, dp_group: ProcessGroup) -> None:
        """把分片的 master 收集起来，写回完整的 compute。
        
        ZeRO-1 step 之后调用——让所有 rank 的 compute 重新一致。
        """
        master = self.master
        if master is None:
            return    # 没有 master，无需同步
        if not self.is_master_sharded:
            # no-shard 路径：直接 cast
            with torch.no_grad():
                self.compute.copy_(master)
            return
        
        # 分片路径：all-gather
        spec = self.master_spec
        assert spec is not None, "is_master_sharded=True implies master_spec is not None"

        gathered = torch.empty(
            spec.flat_size,
            dtype=master.dtype,
            device=master.device,
        )
        dist.all_gather_into_tensor(gathered, master, group=dp_group)
        
        # 截掉 padding，reshape 回完整形状
        unpadded = gathered[: spec.flat_size - spec.pad_size]
        full = unpadded.view(spec.full_shape).to(self.compute.dtype)
        with torch.no_grad():
            self.compute.copy_(full)

    def zero_grad(self) -> None:
        """清空本 group 的 master grad。
        
        注意:compute.grad 由 GradAccumulator.reset() 清理,不在这里处理。
        PyTorch 设计上 compute.grad 是 autograd 引擎管理的,GradAccumulator 持有引用控制其生命周期。
        """
        target = self.master if self.master is not None else self.compute
        target.grad = None