import torch
from torch import nn, Tensor
from dataclasses import field

from femtotron.training.train_config import TrainConfig
from femtotron.model.parallel_plan import ParallelPlan
from femtotron.parallel_context import ParallelContext

class ParamGroup:
    """一个参数的多份物理存储。"""
    name: str                       # 调试用
    compute: nn.Parameter           # forward/backward 用，dtype = param_dtype
    master: Tensor | None           # optimizer 用，dtype = master_dtype；None 表示无 master
    opt_config: dict = field(default_factory=dict)  # {"weight_decay": 0.01} 等
    
    # 这些字段靠ParallelPlan和ParallelContext
    is_tp_sharded: bool                    # 这个参数在 TP 维度上是否分片
    tp_shard_dim: int | None               # 沿哪个 dim 切分（用于反推全局形状）
    is_replicated_across_dp: bool = True   # DP 副本是否需要保持一致（基本永远 True）
    
    def __init__(self, name: str, 
                 compute: nn.Parameter, 
                 master: Tensor | None, 
                 opt_config: dict,
                 parallel_ctx: ParallelContext,
                 parallel_plan: ParallelPlan) -> None:
        self.name = name
        self.compute = compute
        self.master = master
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