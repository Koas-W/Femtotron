
class ParamGroupCluster:
    """占位 - 下一步实现。
    
    必须提供的接口(被 MixedPrecisionManager 调用):
    
    构造时:
    - 在 __init__ 内部修改被接管 ParamGroup.master 为 None
    - 创建 flat_param_shard / flat_grad_shard / master(后两者可延后)
    - 创建 per-param view 到 master 上
    
    属性:
    - master: Tensor               # cluster 自己的 flat master(fp32)
    - opt_config: dict              # cluster 内 ParamGroup 必须有相同 opt_config
    
    方法:
    - get_optimizable_views() -> list[tuple[ParamGroup, Tensor]]
        返回 (pg, view) 对,view 是本 rank 持有的 master slice
    
    - populate_master_grad() -> None
        flat_grad_shard → master.grad(fp32 cast),同时配置 view.grad
    
    - sync_master_to_compute() -> None
        master → flat_param_shard(bf16 cast)
    
    - zero_grad() -> None
        清 master.grad、view.grad、flat_grad_shard
    """