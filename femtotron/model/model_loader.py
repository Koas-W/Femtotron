import os, sys
import torch
from torch import nn, Tensor
from torch import distributed as dist
import torch.nn.functional as F
import json
from pathlib import Path
from collections import defaultdict
from safetensors import safe_open
from typing import cast

from femtotron.parallel_context import ParallelContext
from femtotron.model.parallel_plan import ParallelPlan, ParallelRule
from femtotron.model.shard_loader import ShardLoader, ReplicateLoader, DimShardLoader, _LOADER_REGISTRY

class ModelLoader:
    """
    从 HuggingFace checkpoint 加载权重并按 TP 拓扑分发到各 rank。
    
    行为：
    - 读磁盘，各个rank分布式加载自己的分片参数
    - 支持 safetensors 格式（HuggingFace 的标准格式）
    """
    
    def __init__(self, parallel_ctx: ParallelContext):
        self.ctx = parallel_ctx
    
    def load_and_distribute(self, 
                            model: nn.Module,
                            model_name_or_path: str,
                            parallel_plan: ParallelPlan,
                            device: torch.device,):
        """
        将 checkpoint 中的权重加载并切分到 model 的各个参数中。
        
        model: 分布式的模型
        model_name_or_path: 权重加载的路径
        paprallel_plan: 传入的相应模型的切分方式规则
        model 此时应该已经经过 parallelize_model() 处理，
        其中的 ColumnParallelLinear 等已经知道自己并行化后的 shape。
        参数当前在 meta device 上。
        
        行为：
        1. 加载 checkpoint 的索引（safetensors index），
           得到参数名 → 文件名的映射
        2. 构建模型参数名 → checkpoint 参数名的映射
           （HuggingFace 的参数名和模型的参数名可能有差异）
        3. 遍历模型的每个参数：
           a. 所有 rank: 从 safetensors 文件中加载该参数到 GPU
           b. 所有 rank: 根据该参数所属的层类型，决定切分方式和维度
           c. 所有 rank: 加载切片
              - 不需要切分的参数: 加载完整参数
              - 需要切分的参数: 加载自身需要的参数
        """
        # 1. 读索引
        index_file = Path(model_name_or_path) / "model.safetensors.index.json"
        if index_file.is_file():
            with open(index_file) as f:
                weight_map: dict[str, str] = json.load(f)["weight_map"]
        else:
            # 单文件情况
            weight_map = {name: "model.safetensors" for name in model.state_dict()}
        
        # 2. 按文件分组，避免一个文件被多次 open
        file_to_params: dict[str, list[str]] = defaultdict(list)
        for param_name, file_name in weight_map.items():
            file_to_params[file_name].append(param_name)

        # 3. 获取 model 的 param 引用
        state: dict[str, torch.Tensor] = {}
        state.update(model.named_parameters())
        state.update(model.named_buffers())
    
        # 4. 逐文件加载本 rank 需要的切片
        tp_world_size = self.ctx.tp_size
        tp_rank = self.ctx.tp_rank
        for file_name, param_names in file_to_params.items():
            file_path = os.path.join(model_name_or_path, file_name)
            # device 直接传 cuda:N，safetensors 会零拷贝到 GPU
            with safe_open(file_path, framework="pt", device=str(device)) as f:
                for name in param_names:
                    if name not in state:
                        continue  # 模型不要这个参数（比如 lm_head tie weights）

                    rule = parallel_plan.get_rule(name)
                    tensor = load_one_param(f, name, rule, tp_rank, tp_world_size)

                    # 写入 model 参数（确保 dtype/device 一致）
                    target = state[name]
                    with torch.no_grad():
                        target.copy_(tensor)

        # 5. 等所有 rank 加载完
        if dist.is_initialized():
            dist.barrier()
    
    # def _get_param_parallel_info(self, module: nn.Module) -> tuple[bool, int | None]:
    #     """
    #     判断一个 module 的权重是否需要切分，以及沿哪个维度。
        
    #     返回:
    #       (needs_split, split_dim)
    #       - (False, None): 不切分，broadcast
    #       - (True, 0): 沿 dim 0 切分
    #       - (True, 1): 沿 dim 1 切分
        
    #     判断依据：
    #       module 是 ColumnParallelLinear → (True, 0)
    #       module 是 RowParallelLinear → (True, 1)  
    #       module 是 VocabParallelEmbedding → (True, 0)
    #       其他 → (False, None)
        
    #     或者更简洁：直接读 module.parallel_dim 属性
    #     （你在 1.2 中设计的，ColumnParallel.parallel_dim=0 等）
    #     如果 module 没有 parallel_dim 属性 → 不切分
    #     """
        
    # def _distribute_param(self,
    #                       param_name: str,
    #                       full_weight: torch.Tensor | None,  
    #                       # rank 0 上是完整 CPU tensor，其他 rank 上是 None
    #                       target_module: nn.Module,
    #                       param_attr: str,  # "weight" 或 "bias"
    #                       ):
    #     """
    #     分发一个参数到所有 rank。
        
    #     根据 target_module 的类型决定切分方式：
    #     - ColumnParallelLinear: 沿 dim=0 切分（weight shape [out, in]，切 out）
    #     - RowParallelLinear: 沿 dim=1 切分（weight shape [out, in]，切 in）
    #     - VocabParallelEmbedding: 沿 dim=0 切分（weight shape [vocab, hidden]）
    #     - 其他（RMSNorm, 普通 Linear 等）: 不切分，broadcast 完整副本
        
    #     行为：
    #     - 不切分: dist.broadcast(tensor, src=0, group=tp_group)
    #     - 切分: rank 0 在 CPU 上 chunk，
    #             然后用 dist.scatter 或逐 rank 的 dist.send/recv
    #             （scatter 更优雅但要求所有 chunk 大小一致）
        
    #     分发完成后，用 module._parameters[param_attr] = nn.Parameter(gpu_tensor)
    #     替换 meta device 上的占位参数。
    #     """
    
    
#################################
# 和 shard_loader 工厂类联动
#################################
def _resolve_loader(rule: ParallelRule | None, name: str) -> ShardLoader:
    if rule is None:
        return ReplicateLoader()
    suffix = "." + name.rsplit(".", 1)[-1]   # ".weight" / ".bias" 等
    factory = _LOADER_REGISTRY.get(rule.parallel_type)
    if factory is None:
        raise ValueError(f"unknown parallel kind: {rule.parallel_type}")
    return factory(rule, suffix)


def load_one_param(f, name: str, rule: ParallelRule | None, rank: int, world_size: int) -> Tensor:
    loader = _resolve_loader(rule, name)
    if isinstance(loader, ReplicateLoader):
        # 小优化：replicate 直接 get_tensor 比 get_slice + 全切片少一层包装
        tensor = cast(Tensor, f.get_tensor(name))
        return tensor
    handle = f.get_slice(name)
    return loader.load(handle, rank, world_size)