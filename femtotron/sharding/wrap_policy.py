"""ZeRO-3 wrap policy:决定哪些 module 作为 FSDP unit。"""

import torch.nn as nn


def llama_wrap_policy(module: nn.Module) -> bool:
    """Llama 模型的 wrap policy:每个 transformer block 一个 unit。
    """
    # # 优先尝试 femtotron 自己的实现
    # try:
    #     from femtotron.parallel.tensor_parallel.linear import ColumnParallelLinear, RowParallelLinear
    #     if isinstance(module, (ColumnParallelLinear, RowParallelLinear)):
    #         return True
    # except ImportError:
    #     pass
    
    # fallback:HuggingFace transformers 的实现
    try:
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer as HFLayer
        if isinstance(module, HFLayer):
            return True
    except ImportError:
        pass
    
    return False


def make_class_wrap_policy(target_classes: tuple[type, ...]):
    """通用工厂:wrap 给定类型的所有实例。
    
    用法:
        policy = make_class_wrap_policy((MyBlock, OtherBlock))
    """
    def policy(module: nn.Module) -> bool:
        return isinstance(module, target_classes)
    return policy