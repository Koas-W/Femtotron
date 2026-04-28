import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F

from femtotron.parallel_context import ParallelContext
from femtotron.model.parallel_plan import ParallelPlan, ParallelRule
from femtotron.model.parallel_module_builder import ParallelBuilder, _BUILDER_REGISTRY

def parallelize_model(model: nn.Module, 
                      parallel_plan: ParallelPlan, 
                      parallel_ctx: ParallelContext) -> nn.Module:
    """
    根据 parallel_plan，遍历模型，将匹配的 nn.Linear / nn.Embedding
    按照 TP 版本规则生成。不匹配的层保持不变生成。
    
    注意：此时模型可能在 meta device 上（参数没有真实数据），
    替换只改变层的类型和参数的 shape，不涉及实际权重。
    权重加载由 ModelLoader 单独处理。
    
    具体做的事：
    1. 遍历 model 的所有 named_modules
    2. 对每个 module 中的每个子 module：
       - 如果子 module 是 nn.Linear 且匹配 plan 中的某条规则
       - 创建对应的 TP 版本（ColumnParallel / RowParallel），
         参数 shape 已经是切分后的（out_features//tp 或 in_features//tp）
       - 替换原来的子 module
    3. 对 nn.Embedding 做类似处理
    4. 返回并行化后的 model
    
    替换 sub-module 的方式：
      parent_module 是持有 child 的那个 module，
      通过 setattr(parent, child_name, new_child) 替换。
    """
    # 收集要替换的 (parent, attr_name, old_module, rule)
    # 收集完再统一替换，避免在迭代过程中改 _modules 字典
    replacements: list[tuple[nn.Module, str, nn.Module, ParallelRule]] = []

    for name, module in model.named_modules():
        rule = parallel_plan.get_rule(name)
        if rule is None:
            continue
        parent, attr = _resolve_parent(model, name)
        replacements.append((parent, attr, module, rule))

    for parent, attr, old, rule in replacements:
        new = _build_parallel_module(old, parallel_ctx, rule)
        if attr.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
            parent[int(attr)] = new
        else:
            setattr(parent, attr, new)
    
    return model

def _resolve_parent(root: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    parent = root
    for p in parts[:-1]:
        if p.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    return parent, parts[-1]

def _build_parallel_module(
    old: nn.Module,
    parallel_ctx: ParallelContext,
    rule: ParallelRule,
) -> nn.Module:
    builder = _resolve_builder(rule.parallel_type, type(old))
    return builder(old, parallel_ctx, rule)

def _resolve_builder(kind: str, source_type: type[nn.Module]) -> ParallelBuilder:
    # 精确匹配
    builder = _BUILDER_REGISTRY.get((kind, source_type))
    if builder is not None:
        return builder
    # 退而求其次：按 MRO 找父类匹配
    for base in source_type.__mro__[1:]:
        if not issubclass(base, nn.Module):
            continue
        builder = _BUILDER_REGISTRY.get((kind, base))
        if builder is not None:
            return builder
    raise ValueError(
        f"no builder registered for kind={kind!r}, source_type={source_type.__name__}"
    )