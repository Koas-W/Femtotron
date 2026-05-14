"""Policy factories for activation checkpointing.

提供少量原子 policy 工厂 + 组合算子。每个工厂返回 CheckpointPolicy
(即 Callable[[nn.Module], bool])。

典型用法:
    >>> policy = by_class(LlamaDecoderLayer)                       # 最常见
    >>> policy = any_of(by_class(LlamaDecoderLayer), by_class(...))
    >>> policy = all_of(by_class(...), not_of(by_class_name("LastLayer")))
"""

from __future__ import annotations

from typing import Final

from torch import nn

from femtotron.training.activation_ckpt import CheckpointPolicy


# ─────────────────────────────────────────
# 原子 policy
# ─────────────────────────────────────────

def by_class(*classes: type[nn.Module]) -> CheckpointPolicy:
    """匹配 `classes` 中任一类的实例。最常见的 policy。
    
    Example:
        >>> policy = by_class(LlamaDecoderLayer)
        >>> policy = by_class(LlamaDecoderLayer, MoEBlock)   # 多类
    """
    if not classes:
        raise ValueError("by_class() requires at least one class")
    
    classes_tuple = tuple(classes)   # 冻结
    
    def policy(module: nn.Module) -> bool:
        return isinstance(module, classes_tuple)
    
    policy.__name__ = f"by_class({', '.join(c.__name__ for c in classes_tuple)})"
    return policy


def by_class_name(*names: str) -> CheckpointPolicy:
    """匹配 `type(module).__name__` 在 `names` 中的实例。
    
    `by_class` 的字符串版本。两种场景:
    1. 类不能直接 import(避免循环依赖)
    2. 从 config 文件配置(YAML/JSON 里没法序列化 class)
    
    Example:
        >>> policy = by_class_name("LlamaDecoderLayer")
    """
    if not names:
        raise ValueError("by_class_name() requires at least one name")
    
    names_set = frozenset(names)
    
    def policy(module: nn.Module) -> bool:
        return type(module).__name__ in names_set
    
    policy.__name__ = f"by_class_name({', '.join(repr(n) for n in names)})"
    return policy


# ─────────────────────────────────────────
# 组合算子
# ─────────────────────────────────────────

def any_of(*policies: CheckpointPolicy) -> CheckpointPolicy:
    """若 *任一* policy match 则 match(逻辑 OR)。
    
    Example:
        >>> policy = any_of(by_class(LlamaDecoderLayer), by_class(MoEBlock))
    """
    if not policies:
        raise ValueError("any_of() requires at least one policy")
    if len(policies) == 1:
        return policies[0]
    
    policies_tuple = tuple(policies)
    
    def policy(module: nn.Module) -> bool:
        return any(p(module) for p in policies_tuple)
    
    policy.__name__ = f"any_of({', '.join(_name(p) for p in policies_tuple)})"
    return policy


def all_of(*policies: CheckpointPolicy) -> CheckpointPolicy:
    """若 *全部* policy match 则 match(逻辑 AND)。
    
    Example:
        >>> # 只 checkpoint decoder layer 中的非 final 层
        >>> policy = all_of(
        ...     by_class(LlamaDecoderLayer),
        ...     not_of(by_class_name("FinalDecoderLayer")),
        ... )
    """
    if not policies:
        raise ValueError("all_of() requires at least one policy")
    if len(policies) == 1:
        return policies[0]
    
    policies_tuple = tuple(policies)
    
    def policy(module: nn.Module) -> bool:
        return all(p(module) for p in policies_tuple)
    
    policy.__name__ = f"all_of({', '.join(_name(p) for p in policies_tuple)})"
    return policy


def not_of(policy: CheckpointPolicy) -> CheckpointPolicy:
    """逻辑 NOT。
    
    Example:
        >>> policy = not_of(by_class_name("EmbedLayer"))
    """
    def negated(module: nn.Module) -> bool:
        return not policy(module)
    
    negated.__name__ = f"not_of({_name(policy)})"
    return negated


# ─────────────────────────────────────────
# 平凡 policy
# ─────────────────────────────────────────

def never(module: nn.Module) -> bool:
    """什么都不 match。等价于禁用 AC。"""
    return False


def always(module: nn.Module) -> bool:
    """匹配所有 module。几乎肯定会触发 nested wrap 报错,不要用。"""
    return True


NEVER: Final[CheckpointPolicy] = never
ALWAYS: Final[CheckpointPolicy] = always


# ─────────────────────────────────────────
# 工具
# ─────────────────────────────────────────

def _name(policy: CheckpointPolicy) -> str:
    """安全拿一个 policy 的名字(用于 repr)。"""
    return getattr(policy, "__name__", "<unnamed>")