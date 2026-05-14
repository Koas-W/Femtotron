"""Activation checkpointing wrapper + apply utility.

与 sharding strategy 完全解耦——可以和任何并行配置组合
(baseline / ZeRO-1 / ZeRO-2 / ZeRO-3)。

典型用法:
    >>> def is_decoder_layer(m: nn.Module) -> bool:
    ...     return isinstance(m, LlamaDecoderLayer)
    >>> n_wrapped = apply_activation_checkpointing(model, is_decoder_layer)
    >>> print(f"Wrapped {n_wrapped} modules for activation checkpointing")
"""

from __future__ import annotations

from typing import Any, Callable, Final

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as _torch_checkpoint


# Policy 类型别名:接收 module,返回是否要 wrap
CheckpointPolicy = Callable[[nn.Module], bool]

# 用户可自定义的 checkpoint 实现,签名必须和 torch.utils.checkpoint.checkpoint 一致
CheckpointFn = Callable[..., Any]


# state_dict key 中标识 wrapper 的前缀,通过 hooks 在 save/load 时透明处理
_WRAPPED_MODULE_KEY: Final[str] = "_checkpoint_wrapped_module"
_WRAPPED_MODULE_PREFIX: Final[str] = _WRAPPED_MODULE_KEY + "."


class ActivationCheckpointWrapper(nn.Module):
    """将一个 module 包装为 activation checkpoint。
    
    Forward 调用通过 checkpoint function 转发,中间 activation 不保留;
    backward 时,该 unit 的 forward 会被重做一次以重建 saved tensors。
    
    透明性保证:
        - Parameter 和 buffer 仍然属于内部 module(`model.parameters()` 正常迭代)
        - state_dict 的 key 不含 wrapper prefix(通过 state_dict hooks 处理)
        - 原始 module 通过 `.inner_module` 属性访问
    
    Attributes:
        inner_module: 被包装的原始 module
        checkpoint_fn: 实际的 checkpoint 实现
        use_reentrant: 传递给 checkpoint_fn 的 flag
        preserve_rng_state: 传递给 checkpoint_fn 的 flag
    """
    
    def __init__(
        self,
        module: nn.Module,
        *,
        checkpoint_fn: CheckpointFn = _torch_checkpoint,
        use_reentrant: bool = False,
        preserve_rng_state: bool = False,
    ) -> None:
        super().__init__()
        # 用 setattr 走 nn.Module 的正常 children 注册路径
        # ⇒ 内部 module 在 _modules dict 里,parameters() 等正常工作
        setattr(self, _WRAPPED_MODULE_KEY, module)
        
        self.checkpoint_fn = checkpoint_fn
        self.use_reentrant = use_reentrant
        self.preserve_rng_state = preserve_rng_state
        
        # 注册 state_dict hooks,让外部看不见 wrapper 的存在
        self._register_state_dict_hook(_post_state_dict_hook)
        self._register_load_state_dict_pre_hook(_pre_load_state_dict_hook)
    
    @property
    def inner_module(self) -> nn.Module:
        """获取被包装的原始 module。"""
        return getattr(self, _WRAPPED_MODULE_KEY)
    
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.checkpoint_fn(
            self.inner_module,
            *args,
            use_reentrant=self.use_reentrant,
            preserve_rng_state=self.preserve_rng_state,
            debug=True,
            **kwargs,
        )
    
    def extra_repr(self) -> str:
        return f"use_reentrant={self.use_reentrant}, preserve_rng_state={self.preserve_rng_state}"


def apply_activation_checkpointing(
    model: nn.Module,
    policy: CheckpointPolicy,
    *,
    checkpoint_fn: CheckpointFn = _torch_checkpoint,
    use_reentrant: bool = False,
    preserve_rng_state: bool = False,
) -> int:
    """对 `model` 中所有满足 `policy` 的 module 应用 activation checkpoint。
    
    替换是 in-place 的——匹配的 module 在其 parent 中被替换为
    `ActivationCheckpointWrapper`。Parameter / buffer 在 state_dict 中
    保持原始路径不变(通过 wrapper 的 state_dict hooks 实现)。
    
    Args:
        model: 目标 model(in-place 修改)
        policy: callable,接收 `nn.Module` 返回 `bool`,True 表示需要 wrap
        checkpoint_fn: 实际的 checkpoint 实现。默认 torch.utils.checkpoint.checkpoint。
            用户可以传入自定义实现(例如做 CPU offload 或 selective AC)。
        use_reentrant: 传给 checkpoint_fn。默认 `False`(现代 API)。
        preserve_rng_state: 传给 checkpoint_fn。默认 `False`。
    
    Returns:
        被 wrap 的 module 数量。
    
    Raises:
        ValueError: 当 policy 选中了一个已包含 `ActivationCheckpointWrapper`
            的 module 时(嵌套 AC 不支持,容易导致 buffer 重复 allocate)。
    """
    targets = _collect_wrap_targets(model, policy)
    
    for parent, name, child in targets:
        wrapper = ActivationCheckpointWrapper(
            child,
            checkpoint_fn=checkpoint_fn,
            use_reentrant=use_reentrant,
            preserve_rng_state=preserve_rng_state,
        )
        setattr(parent, name, wrapper)
    
    return len(targets)


def remove_activation_checkpointing(model: nn.Module) -> int:
    """移除 `model` 中所有 `ActivationCheckpointWrapper`,还原原始 module。
    
    幂等:对已经没有 wrapper 的 model 调用返回 `0`。
    
    Returns:
        被还原的 wrapper 数量。
    """
    targets: list[tuple[nn.Module, str, ActivationCheckpointWrapper]] = []
    
    for parent in model.modules():
        for name, child in parent.named_children():
            if isinstance(child, ActivationCheckpointWrapper):
                targets.append((parent, name, child))
    
    for parent, name, wrapper in targets:
        setattr(parent, name, wrapper.inner_module)
    
    return len(targets)


def is_activation_checkpointed(module: nn.Module) -> bool:
    """是否被 activation checkpoint wrap。"""
    return isinstance(module, ActivationCheckpointWrapper)


# ─────────────────────────────────────────
# 实现细节
# ─────────────────────────────────────────

def _collect_wrap_targets(
    model: nn.Module,
    policy: CheckpointPolicy,
) -> list[tuple[nn.Module, str, nn.Module]]:
    """收集所有需要 wrap 的 (parent, name, child) triples。
    
    分两步(先收集再修改),避免遍历的同时修改 _modules dict 导致行为未定义。
    """
    targets: list[tuple[nn.Module, str, nn.Module]] = []
    
    for parent in model.modules():
        for name, child in parent.named_children():
            if isinstance(child, ActivationCheckpointWrapper):
                continue   # 已经 wrap 过,跳过
            if not policy(child):
                continue
            
            # 防御:不允许 wrap 嵌套 wrapper
            for sub in child.modules():
                if sub is child:
                    continue
                if isinstance(sub, ActivationCheckpointWrapper):
                    raise ValueError(
                        f"Cannot wrap {type(child).__name__!r} for activation "
                        f"checkpointing: it contains a nested "
                        f"ActivationCheckpointWrapper around "
                        f"{type(sub.inner_module).__name__!r}. "
                        f"Nested checkpointing causes redundant recompute and "
                        f"is not supported."
                    )
            
            targets.append((parent, name, child))
    
    return targets


def _post_state_dict_hook(
    module: ActivationCheckpointWrapper,
    state_dict: dict[str, Any],
    prefix: str,
    local_metadata: dict[str, Any],
) -> None:
    """保存 state_dict 时,从 keys 移除 wrapper 的前缀。
    
    `layer_0._checkpoint_wrapped_module.weight` → `layer_0.weight`
    """
    wrapped_prefix = prefix + _WRAPPED_MODULE_PREFIX
    for key in list(state_dict.keys()):
        if key.startswith(wrapped_prefix):
            new_key = prefix + key[len(wrapped_prefix):]
            state_dict[new_key] = state_dict.pop(key)


def _pre_load_state_dict_hook(
    module: ActivationCheckpointWrapper,
    state_dict: dict[str, Any],
    prefix: str,
    local_metadata: dict[str, Any],
    strict: bool,
    missing_keys: list[str],
    unexpected_keys: list[str],
    error_msgs: list[str],
) -> None:
    """加载 state_dict 时,给 wrapper 路径下的 keys 补回前缀。
    
    `layer_0.weight` → `layer_0._checkpoint_wrapped_module.weight`
    """
    wrapped_prefix = prefix + _WRAPPED_MODULE_PREFIX
    for key in list(state_dict.keys()):
        if key.startswith(prefix) and not key.startswith(wrapped_prefix):
            new_key = wrapped_prefix + key[len(prefix):]
            state_dict[new_key] = state_dict.pop(key)