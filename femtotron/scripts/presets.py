"""Policy presets for ZeRO wrap_policy and activation checkpointing.

YAML configs reference policies by string name,这里把 string 解析成
真正的 Callable[[nn.Module], bool]。

新增预设:加一项 dict entry。需要自定义 policy 的用户可以绕过
这个 registry,直接在脚本里传 Callable。
"""

from __future__ import annotations

from typing import Callable

import torch.nn as nn


# ─── Wrap policy 预设(给 ZeRO-3 用) ───

def _wrap_llama_decoder_layer() -> Callable[[nn.Module], bool]:
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    
    def policy(m: nn.Module) -> bool:
        return isinstance(m, LlamaDecoderLayer)
    
    policy.__name__ = "wrap_llama_decoder_layer"
    return policy


WRAP_POLICY_PRESETS: dict[str, Callable[[], Callable[[nn.Module], bool]]] = {
    "llama_decoder_layer": _wrap_llama_decoder_layer,
}


def get_wrap_policy(name: str) -> Callable[[nn.Module], bool]:
    if name not in WRAP_POLICY_PRESETS:
        raise ValueError(
            f"Unknown wrap_policy preset {name!r}. "
            f"Available: {list(WRAP_POLICY_PRESETS)}"
        )
    return WRAP_POLICY_PRESETS[name]()


# ─── AC policy 预设 ───

def _ac_llama_decoder_layer() -> Callable[[nn.Module], bool]:
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    from femtotron.training.ckpt_policy import by_class
    return by_class(LlamaDecoderLayer)


AC_POLICY_PRESETS: dict[str, Callable[[], Callable[[nn.Module], bool]]] = {
    "llama_decoder_layer": _ac_llama_decoder_layer,
}

def get_ac_policy(name: str) -> Callable[[nn.Module], bool]:
    """从 preset registry 解析 AC policy。未知名字直接 raise。"""
    if name not in AC_POLICY_PRESETS:
        raise ValueError(
            f"Unknown ac_policy preset {name!r}. "
            f"Available: {list(AC_POLICY_PRESETS)}"
        )
    return AC_POLICY_PRESETS[name]()
