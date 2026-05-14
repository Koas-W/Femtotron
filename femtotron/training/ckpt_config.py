"""Optional: construct a CheckpointPolicy from a dict config.

只有从 YAML/JSON 配置加载时需要。Python 代码里直接用工厂函数更好。

Config schema:
    {"type": "never"}
    {"type": "always"}
    {"type": "by_class_name", "names": ["LlamaDecoderLayer"]}
    {"type": "any_of", "policies": [<config>, <config>, ...]}
    {"type": "all_of", "policies": [<config>, <config>, ...]}
    {"type": "not_of", "policy": <config>}
"""

from __future__ import annotations
from typing import Any

from femtotron.training.activation_ckpt import CheckpointPolicy
from femtotron.training.ckpt_policy import (
    by_class_name, any_of, all_of, not_of, NEVER, ALWAYS,
)


def policy_from_config(config: dict[str, Any] | None) -> CheckpointPolicy:
    """从 dict config 构造 policy。`None` 等价于 NEVER。
    
    注意:
    - 不支持 `by_class`(类对象不可序列化),只支持 `by_class_name`
    - 嵌套 config 递归解析
    """
    if config is None:
        return NEVER
    
    kind = config.get("type")
    
    if kind == "never":
        return NEVER
    
    if kind == "always":
        return ALWAYS
    
    if kind == "by_class_name":
        names = config["names"]
        if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
            raise ValueError(
                f"by_class_name expects 'names' to be a list of strings, "
                f"got {names!r}"
            )
        return by_class_name(*names)
    
    if kind == "any_of":
        sub = [policy_from_config(c) for c in config["policies"]]
        return any_of(*sub)
    
    if kind == "all_of":
        sub = [policy_from_config(c) for c in config["policies"]]
        return all_of(*sub)
    
    if kind == "not_of":
        sub = policy_from_config(config["policy"])
        return not_of(sub)
    
    raise ValueError(
        f"Unknown policy type {kind!r}. "
        f"Valid types: never, always, by_class_name, any_of, all_of, not_of."
    )