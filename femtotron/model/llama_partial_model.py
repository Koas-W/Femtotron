"""LlamaPartialModel: HF LlamaModel 的 PP-aware 等价物。

设计目标:
1. 在退化情况(layer_range=None, is_first=True, is_last=True)下,
   行为完全等价于 transformers.LlamaModel — 数学位级一致
2. 在 PP 启用时,只持有 layer_range 内的 layers;
   first stage 额外持有 embed_tokens,last stage 额外持有 norm
3. 参数命名完全兼容 HF LlamaModel:
   - embed_tokens.weight (first only)
   - layers.{global_idx}.{...}.weight
   - norm.weight (last only)
   这样 ModelLoader / parallelize_model 等依赖参数名匹配的下游组件 0 改动

不暴露 HF 的 BaseModelOutputWithPast 风格输出 — forward 直接返回 hidden_states tensor。

需要 transformers >= 4.38(为了 model-level rotary_emb 和 position_embeddings 参数)。
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    import transformers
    _HF_VERSION = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    if _HF_VERSION < (4, 38):
        raise ImportError(
            f"LlamaPartialModel 需要 transformers >= 4.38(model-level rotary_emb),"
            f"当前 {transformers.__version__}"
        )
except ImportError:
    raise

from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRotaryEmbedding,
    LlamaRMSNorm,
)

from femtotron.parallel_context import ParallelContext


class LlamaPartialModel(nn.Module):
    """LlamaModel 的 PP-aware 等价物。
    
    参数命名完全兼容 HF LlamaModel:
        embed_tokens.weight       (仅 first stage)
        layers.{i}.self_attn.{q,k,v,o}_proj.weight     (i 是全局 layer index)
        layers.{i}.mlp.{gate,up,down}_proj.weight
        layers.{i}.input_layernorm.weight
        layers.{i}.post_attention_layernorm.weight
        norm.weight               (仅 last stage)
    
    使用样例:
        # 退化(等价于 HF LlamaModel)
        model = LlamaPartialModel(config, parallel_ctx)
        
        # PP stage 0/2:layers 0-3,持有 embed
        model = LlamaPartialModel(
            config, parallel_ctx,
            layer_range=range(0, 4), is_first=True, is_last=False,
        )
        
        # PP stage 1/2:layers 4-7,持有 norm
        model = LlamaPartialModel(
            config, parallel_ctx,
            layer_range=range(4, 8), is_first=False, is_last=True,
        )
    """
    
    def __init__(
        self,
        model_config: LlamaConfig,
        parallel_ctx: ParallelContext,
        layer_range: range | None = None,
        is_first: bool | None = None,
        is_last: bool | None = None,  # None = 从 layer_range 自动推导
    ):
        """
        Args:
            model_config: HF LlamaConfig
            parallel_ctx: 并行上下文。目前仅存储,未在 forward 中使用;
                          预留给未来需要 PP/TP-aware 行为的扩展
            layer_range: 本 stage 持有的 layer indices(全局编号,
                          就是 LlamaDecoderLayer.layer_idx 的值)。
                          None 表示持有全部 layers
            is_first: 是否第一个 stage(持有 embed_tokens)
            is_last: 是否最后一个 stage(持有 final norm)
        
        Raises:
            ValueError: layer_range 越界,或 layer_range 不是 range 类型
        """
        super().__init__()
        
        # 默认 + 校验
        if layer_range is None:
            layer_range = range(model_config.num_hidden_layers)
        if not isinstance(layer_range, range):
            raise ValueError(
                f"layer_range 必须是 range,得到 {type(layer_range).__name__}"
            )
        if layer_range.start < 0 or layer_range.stop > model_config.num_hidden_layers:
            raise ValueError(
                f"layer_range {layer_range} 越界,模型总层数 "
                f"{model_config.num_hidden_layers}"
            )
        if layer_range.step != 1:
            raise ValueError(
                f"layer_range 必须是连续的(step=1),得到 step={layer_range.step}"
            )
        
        if is_first is None:
            is_first = (layer_range.start == 0)
        if is_last is None:
            is_last = (layer_range.stop == model_config.num_hidden_layers)
        
        # ★ 手动同步 _attn_implementation,弥补不继承 PreTrainedModel 的 gap
        # PreTrainedModel 会做这件事,我们没继承,需手动补
        if getattr(model_config, "_attn_implementation", None) is None:
            # 优先用 config.attn_implementation(用户构造时传的),否则默认 sdpa
            impl = getattr(model_config, "attn_implementation", None) or "sdpa"
            model_config._attn_implementation = impl
    
        self.config = model_config
        self.parallel_ctx = parallel_ctx
        self.layer_range = layer_range
        self.is_first = is_first
        self.is_last = is_last
        
        # First stage: embed_tokens
        # padding_idx 可以是 None(nn.Embedding 接受 None)
        if is_first:
            self.embed_tokens = nn.Embedding(
                model_config.vocab_size,
                model_config.hidden_size,
                padding_idx=model_config.pad_token_id,
            )
        
        # 所有 stage:本 range 内的 layers
        # layer_idx 用全局编号(HF LlamaDecoderLayer 用它做 KV cache 索引,
        # 训练时虽然不用 cache,保留全局编号便于和 HF 行为对齐)
        self.layers = nn.ModuleDict({
            str(idx): LlamaDecoderLayer(model_config, layer_idx=idx)
            for idx in layer_range
        })
        
        # Rotary embedding:每个 stage 持有一份(无可训练参数,只是 inv_freq buffer)
        # HF >= 4.38 在 LlamaModel level 持有 rotary_emb,我们 mirror 这个设计
        self.rotary_emb = LlamaRotaryEmbedding(config=model_config)
        
        # Last stage: final RMSNorm
        if is_last:
            self.norm = LlamaRMSNorm(
                model_config.hidden_size,
                eps=model_config.rms_norm_eps,
            )
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:
                - is_first=True:  input_ids,LongTensor[B, S]
                - is_first=False: hidden_states,float[B, S, H]
            attention_mask: 默认 None
                SDPA 在 mask=None 且 q_len>1 时自动 is_causal=True,
                正好是因果训练所需。HF LlamaModel 默认行为也是这个
            position_ids: 默认 None,自动按 arange(S).unsqueeze(0) 生成
        
        Returns:
            hidden_states float[B, S, H]
            - 非 last stage: layers 输出(未经 final norm)
            - last stage: 经过 final norm 的 hidden_states
        """
        # Embed (first stage only)
        if self.is_first:
            hidden_states = self.embed_tokens(x)
        else:
            hidden_states = x
        
        bsz, seqlen, _ = hidden_states.shape
        
        if position_ids is None:
            position_ids = torch.arange(
                seqlen, device=hidden_states.device, dtype=torch.long,
            ).unsqueeze(0)
        
        # 用 HF 自己的 mask 构造工具,完全对齐
        from transformers.masking_utils import create_causal_mask
        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=position_ids,
        )
        
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)
        
        for layer in self.layers.values():
            # 5.x: layer 直接返回 tensor(不是 tuple),kwargs 也精简了
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=None,         # 复数!不是 past_key_value
                use_cache=False,
            )
            # 没有 layer_outputs[0] 这一步——5.x 直接是 tensor
        
        if self.is_last:
            hidden_states = self.norm(hidden_states)
        
        return hidden_states