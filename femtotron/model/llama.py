import torch
from torch import Tensor, dtype
import torch.distributed as dist
from torch.distributed import ReduceOp, ProcessGroup
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

from transformers import AutoConfig
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama import LlamaModel

from femtotron.parallel_context import ParallelContext
from femtotron.model.parallel_plan import ParallelPlan, ParallelRule, get_llama_parallel_plan
from femtotron.model.parallelize_model import parallelize_model
from femtotron.model.layer_factory import LayerFactory, DefaultLayerFactory

class LlamaForTraining(nn.Module):
    """
    在 HuggingFace LlamaModel 基础上，加上 loss 计算。
    
    为什么不直接用 HuggingFace 的 LlamaForCausalLM：
    1. HF 的 LlamaForCausalLM 内部的 lm_head 是普通 nn.Linear，
       我们需要替换成 ColumnParallelLinear (或 VocabParallelEmbedding 共享权重)
    2. loss 计算在 TP 下有特殊处理。
       如果 lm_head 是 gather_output=True，loss 在完整 logits 上算，简单直接
       如果想更高效（避免 gather 完整 vocab 的 logits），
       可以用 parallel cross-entropy（每个 rank 在自己的 vocab 切片上算部分 loss）
       但这是优化项，初期用 gather_output=True 就够了
    3. 我们需要控制 forward 的输入输出格式，和我们的训练循环对接
    """
    
    def __init__(
        self,
        model_config: LlamaConfig,
        parallel_ctx: ParallelContext,
    ):
        super().__init__()
        """
        参数：
            model_config: HuggingFace 的 LlamaConfig
            parallel_ctx: ParallelContext
        
        内部创建：
            self.model: 经过 parallelize_model 处理的 LlamaModel（不含 lm_head）
            self.lm_head: ColumnParallelLinear(hidden, vocab, gather_output=True)
                          或者如果 tie_word_embeddings=True，
                          和 embed_tokens 共享权重
        """
        
        self.config = model_config
        self.parallel_ctx = parallel_ctx

        # backbone（不含 lm_head）
        self.model = LlamaModel(model_config)
        self.lm_head = nn.Linear(model_config.hidden_size, model_config.vocab_size, bias=False)
        if model_config.tie_word_embeddings:
            self.tie_weights()
            
        # 关键:训练时禁用 KV cache,否则和 activation checkpointing 不兼容
        self.model.config.use_cache = False

    def tie_weights(self):
        """Tie lm_head 和 embed_tokens 的权重。to_empty 后需要重新调用。"""
        self.lm_head.weight = self.model.embed_tokens.weight
    
    def forward(self, input_ids: torch.Tensor, 
                labels: torch.Tensor | None = None) -> dict:
        """
        参数：
            input_ids: [batch, seq_len] token IDs
            labels:    [batch, seq_len] 目标 token IDs，
                       不需要 loss 的位置设为 -100
        
        流程：
            1. hidden = self.model(input_ids)     # [B, S, H]
            2. logits = self.lm_head(hidden)      # [B, S, vocab]
            3. 如果 labels 不为 None：
               - shift: logits[..., :-1, :] 和 labels[..., 1:]
                 （next-token prediction: 用位置 t 的 logits 预测位置 t+1 的 token）
               - loss = F.cross_entropy(logits_shifted, labels_shifted)
            4. 返回 {"loss": loss, "logits": logits}
        """
        hidden = self.model(input_ids).last_hidden_state   # [B, S, H]
        logits = self.lm_head(hidden)                      # [B, S, V] 或 [B, S, V/tp]，看 gather_output

        out: dict = {"logits": logits}
        if labels is not None:
            # shift: 用位置 t 的 logits 预测位置 t+1 的 token
            shift_logits = logits[..., :-1, :].contiguous()    # [B, S-1, V or V/tp]
            shift_labels = labels[..., 1:].contiguous()         # [B, S-1]

            loss = self._compute_loss(shift_logits, shift_labels)
            out["loss"] = loss

        return out
    
    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # logits: [B, S-1, V] 完整 vocab
        # labels: [B, S-1]，-100 位置忽略
        return F.cross_entropy(
            logits.float().view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

# class LlamaBackbone(nn.Module):
#     def __init__(self, config, parallel_ctx, factory, *, dtype, device):
#         super().__init__()
#         self.embed_tokens = factory.make_embedding(config, parallel_ctx, dtype=dtype, device=device)
#         self.layers = nn.ModuleList([
#             factory.make_decoder_layer(config, parallel_ctx, layer_idx=i, dtype=dtype, device=device)
#             for i in range(config.num_hidden_layers)
#         ])
#         self.norm = factory.make_norm(config, dtype=dtype, device=device)

#     def forward(
#     self,
#     input_ids: Tensor,                          # [batch, seq]
#     position_ids: Tensor | None = None,         # [batch, seq]，None 时自动生成
#     attention_mask: Tensor | None = None,       # 因果 mask 通常由 attention 内部处理
#     ) -> Tensor:                                    # [batch, seq, hidden]
#         # 1. 词嵌入
#         hidden_states = self.embed_tokens(input_ids)

#         # 2. position_ids 默认 [0, 1, ..., seq-1]
#         if position_ids is None:
#             seq_len = input_ids.shape[1]
#             position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

#         # 3. 逐层 transformer block
#         for layer in self.layers:
#             hidden_states = layer(
#                 hidden_states,
#                 position_ids=position_ids,
#                 attention_mask=attention_mask,
#             )

#         # 4. 最终 norm
#         hidden_states = self.norm(hidden_states)
#         return hidden_states

def build_llama_model(model_config: LlamaConfig, parallel_ctx: ParallelContext) -> LlamaForTraining:
    """
    构建训练用模型的完整流程。
    
    1. 从 HuggingFace 加载 config（只是配置，不加载权重）
    2. 在 meta device 上创建 LlamaForTraining
       内部创建 HuggingFace 的 LlamaModel，但参数都在 meta device 上
    3. parallelize_model() 替换线性层为 TP 版本
       参数仍在 meta device 上，只是类型和 shape 变了
    4. 返回 model（尚未加载权重）
    """
    # config: LlamaConfig = LlamaConfig.from_pretrained(model_name)
    
    with torch.device("meta"):
        model = LlamaForTraining(model_config, parallel_ctx)
    
    parallel_plan = get_llama_parallel_plan()
    parallelize_model(model, parallel_plan, parallel_ctx)
    return model