# femtotron/model/llama_causal.py
from __future__ import annotations

from collections import OrderedDict
from torch import nn
import torch.nn.functional as F

from transformers import LlamaConfig
from femtotron.model.base import BaseCausalLMPipeline
from femtotron.model.llama_partial_model import LlamaPartialModel
from femtotron.parallel_context import ParallelContext
    

class LlamaForCausalLM(BaseCausalLMPipeline):
    """Llama causal LM,PP-aware。
    
    pp_size=1 + layer_range=None 时,行为跟 LlamaForTraining 完全一致
    (full backbone + lm_head + loss)。
    """
    def __init__(
        self,
        config: LlamaConfig,
        parallel_ctx: ParallelContext,
        layer_range: range | None = None,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        
        self.model = LlamaPartialModel(
            model_config=config,
            parallel_ctx=parallel_ctx,
            layer_range=layer_range,
        )
        # is_first/is_last 从 backbone 同步
        self.is_first = self.model.is_first
        self.is_last = self.model.is_last
        
        if self.is_last:
            self.lm_head = nn.Linear(
                config.hidden_size, config.vocab_size, bias=False,
            )
        # tie_word_embeddings:暂不支持,在 trainer 阶段检测到就报错