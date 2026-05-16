# femtotron/model/base.py
from __future__ import annotations

from torch import nn
import torch.nn.functional as F

class BaseCausalLMPipeline(nn.Module):
    """PP-aware causal LM 基类。子类提供 self.model + self.lm_head,
    base 提供 forward + loss。
    """
    is_first: bool
    is_last: bool
    hidden_size: int
    
    def forward(self, x, labels=None, materialize_logits: bool = True) -> dict:
        hidden = self.model(x)
        if not self.is_last:
            return {"loss": None, "logits": None, "hidden_states": hidden}
        
        logits = self.lm_head(hidden)
        loss = self._compute_loss(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits, "hidden_states": None}
    
    def _compute_loss(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return F.cross_entropy(
            shift_logits.float().view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )