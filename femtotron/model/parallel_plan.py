from dataclasses import dataclass, field

@dataclass
class ParallelRule:
    """
    一条并行化替换规则。
    
    parallel_type 的取值和含义：
    
        "column"       → 替换为 ColumnParallelLinear
                         适用于：Q/K/V projection, gate_proj, up_proj, lm_head
                         kwargs 可传：gather_output (bool)
        
        "row"          → 替换为 RowParallelLinear
                         适用于：O projection, down_proj
                         kwargs 可传：scatter_input (bool)
        
        "vocab_embed"  → 替换为 VocabParallelEmbedding
                         适用于：token embedding
                         kwargs 通常为空
        
        "replicate"    → 不替换，每个 TP rank 持有完整副本
                         适用于：RMSNorm, bias 等小参数
                         这是默认行为，不在 plan 中的层就是 replicate
                         显式写出来是为了文档清晰和意图明确
    
    kwargs 应该只放和"并行行为"直接相关的配置，不放和模型结构相关的信息。
    """
    parallel_type: str
    kwargs: dict = field(default_factory=dict)

# 切分策略的定义
class ParallelPlan:
    """
    描述一个模型中每个需要并行化的参数的切分方式。
    
    本质上是一个字典：参数名 pattern → 切分类型。
    pattern 用于匹配 HuggingFace 模型的参数命名规则。
    """
    
    def __init__(self, rules: dict[str, ParallelRule]):
        """
        rules: [(pattern, ParallelRule), ...]
               按顺序匹配，命中第一条就停。
        
        ParallelRule 包含：
        - layer_type: "column", "row", "vocab_embed", "no_parallel"
        - 附加信息（gather_output, scatter_input 等）
               按顺序匹配，命中第一条就停。
        """
        self.rules = rules
    
    def get_rule(self, param_name: str) -> ParallelRule | None:
        """
        给定一个参数的完整名称（如 "model.layers.0.self_attn.q_proj"），
        返回匹配的 ParallelRule，未匹配返回 None（意味着 replicate）。
        """
        for pattern, rule in self.rules.items():
            if param_name.endswith(pattern):
                return rule
        return None
    
# LLaMA 的并行计划
def get_llama_parallel_plan() -> ParallelPlan:
    """
    返回 LLaMA 架构的 TP 切分规则。
    
    规则如下：
    
    Embedding:
      model.embed_tokens → VocabParallelEmbedding
    
    每一层的 Attention:
      self_attn.q_proj → ColumnParallel (gather_output=False)
      self_attn.k_proj → ColumnParallel (gather_output=False)
      self_attn.v_proj → ColumnParallel (gather_output=False)
      self_attn.o_proj → RowParallel    (scatter_input=False)
    
    每一层的 FFN (SwiGLU):
      mlp.gate_proj → ColumnParallel (gather_output=False)
      mlp.up_proj   → ColumnParallel (gather_output=False)
      mlp.down_proj → RowParallel    (scatter_input=False)
    
    不切分的：
      input_layernorm  → 每个 rank 持有完整副本
      post_attention_layernorm → 每个 rank 持有完整副本
      model.norm → 每个 rank 持有完整副本
    
    输出层：
      lm_head → ColumnParallel (gather_output=True)
               （需要完整的 vocab logits 来算 loss）
               如果 tie_word_embeddings=True，lm_head 和 embed_tokens 共享权重
    """
    parallel_plan_list = [
        # Embedding
        ("model.embed_tokens", ParallelRule("vocab_embed")),
        
        # Attention，注意每一层均使用相同匹配规则和结果
        ("self_attn.q_proj", ParallelRule("column", {"gather_output": False})),
        ("self_attn.k_proj", ParallelRule("column", {"gather_output": False})),
        ("self_attn.v_proj", ParallelRule("column", {"gather_output": False})),
        ("self_attn.o_proj", ParallelRule("row",    {"scatter_input": False})),
        
        # FFN (SwiGLU)
        ("mlp.gate_proj",    ParallelRule("column", {"gather_output": False})),
        ("mlp.up_proj",      ParallelRule("column", {"gather_output": False})),
        ("mlp.down_proj",    ParallelRule("row",    {"scatter_input": False})),
        
        # 输出层
        ("lm_head",          ParallelRule("column", {"gather_output": True})),
        
        # RMSNorm 等层为 replicate（默认行为）
        # ("input_layernorm",           ParallelRule("replicate")),
        # ("post_attention_layernorm",  ParallelRule("replicate")),
        # ("model.norm",                ParallelRule("replicate")),
    ]
    parallel_plan_rule = dict(parallel_plan_list)
    return ParallelPlan(rules=parallel_plan_rule)