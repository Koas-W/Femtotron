import torch
from torch import Tensor
import torch.distributed as dist
from torch.distributed import ReduceOp, ProcessGroup
from torch import nn
import torch.nn.functional as F
from .comm_ops import CopyToTPRegion, GatherFromTPRegion, ReduceFromTPRegion, ScatterToTPRegion
from femtotron.parallel_context import ParallelContext

class VocabParallelEmbedding(nn.Module):
    """
    沿 vocab 维度切分的 Embedding 层。

    完整 embedding table: [vocab_size, hidden_size]
    每个 rank 持有: table[rank*chunk : (rank+1)*chunk, :]
    其中 chunk = vocab_size // tp_size

    Forward 行为：
        输入:  input_ids [batch, seq_len] — token ID
        输出:  embeddings [batch, seq_len, hidden_size] — 完整的

    实现关键：
        每个 rank 只持有部分 vocab 的 embedding。
        对于不在本 rank 范围内的 token ID，输出全零。
        最后 all-reduce 汇总各 rank 的结果（每个 token 只有一个 rank 输出非零）。

    权重加载：
        from_embedding(embedding, parallel_ctx) 类方法：
        接收完整的 nn.Embedding，按 vocab 切分。
    """

    def __init__(self, 
                 vocab_size : int, 
                 hidden_size : int, 
                 parallel_ctx : ParallelContext, 
                 parallel_dim_name="tp",
                 device=None, 
                 dtype=None,
                 ):
        super().__init__()
        # vocab_size 必须能被 tp_size 整除
        # 如果不能整除，需要 padding 到能整除的最近值
        # self.weight: nn.Parameter, shape [vocab_size // tp_size, hidden_size]
        # self.vocab_start_idx = tp_rank * (vocab_size // tp_size)
        # self.vocab_end_idx = (tp_rank + 1) * (vocab_size // tp_size)
        self.group = parallel_ctx.get_group(parallel_dim_name)
        self.world_size = parallel_ctx.get_size(parallel_dim_name)
        self.rank = parallel_ctx.get_rank(parallel_dim_name)

        self.weight = torch.nn.Parameter(torch.randn(vocab_size // self.world_size, hidden_size, device=device, dtype=dtype))
        self.vocab_start_idx = self.rank * (vocab_size // self.world_size)
        self.vocab_end_idx = (self.rank + 1) * (vocab_size // self.world_size)

    def forward(self, input_ids : Tensor):
        # 1. 将 input_ids 中不在 [vocab_start, vocab_end) 范围内的 ID mask 掉
        # 2. 对范围内的 ID 减去 vocab_start_idx 得到本地索引
        # 3. F.embedding 查表
        # 4. mask 掉的位置输出全零
        # 5. all-reduce 汇总各 rank 的结果
        # 6. 返回完整的 embeddings
        # 1+2: 计算算 mask 和本地索引
        mask = (input_ids < self.vocab_start_idx) | (input_ids >= self.vocab_end_idx)
        masked_input = input_ids - self.vocab_start_idx          # 新 tensor，无需 clone
        masked_input.masked_fill_(mask, 0)              # in-place，越界位置指向 0 行

        # 3: 查表
        output = F.embedding(masked_input, self.weight)      # [*, hidden_dim]

        # 4: 越界位置写 0
        output.masked_fill_(mask.unsqueeze(-1), 0)

        # 5: all-reduce
        ReduceFromTPRegion.apply(output, self.group)
        return output

    @classmethod
    def from_embedding(cls, embedding: nn.Embedding, parallel_ctx):
        """
        从完整的 nn.Embedding 构造 VocabParallelEmbedding。
        沿 vocab 维度（dim=0）切分权重。
        
        embedding.weight shape: [vocab_size, hidden_size]
        切分后每个 rank:        [vocab_size // tp_size, hidden_size]
        
        ⚠️ 仅用于测试！正式训练应使用 ModelLoader。
        """
        tp_size = parallel_ctx.get_size("tp")
        tp_rank = parallel_ctx.get_rank("tp")
        
        vocab_size = embedding.num_embeddings
        hidden_size = embedding.embedding_dim
        
        tp_embed = cls(vocab_size, hidden_size, parallel_ctx,
                       device=embedding.weight.device, dtype=embedding.weight.dtype)
        
        # 权重 [vocab, hidden] 沿 vocab 切分
        chunk = vocab_size // tp_size
        tp_embed.weight.data.copy_(
            embedding.weight.data[tp_rank * chunk : (tp_rank + 1) * chunk, :]
        )
        
        return tp_embed