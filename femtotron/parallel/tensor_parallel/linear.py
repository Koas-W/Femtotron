import torch
from torch import Tensor
import torch.distributed as dist
from torch.distributed import ReduceOp, ProcessGroup
from torch import nn
import torch.nn.functional as F

from .comm_ops import CopyToTPRegion, GatherFromTPRegion, ReduceFromTPRegion, ScatterToTPRegion
from femtotron.parallel_context import ParallelContext
from femtotron.model.parallel_plan import ParallelPlan, ParallelRule

class ColumnParallelLinear(nn.Module):
    """
    沿输出维度（列）切分的线性层。

    完整权重 W: [in_features, out_features]
    每个 rank 持有: W[:, rank*chunk : (rank+1)*chunk]
    其中 chunk = out_features // tp_size

    参数：
        in_features:  输入维度（不切分）
        out_features: 输出维度（沿 TP 切分，必须能被 tp_size 整除）
        parallel_ctx: ParallelContext 实例
        bias:         是否有 bias（默认 False，LLaMA 不用 bias）
        gather_output: 是否在 forward 结尾 all-gather 拼回完整输出
                       默认 False（输出保持切分，直接喂给下游 RowParallel）
                       设为 True 用于需要完整输出的场景（如 lm_head）

    Forward 行为：
        输入:  x [batch, seq_len, in_features] — 每个 rank 持有完整输入
        输出:  y [batch, seq_len, out_features // tp_size] — 切分的
               或 y [batch, seq_len, out_features] — 如果 gather_output=True

    权重加载：
        from_linear(linear, parallel_ctx) 类方法：
        接收一个完整的 nn.Linear，按列切分权重，返回 ColumnParallelLinear。
        用于从 HuggingFace 模型转换。
    """

    def __init__(self, 
                 in_features : int, 
                 out_features : int, 
                 parallel_ctx : ParallelContext,
                 parallel_dim_name="tp",
                 bias=False, 
                 gather_output=False, 
                 device=None, 
                 dtype=None):
        super().__init__()
        # out_features 必须能被 tp_size 整除
        # self.weight: nn.Parameter, shape [out_features // tp_size, in_features]
        # （注意 PyTorch 的 Linear 权重是 [out, in] 排列）
        self.group = parallel_ctx.get_group(parallel_dim_name)
        self.world_size = parallel_ctx.get_size(parallel_dim_name)
        self.rank = parallel_ctx.get_rank(parallel_dim_name)
        self.weight = torch.nn.Parameter(torch.randn(out_features // self.world_size, in_features, device=device, dtype=dtype))
        if bias:
            self.bias = torch.nn.Parameter(torch.randn(out_features, device=device, dtype=dtype))
        else:
            self.bias = None
        self.parallel_ctx = parallel_ctx
        self.gather_output = gather_output

    def forward(self, x : Tensor):
        # 1. x 通过 CopyToTPRegion（forward identity, backward all-reduce）
        # 2. y_partial = F.linear(x, self.weight, self.bias)
        # 3. 如果 gather_output: y = GatherFromTPRegion(y_partial)
        # 4. 返回 y_partial 或 y
        x = CopyToTPRegion.apply(x, self.group)
        y_partial = F.linear(x, self.weight, self.bias)
        if self.gather_output:
            y = GatherFromTPRegion.apply(y_partial, self.group)
        else :
            y = y_partial
        return y


    # ColumnParallelLinear
    @classmethod
    def from_linear(cls, linear: nn.Linear, parallel_ctx: ParallelContext, rule: ParallelRule) -> "ColumnParallelLinear":
        """
        从完整的 nn.Linear 构造 ColumnParallelLinear。
        沿输出维度（dim=0）切分权重。
        
        linear.weight shape: [out_features, in_features]
        切分后每个 rank:     [out_features // tp_size, in_features]
        """
        world_size = parallel_ctx.tp_size
        assert linear.out_features % world_size == 0
        return cls(
            in_features=linear.in_features,
            out_features=linear.out_features // world_size,
            bias=linear.bias is not None,
            parallel_ctx=parallel_ctx,
            gather_output=rule.kwargs["gather_output"],
            device=linear.weight.device,   # 自动推断
            dtype=linear.weight.dtype,     # 自动推断
        )
    
    # ColumnParallelLinear
    @classmethod
    def from_linear_temp(cls, linear: nn.Linear, parallel_ctx, gather_output=False):
        """
        从完整的 nn.Linear 构造 ColumnParallelLinear。
        沿输出维度（dim=0）切分权重。
        
        linear.weight shape: [out_features, in_features]
        切分后每个 rank:     [out_features // tp_size, in_features]
        
        ⚠️ 仅用于测试！正式训练应使用 ModelLoader。
        """
        tp_size = parallel_ctx.get_size("tp")
        tp_rank = parallel_ctx.get_rank("tp")
        
        in_features = linear.in_features
        out_features = linear.out_features
        has_bias = linear.bias is not None
        
        col_linear = cls(
            in_features, out_features, parallel_ctx,
            bias=has_bias, gather_output=gather_output,
            device=linear.weight.device,   # 自动推断
            dtype=linear.weight.dtype,     # 自动推断
        )
        
        # 权重 [out, in] 沿 out 切分
        chunk = out_features // tp_size
        col_linear.weight.data.copy_(
            linear.weight.data[tp_rank * chunk : (tp_rank + 1) * chunk, :]
        )
        
        # bias 也沿 out 切分（如果有的话）
        if has_bias:
            assert col_linear.bias is not None
            col_linear.bias.data.copy_(
                linear.bias.data[tp_rank * chunk : (tp_rank + 1) * chunk]
            )
        
        return col_linear

class RowParallelLinear(nn.Module):
    """
    沿输入维度（行）切分的线性层。

    完整权重 W: [in_features, out_features]
    每个 rank 持有: W[rank*chunk : (rank+1)*chunk, :]
    其中 chunk = in_features // tp_size

    参数：
        in_features:   输入维度（沿 TP 切分，必须能被 tp_size 整除）
        out_features:  输出维度（不切分）
        parallel_ctx:  ParallelContext 实例
        bias:          是否有 bias（默认 False）
        scatter_input:  是否在 forward 开头 scatter 切分输入
                        默认 False（假设输入已经是切分的，来自上游 ColumnParallel）
                        设为 True 用于输入是完整的场景

    Forward 行为：
        输入:  x_partial [batch, seq_len, in_features // tp_size] — 切分的
               或 x [batch, seq_len, in_features] — 如果 scatter_input=True
        输出:  y [batch, seq_len, out_features] — 完整的（经过 all-reduce）

    权重加载：
        from_linear(linear, parallel_ctx) 类方法：
        接收一个完整的 nn.Linear，按行切分权重，返回 RowParallelLinear。
    """

    def __init__(self, 
                 in_features : int, 
                 out_features : int, 
                 parallel_ctx : ParallelContext,
                 parallel_dim_name="tp",
                 bias=False, 
                 scatter_input=False,
                 device=None, 
                 dtype=None):
        super().__init__()
        # in_features 必须能被 tp_size 整除
        # self.weight: nn.Parameter, shape [out_features, in_features // tp_size]
        self.group = parallel_ctx.get_group(parallel_dim_name)
        self.world_size = parallel_ctx.get_size(parallel_dim_name)
        self.rank = parallel_ctx.get_rank(parallel_dim_name)
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features // self.world_size, device=device, dtype=dtype))
        if bias:
            self.bias = torch.nn.Parameter(torch.randn(out_features, device=device, dtype=dtype))
        else:
            self.bias = None
        self.parallel_ctx = parallel_ctx
        self.scatter_input = scatter_input
        

    def forward(self, x : Tensor):
        # 1. 如果 scatter_input: x_partial = ScatterToTPRegion(x)
        #    否则: x_partial = x（假设已经切分好了）
        # 2. y_partial = F.linear(x_partial, self.weight)
        # 3. y = ReduceFromTPRegion(y_partial)（all-reduce 求和）
        # 4. 如果有 bias: y = y + self.bias
        #    （bias 不切分，只在 all-reduce 之后加，避免重复加）
        # 5. 返回 y
        if self.scatter_input:
            x_partial = ScatterToTPRegion.apply(x, self.group)
        else:
            x_partial = x
        y_partial = F.linear(x_partial, self.weight)
        y = ReduceFromTPRegion.apply(y_partial, self.group)
        if self.bias is not None:
            y = y + self.bias
        return y

    
    # RowParallelLinear
    @classmethod
    def from_linear(cls, linear: nn.Linear, parallel_ctx: ParallelContext, rule: ParallelRule) -> "RowParallelLinear":
        """
        从完整的 nn.Linear 构造 RowParallelLinear
        沿输入维度（dim=1）切分权重。
        
        linear.weight shape: [out_features, in_features]
        切分后每个 rank:     [out_features, in_features // tp_size]
        """
        world_size = parallel_ctx.tp_size
        assert linear.out_features % world_size == 0
        return cls(
            in_features=linear.in_features // world_size,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            parallel_ctx=parallel_ctx,
            scatter_input=rule.kwargs["scatter_input"],
            device=linear.weight.device,   # 自动推断
            dtype=linear.weight.dtype,     # 自动推断
        )

    # RowParallelLinear
    @classmethod
    def from_linear_temp(cls, linear: nn.Linear, parallel_ctx, scatter_input=False):
        """
        从完整的 nn.Linear 构造 RowParallelLinear。
        沿输入维度（dim=1）切分权重。
        
        linear.weight shape: [out_features, in_features]
        切分后每个 rank:     [out_features, in_features // tp_size]
        
        ⚠️ 仅用于测试！正式训练应使用 ModelLoader。
        """
        tp_size = parallel_ctx.get_size("tp")
        tp_rank = parallel_ctx.get_rank("tp")
        
        in_features = linear.in_features
        out_features = linear.out_features
        has_bias = linear.bias is not None
        
        row_linear = cls(
            in_features, out_features, parallel_ctx,
            bias=has_bias, scatter_input=scatter_input,
            device=linear.weight.device,   # 自动推断
            dtype=linear.weight.dtype,     # 自动推断
        )
        
        # 权重 [out, in] 沿 in 切分
        chunk = in_features // tp_size
        row_linear.weight.data.copy_(
            linear.weight.data[:, tp_rank * chunk : (tp_rank + 1) * chunk]
        )
        
        # bias 不切分——每个 rank 持有完整副本
        # （bias 在 all-reduce 之后加，只加一次）
        if has_bias:
            assert row_linear.bias is not None
            row_linear.bias.data.copy_(linear.bias.data)
        
        return row_linear