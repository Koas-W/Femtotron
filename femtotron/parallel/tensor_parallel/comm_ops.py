import torch
from torch import Tensor
import torch.distributed as dist
from torch.distributed import ReduceOp, ProcessGroup

class CopyToTPRegion(torch.autograd.Function):
    """
    forward: identity，输入原样输出
    backward: all-reduce，将各 rank 的梯度求和

    ctx：上下文，torch.autograd.function.FunctionCtx类型，用于保存前向传播过程当中需要保存的东西

    用途：ColumnParallelLinear 的输入端。
    ColumnParallel 中每个 rank 用完整输入 x 乘以
    自己的 W_partial，backward 时 dx = dy @ W_partial^T，
    每个 rank 算出的 dx 只是部分梯度，需要 all-reduce 汇总。
    但 forward 时 x 不需要任何通信，每个 rank 已经持有完整 x。
    所以 forward 是 identity，backward 是 all-reduce。
    """
    @staticmethod
    def forward(ctx, x : Tensor, tp_group : ProcessGroup):
        # 保存 group 供 backward 使用
        # 返回 x 本身（identity）
        ctx.tp_group = tp_group # 保存上下文
        return x

    @staticmethod
    def backward(ctx, *grad_outputs : Tensor):
        # 在 tp_group 上 all-reduce grad_output
        # 返回 (all_reduced_grad, None)
        # None 对应 tp_group 参数的梯度（不需要）
        grad_output, *_ = grad_outputs
        dist.all_reduce(tensor=grad_output, op=ReduceOp.SUM, group=ctx.tp_group)
        return grad_output, None


class GatherFromTPRegion(torch.autograd.Function):
    """
    forward: all-gather，将各 rank 的部分 tensor 沿指定维度拼接
    backward: split，将梯度沿同一维度切开，每个 rank 只取自己的部分

    用途：当 ColumnParallelLinear 需要输出完整结果时
    （比如最后一层需要完整的 logits 做 loss 计算）。
    大部分中间层不需要这个，Column 的切分输出直接喂给 Row。
    """
    @staticmethod
    def forward(ctx, x_partial : Tensor, tp_group : ProcessGroup, gather_dim : int = -1):
        # 保存 gather_dim
        # 在 tp_group 上沿 gather_dim 做 all-gather
        # 返回拼接后的完整 tensor
        # x_list:list[Tensor] = [torch.empty_like(x_partial) for i in range(tp_group.size())]
        world_size = dist.get_world_size(tp_group)
        tp_rank = dist.get_rank(tp_group)
        chunk_size = x_partial.shape[gather_dim]
        ctx.gather_dim = gather_dim
        ctx.world_size = world_size
        ctx.chunk_size = chunk_size
        ctx.tp_rank = tp_rank
        ctx.tp_group = tp_group
        if world_size == 1:
            return x_partial
        if gather_dim < 0:
            gather_dim += x_partial.dim()

        # Fast path: gather_dim == 0
        if gather_dim == 0:
            x = x_partial.contiguous()
            x_all = torch.empty(
                (world_size * x.shape[0], *x.shape[1:]),
                dtype=x.dtype, device=x.device,
            )
            dist.all_gather_into_tensor(output_tensor=x_all, input_tensor=x, group=tp_group)
            return x_all
        
        # Normal path: gather_dim > 0，需要转置
        x = x_partial.transpose(0, gather_dim).contiguous()
        x_all = torch.empty(
            (world_size * x.shape[0], *x.shape[1:]),
            dtype=x.dtype, device=x.device,
        )
        dist.all_gather_into_tensor(output_tensor=x_all, input_tensor=x, group=tp_group)
        return x_all.transpose(0, gather_dim).contiguous()

    @staticmethod
    def backward(ctx, *grad_outputs : Tensor):
        # 沿 gather_dim 切分 grad_output
        # 当前 rank 只取自己对应的那一片
        # 返回 (grad_partial, None, None)
        grad_output, *_ = grad_outputs
        grad_partial = torch.narrow(input=grad_output, 
                                    dim=ctx.gather_dim,
                                    start=ctx.tp_rank * ctx.chunk_size, 
                                    length=ctx.chunk_size,).contiguous()
        return grad_partial, None, None


class ReduceFromTPRegion(torch.autograd.Function):
    """
    forward: all-reduce，将各 rank 的部分和求总和
    backward: identity，梯度原样传回

    用途：RowParallelLinear 的输出端。
    RowParallel 中每个 rank 算出 y_partial = x_partial @ W_partial，
    这只是最终结果的一部分（部分和），需要 all-reduce 求总和得到完整 y。
    backward 时 dy 已经是完整的梯度，每个 rank 各自用它算 dx 和 dW，
    不需要通信。
    """
    @staticmethod
    def forward(ctx, y_partial : Tensor, tp_group : ProcessGroup):
        # 在 tp_group 上 all-reduce y_partial
        # 返回求和后的结果
        world_size = dist.get_world_size(tp_group)
        tp_rank = dist.get_rank(tp_group)
        ctx.world_size = world_size
        ctx.tp_rank = tp_rank
        ctx.tp_group = tp_group
        dist.all_reduce(tensor=y_partial, op=ReduceOp.SUM, group=tp_group)
        return y_partial

    @staticmethod
    def backward(ctx, *grad_outputs : Tensor):
        # identity，直接返回 (grad_output, None)
        grad_output, *_ = grad_outputs
        return grad_output, None

class ScatterToTPRegion(torch.autograd.Function):
    """
    forward: split，将输入沿指定维度切开，每个 rank 取自己的部分
    backward: all-gather，将各 rank 的梯度拼接回完整

    用途：当 RowParallelLinear 接收完整输入时
    （而不是来自上游 ColumnParallel 的切分输出）。
    比如模型的第一层，输入 embedding 是完整的，需要先 scatter 再喂给 Row。
    """
    @staticmethod
    def forward(ctx, x : Tensor, tp_group : ProcessGroup, split_dim : int = -1):
        # 沿 split_dim 切分 x，当前 rank 取第 tp_rank 片
        # 返回 x_partial
        world_size = dist.get_world_size(tp_group)
        tp_rank = dist.get_rank(tp_group)
        chunk_size = x.shape[split_dim]
        ctx.split_dim = split_dim
        ctx.world_size = world_size
        ctx.chunk_size = chunk_size
        ctx.tp_rank = tp_rank
        ctx.tp_group = tp_group
        x_partial = torch.narrow(input=x, 
                                 dim=split_dim,
                                 start=tp_rank * chunk_size, 
                                 length=chunk_size,).contiguous()
        return x_partial

    @staticmethod
    def backward(ctx, *grad_outputs : Tensor):
        # 在 tp_group 上沿 split_dim 做 all-gather
        # 返回 (grad_full, None, None)
        world_size = ctx.world_size
        tp_group = ctx.tp_group
        grad_output, *_ = grad_outputs
        if ctx.world_size == 1:
            return grad_output, None, None
        split_dim = ctx.split_dim
        if split_dim < 0:
            split_dim += grad_output.dim()

        # Fast path: split_dim == 0
        if split_dim == 0:
            grad = grad_output.contiguous()
            grad_all = torch.empty(
                (world_size * grad.shape[0], *grad.shape[1:]),
                dtype=grad.dtype, device=grad.device,
            )
            dist.all_gather_into_tensor(output_tensor=grad_all, input_tensor=grad, group=tp_group)
            return grad_all
        
        # Normal path: split_dim > 0，需要转置
        grad = grad_output.transpose(0, split_dim).contiguous()
        grad_all = torch.empty(
            (world_size * grad.shape[0], *grad.shape[1:]),
            dtype=grad.dtype, device=grad.device,
        )
        dist.all_gather_into_tensor(output_tensor=grad_all, input_tensor=grad, group=tp_group)
        return grad_all.transpose(0, split_dim).contiguous()