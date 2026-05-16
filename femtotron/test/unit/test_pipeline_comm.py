"""
femtotron/test/integration/test_pipeline_comm.py

跑法:
    torchrun --nproc_per_node=4 femtotron/test/integration/test_pipeline_comm.py
"""

from collections import OrderedDict
import os
import signal
import sys

import torch
import torch.distributed as dist

from femtotron.parallel_context import ParallelContext
from femtotron.parallel.pipeline_parallel.comm_ops import PipelineComm


# ────────────────────────────────────────────────────────────────
# 工具函数(对齐项目里其它脚本)
# ────────────────────────────────────────────────────────────────

def init_distributed():
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(
        backend="nccl",
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    torch.cuda.set_device(local_rank)
    return local_rank


def log(msg):
    if dist.get_rank() == 0:
        print(msg, flush=True)

def get_device():
    return torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")

def make_comm(parallel_ctx, dtype=torch.bfloat16):
    return PipelineComm(
        parallel_ctx=parallel_ctx,
        microbatch_size=2,
        seqlen=8,
        hidden_size=16,
        dtype=dtype,
    )


# ────────────────────────────────────────────────────────────────
# Test 1: 单向 round-trip(forward 方向 + backward 方向)
# ────────────────────────────────────────────────────────────────

def test_round_trip(ctx, comm):
    log("Test 1: round-trip both directions")

    # Forward direction: rank0 → rank1 → ... → rank(N-1)
    if comm.is_first_stage:
        sent = torch.full(comm.act_shape, 42.0, dtype=comm.dtype, device="cuda")
        comm.send_forward(sent)
    else:
        recv = comm.recv_forward()
        assert torch.all(recv == 42.0), \
            f"rank{ctx.pp_rank} fwd got wrong: {recv.flatten()[:5]}"
        if not comm.is_last_stage:
            comm.send_forward(recv)

    # Backward direction: rank(N-1) → ... → rank0
    if comm.is_last_stage:
        sent_back = torch.full(comm.act_shape, 99.0, dtype=comm.dtype, device="cuda")
        comm.send_backward(sent_back)
    else:
        recv = comm.recv_backward()
        assert torch.all(recv == 99.0), \
            f"rank{ctx.pp_rank} bwd got wrong: {recv.flatten()[:5]}"
        if not comm.is_first_stage:
            comm.send_backward(recv)

    dist.barrier()
    log("  ✓ passed")


# ────────────────────────────────────────────────────────────────
# Test 2: 死锁压力测试(combined ops)
# ────────────────────────────────────────────────────────────────

def test_no_deadlock_combined(ctx, comm):
    log("Test 2: deadlock pressure with combined ops")

    # 每个 rank 用自己的值作 sentinel,方便对端验证
    my_val = float(ctx.pp_rank * 10 + 1)
    payload = torch.full(comm.act_shape, my_val, dtype=comm.dtype, device="cuda")

    if comm.is_first_stage:
        # First: 只跟下游通(combined)
        recv_bwd = comm.send_forward_recv_backward(payload)
        expected = float((ctx.pp_rank + 1) * 10 + 1)
        assert torch.all(recv_bwd == expected), \
            f"first stage got {recv_bwd.flatten()[:3]}, expected {expected}"
    elif comm.is_last_stage:
        # Last: 只跟上游通
        recv_fwd = comm.recv_forward()
        comm.send_backward(payload)
        expected = float((ctx.pp_rank - 1) * 10 + 1)
        assert torch.all(recv_fwd == expected), \
            f"last stage got {recv_fwd.flatten()[:3]}, expected {expected}"
    else:
        # Mid: 同时跟上下游通(两个 combined call)
        recv_fwd = comm.send_backward_recv_forward(payload)
        recv_bwd = comm.send_forward_recv_backward(payload)
        expected_fwd = float((ctx.pp_rank - 1) * 10 + 1)
        expected_bwd = float((ctx.pp_rank + 1) * 10 + 1)
        assert torch.all(recv_fwd == expected_fwd), \
            f"mid rank{ctx.pp_rank} fwd got wrong"
        assert torch.all(recv_bwd == expected_bwd), \
            f"mid rank{ctx.pp_rank} bwd got wrong"

    dist.barrier()
    log("  ✓ passed")


# ────────────────────────────────────────────────────────────────
# Test 3: 边界 stage 行为
# ────────────────────────────────────────────────────────────────

def test_edge_stage_noop(ctx, comm):
    log("Test 3: edge stage no-op behavior")

    dummy = torch.zeros(comm.act_shape, dtype=comm.dtype, device="cuda")

    if comm.is_first_stage:
        assert comm.recv_forward() is None
        comm.send_backward(dummy)  # no-op,不应 crash
        assert comm.send_backward_recv_forward(dummy) is None

    if comm.is_last_stage:
        assert comm.recv_backward() is None
        comm.send_forward(dummy)
        assert comm.send_forward_recv_backward(dummy) is None

    dist.barrier()
    log("  ✓ passed")


# ────────────────────────────────────────────────────────────────
# Test 4: buffer 校验
# ────────────────────────────────────────────────────────────────

def test_buffer_validation(ctx, comm):
    log("Test 4: buffer validation")

    if not comm.is_first_stage:
        # Wrong shape
        bad_shape = torch.empty(
            (comm.act_shape[0], comm.act_shape[1], comm.act_shape[2] + 1),
            dtype=comm.dtype, device="cuda",
        )
        try:
            comm.recv_forward(out=bad_shape)
            assert False, "expected ValueError on wrong shape"
        except ValueError as e:
            assert "shape" in str(e)

        # Wrong dtype
        bad_dtype = torch.empty(comm.act_shape, dtype=torch.float32, device="cuda")
        try:
            comm.recv_forward(out=bad_dtype)
            assert False, "expected ValueError on wrong dtype"
        except ValueError as e:
            assert "dtype" in str(e)

        # CPU buffer
        bad_device = torch.empty(comm.act_shape, dtype=comm.dtype, device="cpu")
        try:
            comm.recv_forward(out=bad_device)
            assert False, "expected ValueError on CPU buffer"
        except ValueError as e:
            assert "cuda" in str(e).lower() or "CUDA" in str(e)

    dist.barrier()
    log("  ✓ passed")


# ────────────────────────────────────────────────────────────────
# Test 5: 不同 dtype(bf16 / fp32 都要能跑通)
# ────────────────────────────────────────────────────────────────

def test_different_dtypes(ctx):
    log("Test 5: different dtypes (bf16 / fp32)")

    for dtype in (torch.bfloat16, torch.float32):
        comm = make_comm(ctx, dtype=dtype)

        if comm.is_first_stage:
            sent = torch.full(comm.act_shape, 7.0, dtype=dtype, device="cuda")
            comm.send_forward(sent)
        else:
            recv = comm.recv_forward()
            assert recv.dtype == dtype
            assert torch.all(recv == 7.0)
            if not comm.is_last_stage:
                comm.send_forward(recv)

        dist.barrier()

    log("  ✓ passed")


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────

def main():
    # 死锁兜底:60s 没跑完就自爆,免得 CI 挂死
    def timeout_handler(signum, frame):
        rank = int(os.environ.get("RANK", 0))
        print(f"[rank{rank}] TIMEOUT - likely deadlock!", flush=True)
        sys.exit(1)

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(60)

    init_distributed()
    world_size = dist.get_world_size()
    print(f"Initialized distributed with world size {world_size}")
    device = get_device()

    parallel_ctx = ParallelContext(OrderedDict([("pp", world_size)]))
    comm = make_comm(parallel_ctx)
    rank = dist.get_rank()
    print(
        f"[rank{rank}] pp_size={parallel_ctx.pp_size} "
        f"pp_rank={parallel_ctx.pp_rank} "
        f"pp_prev_rank={parallel_ctx.pp_prev_rank} "
        f"pp_next_rank={parallel_ctx.pp_next_rank} "
        f"is_first={comm.is_first_stage} "
        f"is_last={comm.is_last_stage}",
        flush=True,
    )
    dist.barrier()

    try:
        test_round_trip(parallel_ctx, comm)
        test_no_deadlock_combined(parallel_ctx, comm)
        test_edge_stage_noop(parallel_ctx, comm)
        test_buffer_validation(parallel_ctx, comm)
        test_different_dtypes(parallel_ctx)

        signal.alarm(0)
        log("\n[All PipelineComm tests passed ✓]")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()