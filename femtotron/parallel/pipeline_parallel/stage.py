"""
femtotron/parallel/pipeline/stage.py

PipelineStage:single-rank execution unit for pipeline parallelism.

持有:
    - model (PipelineCausalLM protocol:is_first/is_last/forward(x, labels=None))
    - per-microbatch tensor state(keyed by mb_id)

不持有 / 不知道:
    - Schedule (action 顺序由 Runner 决定)
    - Comm (P2P 由 Runner 调用)
    - mb_id 的语义(只是 dict key)

Lifecycle for one microbatch on a non-first non-last stage:
    1. Runner: stage_input(mb_id, x_from_comm)
       Stage: detach + requires_grad_(True),cache in _inputs
    2. Runner: forward(mb_id)
       Stage: model(x) → cache output in _outputs
    3. Runner: get_output(mb_id) → send via comm
    4. Runner: stage_grad(mb_id, g_from_comm) → cache in _pending_grad_outputs
    5. Runner: backward(mb_id)
       Stage: pop input/output/grad,output.backward(grad),cache input.grad in _input_grads
    6. Runner: get_input_grad(mb_id) → send via comm

边界 stage:
    - First stage:跳过 step 1 的 requires_grad,跳过 step 5b/6(没有 input grad)
    - Last stage:forward 用 labels,backward 用内部 loss(没有 step 4)
    - pp_size=1:both first and last 同时成立

Rigor invariants:
    - 每个 cache entry 都有明确的 owner-pops-it lifecycle
    - 错误的调用顺序触发 RuntimeError(带 mb_id + method 信息)
    - assert_clean() 在 train_step 末尾检测残留,catch schedule bug
    - 无 silent fallback(没有 default=None 兜底)
"""

from __future__ import annotations

import torch
from torch import nn

from femtotron.parallel_context import ParallelContext


class PipelineStage:
    """Per-rank execution unit for PP. See module docstring for invariants."""

    def __init__(
        self,
        model: nn.Module,
        parallel_ctx: ParallelContext,
        *,
        loss_scale: float = 1.0,
    ) -> None:
        """
        Args:
            model: 必须实现 PipelineCausalLM protocol:
                - is_first: bool
                - is_last: bool
                - forward(x, labels=None) -> Tensor:
                    * non-last:        return hidden_states (preserves autograd graph)
                    * last + labels:   return scalar loss
                    * last + no labels: return logits
            parallel_ctx: 保留作 introspection / debug(核心逻辑不用)
            loss_scale: 在 last stage forward 完后,**对 loss 做 multiplicative scaling**
                再 backward。用于 grad accumulation 平衡:N 个 microbatch 时设 1/N,
                这样 N 次 backward 累加的 grad == grad(mean loss)。

                注意:这不是 fp16 dynamic loss scaling(bf16 不需要),
                只是 grad accumulation 的数学等价补偿。
        """
        # Protocol validation
        for attr in ("is_first", "is_last"):
            if not hasattr(model, attr):
                raise TypeError(
                    f"model must implement PipelineCausalLM protocol; "
                    f"missing attribute '{attr}'"
                )
        if not callable(getattr(model, "forward", None)):
            raise TypeError("model must have a callable .forward method")

        self.model = model
        self.parallel_ctx = parallel_ctx
        self.loss_scale = float(loss_scale)

        # is_first/is_last 在构造时锁定:Stage 假设这两个属性在生命周期内不变
        self.is_first: bool = bool(model.is_first)
        self.is_last: bool = bool(model.is_last)

        # Per-microbatch state (mb_id -> tensor)
        # 每个字典的 owner-pops-it 责任见各方法 docstring
        self._inputs: dict[int, torch.Tensor] = {}
        self._outputs: dict[int, torch.Tensor] = {}
        self._labels: dict[int, torch.Tensor] = {}
        self._pending_grad_outputs: dict[int, torch.Tensor] = {}
        self._input_grads: dict[int, torch.Tensor] = {}

        # Loss values (detached, no graph) for last stage's logging.
        # 由 pop_all_losses() 在 train_step 末尾批量取走。
        self._loss_values: dict[int, torch.Tensor] = {}

    # ════════════════════════════════════════════════════════════════
    # Input staging
    # ════════════════════════════════════════════════════════════════

    def stage_input(self, mb_id: int, x: torch.Tensor) -> None:
        """Cache input for forward.

        Non-first stage:
            - x 必须是 leaf tensor(grad_fn is None);comm.recv_forward 给的就是
            - x 必须是 float dtype(hidden_states)
            - Stage 通过 detach + requires_grad_(True) 构造一个新 leaf 缓存,
              **不修改 caller 的 x**(non-destructive)
        First stage:
            - x 通常是 LongTensor input_ids,不做 requires_grad 处理
        """
        if mb_id in self._inputs:
            raise RuntimeError(
                f"stage_input({mb_id}): input already staged; "
                f"schedule bug or missing reset()"
            )

        if not self.is_first:
            if x.grad_fn is not None:
                raise RuntimeError(
                    f"stage_input({mb_id}): non-first stage expects leaf tensor "
                    f"(grad_fn=None), got grad_fn={x.grad_fn}. "
                    f"Comm.recv_forward should return leaf tensors."
                )
            if not x.dtype.is_floating_point:
                raise TypeError(
                    f"stage_input({mb_id}): non-first stage expects float input "
                    f"(hidden_states), got dtype={x.dtype}"
                )
            # Non-destructive:新 leaf 共享 storage,caller 的 x 属性不变
            x = x.detach()
            x.requires_grad_(True)

        self._inputs[mb_id] = x

    def stage_labels(self, mb_id: int, labels: torch.Tensor) -> None:
        """Cache labels(last stage only)。
        
        Runner 在 train_step 开头给 last stage 准备所有 microbatch 的 labels,
        forward(mb_id) 时被 pop 消费。
        """
        if not self.is_last:
            raise RuntimeError(
                f"stage_labels({mb_id}): only valid on last stage "
                f"(is_last={self.is_last})"
            )
        if mb_id in self._labels:
            raise RuntimeError(
                f"stage_labels({mb_id}): labels already staged"
            )
        self._labels[mb_id] = labels

    # ════════════════════════════════════════════════════════════════
    # Forward
    # ════════════════════════════════════════════════════════════════

    def forward(self, mb_id: int) -> None:
        """Run model.forward;cache output in _outputs。

        Last stage with labels:
            output 是 scalar loss,**在 cache 前乘 loss_scale**
        Non-last stage:
            output 是 hidden_states(保留 autograd graph 用于 backward)

        Forward 成功完成后才修改 cache:如果 model.forward 抛异常,
        _inputs/_labels 不动,reset() 可以干净恢复。
        """
        if mb_id not in self._inputs:
            raise RuntimeError(
                f"forward({mb_id}): no input staged; call stage_input({mb_id}) first"
            )
        if mb_id in self._outputs:
            raise RuntimeError(
                f"forward({mb_id}): output already exists (duplicate forward?)"
            )

        x = self._inputs[mb_id]

        if self.is_last:
            if mb_id not in self._labels:
                raise RuntimeError(
                    f"forward({mb_id}): last stage requires labels; "
                    f"call stage_labels({mb_id}) first"
                )
            # peek,在 model 成功 forward 后才 pop(失败时 labels 不丢)
            labels = self._labels.pop(mb_id, None)
        else:
            labels = None

        # 跑 model。如果抛异常,下面 commit 不会发生。
        output_dict = self.model(x, labels=labels)

        if self.is_last:
            if labels is not None:
                # 训练模式:取 loss,按 loss_scale 缩放(guard 1.0 避免引入 MulBackward)
                loss = output_dict["loss"]
                if self.loss_scale != 1.0:
                    scaled = loss * self.loss_scale
                else:
                    scaled = loss
                self._outputs[mb_id] = scaled
            else:
                # 推理模式:存 logits,backward 不会被调用
                self._outputs[mb_id] = output_dict["logits"]
        else:
            # 中间/首段 stage:存 hidden_states,等接收 grad_output
            self._outputs[mb_id] = output_dict["hidden_states"]

    def get_output(self, mb_id: int) -> torch.Tensor:
        """Retrieve output for Runner's SendForward。**不 pop**(backward 时 pop)。"""
        if mb_id not in self._outputs:
            raise RuntimeError(
                f"get_output({mb_id}): no output cached; call forward({mb_id}) first"
            )
        return self._outputs[mb_id]

    # ════════════════════════════════════════════════════════════════
    # Backward
    # ════════════════════════════════════════════════════════════════

    def stage_grad(self, mb_id: int, grad: torch.Tensor) -> None:
        """Cache grad_output(from downstream stage)for backward。
        
        Last stage 不需要(backward 起点是内部 loss)。
        """
        if self.is_last:
            raise RuntimeError(
                f"stage_grad({mb_id}): last stage doesn't accept external grad "
                f"(backward starts from internal loss)"
            )
        if mb_id in self._pending_grad_outputs:
            raise RuntimeError(
                f"stage_grad({mb_id}): grad already staged"
            )
        self._pending_grad_outputs[mb_id] = grad

    def backward(self, mb_id: int) -> None:
        """Run backward。

        Pops: _outputs[mb_id], _inputs[mb_id], _pending_grad_outputs[mb_id](non-last)
        Pushes: _input_grads[mb_id](non-first), _loss_values[mb_id](last)
        Side effect: param.grad 累加,autograd graph 释放(retain_graph=False)
        """
        if mb_id not in self._outputs:
            raise RuntimeError(
                f"backward({mb_id}): no output; call forward({mb_id}) first"
            )
        if mb_id not in self._inputs:
            # Shouldn't happen if forward succeeded;防御性检查
            raise RuntimeError(
                f"backward({mb_id}): input missing but output exists; state corrupted"
            )

        output = self._outputs.pop(mb_id)
        input_tensor = self._inputs.pop(mb_id)

        if self.is_last:
            # 保留 loss value(detach 掉 graph)给 Runner 收集
            self._loss_values[mb_id] = output.detach()
            # Scalar loss:backward without grad_output arg
            output.backward()
        else:
            if mb_id not in self._pending_grad_outputs:
                raise RuntimeError(
                    f"backward({mb_id}): non-last stage requires grad_output; "
                    f"call stage_grad({mb_id}) first"
                )
            grad_output = self._pending_grad_outputs.pop(mb_id)
            output.backward(grad_output)

        # 提取 input.grad 给上游(non-first stage)
        if not self.is_first:
            if input_tensor.grad is None:
                raise RuntimeError(
                    f"backward({mb_id}): input.grad is None on non-first stage. "
                    f"Possible causes:\n"
                    f"  - model.forward 没真正用到 input(layer_range 错配?)\n"
                    f"  - input 不是 leaf 或 requires_grad=False(stage_input 哪里出错?)\n"
                    f"  - retain_graph 相关问题\n"
                    f"This is a correctness bug; upstream would receive None."
                )
            self._input_grads[mb_id] = input_tensor.grad
            # 显式释放 grad buffer 引用,允许 GC
            input_tensor.grad = None

    def get_input_grad(self, mb_id: int) -> torch.Tensor:
        """Retrieve input.grad for Runner's SendBackward。**Pop**。
        
        First stage 没有 upstream,调用是错误。
        """
        if self.is_first:
            raise RuntimeError(
                f"get_input_grad({mb_id}): first stage has no upstream"
            )
        if mb_id not in self._input_grads:
            raise RuntimeError(
                f"get_input_grad({mb_id}): no input_grad cached; "
                f"call backward({mb_id}) first"
            )
        return self._input_grads.pop(mb_id)

    # ════════════════════════════════════════════════════════════════
    # Loss collection (last stage only)
    # ════════════════════════════════════════════════════════════════

    def pop_all_losses(self) -> dict[int, torch.Tensor]:
        """Return all loss values and clear buffer。
        
        Returns:
            dict[mb_id, scalar_loss_tensor],non-last stage 返回空 dict。
            Tensor 是 detached(no graph),可以放心 .item() 或 mean。
        """
        losses = self._loss_values
        self._loss_values = {}
        return losses

    # ════════════════════════════════════════════════════════════════
    # State management
    # ════════════════════════════════════════════════════════════════

    def assert_clean(self) -> None:
        """Verify per-microbatch caches are empty。Runner 在 train_step 末尾调。
        
        非空 cache = schedule 漏了某个 action(e.g., 有 Forward 没 Backward)。
        是 schedule 正确性的最后防线。
        """
        leftover = []
        for name, d in [
            ("_inputs", self._inputs),
            ("_outputs", self._outputs),
            ("_labels", self._labels),
            ("_pending_grad_outputs", self._pending_grad_outputs),
            ("_input_grads", self._input_grads),
        ]:
            if d:
                leftover.append(f"  {name}: mb_ids={sorted(d.keys())}")

        if leftover:
            raise RuntimeError(
                "PipelineStage has leftover per-microbatch state:\n"
                + "\n".join(leftover)
                + "\nThis means some action was issued without its consumer "
                "(schedule bug)."
            )

    def reset(self) -> None:
        """Force-clear all per-microbatch state(包括 _loss_values)。
        
        防御性用途:start of train_step 兜底清理,或异常后恢复。
        """
        self._inputs.clear()
        self._outputs.clear()
        self._labels.clear()
        self._pending_grad_outputs.clear()
        self._input_grads.clear()
        self._loss_values.clear()