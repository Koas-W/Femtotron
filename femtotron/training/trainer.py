import os, sys
import time
import random
import json
import numpy as np
import torch
from torch import nn, Tensor
from torch import distributed as dist
from torch.distributed import ProcessGroup
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from safetensors import safe_open
from typing import cast
from torch.optim.lr_scheduler import LRScheduler
from contextlib import contextmanager, nullcontext


from femtotron.parallel_context import ParallelContext
from femtotron.model.parallel_plan import ParallelPlan
from femtotron.training.train_config import TrainConfig
from femtotron.training.mixed_precision_manager import MixedPrecisionManager
from femtotron.data.data_loader import DistributedDataLoader
from femtotron.parallel.data_parallel.gradient_synchronizer import GradientSynchronizer, create_grad_synchronizer
from femtotron.training.grad_accumulator import GradAccumulator
from femtotron.training.grad_transform import GradTransform

class Trainer:
    """
    训练循环的编排器。
    
    职责：
    1. 编排一个完整的训练循环：data → forward → backward → grad sync → optimizer step
    2. 管理 epoch、step 计数
    3. Logging（loss、throughput、grad norm、lr）
    4. Checkpoint save/load
    5. LR schedule 管理
    
    不负责的事：
    - 模型的创建和并行化（在 Trainer 外部完成）
    - 混合精度的内部逻辑（由 MixedPrecisionManager 管理）
    - 数据的预处理（由 DistributedDataLoader 管理）
    """
    
    def __init__(self,
                 model: nn.Module,
                 mp_manager: MixedPrecisionManager,
                 scheduler: LRScheduler,
                 dataloader: DistributedDataLoader,
                 grad_sync: GradientSynchronizer,
                 parallel_ctx: ParallelContext,
                 train_config: TrainConfig):
        """
        config 包含：
        - train_steps: 总训练步数
        - lr, betas, eps: optimizer 超参（已在 mp_manager 中配置）
        - warmup_steps: LR warmup 步数
        - grad_clip: 梯度裁剪阈值
        - log_interval: 每隔多少步打印日志
        - checkpoint_interval: 每隔多少步保存 checkpoint
        - checkpoint_dir: checkpoint 保存路径
        """
        self.model = model
        self.mp_manager = mp_manager   # MixedPrecisionManager
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.parallel_ctx = parallel_ctx
        self.grad_sync = grad_sync
        self.train_config = train_config

        # 训练状态
        self.global_step = 0
        self.epoch = 0
        
        # 用于 throughput 计算
        self._last_log_time: float | None = None
        self._last_log_step: int | None = None
        
        self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
    
    def train(self):
        """
        主训练循环。运行训练直到达到 train_steps。
        
        伪代码：
        for epoch in range(num_epochs):
            dataloader.set_epoch(epoch)
            for step, batch in enumerate(dataloader):
                
                # Forward
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(input_ids=batch["input_ids"],
                                labels=batch["labels"])
                loss = outputs["loss"]
                
                # Backward
                loss.backward()
                
                # DP 梯度同步
                grad_sync.sync_gradients()
                
                # Optimizer step（内部含 copy_grads → clip → step → sync → zero_grad）
                mp_manager.step()
                
                # LR schedule
                scheduler.step()
                
                # Logging
                if global_step % config.log_interval == 0:
                    self._log_step(global_step, loss, grad_norm, lr, ...)
                
                # Checkpoint
                if global_step % config.checkpoint_interval == 0:
                    self._save_checkpoint(global_step)
                
                global_step += 1
                if global_step >= config.train_steps:
                    return
        """
        while self.global_step < self.train_config.train_steps:
            self.dataloader.set_epoch(self.epoch)
            
            data_iter = iter(self.dataloader)
            while self.global_step < self.train_config.train_steps:
                try:
                    info = self._train_one_step(data_iter)
                except StopIteration:
                    break    # 当前 epoch 结束
                
                self.global_step += 1
                
                # Logging
                if self.global_step % self.train_config.log_interval == 0:
                    self._log_step(info)
                
                # Checkpoint
                if self.global_step % self.train_config.checkpoint_interval == 0:
                    self._save_checkpoint()
            
            self.epoch += 1
    
    def _train_one_step(self, data_iter) -> dict:
        """完成一个 optimizer step（含 grad_accum_steps 个 micro batch）。"""
        n = self.train_config.grad_accum_steps
        total_loss = torch.zeros((), device=self.device)
        
        for micro_step in range(n):
            batch = next(data_iter)   # 可能 StopIteration
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            
            is_last = (micro_step == n - 1)
            sync_ctx = nullcontext() if is_last else self.grad_sync.no_sync()
            
            with sync_ctx:
                outputs = self.model(input_ids=batch["input_ids"], labels=batch["labels"])
                loss = outputs["loss"] / n   # 缩放，让累加结果是 micro 平均
                loss.backward()
                total_loss += loss.detach().float() * n   # 还原成原始 loss
        
        # 梯度同步（DP）
        self.grad_sync.sync_gradients()
        
        # Optimizer step（含 grad clip、master update、bf16 sync 等）
        step_info = self.mp_manager.step()
        
        # LR schedule
        self.scheduler.step()
        
        return {
            "loss": (total_loss / n).item(),
            # "grad_norm": step_info.get("grad_norm"),
            "lr": self.scheduler.get_last_lr()[0],
            # "successful": step_info.get("successful", True),
            "successful": step_info,
        }
    
    # ──────────────────────────────────────────
    # Logging
    # ──────────────────────────────────────────
    def _log_step(self, info: dict):
        """
        只在 rank 0 打印日志。
        
        打印内容：
        - step / total_steps
        - loss（当前步）
        - grad norm（clip 前）
        - learning rate
        - throughput: tokens/sec/gpu = tokens_per_step / elapsed / num_gpus
        - 显存峰值: torch.cuda.max_memory_allocated()
        """
        if dist.get_rank() != 0:
            return
        
        # 算 throughput
        now = time.perf_counter()
        if self._last_log_time is not None and self._last_log_step is not None:
            elapsed = now - self._last_log_time
            steps_done = self.global_step - self._last_log_step
            tokens_per_step_per_gpu = (
                self.dataloader.micro_batch_size
                * self.dataloader.dataset.seq_len   # dataset 暴露 seq_len
                * self.train_config.grad_accum_steps
            )
            tps_per_gpu = tokens_per_step_per_gpu * steps_done / elapsed
        else:
            tps_per_gpu = float("nan")
        
        self._last_log_time = now
        self._last_log_step = self.global_step
        
        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        
        msg = (
            f"step {self.global_step:>6}/{self.train_config.train_steps} | "
            f"loss {info['loss']:7.4f} | "
            f"grad_norm {info['grad_norm']:6.3f} | "
            f"lr {info['lr']:.2e} | "
            f"tps/gpu {tps_per_gpu:>7.0f} | "
            f"mem {peak_mem_gb:.1f}GB"
        )
        if not info["successful"]:
            msg += "  [SKIPPED: grad overflow]"
        print(msg, flush=True)
        
    # ──────────────────────────────────────────
    # Checkpoint
    # ──────────────────────────────────────────
    def _save_checkpoint(self):
        """
        保存 checkpoint。
        
        需要保存的状态：
        - model state_dict（BF16 参数）
        - optimizer state_dict（FP32 master weights + Adam states）
        - scheduler state_dict
        - 当前 step 数
        - 当前 epoch 数
        - 数据加载器的位置（用于恢复）
        - config（用于重建训练环境）
        - RNG状态（用于复现训练轨迹）
        
        注意：在 TP 下，每个 rank 的参数不同（切分后的），
        所以每个 rank 各自保存自己的 checkpoint 文件。
        文件名包含 rank 信息：checkpoint_step{step}_rank{rank}.pt
        
        只在 rank 0 保存额外的元信息文件（config、parallel 配置等）。
        """
        ckpt_dir = os.path.join(
            self.train_config.checkpoint_dir, f"step_{self.global_step}"
        )
        ctx = self.parallel_ctx
        
        # 同步：所有 rank 进入 save 阶段
        if dist.is_initialized():
            dist.barrier()
        
        # rank 0 创建目录 + 写 meta
        if dist.get_rank() == 0:
            os.makedirs(ckpt_dir, exist_ok=True)
            meta = {
                "step": self.global_step,
                "epoch": self.epoch,
                "tp_size": ctx.tp_size,
                "dp_size": ctx.dp_size,
                "config": self.train_config.__dict__,
            }
            with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2, default=str)
        
        # 等 rank 0 创建好目录
        if dist.is_initialized():
            dist.barrier()
        
        # 每个 (tp, dp) 各自保存 model + optimizer
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.mp_manager.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "rng": self._rng_state_dict(),
            },
            os.path.join(ckpt_dir, f"shard_tp{ctx.tp_rank}_dp{ctx.dp_rank}.pt"),
        )
        
        # 每个 dp_rank 的 tp_rank=0 保存 dataloader
        if ctx.tp_rank == 0:
            torch.save(
                self.dataloader.state_dict(),
                os.path.join(ckpt_dir, f"dataloader_dp{ctx.dp_rank}.pt"),
            )
        
        # 等所有 rank 写完
        if dist.is_initialized():
            dist.barrier()
        
        if dist.get_rank() == 0:
            print(f"Saved checkpoint to {ckpt_dir}")
    
    def _load_checkpoint(self, ckpt_dir: str):
        """
        从 checkpoint 恢复训练。在 train() 之前调用。
        
        加载的逻辑是 _save_checkpoint 的逆过程：
        - 每个 rank 加载自己的 checkpoint 文件
        - 恢复 model、optimizer、scheduler 的 state
        - 恢复 step 计数和 dataloader 位置
        """
        ctx = self.parallel_ctx
        
        # 一致性检查
        with open(os.path.join(ckpt_dir, "meta.json")) as f:
            meta = json.load(f)
        assert meta["tp_size"] == ctx.tp_size, (
            f"tp_size mismatch: ckpt={meta['tp_size']}, current={ctx.tp_size}"
        )
        assert meta["dp_size"] == ctx.dp_size, (
            f"dp_size mismatch: ckpt={meta['dp_size']}, current={ctx.dp_size}"
        )
        
        shard = torch.load(
            os.path.join(ckpt_dir, f"shard_tp{ctx.tp_rank}_dp{ctx.dp_rank}.pt"),
            map_location=self.device,
        )
        self.model.load_state_dict(shard["model"])
        self.mp_manager.load_state_dict(shard["optimizer"])
        self.scheduler.load_state_dict(shard["scheduler"])
        self._load_rng_state(shard["rng"])
        
        if ctx.tp_rank == 0:
            dl_state = torch.load(
                os.path.join(ckpt_dir, f"dataloader_dp{ctx.dp_rank}.pt")
            )
            self.dataloader.load_state_dict(dl_state)
        # TP rank > 0 的 dataloader 状态从 broadcast 同步——这里简化：每个 tp rank 自己也读一遍
        # 实际上 dataloader 状态在 dp group 内一致即可，但每个 tp rank 都需要它
        # 所以让所有 rank 都读：
        # 上面 if ctx.tp_rank == 0 改成所有 rank 读
        
        self.global_step = meta["step"]
        self.epoch = meta["epoch"]
        
        if dist.get_rank() == 0:
            print(f"Resumed from {ckpt_dir} at step {self.global_step}")
    
    # ──────────────────────────────────────────
    # RNG 辅助
    # ──────────────────────────────────────────
    
    def _rng_state_dict(self) -> dict:
        return {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }
    
    def _load_rng_state(self, state: dict):
        torch.set_rng_state(state["torch"])
        torch.cuda.set_rng_state(state["cuda"])
        np.random.set_state(state["numpy"])
        random.setstate(state["python"])