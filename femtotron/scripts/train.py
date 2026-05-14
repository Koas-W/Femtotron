"""
Femtotron 训练入口脚本
======================

用法：
    # 8卡 DP+TP 训练
    torchrun --nproc_per_node=8 scripts/train.py --config configs/tiny_llama_debug.yaml

    # 2卡纯 DP
    torchrun --nproc_per_node=2 scripts/train.py --config configs/tiny_llama_debug.yaml --dp 2 --tp 1

    # 2卡纯 TP
    torchrun --nproc_per_node=2 scripts/train.py --config configs/tiny_llama_debug.yaml --dp 1 --tp 2

    # 覆盖任意配置
    torchrun --nproc_per_node=8 scripts/train.py --config configs/tiny_llama_debug.yaml --train_steps 1000 --lr 1e-4
"""

import os
import sys
import argparse
import yaml
import time
import torch
import torch.distributed as dist
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from pathlib import Path

from femtotron.sharding.factory import create_sharding_strategy

# 确保项目根目录在 path 上
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from femtotron.parallel_context import ParallelContext
from femtotron.model.parallel_plan import get_llama_parallel_plan
from femtotron.model.model_loader import ModelLoader
from femtotron.model.llama import build_llama_model
from femtotron.training.mixed_precision_manager import MixedPrecisionManager
from femtotron.training.optimizer import get_param_groups
from femtotron.training.lr_schedule import create_lr_schedule
from femtotron.training.trainer import Trainer
from femtotron.training.train_config import TrainConfig
from femtotron.parallel.data_parallel.ddp import DataParallelGradSync
from femtotron.parallel.data_parallel.gradient_synchronizer import GradientSynchronizer
from femtotron.data.data_loader import DistributedDataLoader
from femtotron.data.preprocess import preprocess
from femtotron.data.collator import Collator, simple_pretrain_collator, PadSftCollator

def init_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def log(msg):
    if dist.get_rank() == 0:
        print(msg)


def parse_args():
    parser = argparse.ArgumentParser(description="Femtotron Training")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML 配置文件路径")

    # 并行配置（CLI 覆盖 YAML）
    parser.add_argument("--dp", type=int, default=None)
    parser.add_argument("--tp", type=int, default=None)
    parser.add_argument("--pp", type=int, default=None)

    # ZeRO 配置
    parser.add_argument("--zero_stage", type=int, default=None,
                        help="ZeRO stage: 0=baseline / 1 / 2 / 3")
    parser.add_argument("--zero_wrap_policy", type=str, default=None,
                        help="wrap policy preset name (only for stage 3)")

    # AC 配置  
    parser.add_argument("--ac_enabled", type=lambda x: x.lower() in ('true', '1', 'yes'),
                        default=None, help="启用 activation checkpointing")
    parser.add_argument("--ac_policy", type=str, default=None,
                    help="AC policy preset name")
    
    # 模型配置
    parser.add_argument("--model_name", type=str, default=None,
                        help="HuggingFace 模型名或路径")
    parser.add_argument("--num_hidden_layers", type=int, default=None,
                        help="覆盖层数（调试用小模型）")

    # 训练超参
    parser.add_argument("--train_steps", type=int, default=None)
    parser.add_argument("--micro_batch_size", type=int, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)

    # 数据
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=None)

    # Logging & Checkpoint
    parser.add_argument("--log_interval", type=int, default=None)
    parser.add_argument("--checkpoint_interval", type=int, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="从 checkpoint 恢复训练")

    return parser.parse_args()


def load_config(args) -> dict:
    """加载 YAML 配置，CLI 参数覆盖 YAML 中的值。"""
    config = {}

    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}

    # CLI 参数覆盖 YAML（只覆盖非 None 的值）
    cli_overrides = {k: v for k, v in vars(args).items()
                     if v is not None and k != "config"}
    
    for key, value in cli_overrides.items():
        # 嵌套 key 用 flat 方式处理
        config[key] = value

    return config


def build_all(config: dict):
    """
    根据配置构建所有组件。
    返回训练所需的全部对象。
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")

    # ─── 并行配置 ───
    tp_size = config.get("tp", 1)
    dp_size = config.get("dp", world_size // tp_size)
    pp_size = config.get("pp", 1)

    assert dp_size * tp_size * pp_size == world_size, \
        f"dp({dp_size}) * tp({tp_size}) * pp({pp_size}) = {dp_size*tp_size*pp_size} != world_size({world_size})"

    parallel_ctx = ParallelContext(OrderedDict([
        ("pp", pp_size),
        ("dp", dp_size),
        ("tp", tp_size),
    ]))

    log(f"并行配置: DP={dp_size}, TP={tp_size}, PP={pp_size}")

    # ─── 模型 ───
    model_name = config.get("model_name", None)
    plan = get_llama_parallel_plan()

    if model_name:
        tokenizer_name = model_name
        # 从 HuggingFace 加载真实模型
        from transformers import AutoConfig
        from transformers.models.llama.configuration_llama import LlamaConfig
        model_config = AutoConfig.from_pretrained(model_name)
        if config.get("num_hidden_layers"):
            model_config.num_hidden_layers = config["num_hidden_layers"]

        log(f"模型: {model_name}")
        log(f"  层数: {model_config.num_hidden_layers}")
        log(f"  Hidden: {model_config.hidden_size}")
        log(f"  Heads: {model_config.num_attention_heads}")
        log(f"  Vocab: {model_config.vocab_size}")

        with torch.device("meta"):
            model = build_llama_model(model_config, parallel_ctx)

        loader = ModelLoader(parallel_ctx)
        loader.load_and_distribute(model, model_name, parallel_plan=plan, device=device)
    else:
        # 使用 tiny 模型（调试用）
        from transformers import AutoConfig
        from transformers import AutoTokenizer
        tokenizer_name = config.get("tokenizer", "meta-llama/Llama-2-7b-hf")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        log(f"Tokenizer: {tokenizer_name} (vocab_size={tokenizer.vocab_size})")
        model_config = AutoConfig.for_model(
            "llama",
            hidden_size=config.get("hidden_size", 256),
            intermediate_size=config.get("intermediate_size", 512),
            num_attention_heads=config.get("num_attention_heads", 8),
            num_key_value_heads=config.get("num_key_value_heads", 4),
            num_hidden_layers=config.get("num_hidden_layers", 4),
            max_position_embeddings=config.get("max_position_embeddings", 512),
            vocab_size=tokenizer.vocab_size,  # ← 从 tokenizer 推导
            rms_norm_eps=1e-5,
            hidden_act="silu",
            tie_word_embeddings=config.get("tie_word_embeddings", False),
        )

        log(f"模型: Tiny LLaMA (随机初始化)")
        log(f"  层数: {model_config.num_hidden_layers}, Hidden: {model_config.hidden_size}")

        with torch.device("meta"):
            model = build_llama_model(model_config, parallel_ctx)

        # 随机初始化（不加载预训练权重）
        # 需要把 meta tensor 物化为真实 tensor
        model = model.to_empty(device=device)
        for p in model.parameters():
            if p.requires_grad:
                nn_init_scale = config.get("init_std", 0.02)
                torch.nn.init.normal_(p, mean=0.0, std=nn_init_scale)

    model = model.bfloat16()
    num_params = sum(p.numel() for p in model.parameters())
    log(f"  参数量（本 rank）: {num_params:,}")

    # ─── 训练配置 ───
    train_config = TrainConfig(
        master_dtype=torch.float32,
        grad_clip=config.get("grad_clip", 1.0),
        train_steps=config.get("train_steps", 500),
        log_interval=config.get("log_interval", 10),
        checkpoint_interval=config.get("checkpoint_interval", 100),
        checkpoint_dir=config.get("checkpoint_dir", "./checkpoints"),
        warmup_steps=config.get("warmup_steps", 50),
        min_lr_ratio=config.get("min_lr_ratio", 0.1),
    )

    # ─── 混合精度 + Optimizer ───
    lr = config.get("lr", 3e-4)
    weight_decay = config.get("weight_decay", 0.01)
    compute_param_groups = get_param_groups(model, weight_decay=weight_decay)

    # ─── 新增:ZeRO strategy ───
    from femtotron.sharding.factory import create_sharding_strategy
    from femtotron.parallel.data_parallel.gradient_synchronizer import create_grad_synchronizer
    from femtotron.sharding.zero_config import ZeROConfig
    from femtotron.scripts.presets import get_wrap_policy, get_ac_policy
    
    zero_stage = config.get("zero_stage", 0)
    if zero_stage == 3:
        wrap_policy_name = config.get("zero_wrap_policy")
        if not wrap_policy_name:
            raise ValueError("zero_stage=3 但没指定 zero_wrap_policy")
        wrap_policy = get_wrap_policy(wrap_policy_name)
    else:
        wrap_policy = None
    
    zero_config = ZeROConfig(
        stage=zero_stage,
        wrap_policy=wrap_policy,
    )
    strategy = create_sharding_strategy(parallel_ctx, zero_config)
    
    log(f"ZeRO 配置: stage={zero_stage}, strategy={type(strategy).__name__}")
    
    # ─── MixedPrecisionManager 构造 ───
    mp_manager = MixedPrecisionManager(
        model=model,
        sharding_strategy=strategy,
        parallel_ctx=parallel_ctx,
        parallel_plan=plan,
        config=train_config,
        inner_optimizer_cls=torch.optim.AdamW,
        inner_optimizer_kwargs={
            "lr": lr,
            "betas": (config.get("adam_beta1", 0.9), config.get("adam_beta2", 0.95)),
            "eps": config.get("adam_eps", 1e-8),
        },
        compute_param_groups=compute_param_groups,
    )
    
    # ─── 应用 Activation Checkpointing ───
    if config.get("ac_enabled", False):
        from femtotron.training.activation_ckpt import (
            apply_activation_checkpointing
        )
        ac_policy_name = config.get("ac_policy")
        if not ac_policy_name:
            raise ValueError("ac_enabled=True 但没指定 ac_policy")
        
        ac_policy = get_ac_policy(ac_policy_name)   # 此处类型确定是 Callable
        n_wrapped = apply_activation_checkpointing(
            model,
            ac_policy,
            use_reentrant=config.get("ac_use_reentrant", False),
            preserve_rng_state=config.get("ac_preserve_rng_state", False),
        )
        log(f"  Activation checkpointing: wrapped {n_wrapped} modules")
    else:
        log(f"  Activation checkpointing: disabled")
    

    log(f"训练配置:")
    log(f"  LR: {lr}, Weight Decay: {weight_decay}")
    log(f"  Warmup: {train_config.warmup_steps} steps")
    log(f"  Grad Clip: {train_config.grad_clip}")
    log(f"  精度: BF16 compute + FP32 master")

    # ─── LR Schedule ───
    scheduler = create_lr_schedule(
        mp_manager.inner,  # optimizer
        warmup_steps=train_config.warmup_steps,
        total_steps=train_config.train_steps,
        min_lr_ratio=train_config.min_lr_ratio,
    )

    # ─── 数据 ───
    seq_len = config.get("seq_len", 128)
    micro_batch_size = config.get("micro_batch_size", 4)
    dataset_name = config.get("dataset", "roneneldan/TinyStories")
    if model_name:
        tokenizer_name = model_name
    else:
        # tiny debug 模型：用一个公开的 tokenizer，并让 vocab_size 跟它对齐
        tokenizer_name = config.get("tokenizer", "meta-llama/Llama-2-7b-hf")
    data_dir = config.get("data_dir", "./data")
    
    # 构造缓存路径：包含数据集名和 seq_len，避免混淆
    safe_name = dataset_name.replace("/", "_")
    safe_tok = tokenizer_name.replace("/", "_")
    cache_path = os.path.join(data_dir, f"{safe_name}_{safe_tok}_seqlen{seq_len}.pt")
    
    # 只在 rank 0 做预处理，其他 rank 等待
    if dist.get_rank() == 0:
        if os.path.exists(cache_path):
            log(f"数据: 使用缓存 {cache_path}")
        else:
            log(f"数据: 缓存不存在，开始预处理...")
            os.makedirs(data_dir, exist_ok=True)
            preprocess(
                dataset_name=dataset_name,
                tokenizer_name=tokenizer_name,
                output_path=cache_path,
                seq_len=seq_len,
            )
            log(f"数据: 预处理完成")
    
    dist.barrier()  # 等 rank 0 处理完
    
    train_data = torch.load(cache_path, weights_only=True)
    log(f"数据: {train_data.shape[0]} 条样本, seq_len={train_data.shape[1]}")

    dataloader = DistributedDataLoader(
        dataset=train_data,
        parallel_ctx=parallel_ctx,
        micro_batch_size=micro_batch_size,
        collator=simple_pretrain_collator,  # 定长输入，直接 stack
        sampler=None,  # 使用默认的DistributedSampler
        num_workers=config.get("num_workers", 2),
    )

    tokens_per_step = micro_batch_size * seq_len * dp_size
    log(f"  Tokens/step (全局): {tokens_per_step:,}")

    # ─── DDP 梯度同步 用 factory 构造 grad_sync ───
    grad_sync = create_grad_synchronizer(
        mp_manager.groups, parallel_ctx, strategy
    )
    log(f"  Grad sync: {type(grad_sync).__name__}")

    # ─── Trainer ───
    trainer = Trainer(
        model=model,
        mp_manager=mp_manager,
        scheduler=scheduler,
        dataloader=dataloader,
        grad_sync=grad_sync,
        parallel_ctx=parallel_ctx,
        train_config=train_config,
    )

    return trainer


def main():
    args = parse_args()
    config = load_config(args)

    local_rank = init_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print("=" * 60)
        print("  Femtotron Training")
        print(f"  World size: {world_size}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print("=" * 60)

    dist.barrier()

    # 构建所有组件
    trainer = build_all(config)

    # 从 checkpoint 恢复
    if args.resume:
        log(f"\n从 checkpoint 恢复: {args.resume}")
        trainer._load_checkpoint(args.resume)

    # 开始训练
    log(f"\n{'=' * 60}")
    log(f"  开始训练")
    log(f"{'=' * 60}\n")

    start_time = time.time()
    trainer.train()
    total_time = time.time() - start_time

    if rank == 0:
        print(f"\n{'=' * 60}")
        print(f"  训练完成")
        print(f"  总耗时: {total_time:.1f}s ({total_time/60:.1f}min)")
        print(f"{'=' * 60}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()