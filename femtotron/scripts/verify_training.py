"""
Femtotron 训练结果验证
======================

用法：
    torchrun --nproc_per_node=2 scripts/verify_training.py \
        --config configs/tiny_llama_debug.yaml \
        --checkpoint ./checkpoints/tiny_debug/step_500

验证内容：
    1. 加载 checkpoint，生成文本（应该是基本连贯的英文）
    2. 随机初始化模型，生成文本（应该是乱码）
    3. 对比两者，确认训练确实让模型学到了语言模式
    4. 计算 validation loss，确认和训练末期 loss 一致
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn.functional as F
import torch.distributed as dist
from collections import OrderedDict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from femtotron.parallel_context import ParallelContext
from femtotron.model.parallel_plan import get_llama_parallel_plan
from femtotron.model.model_loader import ModelLoader
from femtotron.model.llama import build_llama_model
from femtotron.training.train_config import TrainConfig


def init_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def log(msg):
    if dist.get_rank() == 0:
        print(msg)


@torch.no_grad()
def generate(model, tokenizer, prompt, device, max_new_tokens=80, temperature=0.8):
    """简单的自回归生成。"""
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        outputs = model(input_ids=generated)
        logits = outputs["logits"][:, -1, :].float()

        # 只在 rank 0 做采样
        if dist.get_rank() == 0:
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            next_token = torch.zeros(1, 1, dtype=torch.long, device=device)

        # 广播给所有 rank，保证生成路径一致
        dist.broadcast(next_token, src=0)

        generated = torch.cat([generated, next_token], dim=-1)

        if next_token.item() == tokenizer.eos_token_id:
            break

        # 防止超过 max_position_embeddings
        if generated.shape[1] >= 512:
            break

    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return text


@torch.no_grad()
def compute_val_loss(model, data_path, device, num_batches=20, batch_size=4):
    """在验证数据上计算 loss。"""
    model.eval()
    data = torch.load(data_path, weights_only=True, mmap=True)

    # 取最后一部分作为 validation（训练用的是前面的）
    val_start = max(0, len(data) - num_batches * batch_size)
    val_data = data[val_start:]

    total_loss = 0.0
    count = 0

    for i in range(0, len(val_data) - batch_size, batch_size):
        batch = val_data[i:i+batch_size].to(device)
        outputs = model(input_ids=batch, labels=batch)
        total_loss += outputs["loss"].float().item()
        count += 1
        if count >= num_batches:
            break

    return total_loss / max(count, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    world_size = dist.get_world_size()
    tp_size = config.get("tp", 1)
    dp_size = config.get("dp", world_size // tp_size)

    parallel_ctx = ParallelContext(OrderedDict([
        ("pp", config.get("pp", 1)),
        ("dp", dp_size),
        ("tp", tp_size),
    ]))
    plan = get_llama_parallel_plan()

    # Tokenizer
    tokenizer_name = config.get("tokenizer", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    from transformers import AutoTokenizer, AutoConfig
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # 模型配置
    if config.get("model_name"):
        model_config = AutoConfig.from_pretrained(config["model_name"])
    else:
        model_config = AutoConfig.for_model(
            "llama",
            hidden_size=config.get("hidden_size", 256),
            intermediate_size=config.get("intermediate_size", 512),
            num_attention_heads=config.get("num_attention_heads", 8),
            num_key_value_heads=config.get("num_key_value_heads", 4),
            num_hidden_layers=config.get("num_hidden_layers", 4),
            max_position_embeddings=config.get("max_position_embeddings", 512),
            vocab_size=tokenizer.vocab_size,
            rms_norm_eps=1e-5,
            hidden_act="silu",
            tie_word_embeddings=config.get("tie_word_embeddings", False),
        )

    prompts = [
        "Once upon a time",
        "The little girl",
        "One day, a boy",
    ]

    # ========== 1. 随机初始化模型的生成 ==========
    if dist.get_rank() == 0:
        print("=" * 60)
        print("验证 1: 随机初始化模型（预期乱码）")
        print("=" * 60)

    with torch.device("meta"):
        random_model = build_llama_model(model_config, parallel_ctx)
    random_model = random_model.to_empty(device=device)
    for p in random_model.parameters():
        torch.nn.init.normal_(p, mean=0.0, std=0.02)
    random_model = random_model.bfloat16()

    for prompt in prompts:
        text = generate(random_model, tokenizer, prompt, device)  # 所有 rank 都执行
        if dist.get_rank() == 0:
            print(f"  Prompt: \"{prompt}\"")
            print(f"  Output: \"{text}\"")
            print()

    del random_model
    torch.cuda.empty_cache()
    dist.barrier()

    # ========== 2. 训练后模型的生成 ==========
    if dist.get_rank() == 0:
        print("=" * 60)
        print("验证 2: 训练后模型（预期基本连贯的英文）")
        print("=" * 60)

    with torch.device("meta"):
        trained_model = build_llama_model(model_config, parallel_ctx)

    # 加载 checkpoint
    checkpoint_dir = args.checkpoint
    tp_rank = parallel_ctx.tp_rank
    dp_rank = parallel_ctx.dp_rank
    
    shard_path = os.path.join(checkpoint_dir, f"shard_tp{tp_rank}_dp{dp_rank}.pt")
    log(f"  加载 checkpoint: {shard_path}")
    
    state = torch.load(shard_path, map_location="cpu", weights_only=False)
    trained_model = trained_model.to_empty(device=device)
    trained_model.load_state_dict(state["model"])  # key 名按你实际保存的调整
    trained_model = trained_model.bfloat16()
    
    for prompt in prompts:
        text = generate(trained_model, tokenizer, prompt, device)  # 所有 rank 都执行
        if dist.get_rank() == 0:
            print(f"  Prompt: \"{prompt}\"")
            print(f"  Output: \"{text}\"")
            print()

    # ========== 3. Validation Loss ==========
    data_path = config.get("data_path")
    if data_path and os.path.exists(data_path):
        if dist.get_rank() == 0:
            print("=" * 60)
            print("验证 3: Validation Loss")
            print("=" * 60)

        val_loss = compute_val_loss(trained_model, data_path, device)
        log(f"  Validation loss: {val_loss:.4f}")
        log(f"  训练末期 loss:   ~5.58 (从训练日志)")
        log(f"  {'✓' if abs(val_loss - 5.58) < 1.0 else '⚠'} "
            f"Val loss 和训练末期 loss 差距 < 1.0")
    else:
        log("\n  跳过 validation loss（未找到数据文件）")

    # ========== 4. 总结 ==========
    if dist.get_rank() == 0:
        print("\n" + "=" * 60)
        print("验证完成")
        print("=" * 60)
        print("""
对比要点：
  - 随机模型应该输出乱码（随机 token 拼接）
  - 训练后模型应该输出基本连贯的英文
    （TinyStories 风格：简单句式、儿童故事语气）
  - 15M 参数模型的语法会有明显错误，这是正常的
  - Validation loss 应该接近训练末期的 loss（~5.58）
""")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()