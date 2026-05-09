# preprocess.py
"""离线把原始数据 tokenize + pack 成定长序列，存到磁盘。"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import cast, Protocol

def preprocess(
    dataset_name: str,
    tokenizer_name: str,
    output_path: str,
    seq_len: int = 4096,
    num_proc: int = 16,
):
    """
    预处理预训练数据：加载 → tokenize → pack → 返回 tensor。
    
    Packing 的含义：
    把多条文本 tokenize 后首尾拼接成一个长 token 流，
    然后切分成固定长度 seq_len 的序列。
    不加分隔符（预训练的标准做法）。
    
    返回 shape: [num_samples, seq_len] 的 LongTensor
    """
    # 1. 加载原始数据
    raw = load_dataset(dataset_name, split="train")
    
    # 2. Tokenize
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    eos_id = tokenizer.eos_token_id
    
    def tokenize_fn(examples):
        # batched=True 时 examples["text"] 是 list[str]
        outputs = tokenizer(examples["text"], add_special_tokens=False)
        return {"input_ids": outputs["input_ids"]}
    
    tokenized = raw.map(
        tokenize_fn,
        batched=True,
        num_proc=num_proc,
        remove_columns=raw.column_names,   # 只保留 input_ids
        desc="Tokenizing",
    )
    
    # 3. Concat + pack
    print("Concatenating...")
    all_tokens: list[int] = []
    for ids in tokenized["input_ids"]:
        all_tokens.extend(ids)
        all_tokens.append(eos_id)
    
    print(f"Total tokens: {len(all_tokens):,}")
    
    # 4. Truncate to multiple of seq_len + reshape
    n_full = len(all_tokens) // seq_len
    all_tokens = all_tokens[:n_full * seq_len]
    packed = torch.tensor(all_tokens, dtype=torch.long).view(n_full, seq_len)
    
    print(f"Packed shape: {packed.shape}")
    
    # 5. Save
    torch.save(packed, output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    preprocess(
        dataset_name="HuggingFaceFW/fineweb-edu",
        tokenizer_name="meta-llama/Meta-Llama-3-8B",
        output_path="packed_4k.pt",
        seq_len=4096,
        num_proc=16,
    )