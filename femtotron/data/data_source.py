# dataset.py
import torch
from torch import Tensor
from torch.utils.data import Dataset


class PackedDataset(Dataset):
    """定长 token 序列的 dataset。
    
    输入是 preprocess.py 产出的 .pt 文件，shape [N, seq_len]。
    """
    
    def __init__(self, path: str, mmap: bool = True):
        # mmap=True 让多个 worker / 多个 rank 共享同一份内存映射
        self.data: Tensor = torch.load(path, mmap=mmap)
        assert self.data.dim() == 2, f"expected 2D tensor, got {self.data.shape}"
        assert self.data.dtype == torch.long, f"expected long, got {self.data.dtype}"
    
    @property
    def seq_len(self) -> int:
        return self.data.shape[1]
    
    def __len__(self) -> int:
        return self.data.shape[0]
    
    def __getitem__(self, idx: int) -> Tensor:
        return self.data[idx]   # [seq_len], long