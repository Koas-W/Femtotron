import torch
from dataclasses import dataclass

@dataclass
class TrainConfig:
    # 预留（现在只支持 "fp32"）
    # 未来: "bf16_mixed", "bf16_pure", "fp8"
    param_dtype: torch.dtype = torch.bfloat16            # forward/backward 用的 weight 精度
    master_dtype: torch.dtype | None = None              # optimizer 看的 master 精度，None 表示不维护 master
    grad_acc_dtype: torch.dtype | None = None            # 梯度累加 buffer 精度，None 表示原地累
    reduce_dtype: torch.dtype | None = None              # 梯度通信精度，None 表示和 grad 一致
    

    sharding: str = "none"    # 未来: "ddp", "zero1", "zero2", "zero3", "fsdp"

    use_loss_scaler: bool = False     # fp16 才开
    initial_scale: float = 2.0 ** 16
    
    optimizer: str = "adamw"
    lr: float = 1e-4