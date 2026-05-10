import torch
from dataclasses import dataclass, field

from femtotron.sharding.zero_config import ZeROConfig

@dataclass
class TrainConfig:

    #####################################################
    # 训练计划相关
    #####################################################
    train_steps: int = 0
    grad_accum_steps: int = 1
    grad_clip: float | None = 1.0
    log_interval: int = 10
    checkpoint_interval: int = 1000
    checkpoint_dir: str = "checkpoints"
    eval_interval: int | None = None   # 可选

    #####################################################
    # 混合精度相关
    #####################################################
    # 预留（现在只支持 "fp32"）
    # 未来: "bf16_mixed", "bf16_pure", "fp8"
    param_dtype: torch.dtype = torch.bfloat16            # forward/backward 用的 weight 精度
    master_dtype: torch.dtype | None = None              # optimizer 看的 master 精度，None 表示不维护 master
    grad_acc_dtype: torch.dtype | None = None            # 梯度累加 buffer 精度，None 表示原地累
    reduce_dtype: torch.dtype | None = None              # 梯度通信精度，None 表示和 grad 一致
    

    #####################################################
    # ZeRO-1 2 3（FSDP）相关
    #####################################################
    ZeRO_config: ZeROConfig = field(default_factory=ZeROConfig)
    # sharding: str = "none"    # 未来: "ddp", "zero1", "zero2", "zero3", "fsdp"

    use_loss_scaler: bool = False     # fp16 才开
    initial_scale: float = 2.0 ** 16
    
    #####################################################
    # 参数优化相关
    #####################################################
    optimizer: str = "adamw"
    lr: float = 1e-4
    warmup_steps: int = 50
    min_lr_ratio: float = 0.1