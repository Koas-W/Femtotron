from dataclasses import dataclass

@dataclass
class ZeROConfig:
    stage: int = 0   # 0 = no shard, 1 = ZeRO-1, 2 = ZeRO-2, 3 = ZeRO-3
