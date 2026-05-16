from collections import OrderedDict
import torch
import torch.distributed as dist
import numpy as np
from typing import cast

# 框架支持的所有并行维度，按标准顺序排列（最外层到最内层）
# 未来加新维度只需要在这里插入到合适的位置
# sequence Parallel（context Parallel）维度和 TP 共用通信组，不需要单独建模
SUPPORTED_PARALLEL_DIMS = ["pp", "dp", "ep", "cp", "tp"]

class ParallelContext:
    def __init__(self, parallel_dims: OrderedDict[str, int]):
        """
        parallel_dims: 有序字典，key 是维度名，value 是该维度的 size。
                       顺序从左到右对应从最外层到最内层。
                       所有 size 的乘积必须等于 world_size。
        
        例如 OrderedDict([("pp", 2), ("dp", 2), ("tp", 2)])
        会创建一个 [2, 2, 2] 的 3D rank 网格。
        """
        ##################################
        # 初始化 torch.dist，检查参数合法性
        ##################################
        self.parallel_dims = self._normalize_parallel_dims(parallel_dims)
        self.world_size = dist.get_world_size()
        self.world_rank = dist.get_rank()
        # self.parallel_dims = parallel_dims
        self.dim_names = list(self.parallel_dims.keys())
        self.dim_sizes = list(self.parallel_dims.values())

        # 检查输入参数一致性
        total_size = 1
        for s in self.dim_sizes:
            total_size *= s
        assert total_size == self.world_size, \
            f"并行维度之积 {total_size} != world_size {self.world_size}"

        ##################################
        # 构建 N 维并行网格 grid
        ##################################
        # 示例：shape = [pp_size, dp_size, tp_size]，右侧为内侧（物理相邻）维度
        self.rank_grid = torch.arange(self.world_size).reshape(self.dim_sizes)
        # 计算本rank的位置
        # self.local_coord = cast(list[int], torch.unravel_index(torch.tensor(self.world_rank), self.dim_sizes))
        # self.local_coord = cast(tuple[int, ...], self.local_coord)
        # print(self.local_coord)
        self.local_coord = [
            int(c) for c in torch.unravel_index(torch.tensor(self.world_rank), self.dim_sizes)
        ]

        # 为每个维度创建 process group
        self.groups = {}      # dim_name -> ProcessGroup
        self.ranks_in = {}    # dim_name -> 当前 rank 所在 group 的 rank 列表

        ##################################
        # 初始化进程组
        ##################################
        for dim_idx, dim_name in enumerate(self.dim_names):
            group, ranks = self._create_groups_along_dim(dim_idx)
            self.groups[dim_name] = group
            self.ranks_in[dim_name] = ranks
        

    def _normalize_parallel_dims(self, parallel_dims: OrderedDict[str, int]) -> OrderedDict[str, int]:
        """
        预处理：检查是否有不支持的维度名，补全缺失的并行维度（默认 size=1），并按标准顺序排列。
        
        用户可以只传关心的维度：
            OrderedDict([("tp", 2)])
        会被补全为：
            OrderedDict([("pp", 1), ("dp", 1), ("ep", 1), ("cp", 1), ("tp", 2)])
        
        size=1 的维度不影响任何计算（group 就是 rank 自己），
        但统一了内部逻辑，不需要到处判断"这个维度存不存在"。
        """
        # 检查用户是否传了不支持的维度
        for name in parallel_dims:
            if name not in SUPPORTED_PARALLEL_DIMS:
                raise ValueError(
                    f"不支持的并行维度 '{name}'。"
                    f"支持的维度: {SUPPORTED_PARALLEL_DIMS}"
                )
        
        # 按标准顺序构建完整的维度列表，缺失的补 1
        normalized = OrderedDict()
        for name in SUPPORTED_PARALLEL_DIMS:
            normalized[name] = parallel_dims.get(name, 1)
        
        return normalized
    

    def _rank_to_coord(self, rank):
        """将 world rank 转换为 N 维坐标。目前不用这个，和前面的写法等效"""
        coord = []
        for size in reversed(self.dim_sizes):
            coord.append(rank % size)
            rank //= size
        return list(reversed(coord))
    
    def _create_groups_along_dim(self, dim_idx):
        """
        沿第 dim_idx 个维度创建 process group。
        
        同一个 group 内的 rank：在其他所有维度上坐标相同，
        只在第 dim_idx 个维度上坐标不同。
        """
        # 遍历所有可能的"其他维度坐标"组合
        # 对每个组合，收集沿 dim_idx 变化的所有 rank
        my_group = None
        my_ranks = None
        
        # 用 itertools 生成其他维度的所有坐标组合
        import itertools
        other_dims_ranges = []
        for i, size in enumerate(self.dim_sizes):
            if i == dim_idx:
                other_dims_ranges.append([None])  # 占位
            else:
                other_dims_ranges.append(range(size))
        
        for combo in itertools.product(*other_dims_ranges):
            # 收集这个 group 内的所有 rank
            ranks = []
            for pos in range(self.dim_sizes[dim_idx]):
                coord = list(combo)
                coord[dim_idx] = pos
                rank = self.rank_grid[tuple(coord)].item()
                ranks.append(rank)
            
            # 创建 group（所有 rank 都必须调用 new_group）
            group = dist.new_group(ranks)
            
            if self.world_rank in ranks:
                my_group = group
                my_ranks = ranks
        
        return my_group, my_ranks
    

    # ========== 通用接口 ==========
    
    def get_rank(self, dim_name: str) -> int:
        """获取当前 rank 在指定维度上的坐标。"""
        idx = self.dim_names.index(dim_name)
        return self.local_coord[idx]
    
    def get_size(self, dim_name: str) -> int:
        """获取指定维度的 size。"""
        return self.parallel_dims[dim_name]
    
    def get_group(self, dim_name: str) -> dist.ProcessGroup:
        """获取当前 rank 在指定维度上所在的 process group。"""
        return self.groups[dim_name]
    
    def get_ranks_in_group(self, dim_name: str) -> list[int]:
        """获取当前 rank 在指定维度上的 group 包含哪些 world rank。"""
        return self.ranks_in[dim_name]
    
    # ========== 便利接口（常用维度的快捷方式）==========
    
    @property
    def dp_rank(self): return self.get_rank("dp")
    @property
    def tp_rank(self): return self.get_rank("tp")
    @property
    def pp_rank(self): return self.get_rank("pp")
    
    @property
    def dp_size(self): return self.get_size("dp")
    @property
    def tp_size(self): return self.get_size("tp")
    @property
    def pp_size(self): return self.get_size("pp")
    
    @property
    def dp_group(self): return self.get_group("dp")
    @property
    def tp_group(self): return self.get_group("tp")
    @property
    def pp_group(self): return self.get_group("pp")
    
    # ========== PP 特有接口 ==========
    
    def get_prev_rank_in(self, dim_name: str) -> int | None:
        """获取指定维度上前一个位置的 world rank。"""
        idx = self.dim_names.index(dim_name)
        if self.local_coord[idx] == 0:
            return None
        prev_coord = self.local_coord.copy()
        prev_coord[idx] -= 1
        return int(self.rank_grid[tuple(prev_coord)].item())
    
    def get_next_rank_in(self, dim_name: str) -> int | None:
        """获取指定维度上后一个位置的 world rank。"""
        idx = self.dim_names.index(dim_name)
        if self.local_coord[idx] == self.dim_sizes[idx] - 1:
            return None
        next_coord = self.local_coord.copy()
        next_coord[idx] += 1
        return int(self.rank_grid[tuple(next_coord)].item())
    
    @property
    def pp_prev_rank(self): return self.get_prev_rank_in("pp")
    @property
    def pp_next_rank(self): return self.get_next_rank_in("pp")

if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    test_context = ParallelContext(OrderedDict([("pp",2),("dp",2),("tp",2)]))
    dist.destroy_process_group()