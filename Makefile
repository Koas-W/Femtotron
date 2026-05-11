# ============================================================
# Femtotron CI Regression Test
# ============================================================
#
# 用法：
#   make test              # 跑全部测试
#   make test-unit         # 只跑单元测试
#   make test-integration  # 只跑集成测试
#   make test-quick        # 快速冒烟测试（单卡 + 2卡核心验证）
#
# 环境变量：
#   GPUS=8                 # 可用 GPU 数量（默认自动检测）
#   TIMEOUT=300            # 单个测试超时秒数（默认300）
#

SHELL := /bin/bash
PYTHON := python
TORCHRUN := torchrun
PROJECT_ROOT := $(shell pwd)
export PYTHONPATH := $(PROJECT_ROOT):$(PYTHONPATH)

# 自动检测 GPU 数量，可通过 GPUS=N 覆盖
GPUS ?= $(shell $(PYTHON) -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
TIMEOUT ?= 300

# torchrun 通用参数
RUN_1 := $(TORCHRUN) --nproc_per_node=1
RUN_2 := $(TORCHRUN) --nproc_per_node=2
RUN_4 := $(TORCHRUN) --nproc_per_node=4
RUN_8 := $(TORCHRUN) --nproc_per_node=8

# 颜色输出
GREEN := \033[0;32m
RED := \033[0;31m
YELLOW := \033[0;33m
NC := \033[0m

# ============================================================
# 主目标
# ============================================================

.PHONY: test test-unit test-integration test-quick clean

test: test-unit test-integration
	@echo ""
	@echo -e "$(GREEN)========================================$(NC)"
	@echo -e "$(GREEN)  全部测试通过$(NC)"
	@echo -e "$(GREEN)========================================$(NC)"

# ============================================================
# 单元测试
# ============================================================

test-unit: test-tp-linear test-model-loading
	@echo -e "$(GREEN)单元测试全部通过 ✓$(NC)"

# # 1.1 ParallelContext
# test-parallel-context:
# 	@echo -e "$(YELLOW)>>> 测试 ParallelContext (1卡)$(NC)"
# 	@timeout $(TIMEOUT) $(RUN_1) femtotron/test/unit/test_parallel_context.py
# ifeq ($(shell test $(GPUS) -ge 2 && echo yes),yes)
# 	@echo -e "$(YELLOW)>>> 测试 ParallelContext (2卡)$(NC)"
# 	@timeout $(TIMEOUT) $(RUN_2) femtotron/test/unit/test_parallel_context.py
# endif
# ifeq ($(shell test $(GPUS) -ge 8 && echo yes),yes)
# 	@echo -e "$(YELLOW)>>> 测试 ParallelContext (8卡)$(NC)"
# 	@timeout $(TIMEOUT) $(RUN_8) femtotron/test/unit/test_parallel_context.py
# endif
# 	@echo -e "$(GREEN)  ParallelContext ✓$(NC)"

# 1.2 TP 线性层
test-tp-linear:
ifeq ($(shell test $(GPUS) -ge 2 && echo yes),yes)
	@echo -e "$(YELLOW)>>> 测试 TP Linear (2卡)$(NC)"
	@timeout $(TIMEOUT) $(RUN_2) femtotron/test/unit/test_tp_linear.py
endif
ifeq ($(shell test $(GPUS) -ge 4 && echo yes),yes)
	@echo -e "$(YELLOW)>>> 测试 TP Linear (4卡)$(NC)"
	@timeout $(TIMEOUT) $(RUN_4) femtotron/test/unit/test_tp_linear.py
endif
	@echo -e "$(GREEN)  TP Linear ✓$(NC)"

# 1.3 模型封装层
test-model-loading:
	@echo -e "$(YELLOW)>>> 测试 Model Loading (1卡, TP=1 退化)$(NC)"
	@timeout $(TIMEOUT) $(RUN_1) femtotron/test/unit/test_model_loading.py --test all
ifeq ($(shell test $(GPUS) -ge 2 && echo yes),yes)
	@echo -e "$(YELLOW)>>> 测试 Model Loading (2卡, TP=2)$(NC)"
	@timeout $(TIMEOUT) $(RUN_2) femtotron/test/unit/test_model_loading.py --test all
endif
	@echo -e "$(GREEN)  Model Loading ✓$(NC)"

# 1.4 混合精度训练
test-model-loading:
	@echo -e "$(YELLOW)>>> 测试 Mixed Precision (1卡, TP=1 退化)$(NC)"
	@timeout $(TIMEOUT) $(RUN_1) femtotron/test/unit/test_mixed_precision.py
ifeq ($(shell test $(GPUS) -ge 2 && echo yes),yes)
	@echo -e "$(YELLOW)>>> 测试 Model Loading (2卡, TP=2)$(NC)"
	@timeout $(TIMEOUT) $(RUN_2) femtotron/test/unit/test_mixed_precision.py
endif
	@echo -e "$(GREEN)  Mixed Precision ✓$(NC)"

# ============================================================
# 集成测试（后续开发时逐步添加）
# ============================================================

test-integration: test-dp-training test-gradient-accum test-zero1-2 # test-tp-training test-pp-training test-3d-parallel

# 1.5 DDP 训练
# test-dp-training:
# 	@echo -e "$(YELLOW)>>> [跳过] DDP 训练测试 (待 1.5 实现)$(NC)"
test-dp-training:
	@echo -e "$(YELLOW)>>> 测试 DP 训练一致性 (2卡)$(NC)"
	@timeout $(TIMEOUT) $(RUN_2) tests/integration/test_dp_training.py
ifeq ($(shell test $(GPUS) -ge 4 && echo yes),yes)
	@echo -e "$(YELLOW)>>> 测试 DP 训练一致性 (4卡)$(NC)"
	@timeout $(TIMEOUT) $(RUN_4) tests/integration/test_dp_training.py
endif
	@echo -e "$(GREEN)  DP Training ✓$(NC)"

test-gradient-accum:
	@echo -e "$(YELLOW)>>> 测试梯度累积一致性 (2卡)$(NC)"
	@timeout $(TIMEOUT) $(RUN_2) tests/integration/test_gradient_accum.py
	@echo -e "$(GREEN)  Gradient Accumulation ✓$(NC)"
	
test-zero1-2:
ifeq ($(shell test $(GPUS) -ge 4 && echo yes),yes)
	@echo -e "$(YELLOW)>>> 测试 ZeRO-1 正确性 (4卡)$(NC)"
	@timeout $(TIMEOUT) $(RUN_4) tests/integration/test_zero1_2.py
else ifeq ($(shell test $(GPUS) -ge 2 && echo yes),yes)
	@echo -e "$(YELLOW)>>> 测试 ZeRO-1 正确性 (2卡)$(NC)"
	@timeout $(TIMEOUT) $(RUN_2) tests/integration/test_zero1_2.py
endif
	@echo -e "$(GREEN)  ZeRO-1 ✓$(NC)"

# test-tp-training:
# 	@echo -e "$(YELLOW)>>> 测试 TP 训练一致性 (2卡)$(NC)"
# 	@timeout $(TIMEOUT) $(RUN_2) femtotron/test/integration/test_tp_training.py

# test-pp-training:
# 	@echo -e "$(YELLOW)>>> 测试 PP 训练一致性 (2卡)$(NC)"
# 	@timeout $(TIMEOUT) $(RUN_2) femtotron/test/integration/test_pp_training.py

# test-3d-parallel:
# 	@echo -e "$(YELLOW)>>> 测试 3D 并行一致性 (8卡)$(NC)"
# 	@timeout $(TIMEOUT) $(RUN_8) femtotron/test/integration/test_3d_parallel.py

# ============================================================
# 快速冒烟测试（CI 或快速验证用）
# ============================================================

test-quick:
	@echo -e "$(YELLOW)>>> 快速冒烟测试$(NC)"
	@timeout $(TIMEOUT) $(RUN_1) femtotron/test/unit/test_model_loading.py --test tp1
ifeq ($(shell test $(GPUS) -ge 2 && echo yes),yes)
	@timeout $(TIMEOUT) $(RUN_2) femtotron/test/unit/test_model_loading.py --test forward
	@timeout $(TIMEOUT) $(RUN_2) femtotron/test/unit/test_model_loading.py --test backward
endif
	@echo -e "$(GREEN)  冒烟测试通过 ✓$(NC)"

# ============================================================
# 清理
# ============================================================

clean:
	@rm -rf checkpoints/ tmp/ __pycache__
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	@echo "清理完成"