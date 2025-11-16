#!/bin/bash

# 优化 CPU 并行训练的环境变量设置
# 适用于多核 CPU 训练

# 设置线程数以匹配 CPU 核心数
# 对于 24 核系统，建议设置为核心数或略少
export OMP_NUM_THREADS=20
export MKL_NUM_THREADS=20
export OPENBLAS_NUM_THREADS=20
export NUMEXPR_NUM_THREADS=20

# 设置 PyTorch 使用多线程
export TORCH_NUM_THREADS=20

# 打印配置
echo "======================================"
echo "CPU 优化配置："
echo "  OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "  MKL_NUM_THREADS: $MKL_NUM_THREADS"
echo "  OPENBLAS_NUM_THREADS: $OPENBLAS_NUM_THREADS"
echo "======================================"

# 运行训练脚本，传递所有参数
python training/train_ppo.py "$@"
