@echo off
REM 优化 CPU 并行训练的环境变量设置
REM 适用于多核 CPU 训练 (Windows版本)

REM 设置线程数以匹配 CPU 核心数
REM 对于 24 核系统，建议设置为核心数或略少
set OMP_NUM_THREADS=20
set MKL_NUM_THREADS=20
set OPENBLAS_NUM_THREADS=20
set NUMEXPR_NUM_THREADS=20
set TORCH_NUM_THREADS=20

echo ======================================
echo CPU 优化配置：
echo   OMP_NUM_THREADS: %OMP_NUM_THREADS%
echo   MKL_NUM_THREADS: %MKL_NUM_THREADS%
echo   OPENBLAS_NUM_THREADS: %OPENBLAS_NUM_THREADS%
echo ======================================
echo.

REM 运行训练脚本，传递所有参数
python training/train_ppo.py %*
