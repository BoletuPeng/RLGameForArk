# CPU 多核优化指南

## 问题描述

当使用 `--n-envs` 参数创建多个并行环境时（例如 22 个），你可能会发现 CPU 利用率并不高，只有部分核心在工作（例如 24 核系统只有 8 核在工作）。

## 原因分析

### 1. PPO 训练的架构

在 Stable-Baselines3 的 PPO 实现中：

- **数据收集阶段**：使用 `SubprocVecEnv` 创建多个子进程，每个进程运行一个环境实例
  - 这部分是多进程的，理论上可以利用多核
  - 但环境本身的计算通常很轻量

- **模型推理阶段**（选择动作）：
  - 在**主进程**中进行批量推理
  - 使用 PyTorch/NumPy 进行张量计算
  - 这是 CPU 密集型操作

- **训练阶段**（更新网络）：
  - 在主进程中进行梯度计算和参数更新
  - 同样是 CPU 密集型操作

### 2. 线程数限制

PyTorch 和 NumPy 底层使用的计算库（OpenBLAS、MKL、OpenMP）默认线程数可能被限制：

- 如果没有设置环境变量，系统可能默认使用较少的线程（例如 8 个）
- 这些线程数控制了矩阵运算、张量操作等的并行度
- 即使有 22 个环境进程，如果神经网络计算只用 8 个线程，CPU 利用率也会受限

## 解决方案

### 方案 1：自动优化（推荐）

修改后的 `train_ppo.py` 已经自动设置了线程数：

```python
# 在代码开头自动检测 CPU 核心数并设置为 80%
setup_cpu_threads()  # 会自动设置为 CPU 核心数的 80%
```

直接运行即可：

```bash
python training/train_ppo.py --network medium --timesteps 10000000 --n-envs 22 --no-eval
```

### 方案 2：使用优化脚本

**Windows 用户：**
```cmd
training\train_ppo_optimized.bat --network medium --timesteps 10000000 --n-envs 22 --no-eval
```

**Linux/Mac 用户：**
```bash
chmod +x training/train_ppo_optimized.sh
./training/train_ppo_optimized.sh --network medium --timesteps 10000000 --n-envs 22 --no-eval
```

### 方案 3：手动设置环境变量

**Windows (cmd):**
```cmd
set OMP_NUM_THREADS=20
set MKL_NUM_THREADS=20
set OPENBLAS_NUM_THREADS=20
python training/train_ppo.py --network medium --timesteps 10000000 --n-envs 22 --no-eval
```

**Windows (PowerShell):**
```powershell
$env:OMP_NUM_THREADS=20
$env:MKL_NUM_THREADS=20
$env:OPENBLAS_NUM_THREADS=20
python training/train_ppo.py --network medium --timesteps 10000000 --n-envs 22 --no-eval
```

**Linux/Mac:**
```bash
export OMP_NUM_THREADS=20
export MKL_NUM_THREADS=20
export OPENBLAS_NUM_THREADS=20
python training/train_ppo.py --network medium --timesteps 10000000 --n-envs 22 --no-eval
```

## 最佳实践

### 线程数设置建议

对于 24 核系统：

1. **保守设置**（推荐）：
   - 线程数：16-20
   - 留一些核心给系统和其他进程
   - 避免过度竞争和上下文切换

2. **激进设置**：
   - 线程数：22-24
   - 最大化 CPU 利用率
   - 可能导致系统响应变慢

3. **并行环境数 vs 线程数**：
   - 并行环境数（`--n-envs`）：控制同时运行的环境实例数
   - 线程数：控制单个神经网络计算的并行度
   - 两者相乘不应超过总核心数太多

### 推荐配置（24 核系统）

```bash
# 平衡配置
python training/train_ppo.py --network medium --timesteps 10000000 --n-envs 16 --no-eval
# 线程数会自动设置为 ~19（24 * 0.8）

# 或者手动控制
export OMP_NUM_THREADS=18
python training/train_ppo.py --network medium --timesteps 10000000 --n-envs 16 --no-eval
```

## 验证效果

### 监控 CPU 使用率

**Windows:**
- 打开任务管理器 → 性能 → CPU
- 观察各个核心的使用情况

**Linux:**
```bash
htop  # 或 top
```

**预期效果：**
- 优化前：8 个核心高使用率，其他核心闲置
- 优化后：大部分核心都有较高使用率（60-90%）

## 常见问题

### Q1: 为什么不直接设置为 CPU 核心数？

超线程核心不等于物理核心，设置过多线程会导致：
- 过度的上下文切换
- 缓存失效增加
- 性能反而下降

建议设置为核心数的 70-80%。

### Q2: 并行环境数应该设置多少？

建议：
- 小环境（轻量计算）：可以设置为核心数
- 大环境（重计算）：设置为核心数的 50-75%
- 本项目的环境比较轻量，可以设置 16-22

### Q3: 如何判断是否需要增加线程数？

观察 CPU 利用率：
- 如果总体利用率低（<50%）→ 增加线程数或环境数
- 如果总体利用率高但训练慢 → 可能是其他瓶颈（I/O、内存等）

## 性能基准

在 24 核系统上的预期性能（本项目）：

| 配置 | 环境数 | 线程数 | CPU利用率 | 训练速度 |
|------|--------|--------|-----------|----------|
| 默认 | 8 | 8 | ~30% | 基准 |
| 优化1 | 16 | 19 | ~70% | 2-2.5x |
| 优化2 | 22 | 19 | ~80% | 2.5-3x |

**注意**：实际性能取决于具体硬件和环境复杂度。
