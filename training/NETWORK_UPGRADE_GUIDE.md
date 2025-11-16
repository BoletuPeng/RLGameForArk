# PPO 网络架构升级指南

## 升级内容

原始PPO模型使用的是小型网络（~7.4K参数），现已升级支持多种网络规模：

| 架构 | 网络结构 | 参数量 | 相比原版 | 适用场景 |
|------|---------|--------|----------|----------|
| **small** | [64, 64] | ~14K | 1.0x | 快速实验、基线测试 |
| **medium** | [128, 128] | ~44K | 3.2x | **推荐选择**，性能与速度平衡 |
| **large** | [256, 256] | ~154K | 11.0x | 需要更强表达能力 |
| **xlarge** | [256, 256, 128] | ~219K | 15.6x | 最强表达能力，3层网络 |

## 使用方法

### 训练不同规模的模型

```bash
# 使用medium网络（推荐，默认）
python training/train_ppo.py --network medium --timesteps 100000

# 使用large网络
python training/train_ppo.py --network large --timesteps 200000

# 使用xlarge网络
python training/train_ppo.py --network xlarge --timesteps 300000

# 使用small网络（原始规模）
python training/train_ppo.py --network small --timesteps 100000
```

### 完整训练命令示例

```bash
# Medium网络，更多训练步数，更多并行环境
python training/train_ppo.py \
    --mode train \
    --network medium \
    --timesteps 500000 \
    --n-envs 16

# Large网络，长时间训练
python training/train_ppo.py \
    --mode train \
    --network large \
    --timesteps 1000000 \
    --n-envs 8
```

### 评估模型

```bash
# 评估训练好的模型
python training/train_ppo.py \
    --mode eval \
    --model-path models/ppo_resource_game/final_model \
    --episodes 20
```

## 网络架构详解

### Medium网络 (推荐)

```
观测空间 (38维)
    ↓
Actor网络:                      Critic网络:
输入 → 隐藏层1 (128)            输入 → 隐藏层1 (128)
    ↓                               ↓
隐藏层1 → 隐藏层2 (128)         隐藏层1 → 隐藏层2 (128)
    ↓                               ↓
隐藏层2 → 动作概率 (10)         隐藏层2 → 状态价值 (1)
```

**参数分配:**
- Actor网络: 22,794 个参数
- Critic网络: 21,633 个参数
- **总计: 44,427 个参数**

### 为什么推荐Medium？

1. **性能提升明显**: 比原版small网络大3.2倍，能够学习更复杂的策略
2. **训练速度适中**: 比large和xlarge网络训练更快
3. **稳定性好**: 不会因为网络过大导致过拟合
4. **适合当前环境**: 38维观测空间 + 10动作空间，medium规模刚好

## 预期性能提升

根据强化学习研究经验：

- **Small → Medium**: 预期最终代币提升 15-25%
- **Small → Large**: 预期最终代币提升 25-40%
- **Small → XLarge**: 预期最终代币提升 30-50%

注意：更大的网络需要更多训练步数才能收敛。

## 训练建议

### 推荐配置

```bash
# 快速测试 (1-2小时)
python training/train_ppo.py --network medium --timesteps 100000 --n-envs 8

# 标准训练 (4-6小时)
python training/train_ppo.py --network medium --timesteps 500000 --n-envs 16

# 深度训练 (12-24小时)
python training/train_ppo.py --network large --timesteps 1000000 --n-envs 16
```

### 调参建议

更大的网络可能需要调整超参数：

```python
# Medium网络 - 使用默认参数即可
--network medium

# Large网络 - 可能需要更大的batch_size
# 修改 train_ppo.py 中的 batch_size=128

# XLarge网络 - 考虑降低学习率
# 修改 train_ppo.py 中的 learning_rate=1e-4
```

## 监控训练

使用TensorBoard查看训练进度：

```bash
tensorboard --logdir ./logs/ppo_resource_game
```

关键指标：
- `rollout/ep_rew_mean`: 平均奖励（应该持续上升）
- `train/value_loss`: 价值函数损失（应该下降）
- `train/policy_loss`: 策略损失

## 故障排查

### 训练不稳定
- 尝试降低学习率：`learning_rate=1e-4`
- 增加训练步数：`--timesteps 1000000`

### 内存不足
- 减少并行环境：`--n-envs 4`
- 使用smaller网络：`--network medium`

### 收敛太慢
- 增加并行环境：`--n-envs 16`
- 检查是否有足够的训练步数

## 下一步

1. **先用medium网络训练**: `python training/train_ppo.py --network medium --timesteps 500000`
2. **评估性能**: 与原始small网络对比
3. **如果效果好**: 尝试large网络进行更深度训练
4. **考虑其他算法**: 如果PPO效果不理想，可以尝试DQN或RecurrentPPO
