# MaskablePPO 迁移指南

## 概述

项目已从标准PPO切换到MaskablePPO,自动支持无效动作掩码(Action Masking),提升训练效率和智能体性能。

## 主要变更

### 1. 依赖更新

```bash
# 新增依赖
pip install sb3-contrib
```

### 2. 代码变更

#### 训练脚本 (training/train_ppo.py)

**之前 (标准PPO)**:
```python
from stable_baselines3 import PPO

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
```

**现在 (MaskablePPO)**:
```python
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

def mask_fn(env):
    """提取action mask的函数"""
    return env.game.get_valid_actions()

# 环境需要用ActionMasker包装
env = ActionMasker(env, mask_fn)

model = MaskablePPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
```

### 3. 自动Masking机制

MaskablePPO会自动:
1. 在训练时调用`mask_fn`获取当前有效动作
2. 将无效动作的概率设为0
3. 重新归一化有效动作的概率分布
4. 确保智能体只选择有效动作

### 4. 性能优势

- **训练效率**: 不会浪费时间探索无效动作
- **收敛速度**: 更快收敛到最优策略
- **策略质量**: 保证永远不会选择无效动作
- **样本效率**: 更有效利用训练数据

## 使用说明

### 基础训练

```bash
# 和之前一样的命令
cd training
python train_ppo.py --mode train --timesteps 100000 --n-envs 8
```

### 评估模型

```bash
# 加载和评估方式不变
python train_ppo.py --mode eval --model-path models/ppo_resource_game/final_model
```

### 自定义mask函数

如果需要自定义mask逻辑:

```python
def custom_mask_fn(env):
    """自定义action mask"""
    valid_actions = env.game.get_valid_actions()

    # 可以添加额外的约束
    # 例如:只在前5回合允许某些动作
    if env.game.current_round <= 5:
        valid_actions[3:] = 0  # 禁用收集动作

    return valid_actions
```

## 向后兼容性

### 模型加载

MaskablePPO模型和标准PPO模型**不兼容**。需要:
- 使用新的MaskablePPO重新训练模型
- 旧的PPO模型无法直接用于新代码

### 环境接口

环境接口保持不变:
- 观测空间: 29维向量 (不变)
- 动作空间: 6个离散动作 (不变)
- `info['action_mask']`: 仍然返回,供前端使用

## 常见问题

### Q: 为什么要切换到MaskablePPO?

A:
1. 游戏中存在大量无效动作(没有对应手牌、不可收集等)
2. 标准PPO会浪费时间探索这些无效动作
3. MaskablePPO直接屏蔽无效动作,提升训练效率

### Q: 需要修改环境代码吗?

A: 不需要。环境的`get_valid_actions()`方法已经存在,只需要在训练时用ActionMasker包装即可。

### Q: 性能提升有多少?

A: 预期:
- 训练速度提升: 20-40%
- 收敛速度: 减少30-50%训练步数
- 最终性能: 提升10-20% (因为不会犯无效动作错误)

### Q: 如何在Web界面中使用?

A: Web界面的推理代码需要更新:

```python
# 加载模型
from sb3_contrib import MaskablePPO

model = MaskablePPO.load("model_path")

# 推理时自动使用mask (环境已用ActionMasker包装)
action, _states = model.predict(obs, deterministic=True)
```

## 技术细节

### ActionMasker包装器

ActionMasker是一个轻量级包装器,它:
1. 在每个step()之前调用mask_fn
2. 将mask存储在环境的私有属性中
3. MaskablePPO在采样动作时自动读取这个mask

### Mask格式

```python
# 6维0/1数组
action_mask = np.array([1, 1, 0, 1, 0, 0], dtype=np.float32)
# 解释: 动作0,1,3有效, 动作2,4,5无效
```

## 参考资源

- [SB3-Contrib文档](https://sb3-contrib.readthedocs.io/)
- [MaskablePPO论文](https://arxiv.org/abs/2006.14171)
- [Action Masking最佳实践](https://costa.sh/blog-a-closer-look-at-invalid-action-masking-in-policy-gradient-algorithms.html)
