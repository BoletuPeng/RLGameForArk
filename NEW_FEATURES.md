# 新功能说明

本次更新添加了以下新功能：

## 1. 辅助奖励系数参数

### 功能说明
添加了辅助奖励系数参数 `auxiliary_reward_coef`，用于控制辅助奖励在训练中的权重。

- **默认值**: 1.0（完整辅助奖励模式）
- **范围**: 0.0 ~ 1.0
  - `1.0`: 使用全部辅助奖励（移动到高价值位置奖励、顾客获得资源奖励、游戏结束额外奖励）
  - `0.0`: 纯token奖励模式（只使用收集token的奖励）
  - `0.0~1.0`: 混合模式（按比例调整辅助奖励）

### 使用方法

#### 训练时指定系数
```bash
# 默认模式（完整辅助奖励）
python training/train_ppo.py --timesteps 100000

# 纯token奖励模式
python training/train_ppo.py --timesteps 100000 --auxiliary-reward-coef 0.0

# 混合模式（50%辅助奖励）
python training/train_ppo.py --timesteps 100000 --auxiliary-reward-coef 0.5
```

#### 代码中使用
```python
from rl_env.game_env import ResourceGameEnv

# 创建环境时指定系数
env = ResourceGameEnv(rounds=10, auxiliary_reward_coef=0.5)
```

## 2. 最佳模型续训

### 功能说明
支持从已训练的模型继续训练，特别是从训练过程中保存的最佳模型（best_model.zip）继续训练。

### 使用方法

#### 从最佳模型续训
```bash
python training/train_ppo.py --timesteps 100000 --resume best
```

#### 从指定模型续训
```bash
python training/train_ppo.py --timesteps 100000 --resume models/ppo_resource_game/final_model.zip
```

#### 组合使用（续训+调整辅助奖励系数）
```bash
# 从最佳模型开始，使用纯token奖励模式继续训练
python training/train_ppo.py --timesteps 100000 --resume best --auxiliary-reward-coef 0.0
```

## 3. 对局记录保存

### 功能说明
游戏结束后，可以保存完整的对局记录到文件，包括所有动作历史、最终代币数等信息。

### 使用方法

#### 前端界面
1. 进行一局游戏直到结束
2. 在游戏结束面板中，点击"保存对局记录"按钮
3. 可选：输入对局记录的名称
4. 对局记录将保存到 `replays/` 目录

#### API调用
```bash
curl -X POST http://localhost:5000/api/game/{game_id}/save_replay \
  -H "Content-Type: application/json" \
  -d '{"name": "my_game"}'
```

#### 对局记录格式
保存的JSON文件包含：
```json
{
  "game_id": "游戏ID",
  "timestamp": "保存时间",
  "total_rounds": 10,
  "final_tokens": 125,
  "action_history": [
    {
      "type": "move",
      "card_value": 3,
      "old_position": 0,
      "new_position": 3
    },
    {
      "type": "collect",
      "card_value": 5,
      "position": 3,
      "tokens_earned": 15
    }
  ],
  "seed": null
}
```

## 4. 基于对局记录的训练

### 功能说明
使用行为克隆（Behavior Cloning）方法，从保存的对局记录中学习，可以用于：
- 预训练模型
- 从人类专家对局中学习
- 微调现有模型

### 使用方法

#### 基础训练
```bash
# 从replays目录的所有对局记录训练
python training/train_from_replay.py --mode train
```

#### 高级选项
```bash
# 指定replay目录和训练参数
python training/train_from_replay.py \
  --mode train \
  --replay-dir replays \
  --epochs 100 \
  --batch-size 64 \
  --learning-rate 0.001 \
  --save-path models/bc_model
```

#### 从已有模型继续训练
```bash
# 在已有PPO模型基础上进行行为克隆微调
python training/train_from_replay.py \
  --mode train \
  --resume models/ppo_resource_game/best_model.zip \
  --epochs 50
```

#### 评估训练的模型
```bash
python training/train_from_replay.py \
  --mode eval \
  --model-path models/ppo_from_replay/bc_model
```

## 完整训练流程示例

### 流程1：常规训练 -> 纯token奖励续训
```bash
# 步骤1：使用完整辅助奖励训练基础模型
python training/train_ppo.py --timesteps 500000 --network medium

# 步骤2：从最佳模型继续，使用纯token奖励模式
python training/train_ppo.py --timesteps 500000 --resume best --auxiliary-reward-coef 0.0
```

### 流程2：人类对局 -> 行为克隆 -> PPO训练
```bash
# 步骤1：玩几局游戏并保存对局记录（通过前端）

# 步骤2：从人类对局中学习（行为克隆）
python training/train_from_replay.py --mode train --epochs 100

# 步骤3：使用行为克隆模型作为基础，继续PPO训练
python training/train_ppo.py \
  --timesteps 1000000 \
  --resume models/ppo_from_replay/bc_model.zip \
  --auxiliary-reward-coef 0.8
```

## 注意事项

1. **辅助奖励系数**
   - 调整系数会改变训练的奖励信号，可能需要重新调整学习率等超参数
   - 建议从默认值1.0开始，逐步调整到目标值

2. **模型续训**
   - 续训时，环境的观测空间和动作空间必须保持一致
   - 可以调整学习率、批次大小等训练参数

3. **对局记录**
   - 对局记录在游戏结束后才能保存
   - 记录文件保存在项目根目录的 `replays/` 文件夹

4. **行为克隆训练**
   - 需要足够的高质量对局记录（建议至少10局以上）
   - 训练效果取决于对局记录的质量
   - 可以与PPO训练结合使用，效果更佳

## 技术细节

### 辅助奖励的计算位置
- `backend/rl_env/game_env.py` 第119-173行
- 所有辅助奖励都乘以 `auxiliary_reward_coef` 系数
- Token奖励不受系数影响（保持为主要奖励信号）

### 模型保存位置
- 训练过程中的最佳模型：`models/ppo_resource_game/best_model.zip`
- 最终模型：`models/ppo_resource_game/final_model.zip`
- 行为克隆模型：`models/ppo_from_replay/bc_model.zip`

### 对局记录保存位置
- 默认目录：`replays/`
- 文件格式：`{名称}_{时间戳}.json` 或 `replay_{时间戳}.json`
