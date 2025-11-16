# Replay格式修改说明

## 修改概述

本次修改完全重新设计了对局记录（Replay）的保存格式，解决了之前无法正确用于强化学习训练的问题。

## 主要问题（修改前）

1. **没有保存观测数据**：replay中完全没有保存38维的observation向量
2. **缺少游戏seed**：无法重现相同的游戏状态
3. **动作记录不精确**：只保存card_value，当手牌有重复时无法确定使用的是哪张牌
4. **无法用于训练**：训练代码尝试通过重放重建observation，但由于缺少seed，重建的状态是错误的

## 新的Replay格式

### 数据结构

```json
{
  "game_id": "string",
  "timestamp": "ISO格式时间戳",
  "seed": 42,                    // 游戏初始seed，用于重现
  "rounds": 10,
  "current_round": 3,
  "final_tokens": 15,
  "transitions": [               // 完整的transition数据
    {
      "step": 0,
      "observation": [float × 38],           // 38维观测向量
      "valid_actions": [bool × 10],          // 10维动作掩码
      "action": 2,                           // 实际选择的动作索引（0-9）
      "action_type": "move",                 // 'move' 或 'collect'
      "card_index": 2,                       // 使用的手牌索引
      "card_value": 3,                       // 卡牌点数
      "reward": 0.0,                         // 本步获得的奖励
      "next_observation": [float × 38],      // 下一步的观测
      "done": false,                         // 是否游戏结束
      "info": {                              // 额外信息
        "message": "使用 3 点卡牌前进 3 步..."
      }
    },
    ...
  ],
  "total_moves": 20,
  "total_collections": 15,
  "action_history": [...]        // 保留旧格式以保持兼容性
}
```

### 观测维度说明（38维）

- 手牌（5维）：第 0-4 维
- 位置（10维，one-hot）：第 5-14 维
- 资源系数（1维）：第 15 维
- 当前回合（1维）：第 16 维
- 是否可收集（1维）：第 17 维
- 上次收集代价（1维）：第 18 维
- 顾客信息（18维）：第 19-36 维
  - 每个顾客 6 维 × 3 = 18 维
- 代币数（1维）：第 37 维

### 动作索引说明（0-9）

- 0-4：移动动作（使用第 0-4 张手牌进行移动）
- 5-9：收集动作（使用第 0-4 张手牌进行收集）

## 代码修改

### 1. `backend/game_core.py`

**修改内容：**
- 在 `ResourceGame.__init__()` 中添加：
  - `self.initial_seed`：保存初始seed
  - `self.transitions = []`：用于存储完整的transition数据
- 在 `ResourceGame.reset()` 中重置 `transitions` 列表

**修改位置：** 第 56-83 行（`__init__`），第 433-455 行（`reset`）

### 2. `backend/app.py`

**修改内容：**

#### a) `perform_action` 函数（执行动作接口）

在每次执行动作时记录完整的transition：

1. **执行前**：记录 `observation`、`valid_actions`、`old_tokens`、`card_value`
2. **执行动作**：调用 `game.move()` 或 `game.collect()`
3. **执行后**：记录 `next_observation`、`reward`、`done`
4. **保存**：将完整的transition添加到 `game.transitions`

**修改位置：** 第 173-259 行

#### b) `save_replay` 函数

- 修改保存的数据结构，使用 `game.transitions` 替代 `game.action_history`
- 添加 `seed` 字段
- 更新返回信息，显示 `transitions_count` 和 `seed`

**修改位置：** 第 309-388 行

### 3. `training/train_from_replay.py`

**修改内容：**

#### a) `replay_to_training_data` 函数

**旧版本问题：**
- 尝试通过重放游戏重建observation
- 由于没有seed，重建的状态不正确
- 通过card_value查找card_index时，总是选择第一个匹配的

**新版本改进：**
- 直接从 `replay['transitions']` 中读取数据
- 不需要重建游戏状态
- 返回3个数组：`observations`, `actions`, `valid_actions_list`

```python
def replay_to_training_data(replays):
    observations = []
    actions = []
    valid_actions_list = []

    for replay in replays:
        transitions = replay.get('transitions', None)

        if transitions is None:
            print("警告：旧格式replay，跳过")
            continue

        for transition in transitions:
            observations.append(np.array(transition['observation']))
            actions.append(transition['action'])
            valid_actions_list.append(np.array(transition['valid_actions']))

    return np.array(observations), np.array(actions), np.array(valid_actions_list)
```

**修改位置：** 第 50-92 行，第 126 行

## 使用方法

### 1. 保存对局记录

通过API保存对局记录（无需修改前端代码）：

```bash
POST /api/game/<game_id>/save_replay
{
  "name": "my_replay"  # 可选
}
```

新的replay文件将自动包含完整的transitions数据。

### 2. 训练模型

使用新格式的replay进行训练：

```bash
python training/train_from_replay.py --mode train --replay-dir replays --epochs 50
```

训练代码会：
1. 自动检测replay格式
2. 跳过旧格式的replay
3. 从新格式的replay中直接提取observation和action
4. 进行行为克隆训练

### 3. 评估模型

```bash
python training/train_from_replay.py --mode eval --model-path models/ppo_from_replay/bc_model
```

## 向后兼容性

- 新的save_replay接口仍然保存 `action_history` 字段（向后兼容）
- 训练代码会检测replay格式，跳过旧格式replay
- 建议废弃所有旧的replay数据，使用新格式重新生成

## 验证

可以使用测试脚本验证新格式：

```bash
python test_new_replay_format.py
```

该脚本会：
1. 创建游戏实例并模拟几步
2. 验证transition的数据结构
3. 检查observation维度（应为38）
4. 检查valid_actions维度（应为10）
5. 检查action索引范围（应为0-9）
6. 保存示例replay文件到 `replays/test_new_format.json`

## 优势

### 修改前 vs 修改后

| 特性 | 修改前 | 修改后 |
|------|--------|--------|
| 保存observation | ✗ | ✓ 完整的38维向量 |
| 保存动作掩码 | ✗ | ✓ 10维布尔数组 |
| 保存seed | ✗ | ✓ 可重现游戏 |
| 动作记录精确性 | ✗ 只有card_value | ✓ 完整的card_index和action索引 |
| 可用于监督学习 | ✗ | ✓ |
| 可用于离线强化学习 | ✗ | ✓ |
| 可用于行为克隆 | ✗ | ✓ |

## 注意事项

1. **旧replay数据将被废弃**：旧格式的replay无法提供正确的训练数据，建议删除
2. **文件大小增加**：由于保存了完整的observation数据，replay文件会变大
3. **只记录成功的动作**：失败的动作不会被记录到transitions中
4. **奖励计算简化**：当前使用tokens变化作为奖励，可以根据需要调整

## 总结

本次修改彻底解决了replay数据不完整的问题，使得保存的对局记录可以真正用于强化学习训练。新格式包含了每一步的完整状态、动作和奖励信息，为行为克隆、监督学习和离线强化学习提供了可靠的数据基础。
