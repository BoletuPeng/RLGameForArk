# PPO模型使用说明

## 概述

本项目现在支持在前端游戏界面中使用训练好的PPO模型进行AI辅助决策。

## 前置要求

1. **安装依赖**
   ```bash
   pip install stable-baselines3 torch
   ```

2. **训练模型**

   运行训练脚本生成模型文件：
   ```bash
   cd training
   python train_ppo.py
   ```

   训练完成后，模型会保存在 `models/ppo_resource_game/` 目录下：
   - `best_model.zip` - 训练过程中验证得分最高的模型
   - `final_model.zip` - 训练结束时的最终模型

## 模型文件结构

确保你的项目根目录下有以下结构：

```
RLGameForArk/
├── models/
│   └── ppo_resource_game/
│       ├── best_model.zip      # 最佳模型（推荐使用）
│       └── final_model.zip     # 最终模型
├── backend/
├── frontend/
└── training/
```

**注意**: `models/` 目录在 `.gitignore` 中被忽略，不会提交到Git仓库。

## 在前端使用PPO模型

1. **启动后端服务器**
   ```bash
   cd backend
   python app.py
   ```

2. **打开游戏界面**

   访问 http://localhost:5000

3. **启用AI辅助**

   - 点击页面顶部的「AI 辅助」按钮
   - 在下拉菜单中选择模型类型：
     - **随机策略** - 随机选择有效动作
     - **规则策略** - 基于手写规则的策略
     - **PPO模型(最佳)** - 使用best_model.zip（推荐）
     - **PPO模型(最终)** - 使用final_model.zip

4. **查看AI建议**

   选择PPO模型后，右侧会显示：
   - AI建议的动作
   - 每个动作的概率分布
   - 当前状态的观测向量
   - 有效动作列表

## 检查模型状态

可以通过API端点检查模型是否正确加载：

```bash
curl http://localhost:5000/api/models/status
```

返回示例：
```json
{
  "has_sb3": true,
  "models": {
    "ppo_best": {
      "available": true,
      "path": "/path/to/models/ppo_resource_game/best_model.zip",
      "loaded": true
    },
    "ppo_final": {
      "available": true,
      "path": "/path/to/models/ppo_resource_game/final_model.zip",
      "loaded": false
    }
  }
}
```

## 故障排除

### 1. 提示"stable-baselines3 not installed"

**原因**: 未安装stable-baselines3库

**解决方案**:
```bash
pip install stable-baselines3 torch
```

### 2. 提示"Failed to load PPO model"

**原因**: 模型文件不存在或路径不正确

**解决方案**:
- 确认 `models/ppo_resource_game/` 目录存在
- 确认模型文件 `best_model.zip` 或 `final_model.zip` 存在
- 如果没有模型文件，运行训练脚本生成：
  ```bash
  cd training
  python train_ppo.py
  ```

### 3. 提示"PPO inference failed"

**原因**: 模型推理过程中出错

**解决方案**:
- 检查后端控制台的详细错误信息
- 确认模型文件未损坏
- 确认观测空间维度与训练时一致（36维）

## 技术细节

### 模型推理流程

1. 前端发送AI预测请求：
   ```javascript
   POST /api/game/{game_id}/ai/predict
   {
     "model_type": "ppo_best"
   }
   ```

2. 后端加载模型（如果未缓存）：
   - 从 `models/ppo_resource_game/best_model.zip` 加载
   - 缓存到内存中供后续使用

3. 模型进行推理：
   - 获取当前游戏状态的观测向量（36维）
   - 通过PPO策略网络计算动作概率分布
   - 应用有效动作掩码（invalid actions概率设为0）
   - 重新归一化概率分布
   - 选择概率最高的有效动作

4. 返回预测结果：
   ```json
   {
     "action": 5,
     "action_info": {
       "type": "collect",
       "card_value": 3,
       "is_combo": false
     },
     "probabilities": [...],
     "observation": [...],
     "valid_actions": [...]
   }
   ```

### 模型缓存机制

- 首次请求时加载模型到内存
- 后续请求直接使用缓存的模型
- 不同模型（best/final）分别缓存
- 服务器重启后需要重新加载

## 相关文件

- `backend/app.py` - 后端API，包含模型加载和推理逻辑
- `frontend/templates/index.html` - 前端界面，包含模型选择器
- `frontend/static/game.js` - 前端逻辑，处理AI决策请求
- `training/train_ppo.py` - PPO训练脚本
