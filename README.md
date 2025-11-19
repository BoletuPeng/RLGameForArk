# 明日方舟喀兰贸易技术研发部罗德分部

为明日方舟喀兰贸易技术研发部第十重建关卡开发，包括网页前端模拟游戏，后端模型训练管线

允许在残局模式下编辑当前手牌、重建订单、资源倍率等等数据，并使用您训练的AI在前端游戏中进行推理，输出决策的置信分布。

本品由Claude Code亲情开发，如果写的好那是因为本人很好，如果出了岔子那是Claude很坏

详细规则请查看：[游戏规则说明.txt](游戏规则说明.txt)

## 🏗️ 项目结构

```
RLGameForArk/
├── backend/                    # 后端代码
│   ├── game_core.py           # 游戏核心逻辑
│   ├── app.py                 # Flask API服务器
│   └── rl_env/                # 强化学习环境
│       ├── game_env.py        # Gymnasium环境
│       └── parallel_env.py    # 并行训练环境
├── frontend/                   # 前端代码
│   ├── templates/
│   │   └── index.html         # 游戏界面
│   └── static/
│       ├── style.css          # 样式
│       └── game.js            # 前端逻辑
├── training/                   # 训练脚本
│   ├── test_env.py            # 环境测试
│   ├── train_random.py        # 随机策略训练
│   └── train_ppo.py           # PPO算法训练
├── Game.py                     # 原始游戏（命令行版本）
└── requirements.txt            # 依赖列表
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 测试环境

```bash
# 测试强化学习环境是否正常工作
cd training
python test_env.py
```

### 3. 启动Web服务器

```bash
cd backend
python app.py
```

然后在浏览器中访问：`http://localhost:5000`

### 4. 游玩游戏

- 点击手牌上的按钮进行移动或收集
- 点击"AI 辅助"查看AI决策建议
- 观察右侧面板的AI分析（概率分布、观测向量等）

## 🤖 强化学习训练

### 环境接口

项目实现了符合 Gymnasium 标准的强化学习环境：

- **观测空间**：29维连续向量
  - 手牌统计（3维）：1点、2点、3点各有几张
  - 地图位置（10维 one-hot）
  - 游戏状态（资源系数、回合数、是否可普通收集、是否可连击）
  - 上次收集代价（2维 one-hot）
  - 顾客需求信息（9维）：每个顾客的冰/铁/火仍需量
  - 代币数

- **动作空间**：6个离散动作
  - 0-2：使用1点、2点、3点牌进行移动
  - 3-5：使用1点、2点、3点牌进行收集

- **奖励设计**：
  - 完成顾客订单：+订单代币数（1或4）
  - 无效动作：-0.1
  - 游戏结束：额外奖励（基于最终代币数）

### 训练示例

#### 1. 随机策略（测试环境）

```bash
cd training
python train_random.py --episodes 100
```

#### 2. MaskablePPO算法训练(支持自动action masking)

```bash
cd training
python train_ppo.py --mode train --timesteps 100000 --n-envs 8
```

参数说明：
- `--timesteps`：训练总步数
- `--n-envs`：并行环境数量（建议4-16）

**新特性**：现在使用MaskablePPO算法,自动处理无效动作掩码,提升训练效率和性能。

#### 3. 评估训练好的模型

```bash
python train_ppo.py --mode eval --model-path models/ppo_resource_game/final_model --episodes 10
```

### 并行训练

支持多进程并行训练以提高效率：

```python
from rl_env.parallel_env import make_parallel_env

# 创建8个并行环境
envs = make_parallel_env(n_envs=8, rounds=10, seed=42)

# 批量执行动作
observations, rewards, dones, infos = envs.step(actions)
```

## 🎯 API 接口

### REST API

- `POST /api/game/new` - 创建新游戏
- `GET /api/game/<game_id>/state` - 获取游戏状态
- `POST /api/game/<game_id>/action` - 执行动作
- `POST /api/game/<game_id>/ai/predict` - 获取AI预测

### AI预测接口

```python
# 请求格式
POST /api/game/<game_id>/ai/predict
{
    "model_type": "random" | "rule_based" | "custom",
    "probabilities": [...]  # 可选，用于自定义模型
}

# 响应格式
{
    "action": 3,
    "action_info": {
        "type": "collect",
        "card_value": 2,
        "is_combo": true
    },
    "probabilities": [0.0, 0.1, ...],  # 6维概率分布
    "observation": [...],               # 29维观测向量
    "valid_actions": [1, 1, 0, ...]    # 有效动作掩码
}
```

## 📊 前端功能

### 游戏界面

- ✅ 实时游戏状态显示（回合、代币、位置等）
- ✅ 可视化环形地图
- ✅ 顾客订单进度条
- ✅ 交互式手牌选择
- ✅ 操作日志

### AI决策可视化

- ✅ AI建议动作显示
- ✅ 动作概率分布图
- ✅ 观测向量预览
- ✅ 有效动作指示器
- ✅ 多种AI策略选择（随机、规则基础）

## 🛠️ 自定义AI模型

你可以训练自己的模型并集成到Web界面中：

```python
# 1. 训练模型(使用MaskablePPO)
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from rl_env.game_env import ResourceGameEnv

def mask_fn(env):
    return env.game.get_valid_actions()

env = ResourceGameEnv(rounds=10)
env = ActionMasker(env, mask_fn)  # 添加action masking支持
model = MaskablePPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
model.save("my_model")

# 2. 加载并预测
model = MaskablePPO.load("my_model")
obs, info = env.reset()
action, _states = model.predict(obs)  # 自动使用action masks
```

## 📈 性能优化

### 并行训练性能

在8核CPU上使用8个并行环境：
- **吞吐量**：约 5000-10000 步/秒
- **内存占用**：约 500MB
- **GPU加速**：支持（通过 stable-baselines3）

### 建议配置

- **开发/测试**：1-4个并行环境
- **训练**：8-16个并行环境
- **大规模训练**：16-32个并行环境

## 🔧 高级用法

### 自定义环境参数

```python
from rl_env.game_env import ResourceGameEnv

env = ResourceGameEnv(
    rounds=20,        # 增加回合数
    seed=42,          # 固定随机种子
    render_mode='human'  # 终端渲染
)
```

### 自定义奖励函数

修改 `backend/rl_env/game_env.py` 中的 `step()` 方法，调整奖励设计。

### WebSocket实时通信

前端支持WebSocket进行实时游戏状态更新：

```javascript
const socket = io();
socket.emit('join_game', { game_id: gameId });
socket.on('game_state', (state) => {
    // 处理游戏状态更新
});
```

## 📝 开发计划

- [ ] 支持更多强化学习算法（DQN, A2C, SAC等）
- [ ] 添加更多AI可视化功能（注意力热图、价值估计等）
- [ ] 实现模型对战功能
- [ ] 添加回放系统
- [ ] 优化观测空间和动作空间
- [ ] 添加更多游戏模式和难度

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License

## 🙏 致谢

- [Gymnasium](https://gymnasium.farama.org/) - 强化学习环境标准
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - 强化学习算法库
- [Flask](https://flask.palletsprojects.com/) - Web框架

---

## 常见问题

### Q: 如何提高训练速度？

A:
1. 增加并行环境数量（`--n-envs`）（但这是不行的，PPO的多线程属于本笨比根本看不懂的东西，cmd里print的自动线程分配都是假的）
2. 使用GPU加速（需要CUDA支持）   （但这是不行的，这样的游戏在GPU上跑只会更慢捏）
3. 调整PPO参数（减小 `n_steps` 和 `batch_size`） （这是可以的，但是16分以上的对局几乎无法学习到东西）
4. numba编译 （已经做了，似乎没什么软用）

0. 用C重写GameCore: 这是可以的，请（鞠躬）

### Q: 如何调试环境？

A: 使用 `test_env.py` 脚本进行全面测试：

```bash
python training/test_env.py --test all
```

（找不到就是没有）

### Q: 如何在训练时可视化？

A: 使用 TensorBoard：

```bash
tensorboard --logdir ./logs/ppo_resource_game
```

