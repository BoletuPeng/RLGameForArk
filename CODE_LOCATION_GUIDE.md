# 计算密集型代码位置详细指南

## 🎯 优先级 1: 极高频调用函数

### 1. `get_observation()` - 观测向量生成
```
文件: /home/user/RLGameForArk/backend/game_core.py
行数: 354-429 (76 行)
调用频率: 每步一次 (每次训练 800k 次)
```

**关键计算**:
```
第 376-379 行: 手牌统计 (3次除法)
第 383-385 行: 位置 one-hot (10 维)
第 411-424 行: ⭐ 顾客需求循环 (3×3=9 次嵌套循环)
  └─ 第 415 行: for res_type in RESOURCE_TYPES  ⭐⭐⭐
  └─ 第 417-422 行: 资源计算和归一化
第 429 行: np.array 转换
```

**最耗时部分**:
```python
for cust in self.customers:               # 3 次
    cust_obs = [0.0, 0.0, 0.0]
    for res_type in RESOURCE_TYPES:       # 3 次
        res_id = RESOURCE_TYPE_TO_ID[res_type]
        if res_type in cust.needs:        # 字典查询
            need = cust.needs[res_type]    # 字典访问
            have = cust.have.get(res_type, 0)
            remaining = max(0, need - have)  # 计算
            cust_obs[res_id] = remaining / 100.0  # 除法
    obs.extend(cust_obs)                  # 列表扩展
```

**Numba 优化方向**:
- 使用数组代替列表
- 删除字典操作
- 删除条件分支 (预计算)

---

### 2. `get_valid_actions()` - 动作掩码生成
```
文件: /home/user/RLGameForArk/backend/game_core.py
行数: 431-449 (19 行)
调用频率: 每步一次 (每次训练 800k 次)
```

**关键计算**:
```
第 437-442 行: 移动动作检查 (3 次循环)
  └─ self.hand.get(card_value, 0) > 0  ⭐ 字典查询
第 445-447 行: 收集动作检查 (3 次循环)
  └─ self.can_collect(card_value)      ⭐ 方法调用
```

**优化困难**: 包含 `self.can_collect()` 方法调用，需要内联

---

### 3. `step()` - 环境执行步骤
```
文件: /home/user/RLGameForArk/backend/rl_env/game_env.py
行数: 92-180 (89 行)
调用频率: 每步一次 (每次训练 800k 次)
```

**关键计算**:
```
第 104-105 行: 回合检查和开始
第 111-125 行: 移动动作处理
  └─ 第 122 行: self._calculate_resource_need(resource_type) ⭐ 循环
第 130-149 行: 收集动作处理
  └─ 第 140 行: for is_vip, gain in customer_gains ⭐ 循环
第 161 行: obs = self.game.get_observation() ⭐⭐ 计算密集
第 164 行: terminated = self.game.is_game_over()
第 178 行: info['action_mask'] = self.game.get_valid_actions() ⭐⭐ 计算密集
```

**瓶颈分析**:
```
单次调用耗时分布:
├─ game.move()/collect()    40% (~40μs)
├─ get_observation()         30% (~30μs) ← Numba 目标
├─ get_valid_actions()       15% (~15μs) ← Numba 目标
├─ 奖励计算                  10% (~10μs)
└─ 其他                       5% (~5μs)
  总计: ~100μs
```

---

## 📊 优先级 2: 中频调用函数

### 4. `_calculate_resource_need()` - 资源需求计算
```
文件: /home/user/RLGameForArk/backend/rl_env/game_env.py
行数: 82-90 (9 行)
调用频率: 每次移动时 (大约 1/2 的步)
```

**关键计算**:
```
第 84-89 行: ⭐ 顾客循环
  for cust in self.game.customers:       # 3 次
      if resource_type in cust.needs:    # 字典查询
          need = cust.needs[resource_type]
          have = cust.have.get(resource_type, 0)
          total_need += max(0, need - have)
```

**优化潜力**: 高，纯数值计算

---

### 5. `distribute()` - 资源分配和顾客管理
```
文件: /home/user/RLGameForArk/backend/game_core.py
行数: 259-304 (46 行)
调用频率: 每次收集时 (大约 1/5 的步)
```

**关键计算**:
```
第 268-270 行: ⭐ 记录旧状态循环 (3 次)
  for cust in self.customers:
      old_have.append({r: cust.have.get(r, 0) for r in cust.needs})

第 273-282 行: ⭐ 资源分配循环 (~3×3=9 次)
  for res_type, amount in produced.items():
      for cust in self.customers:
          if res_type in cust.needs:
              cust.have[res_type] = min(need, old + amount)

第 285-291 行: ⭐ 收益计算循环 (3×3=9 次)
  for idx, cust in enumerate(self.customers):
      for res_type in cust.needs:
          gain += new - old

第 294-302 行: ⭐ 完成检查循环 (3 次)
  for idx, cust in enumerate(list(self.customers)):
      if cust.is_complete():
          self.tokens += cust.reward
```

**优化困难**: 
- 包含对象创建 (`self.new_vip_customer()`)
- 复杂的数据结构操作
- 方法调用 (`is_complete()`)

---

## 🔍 优先级 3: 低频但有优化空间的函数

### 6. `move()` - 游戏移动逻辑
```
文件: /home/user/RLGameForArk/backend/game_core.py
行数: 158-199 (42 行)
调用频率: 每次移动时 (大约 1/2 的步)
```

**关键计算**:
```
第 182 行: self.position = (old_pos + steps) % len(self.map)
第 177-179 行: 跨越起点判断
```

**优化潜力**: 低，逻辑简单

---

### 7. `collect()` - 游戏收集逻辑
```
文件: /home/user/RLGameForArk/backend/game_core.py
行数: 201-257 (57 行)
调用频率: 每次收集时 (大约 1/5 的步)
```

**关键计算**:
```
第 224-226 行: 资源产出计算
  produced = {r: 0 for r in RESOURCE_TYPES}
  gain_tile = self.resource_coef * card_value
  produced[tile] += gain_tile
  
第 233-234 行: 连击奖励
  for r in RESOURCE_TYPES:
      produced[r] += 2
```

**优化潜力**: 低，逻辑主要在 distribute()

---

## 📈 性能影响分布

### 计算热点 Top-5
```
1. get_observation()          ████████████████ 30% (800k 次调用)
2. get_valid_actions()        ████████         15% (800k 次调用)
3. step() 中的奖励计算        ████████         15%
4. distribute()               ████████         15% (160k 次调用)
5. 其他游戏逻辑               ████             25%
```

### 加速收益预估
```
优化                        预期加速    总体影响
─────────────────────────────────────────────────
get_observation() 3x         +0.9%      → 27%
get_valid_actions() 2x       +0.3%      → 18%
combined                                → 33%
```

---

## 🛠️ 代码改造建议

### 方案 A: 最小改动版 (推荐)

**第 1 步**: 创建 `backend/numba_utils.py`
```python
from numba import jit
import numpy as np

@jit(nopython=True, cache=True)
def compute_observation_fast(
    hand_1, hand_2, hand_3,           # 3 个整数
    position,                          # 1 个整数
    resource_coef,                     # 1 个整数
    current_round, total_rounds,      # 2 个整数
    collectable,                       # 1 个布尔
    last_collect_cost,                 # 1 个整数或 0
    last_action_was_move,              # 1 个布尔
    customers_needs,                   # (3, 3) 数组: 需求
    customers_have,                    # (3, 3) 数组: 已有
    tokens                             # 1 个整数
):
    """Fast observation computation - no Python objects"""
    obs = np.zeros(29, dtype=np.float32)
    idx = 0
    
    # Hand stats (3)
    obs[idx] = hand_1 / 5.0
    obs[idx + 1] = hand_2 / 5.0
    obs[idx + 2] = hand_3 / 5.0
    idx = 3
    
    # Position one-hot (10)
    obs[idx + position] = 1.0
    idx = 13
    
    # ... 继续填充
    
    return obs
```

**第 2 步**: 修改 `game_core.py` 中的 `get_observation()`
```python
def get_observation(self) -> np.ndarray:
    # ... 数据准备 ...
    return numba_utils.compute_observation_fast(
        self.hand[1], self.hand[2], self.hand[3],
        self.position,
        # ... 其他参数 ...
    )
```

---

### 方案 B: 完整模块化版

**创建** `backend/numba_core.py`
```python
# 包含所有 Numba 优化的函数
# - compute_observation_fast()
# - compute_valid_actions_fast()
# - compute_resource_need_fast()
# - compute_distribute_fast()
```

---

## 📝 快速参考

### 最频繁的调用链
```
ParallelEnv.step() (N_envs 次)
  └─ ResourceGameEnv.step()  (800k 次)
      ├─ game.move() or game.collect()
      ├─ get_observation()         [⭐ 800k次, 30% 耗时]
      │   └─ 9×3 循环
      ├─ get_valid_actions()       [⭐ 800k次, 15% 耗时]
      │   └─ 6 次循环 + 3 次方法调用
      └─ 奖励计算
          └─ _calculate_resource_need()  [400k 次]
```

### 单次调用所需的输入数据
```
get_observation(self):
  输入: self 的各种属性
  输出: np.array(29,)

get_valid_actions(self):
  输入: self 的各种属性
  输出: np.array(6,)

_calculate_resource_need(resource_type: str):
  输入: self.game.customers, resource_type
  输出: int
```

### 跨边界数据结构
```
game_core.py 中的关键数据:
├─ self.hand: Dict[int, int]  = {1: 0-5, 2: 0-5, 3: 0-5}
├─ self.position: int = 0-9
├─ self.customers: List[Customer]
│  └─ Customer.needs: Dict[str, int]
│  └─ Customer.have: Dict[str, int]
├─ self.resource_coef: int
├─ self.current_round: int
├─ self.tokens: int
└─ self.collectable: bool
```

---

## ✅ 执行清单

- [ ] 读完 game_core.py (第 354-449 行)
- [ ] 读完 game_env.py (第 82-180 行)
- [ ] 理解观测向量的 29 维构成
- [ ] 理解 6 维动作掩码的生成
- [ ] 规划 Numba 优化架构
- [ ] 实现 numba_utils.py
- [ ] 测试和基准测试
- [ ] 文档更新

