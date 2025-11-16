"""
游戏核心逻辑 - 优化版本，用于强化学习训练
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

# 三种资源类型
RESOURCE_TYPES = ['冰', '铁', '火']
RESOURCE_TYPE_TO_ID = {'冰': 0, '铁': 1, '火': 2}


@dataclass
class Customer:
    """顾客类"""
    is_vip: bool                     # 是否为高价值顾客
    needs: Dict[str, int]            # 需求：资源 -> 数量
    have: Dict[str, int] = field(default_factory=dict)  # 已经被满足的数量
    reward: int = 0                  # 完成后给多少代币

    def __post_init__(self):
        # 初始化 have
        for r in self.needs:
            self.have.setdefault(r, 0)
        if self.reward == 0:
            self.reward = 4 if self.is_vip else 1

    def is_complete(self) -> bool:
        """是否已经完全满足需求"""
        return all(self.have.get(r, 0) >= need for r, need in self.needs.items())

    def progress_str(self, label: str = "") -> str:
        """显示进度用的字符串"""
        parts = []
        for r, need in self.needs.items():
            have = self.have.get(r, 0)
            parts.append(f"{have}/{need}{r}")
        t = "高价值" if self.is_vip else "普通"
        head = f"{label}{t}顾客：" if label else f"{t}顾客："
        return head + "，".join(parts) + f"（完成 +{self.reward} 代币）"

    def to_dict(self) -> dict:
        """转换为字典（用于API返回）"""
        return {
            'is_vip': self.is_vip,
            'needs': self.needs,
            'have': self.have,
            'reward': self.reward,
            'is_complete': self.is_complete()
        }


class ResourceGame:
    """资源收集游戏核心逻辑"""

    def __init__(self, rounds: int = 10, seed: Optional[int] = None):
        self.rounds = rounds
        # 地图：10 格环形资源点
        self.map = ['冰', '铁', '火', '冰', '铁', '火', '冰', '铁', '火', '火']
        self.map_ids = [RESOURCE_TYPE_TO_ID[r] for r in self.map]

        self.position = 0             # 当前所在格子（0~9）
        self.resource_coef = 3        # 初始资源系数
        self.current_round = 0        # 当前回合数
        self.rng = np.random.RandomState(seed)

        # 回合内状态
        self.hand: List[int] = []     # 手牌
        self.collectable = False      # 是否处于"可普通收集"状态（刚移动过）
        self.last_collect_cost = None # 上一次收集所用卡牌点数 n
        self.last_action_was_move = False

        # 顾客 & 代币
        self.tokens = 0
        self.customers: List[Customer] = []
        self.init_customers()

        # 统计信息
        self.total_moves = 0
        self.total_collections = 0
        self.action_history = []

    def init_customers(self):
        """初始化三名顾客：1 高价值 + 2 普通"""
        self.customers = [
            self.new_vip_customer(),
            self.new_normal_customer(),
            self.new_normal_customer()
        ]

    def new_vip_customer(self) -> Customer:
        """生成一个高价值顾客：两种资源，100/40 或 70/70，奖励 4 代币"""
        types = self.rng.choice(RESOURCE_TYPES, size=2, replace=False).tolist()
        # 随机选择两种模式之一：100+40 或 70+70
        pattern = self.rng.randint(0, 2)
        if pattern == 0:
            needs = {types[0]: 100, types[1]: 40}
        else:
            needs = {types[0]: 70, types[1]: 70}
        return Customer(is_vip=True, needs=needs, reward=4)

    def new_normal_customer(self) -> Customer:
        """生成一个普通顾客"""
        pattern = self.rng.randint(0, 3)
        types = self.rng.choice(RESOURCE_TYPES, size=2, replace=False).tolist()
        if pattern == 0:
            t = self.rng.choice(RESOURCE_TYPES)
            needs = {t: 35}
        elif pattern == 1:
            needs = {types[0]: 20, types[1]: 20}
        else:
            needs = {types[0]: 30, types[1]: 10}
        return Customer(is_vip=False, needs=needs, reward=1)

    def deal_hand(self):
        """发 5 张 1~3 点的随机手牌"""
        self.hand = self.rng.randint(1, 4, size=5).tolist()

    def tile_type(self) -> str:
        """当前位置的资源类型"""
        return self.map[self.position]

    def tile_id(self) -> int:
        """当前位置的资源类型ID"""
        return self.map_ids[self.position]

    def can_move(self) -> bool:
        """是否可以移动（有手牌）"""
        return len(self.hand) > 0

    def can_collect(self, card_index: int) -> bool:
        """是否可以使用指定手牌进行收集"""
        if card_index < 0 or card_index >= len(self.hand):
            return False

        # 检查是否可以连击
        combo_indices = self.can_combo_indices()
        if card_index in combo_indices:
            return True

        # 检查是否可以普通收集
        return self.collectable

    def can_combo_indices(self) -> List[int]:
        """当前可用于连击的手牌索引列表"""
        if self.last_collect_cost is None or self.last_action_was_move:
            return []
        target = self.last_collect_cost + 1
        return [i for i, v in enumerate(self.hand) if v == target]

    def move(self, card_index: int) -> Tuple[bool, str]:
        """
        使用一张牌进行移动
        返回：(成功, 消息)
        """
        if card_index < 0 or card_index >= len(self.hand):
            return False, "卡牌索引无效"

        value = self.hand.pop(card_index)
        old_pos = self.position
        steps = value

        # 判断是否绕过起点（0 号格）
        crossed_start = False
        if old_pos + steps >= len(self.map):
            self.resource_coef += 2
            crossed_start = True

        # 环形前进
        self.position = (old_pos + steps) % len(self.map)
        self.collectable = True
        self.last_action_was_move = True
        self.total_moves += 1

        msg = f"使用 {value} 点卡牌前进 {steps} 步，到达 [{self.position}] {self.tile_type()}"
        if crossed_start:
            msg += f"。绕过起点，资源系数 +2，当前：{self.resource_coef}"

        self.action_history.append({
            'type': 'move',
            'card_value': value,
            'old_position': old_pos,
            'new_position': self.position,
            'crossed_start': crossed_start
        })

        return True, msg

    def collect(self, card_index: int) -> Tuple[bool, str, int]:
        """
        使用一张牌进行收集
        返回：(成功, 消息, 获得的代币数)
        """
        if card_index < 0 or card_index >= len(self.hand):
            return False, "卡牌索引无效", 0

        # 检查是否可以收集
        combo_indices = self.can_combo_indices()
        is_combo = card_index in combo_indices

        if not is_combo and not self.collectable:
            return False, "当前不能进行普通收集", 0

        value = self.hand.pop(card_index)
        tile = self.tile_type()

        # 本次产出的资源
        produced = {r: 0 for r in RESOURCE_TYPES}
        gain_tile = self.resource_coef * value
        produced[tile] += gain_tile

        msg_parts = []
        if not is_combo:
            msg_parts.append(f"普通收集：在 {tile} 上使用 {value} 点卡牌，产出 {gain_tile} 个{tile}")
        else:
            # 连击：额外 +2 冰/+2 铁/+2 火
            for r in RESOURCE_TYPES:
                produced[r] += 2
            msg_parts.append(f"连击收集：在 {tile} 上使用 {value} 点卡牌，产出 {gain_tile} 个{tile}，额外 +2 全资源")

        self.last_collect_cost = value
        self.collectable = False
        self.last_action_was_move = False
        self.total_collections += 1

        # 分配资源并检查完成情况
        tokens_earned = self.distribute(produced)

        if tokens_earned > 0:
            msg_parts.append(f"完成订单，获得 {tokens_earned} 代币！")

        self.action_history.append({
            'type': 'collect',
            'card_value': value,
            'position': self.position,
            'is_combo': is_combo,
            'produced': produced.copy(),
            'tokens_earned': tokens_earned
        })

        return True, "；".join(msg_parts), tokens_earned

    def distribute(self, produced: Dict[str, int]) -> int:
        """
        把本次产出的资源分配给所有顾客
        返回：本次获得的代币数
        """
        tokens_earned = 0

        # 先分配
        for res_type, amount in produced.items():
            if amount <= 0:
                continue
            for cust in self.customers:
                if res_type in cust.needs:
                    old = cust.have.get(res_type, 0)
                    need = cust.needs[res_type]
                    if old >= need:
                        continue
                    cust.have[res_type] = min(need, old + amount)

        # 再检查完成情况
        for idx, cust in enumerate(list(self.customers)):
            if cust.is_complete():
                self.tokens += cust.reward
                tokens_earned += cust.reward
                # 换成同类型新顾客
                if cust.is_vip:
                    self.customers[idx] = self.new_vip_customer()
                else:
                    self.customers[idx] = self.new_normal_customer()

        return tokens_earned

    def start_round(self) -> bool:
        """
        开始新回合
        返回：是否还有回合可玩
        """
        self.current_round += 1
        if self.current_round > self.rounds:
            return False

        self.deal_hand()
        self.collectable = False
        self.last_collect_cost = None
        self.last_action_was_move = False
        return True

    def is_round_over(self) -> bool:
        """当前回合是否结束（无牌可出）"""
        return len(self.hand) == 0

    def is_game_over(self) -> bool:
        """游戏是否结束"""
        return self.current_round >= self.rounds and self.is_round_over()

    def get_state(self) -> dict:
        """
        获取完整游戏状态（用于API和可视化）
        """
        return {
            'current_round': self.current_round,
            'total_rounds': self.rounds,
            'position': self.position,
            'resource_type': self.tile_type(),
            'resource_coef': self.resource_coef,
            'hand': self.hand.copy(),
            'collectable': self.collectable,
            'last_collect_cost': self.last_collect_cost,
            'can_combo_indices': self.can_combo_indices(),
            'customers': [c.to_dict() for c in self.customers],
            'tokens': self.tokens,
            'is_round_over': self.is_round_over(),
            'is_game_over': self.is_game_over(),
            'map': self.map,
            'stats': {
                'total_moves': self.total_moves,
                'total_collections': self.total_collections
            }
        }

    def get_observation(self) -> np.ndarray:
        """
        获取观测向量（用于强化学习）

        观测维度：
        - 手牌（5维，每个0-3）
        - 位置（10维，one-hot）
        - 资源系数（1维）
        - 当前回合（1维）
        - 是否可收集（1维）
        - 上次收集代价（1维，0-3）
        - 顾客需求和进度（每个顾客6维 x 3 = 18维）
          - 需求资源1类型（3维one-hot）
          - 需求资源1数量（1维，归一化）
          - 需求资源1进度（1维，归一化）
          - （如果有第二种资源，类似）
        - 代币数（1维）

        总维度：5 + 10 + 1 + 1 + 1 + 1 + 18 + 1 = 38
        """
        obs = []

        # 手牌（5维）
        hand_padded = self.hand + [0] * (5 - len(self.hand))
        obs.extend(hand_padded[:5])

        # 位置（10维one-hot）
        pos_onehot = [0] * 10
        pos_onehot[self.position] = 1
        obs.extend(pos_onehot)

        # 资源系数（归一化，最大假设为15）
        obs.append(self.resource_coef / 15.0)

        # 当前回合（归一化）
        obs.append(self.current_round / self.rounds)

        # 是否可收集
        obs.append(1.0 if self.collectable else 0.0)

        # 上次收集代价（归一化）
        obs.append((self.last_collect_cost or 0) / 3.0)

        # 顾客信息（每个6维 x 3）
        for cust in self.customers:
            cust_obs = [0] * 6

            # 提取需求信息
            needs_items = list(cust.needs.items())
            if len(needs_items) >= 1:
                res1_type, res1_need = needs_items[0]
                res1_have = cust.have.get(res1_type, 0)
                # 资源类型（不编码one-hot，直接用ID/2归一化）
                cust_obs[0] = RESOURCE_TYPE_TO_ID[res1_type] / 2.0
                # 需求数量（归一化，最大假设100）
                cust_obs[1] = res1_need / 100.0
                # 进度比例
                cust_obs[2] = res1_have / res1_need if res1_need > 0 else 0

            if len(needs_items) >= 2:
                res2_type, res2_need = needs_items[1]
                res2_have = cust.have.get(res2_type, 0)
                cust_obs[3] = RESOURCE_TYPE_TO_ID[res2_type] / 2.0
                cust_obs[4] = res2_need / 100.0
                cust_obs[5] = res2_have / res2_need if res2_need > 0 else 0

            obs.extend(cust_obs)

        # 代币数（归一化，最大假设20）
        obs.append(self.tokens / 20.0)

        return np.array(obs, dtype=np.float32)

    def get_valid_actions(self) -> np.ndarray:
        """
        获取有效动作掩码（用于强化学习）
        动作空间：[move_0, move_1, move_2, move_3, move_4,
                  collect_0, collect_1, collect_2, collect_3, collect_4]
        返回：10维的0/1数组，1表示该动作有效
        """
        valid = np.zeros(10, dtype=np.float32)

        # 移动动作：只要有对应手牌就有效
        for i in range(min(5, len(self.hand))):
            valid[i] = 1

        # 收集动作：需要检查是否可收集
        for i in range(min(5, len(self.hand))):
            if self.can_collect(i):
                valid[5 + i] = 1

        return valid

    def reset(self, seed: Optional[int] = None):
        """重置游戏"""
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        self.position = 0
        self.resource_coef = 3
        self.current_round = 0
        self.hand = []
        self.collectable = False
        self.last_collect_cost = None
        self.last_action_was_move = False
        self.tokens = 0
        self.total_moves = 0
        self.total_collections = 0
        self.action_history = []

        self.init_customers()
        self.start_round()

        return self.get_observation()
