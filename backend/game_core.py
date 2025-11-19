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
        self.initial_seed = seed      # 保存初始seed，用于replay
        # 地图：10 格环形资源点
        self.map = ['冰', '铁', '火', '冰', '铁', '火', '冰', '铁', '火', '火']
        self.map_ids = [RESOURCE_TYPE_TO_ID[r] for r in self.map]

        self.position = 0             # 当前所在格子（0~9）
        self.resource_coef = 3        # 初始资源系数
        self.current_round = 0        # 当前回合数
        self.rng = np.random.RandomState(seed)

        # 回合内状态
        self.hand: Dict[int, int] = {1: 0, 2: 0, 3: 0}  # 手牌统计：点数 -> 数量
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
        self.action_history = []      # 旧的action历史（保持兼容性）
        self.transitions = []         # 新的完整transition记录（用于强化学习）

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
        """发 5 张 1~3 点的随机手牌，并统计每个点数的数量"""
        cards = self.rng.randint(1, 4, size=5)
        self.hand = {1: 0, 2: 0, 3: 0}
        for card in cards:
            self.hand[card] += 1

    def tile_type(self) -> str:
        """当前位置的资源类型"""
        return self.map[self.position]

    def tile_id(self) -> int:
        """当前位置的资源类型ID"""
        return self.map_ids[self.position]

    def can_move(self) -> bool:
        """是否可以移动（有手牌）"""
        return any(count > 0 for count in self.hand.values())

    def can_collect(self, card_value: int) -> bool:
        """是否可以使用指定点数的卡牌进行收集"""
        if card_value not in [1, 2, 3] or self.hand.get(card_value, 0) == 0:
            return False

        # 检查是否可以连击
        combo_values = self.can_combo_values()
        if card_value in combo_values:
            return True

        # 检查是否可以普通收集
        return self.collectable

    def can_combo_values(self) -> List[int]:
        """当前可用于连击的卡牌点数列表"""
        if self.last_collect_cost is None or self.last_action_was_move:
            return []
        target = self.last_collect_cost + 1
        if target in [1, 2, 3] and self.hand.get(target, 0) > 0:
            return [target]
        return []

    def move(self, card_value: int) -> Tuple[bool, str]:
        """
        使用指定点数的卡牌进行移动
        返回：(成功, 消息)
        """
        if card_value not in [1, 2, 3]:
            return False, "卡牌点数无效"

        if self.hand.get(card_value, 0) == 0:
            return False, f"没有 {card_value} 点的卡牌"

        # 使用一张该点数的卡
        self.hand[card_value] -= 1

        old_pos = self.position
        steps = card_value

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

        msg = f"使用 {card_value} 点卡牌前进 {steps} 步，到达 [{self.position}] {self.tile_type()}"
        if crossed_start:
            msg += f"。绕过起点，资源系数 +2，当前：{self.resource_coef}"

        self.action_history.append({
            'type': 'move',
            'card_value': card_value,
            'old_position': old_pos,
            'new_position': self.position,
            'crossed_start': crossed_start
        })

        return True, msg

    def collect(self, card_value: int) -> Tuple[bool, str, int, List[Tuple[bool, int]]]:
        """
        使用指定点数的卡牌进行收集
        返回：(成功, 消息, 获得的代币数, 每个顾客的(是否VIP, 获得的有效资源量)列表)
        """
        if card_value not in [1, 2, 3]:
            return False, "卡牌点数无效", 0, []

        if self.hand.get(card_value, 0) == 0:
            return False, f"没有 {card_value} 点的卡牌", 0, []

        # 检查是否可以收集
        combo_values = self.can_combo_values()
        is_combo = card_value in combo_values

        if not is_combo and not self.collectable:
            return False, "当前不能进行普通收集", 0, []

        # 使用一张该点数的卡
        self.hand[card_value] -= 1
        tile = self.tile_type()

        # 本次产出的资源
        produced = {r: 0 for r in RESOURCE_TYPES}
        gain_tile = self.resource_coef * card_value
        produced[tile] += gain_tile

        msg_parts = []
        if not is_combo:
            msg_parts.append(f"普通收集：在 {tile} 上使用 {card_value} 点卡牌，产出 {gain_tile} 个{tile}")
        else:
            # 连击：额外 +2 冰/+2 铁/+2 火
            for r in RESOURCE_TYPES:
                produced[r] += 2
            msg_parts.append(f"连击收集：在 {tile} 上使用 {card_value} 点卡牌，产出 {gain_tile} 个{tile}，额外 +2 全资源")

        self.last_collect_cost = card_value
        self.collectable = False
        self.last_action_was_move = False
        self.total_collections += 1

        # 分配资源并检查完成情况
        tokens_earned, customer_gains = self.distribute(produced)

        if tokens_earned > 0:
            msg_parts.append(f"完成订单，获得 {tokens_earned} 代币！")

        self.action_history.append({
            'type': 'collect',
            'card_value': card_value,
            'position': self.position,
            'is_combo': is_combo,
            'produced': produced.copy(),
            'tokens_earned': tokens_earned
        })

        return True, "；".join(msg_parts), tokens_earned, customer_gains

    def distribute(self, produced: Dict[str, int]) -> Tuple[int, List[Tuple[bool, int]]]:
        """
        把本次产出的资源分配给所有顾客
        返回：(本次获得的代币数, 每个顾客的(是否VIP, 获得的有效资源量)列表)
        """
        tokens_earned = 0
        customer_gains = []

        # 先记录每个顾客的旧状态
        old_have = []
        for cust in self.customers:
            old_have.append({r: cust.have.get(r, 0) for r in cust.needs})

        # 分配资源
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

        # 计算每个顾客实际获得的有效资源量
        for idx, cust in enumerate(self.customers):
            gain = 0
            for res_type in cust.needs:
                old = old_have[idx].get(res_type, 0)
                new = cust.have.get(res_type, 0)
                gain += new - old
            customer_gains.append((cust.is_vip, gain))

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

        return tokens_earned, customer_gains

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
        return all(count == 0 for count in self.hand.values())

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
            'can_combo_values': self.can_combo_values(),
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
        - 手牌统计（3维）：1点、2点、3点各有几张（0-5）
        - 位置（10维，one-hot）
        - 资源系数（1维）
        - 当前回合（1维）
        - 是否可普通收集（1维）
        - 是否可连击（1维）
        - 上次收集代价（2维 one-hot）：[收集1?, 收集2?]
          - 如果上次收集用1点：[1, 0]
          - 如果上次收集用2点：[0, 1]
          - 如果上次收集用3点或上次动作不是收集：[0, 0]
        - 顾客需求信息（9维 = 每个顾客3维 x 3个顾客）
          - 每个顾客3维：[冰仍需量, 铁仍需量, 火仍需量]
        - 代币数（1维）

        总维度：3 + 10 + 1 + 1 + 1 + 1 + 2 + 9 + 1 = 29
        """
        obs = []

        # 手牌统计（3维）：归一化到0-1之间，最大5张
        obs.append(self.hand.get(1, 0) / 5.0)
        obs.append(self.hand.get(2, 0) / 5.0)
        obs.append(self.hand.get(3, 0) / 5.0)

        # 位置（10维one-hot）
        pos_onehot = [0] * 10
        pos_onehot[self.position] = 1
        obs.extend(pos_onehot)

        # 资源系数（归一化，最大假设为15）
        obs.append(self.resource_coef / 15.0)

        # 当前回合（归一化）
        obs.append(self.current_round / self.rounds)

        # 是否可普通收集
        obs.append(1.0 if self.collectable else 0.0)

        # 是否可连击
        can_combo = len(self.can_combo_values()) > 0
        obs.append(1.0 if can_combo else 0.0)

        # 上次收集代价（2维 one-hot）
        last_collect_1 = 0.0
        last_collect_2 = 0.0
        if self.last_collect_cost == 1 and not self.last_action_was_move:
            last_collect_1 = 1.0
        elif self.last_collect_cost == 2 and not self.last_action_was_move:
            last_collect_2 = 1.0
        obs.append(last_collect_1)
        obs.append(last_collect_2)

        # 顾客信息（每个顾客3维：冰/铁/火仍需量）
        for cust in self.customers:
            cust_obs = [0.0, 0.0, 0.0]  # [冰, 铁, 火]

            # 计算每种资源的仍需量（需求量 - 已有量）
            for res_type in RESOURCE_TYPES:
                res_id = RESOURCE_TYPE_TO_ID[res_type]
                if res_type in cust.needs:
                    need = cust.needs[res_type]
                    have = cust.have.get(res_type, 0)
                    remaining = max(0, need - have)
                    # 归一化（最大假设100）
                    cust_obs[res_id] = remaining / 100.0

            obs.extend(cust_obs)

        # 代币数（归一化，最大假设20）
        obs.append(self.tokens / 20.0)

        return np.array(obs, dtype=np.float32)

    def get_valid_actions(self) -> np.ndarray:
        """
        获取有效动作掩码（用于强化学习）
        动作空间：[move_1, move_2, move_3, collect_1, collect_2, collect_3]
        返回：6维的0/1数组，1表示该动作有效
        """
        valid = np.zeros(6, dtype=np.float32)

        # 移动动作（0-2）：只要有对应点数的牌就有效
        for card_value in [1, 2, 3]:
            if self.hand.get(card_value, 0) > 0:
                valid[card_value - 1] = 1

        # 收集动作（3-5）：需要检查是否可收集
        for card_value in [1, 2, 3]:
            if self.can_collect(card_value):
                valid[3 + card_value - 1] = 1

        return valid

    def reset(self, seed: Optional[int] = None):
        """重置游戏"""
        if seed is not None:
            self.initial_seed = seed
            self.rng = np.random.RandomState(seed)

        self.position = 0
        self.resource_coef = 3
        self.current_round = 0
        self.hand = {1: 0, 2: 0, 3: 0}
        self.collectable = False
        self.last_collect_cost = None
        self.last_action_was_move = False
        self.tokens = 0
        self.total_moves = 0
        self.total_collections = 0
        self.action_history = []
        self.transitions = []         # 重置transitions

        self.init_customers()
        self.start_round()

        return self.get_observation()
