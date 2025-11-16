import random
from dataclasses import dataclass, field
from typing import Dict, List

# 三种资源类型
RESOURCE_TYPES = ['冰', '铁', '火']


@dataclass
class Customer:
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


class ResourceGame:
    def __init__(self, rounds: int = 10, seed=None):
        self.rounds = rounds
        # 地图：10 格环形资源点
        self.map = ['冰', '铁', '火', '冰', '铁', '火', '冰', '铁', '火', '火']
        self.position = 0             # 当前所在格子（0~9）
        self.resource_coef = 3        # 初始资源系数
        self.rng = random.Random(seed)

        # 回合内状态
        self.hand: List[int] = []     # 手牌
        self.collectable = False      # 是否处于“可普通收集”状态（刚移动过）
        self.last_collect_cost = None # 上一次收集所用卡牌点数 n
        self.last_action_was_move = False

        # 顾客 & 代币
        self.tokens = 0
        self.customers: List[Customer] = []
        self.init_customers()

    # ========== 顾客相关 ==========

    def init_customers(self):
        """初始化三名顾客：1 高价值 + 2 普通"""
        self.customers = [
            self.new_vip_customer(),
            self.new_normal_customer(),
            self.new_normal_customer()
        ]

    def new_vip_customer(self) -> Customer:
        """生成一个高价值顾客：两种资源，100 / 40，奖励 4 代币"""
        types = self.rng.sample(RESOURCE_TYPES, 2)
        needs = {types[0]: 100, types[1]: 40}
        return Customer(is_vip=True, needs=needs, reward=4)

    def new_normal_customer(self) -> Customer:
        """生成一个普通顾客：
        三种模式之一：
        1) 随机一种资源 35 个
        2) 随机两种资源，各 20 个
        3) 随机两种资源，第一种 30 个，第二种 10 个
        """
        pattern = self.rng.randint(0, 2)
        types = self.rng.sample(RESOURCE_TYPES, 2)
        if pattern == 0:
            t = self.rng.choice(RESOURCE_TYPES)
            needs = {t: 35}
        elif pattern == 1:
            needs = {types[0]: 20, types[1]: 20}
        else:
            needs = {types[0]: 30, types[1]: 10}
        return Customer(is_vip=False, needs=needs, reward=1)

    # ========== 核心游戏机制 ==========

    def deal_hand(self):
        """发 5 张 1~3 点的随机手牌"""
        self.hand = [self.rng.randint(1, 3) for _ in range(5)]

    def tile_type(self) -> str:
        """当前位置的资源类型"""
        return self.map[self.position]

    def move(self, card_index: int):
        """使用一张牌进行移动"""
        value = self.hand.pop(card_index)
        old_pos = self.position
        steps = value

        # 判断是否绕过起点（0 号格）
        if old_pos + steps >= len(self.map):
            self.resource_coef += 2
            print(f"你绕过了起始点，资源系数 +2！当前资源系数：{self.resource_coef}")

        # 环形前进
        self.position = (old_pos + steps) % len(self.map)
        self.collectable = True
        self.last_action_was_move = True
        print(f"你使用 {value} 点卡牌前进 {steps} 步，停在 [{self.position}] {self.tile_type()}。")

    def can_combo_indices(self) -> List[int]:
        """当前可用于连击的手牌索引列表"""
        if self.last_collect_cost is None or self.last_action_was_move:
            return []
        target = self.last_collect_cost + 1
        return [i for i, v in enumerate(self.hand) if v == target]

    def collect(self, card_index: int, is_combo: bool = False):
        """使用一张牌进行收集（普通或连击）"""
        value = self.hand.pop(card_index)
        tile = self.tile_type()

        # 本次“理论上”产出的资源（不入背包，直接给顾客）
        produced = {r: 0 for r in RESOURCE_TYPES}
        gain_tile = self.resource_coef * value
        produced[tile] += gain_tile

        if not is_combo:
            print(f"普通收集：在 {tile} 上使用 {value} 点卡牌，产出 {gain_tile} 个{tile}。")
        else:
            # 连击：在基础产出的基础上，额外 +2 冰/+2 铁/+2 火
            for r in RESOURCE_TYPES:
                produced[r] += 2
            print(
                f"连击收集：在 {tile} 上使用 {value} 点卡牌，产出 {gain_tile} 个{tile}，"
                f"额外 +2 冰/+2 铁/+2 火（对每位需要此资源的顾客都生效）。"
            )

        self.last_collect_cost = value
        self.collectable = False
        self.last_action_was_move = False

        # 显示本次总产出
        print("本次总资源产出：", "，".join(f"{k}:{v}" for k, v in produced.items() if v > 0))

        # 把这些资源“复制”给所有顾客
        self.distribute(produced)

    def distribute(self, produced: Dict[str, int]):
        """把本次产出的资源分配给所有顾客（按你示例的规则：同样数量同时给所有需要该资源的顾客）"""
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
                print(f"=== 顾客{idx + 1}订单完成！获得 {cust.reward} 个代币！===")
                # 换成同类型新顾客（新顾客不会吃到这次产出，只从下一次收集开始算）
                if cust.is_vip:
                    self.customers[idx] = self.new_vip_customer()
                else:
                    self.customers[idx] = self.new_normal_customer()
                print(f"顾客{idx + 1}已更换为新订单：{self.customers[idx].progress_str(label=f'顾客{idx+1}-')}")

    # ========== UI 辅助 ==========

    def print_customers(self):
        print("\n当前顾客状态：")
        for i, cust in enumerate(self.customers, start=1):
            print("  " + cust.progress_str(label=f"顾客{i}-"))
        print(f"当前代币数：{self.tokens}")

    def print_status(self, round_no: int):
        print("\n" + "=" * 50)
        print(
            f"第 {round_no} 回合 / {self.rounds} 回合  |  "
            f"当前位置：[{self.position}] {self.tile_type()}  |  资源系数：{self.resource_coef}"
        )
        print("手牌：", end="")
        if not self.hand:
            print("（无）")
        else:
            print("  ".join(f"[{i}] {v}" for i, v in enumerate(self.hand)))

        if self.collectable:
            print("当前格子可进行【普通收集】。")
        combo_idxs = self.can_combo_indices()
        if combo_idxs:
            target = self.last_collect_cost + 1
            print(f"可进行【连击收集】（需 {target} 点牌），可用牌序号：{combo_idxs}")

        self.print_customers()
        print("操作说明：m X  -> 用序号 X 的牌移动")
        print("          c X  -> 用序号 X 的牌收集 / 连击")
        print("          e    -> 提前结束本回合")
        print("=" * 50)

    # ========== 回合与总流程 ==========

    def play_round(self, round_no: int):
        self.deal_hand()
        self.collectable = False
        self.last_collect_cost = None
        self.last_action_was_move = False

        print(f"\n=== 第 {round_no} 回合开始，新手牌：{self.hand} ===")

        while self.hand:
            self.print_status(round_no)
            cmd = input("请输入指令（例如 m 0 或 c 2）：").strip().lower()
            if not cmd:
                continue

            if cmd == "e":
                print("你选择提前结束本回合。")
                break

            # 支持两种输入： "m 0" / "c 2" 或 "m0" / "c2"
            parts = cmd.split()
            if len(parts) == 1:
                action = cmd[0]
                idx_str = cmd[1:]
            else:
                action, idx_str = parts[0], parts[1]

            if action not in ("m", "c"):
                print("指令错误，请使用 m 或 c。")
                continue
            if not idx_str.isdigit():
                print("卡牌序号必须是数字。")
                continue

            card_index = int(idx_str)
            if not (0 <= card_index < len(self.hand)):
                print("卡牌序号不存在。")
                continue

            if action == "m":
                self.move(card_index)
            else:  # 收集
                combo_idxs = self.can_combo_indices()
                is_combo = card_index in combo_idxs
                if is_combo:
                    self.collect(card_index, is_combo=True)
                else:
                    if not self.collectable:
                        print("当前不能进行普通收集（需要先移动，或者使用可连击的牌）。")
                        continue
                    self.collect(card_index, is_combo=False)

        print(f"第 {round_no} 回合结束，本回合剩余手牌清空。")

    def play(self):
        print("欢迎来到【环形资源连击 · 顾客代币版】！")
        print("地图顺序：", " -> ".join(f"[{i}] {t}" for i, t in enumerate(self.map)))
        print("你总是面对 3 位顾客：")
        print("  顾客1：高价值顾客，需求 2 种资源（100 / 40），完成给 4 代币；")
        print("  顾客2 & 3：普通顾客，从几种需求模式里随机生成，完成各给 1 代币。")
        print("每次你收集到的资源，不再进入背包，而是同时喂给所有需要该资源的顾客。")
        print("游戏共 10 个回合，最后看你拿了多少代币。\n")

        for r in range(1, self.rounds + 1):
            self.play_round(r)

        print("\n=== 游戏结束 ===")
        print(f"你最终获得了 {self.tokens} 个代币。谢谢游玩！")


if __name__ == "__main__":
    # 可以指定 seed 方便复盘：例如 ResourceGame(seed=0)
    game = ResourceGame(rounds=10, seed=None)
    game.play()
