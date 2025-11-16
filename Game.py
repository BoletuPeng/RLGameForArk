import random

# ================= 基础配置 =================
BOARD = ["冰", "铁", "火", "冰", "铁", "火", "冰", "铁", "火", "火"]
RESOURCE_TYPES = ["冰", "铁", "火"]

# ================= 数据结构 =================
class GameState:
    def __init__(self):
        self.position = 0              # 当前所在格子索引（0~9）
        self.coeff = 3                 # 资源系数，初始为 3
        self.round_no = 1              # 当前回合（1~10）
        self.hand = []                 # 手牌列表，每张牌是 1~3
        self.resources = {t: 0 for t in RESOURCE_TYPES}  # 拥有的资源数量
        self.can_collect = False       # 是否可以进行普通收集
        self.last_collect_value = None # 上一次收集使用的牌点数（用于判断连击）

    def draw_hand(self):
        """发 5 张新牌（每张 1~3），清空收集状态。"""
        self.hand = [random.randint(1, 3) for _ in range(5)]
        self.can_collect = False
        self.last_collect_value = None

    @property
    def current_tile(self):
        return BOARD[self.position]

# ================= 核心逻辑 =================
def move_player(gs: GameState, card_index: int):
    """使用一张牌进行移动。"""
    if card_index < 0 or card_index >= len(gs.hand):
        print("索引超出范围，请重新输入。")
        return False

    value = gs.hand.pop(card_index)
    start_pos = gs.position
    steps = value
    size = len(BOARD)

    new_pos = (start_pos + steps) % size
    wraps = (start_pos + steps) // size   # 是否绕过了起点

    gs.position = new_pos
    if wraps > 0:
        gs.coeff += 2 * wraps
        print(f"你绕过了起点，资源系数 +2，现在为 {gs.coeff}。")

    gs.can_collect = True           # 移动后可以进行一次普通收集
    gs.last_collect_value = None    # 移动会重置收集连击链

    print(f"你移动了 {steps} 步，来到了格子 {gs.position}（{gs.current_tile}）。")
    return True

def collect_resource(gs: GameState, card_index: int):
    """使用一张牌进行资源收集（普通或连击）。"""
    if card_index < 0 or card_index >= len(gs.hand):
        print("索引超出范围，请重新输入。")
        return False

    card_value = gs.hand[card_index]

    # 判断是否为普通收集或连击
    is_combo = False
    if gs.can_collect:
        # 正常收集（必须在刚刚移动之后）
        is_combo = False
    elif gs.last_collect_value is not None and card_value == gs.last_collect_value + 1:
        # 连击收集（n -> n+1）
        is_combo = True
    else:
        print("现在不能用这张牌收集：需要先移动，或者使用上一张收集牌点数 +1 的牌进行连击。")
        return False

    # 真正消耗该牌
    gs.hand.pop(card_index)
    tile = gs.current_tile
    base_gain = gs.coeff * card_value
    gs.resources[tile] += base_gain

    print(f"你在 {tile} 格子用点数 {card_value} 的牌进行{'【连击】' if is_combo else '收集'}，获得 {base_gain} 个 {tile}。")

    # 连击额外奖励：+2 冰、+2 铁、+2 火
    if is_combo:
        for t in RESOURCE_TYPES:
            gs.resources[t] += 2
        print("连击奖励：额外获得 2 冰、2 铁、2 火！")

    # 收集结束后，不能再次普通收集，但可以继续尝试使用 n+1 进行连击
    gs.can_collect = False
    gs.last_collect_value = card_value

    return True

# ================= 辅助显示 =================
def show_state(gs: GameState):
    print("-" * 40)
    print(f"第 {gs.round_no} 回合 / 10 回合")
    print(f"当前位置：格子 {gs.position}（{gs.current_tile}）")
    print(f"当前资源系数：{gs.coeff}")
    print("当前资源：", end="")
    print("，".join(f"{t}:{gs.resources[t]}" for t in RESOURCE_TYPES))

    # 显示手牌
    print("手牌：", end="")
    if not gs.hand:
        print("（无牌）")
    else:
        # 显示 索引:点数
        hand_str = "  ".join(f"[{i}] {v}" for i, v in enumerate(gs.hand))
        print(hand_str)

    # 显示收集状态提示
    if gs.can_collect:
        print("现在可以进行【普通收集】或继续【移动】。")
    elif gs.last_collect_value is not None and any(
        v == gs.last_collect_value + 1 for v in gs.hand
    ):
        print(f"你可以使用点数为 {gs.last_collect_value + 1} 的牌在原地继续【连击收集】。")
    else:
        print("现在不能收集，只能【移动】（如果还有牌）。")

    print("操作说明：")
    print("  m X  -> 使用索引为 X 的牌进行移动")
    print("  c X  -> 使用索引为 X 的牌进行收集 / 连击")
    print("  e    -> 提前结束本回合（放弃剩余牌）")
    print("-" * 40)

# ================= 主流程 =================
def main():
    print("==== 环形资源连击小游戏 ====")
    print("共有 10 个回合，每回合会发给你 5 张 1~3 点的手牌。")
    print("通过移动和收集，在冰 / 铁 / 火三种资源中刷出高分吧！\n")

    gs = GameState()

    while gs.round_no <= 10:
        print(f"\n===== 开始第 {gs.round_no} 回合 =====")
        gs.draw_hand()
        print(f"本回合新手牌：{gs.hand}")

        # 当回合：直到牌用完或玩家主动结束
        while gs.hand:
            show_state(gs)
            cmd = input("请输入指令（例如 m 0 或 c 2）：").strip().lower()

            if not cmd:
                continue

            if cmd == "e":
                print("你选择提前结束本回合。")
                break

            parts = cmd.split()
            if len(parts) != 2:
                print("指令格式错误，请使用 m X 或 c X。")
                continue

            action, idx_str = parts
            if not idx_str.isdigit():
                print("索引必须是数字。")
                continue

            idx = int(idx_str)

            if action == "m":
                move_player(gs, idx)
            elif action == "c":
                collect_resource(gs, idx)
            else:
                print("未知指令，请使用 m 或 c。")

        print(f"第 {gs.round_no} 回合结束。当前总资源：冰 {gs.resources['冰']}，铁 {gs.resources['铁']}，火 {gs.resources['火']}")
        gs.round_no += 1

    print("\n==== 游戏结束 ====")
    print(f"最终资源统计：冰 {gs.resources['冰']}，铁 {gs.resources['铁']}，火 {gs.resources['火']}")
    total_score = gs.resources['冰'] + gs.resources['铁'] + gs.resources['火']
    print(f"总资源和（可理解为分数）：{total_score}")

if __name__ == "__main__":
    main()
