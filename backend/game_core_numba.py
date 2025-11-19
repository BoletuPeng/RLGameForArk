"""
Numba 加速的游戏核心计算函数
将计算密集型操作提取为纯数值计算，使用 numba JIT 编译加速
"""
import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def compute_customer_observation_fast(
    customer_needs: np.ndarray,  # shape: (3, 3) - 3个顾客 x 3种资源的需求量
    customer_have: np.ndarray    # shape: (3, 3) - 3个顾客 x 3种资源的已有量
) -> np.ndarray:
    """
    快速计算顾客需求观测向量

    参数:
        customer_needs: (3, 3) 数组，表示3个顾客对3种资源的需求量
        customer_have: (3, 3) 数组，表示3个顾客对3种资源的已有量

    返回:
        (9,) 数组，每个顾客3维 [冰仍需量, 铁仍需量, 火仍需量]，归一化到 0-1
    """
    result = np.zeros(9, dtype=np.float32)

    for cust_idx in range(3):
        for res_idx in range(3):
            need = customer_needs[cust_idx, res_idx]
            have = customer_have[cust_idx, res_idx]
            remaining = max(0.0, need - have)
            # 归一化（最大假设100）
            result[cust_idx * 3 + res_idx] = remaining / 100.0

    return result


@jit(nopython=True, cache=True)
def compute_observation_vector_fast(
    hand_1: int,
    hand_2: int,
    hand_3: int,
    position: int,
    resource_coef: int,
    current_round: int,
    total_rounds: int,
    collectable: bool,
    can_combo: bool,
    last_collect_cost: int,
    last_action_was_move: bool,
    customer_needs: np.ndarray,
    customer_have: np.ndarray,
    tokens: int
) -> np.ndarray:
    """
    快速计算完整的观测向量（29维）

    这个函数使用纯数值计算，避免了字典访问和对象操作，适合 numba 加速
    """
    obs = np.zeros(29, dtype=np.float32)
    idx = 0

    # 手牌统计（3维）
    obs[idx] = hand_1 / 5.0
    obs[idx + 1] = hand_2 / 5.0
    obs[idx + 2] = hand_3 / 5.0
    idx += 3

    # 位置（10维 one-hot）
    obs[idx + position] = 1.0
    idx += 10

    # 资源系数（1维）
    obs[idx] = resource_coef / 15.0
    idx += 1

    # 当前回合（1维）
    obs[idx] = current_round / total_rounds
    idx += 1

    # 是否可普通收集（1维）
    obs[idx] = 1.0 if collectable else 0.0
    idx += 1

    # 是否可连击（1维）
    obs[idx] = 1.0 if can_combo else 0.0
    idx += 1

    # 上次收集代价（2维 one-hot）
    if last_collect_cost == 1 and not last_action_was_move:
        obs[idx] = 1.0
    elif last_collect_cost == 2 and not last_action_was_move:
        obs[idx + 1] = 1.0
    idx += 2

    # 顾客需求信息（9维）
    customer_obs = compute_customer_observation_fast(customer_needs, customer_have)
    obs[idx:idx + 9] = customer_obs
    idx += 9

    # 代币数（1维）
    obs[idx] = tokens / 20.0

    return obs


@jit(nopython=True, cache=True)
def compute_valid_actions_fast(
    hand_1: int,
    hand_2: int,
    hand_3: int,
    collectable: bool,
    can_combo_1: bool,
    can_combo_2: bool,
    can_combo_3: bool
) -> np.ndarray:
    """
    快速计算有效动作掩码

    动作空间：[move_1, move_2, move_3, collect_1, collect_2, collect_3]

    参数:
        hand_1, hand_2, hand_3: 手牌中各点数的数量
        collectable: 是否可以普通收集
        can_combo_1, can_combo_2, can_combo_3: 各点数是否可以连击

    返回:
        (6,) 数组，1表示该动作有效，0表示无效
    """
    valid = np.zeros(6, dtype=np.float32)

    # 移动动作（0-2）
    if hand_1 > 0:
        valid[0] = 1.0
    if hand_2 > 0:
        valid[1] = 1.0
    if hand_3 > 0:
        valid[2] = 1.0

    # 收集动作（3-5）
    # 需要检查是否可收集：有对应手牌 且 (可普通收集 或 可连击)
    if hand_1 > 0 and (collectable or can_combo_1):
        valid[3] = 1.0
    if hand_2 > 0 and (collectable or can_combo_2):
        valid[4] = 1.0
    if hand_3 > 0 and (collectable or can_combo_3):
        valid[5] = 1.0

    return valid


@jit(nopython=True, cache=True)
def compute_resource_need_fast(
    customer_needs: np.ndarray,  # shape: (3, 3)
    customer_have: np.ndarray,   # shape: (3, 3)
    resource_idx: int            # 0=冰, 1=铁, 2=火
) -> int:
    """
    快速计算所有顾客对某个资源的总需求量（需求量 - 已拥有量）

    参数:
        customer_needs: (3, 3) 数组
        customer_have: (3, 3) 数组
        resource_idx: 资源索引 (0=冰, 1=铁, 2=火)

    返回:
        总需求量
    """
    total_need = 0
    for cust_idx in range(3):
        need = customer_needs[cust_idx, resource_idx]
        have = customer_have[cust_idx, resource_idx]
        remaining = max(0, int(need - have))
        total_need += remaining
    return total_need


@jit(nopython=True, cache=True)
def distribute_resources_fast(
    customer_needs: np.ndarray,   # shape: (3, 3)
    customer_have: np.ndarray,    # shape: (3, 3) - will be modified in-place
    customer_is_vip: np.ndarray,  # shape: (3,) - bool array
    produced: np.ndarray          # shape: (3,) - [冰, 铁, 火] 本次产出
) -> tuple:
    """
    快速分配资源给顾客

    参数:
        customer_needs: (3, 3) 数组
        customer_have: (3, 3) 数组，会被原地修改
        customer_is_vip: (3,) bool 数组
        produced: (3,) 数组，本次产出的 [冰, 铁, 火]

    返回:
        (customer_gains,) - (3, 2) 数组，每行为 [is_vip, gain]
    """
    # 记录每个顾客的旧状态
    old_have = customer_have.copy()

    # 分配资源
    for res_idx in range(3):
        amount = produced[res_idx]
        if amount <= 0:
            continue

        for cust_idx in range(3):
            need = customer_needs[cust_idx, res_idx]
            if need <= 0:
                continue

            old = old_have[cust_idx, res_idx]
            if old >= need:
                continue

            # 更新已有量
            customer_have[cust_idx, res_idx] = min(need, old + amount)

    # 计算每个顾客实际获得的有效资源量
    customer_gains = np.zeros((3, 2), dtype=np.float32)
    for cust_idx in range(3):
        gain = 0.0
        for res_idx in range(3):
            if customer_needs[cust_idx, res_idx] > 0:
                old = old_have[cust_idx, res_idx]
                new = customer_have[cust_idx, res_idx]
                gain += new - old

        customer_gains[cust_idx, 0] = 1.0 if customer_is_vip[cust_idx] else 0.0
        customer_gains[cust_idx, 1] = gain

    return customer_gains
