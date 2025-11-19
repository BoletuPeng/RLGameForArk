"""
测试新的29维观测空间
"""
import sys
sys.path.append('backend')

from game_core import ResourceGame, RESOURCE_TYPES, RESOURCE_TYPE_TO_ID

# 创建游戏实例
game = ResourceGame(rounds=10, seed=42)

# 获取观测
obs = game.get_observation()

print(f'观测空间维度: {len(obs)}')
print()

# 详细解析观测向量
print('=' * 60)
print('观测向量详细解析：')
print('=' * 60)
idx = 0

# 手牌统计
print(f'\n[索引 {idx:2d}-{idx+2:2d}] 手牌统计:')
print(f'  1点卡: {game.hand.get(1, 0)}张 -> {obs[idx]:.2f}')
print(f'  2点卡: {game.hand.get(2, 0)}张 -> {obs[idx+1]:.2f}')
print(f'  3点卡: {game.hand.get(3, 0)}张 -> {obs[idx+2]:.2f}')
idx += 3

# 位置
print(f'\n[索引 {idx:2d}-{idx+9:2d}] 位置 one-hot:')
print(f'  当前位置: {game.position} ({game.tile_type()})')
print(f'  One-hot向量: {list(obs[idx:idx+10])}')
idx += 10

# 资源系数
print(f'\n[索引 {idx:2d}] 资源系数:')
print(f'  实际值: {game.resource_coef} -> 观测值: {obs[idx]:.3f}')
idx += 1

# 当前回合
print(f'\n[索引 {idx:2d}] 当前回合:')
print(f'  实际值: {game.current_round}/{game.rounds} -> 观测值: {obs[idx]:.2f}')
idx += 1

# 是否可普通收集
print(f'\n[索引 {idx:2d}] 是否可普通收集:')
print(f'  实际值: {game.collectable} -> 观测值: {obs[idx]:.1f}')
idx += 1

# 是否可连击
can_combo = len(game.can_combo_values()) > 0
print(f'\n[索引 {idx:2d}] 是否可连击:')
print(f'  实际值: {can_combo} -> 观测值: {obs[idx]:.1f}')
idx += 1

# 上次收集代价
print(f'\n[索引 {idx:2d}-{idx+1:2d}] 上次收集代价 one-hot:')
print(f'  上次收集点数: {game.last_collect_cost}')
print(f'  上次动作是移动: {game.last_action_was_move}')
print(f'  [收集1?, 收集2?]: [{obs[idx]:.0f}, {obs[idx+1]:.0f}]')
idx += 2

# 顾客信息
print(f'\n[索引 {idx:2d}-{idx+8:2d}] 顾客需求信息:')
for i, cust in enumerate(game.customers, 1):
    cust_type = "VIP" if cust.is_vip else "普通"
    print(f'\n  顾客{i} ({cust_type}):')

    # 显示需求
    needs_str = []
    for res_type, amount in cust.needs.items():
        have = cust.have.get(res_type, 0)
        needs_str.append(f'{res_type}[{have}/{amount}]')
    print(f'    需求: {", ".join(needs_str)}')

    # 显示观测值
    cust_start = idx + (i-1) * 3
    for j, res_type in enumerate(RESOURCE_TYPES):
        res_id = RESOURCE_TYPE_TO_ID[res_type]
        obs_val = obs[cust_start + res_id]

        if res_type in cust.needs:
            need = cust.needs[res_type]
            have = cust.have.get(res_type, 0)
            remaining = need - have
            print(f'    {res_type}仍需: {remaining} -> 观测值: {obs_val:.3f}')
        else:
            print(f'    {res_type}仍需: 0 -> 观测值: {obs_val:.3f}')
idx += 9

# 代币数
print(f'\n[索引 {idx:2d}] 代币数:')
print(f'  实际值: {game.tokens} -> 观测值: {obs[idx]:.2f}')

print('\n' + '=' * 60)
print(f'✓ 总维度: {len(obs)} (预期29维)')
print('=' * 60)

# 验证观测空间
assert len(obs) == 29, f"观测空间维度错误！预期29维，实际{len(obs)}维"
print('\n✓ 观测空间维度验证通过！')

# 测试一个特定的场景
print('\n' + '=' * 60)
print('测试场景：模拟顾客需求')
print('=' * 60)

# 假设顾客1：火[0/20]，冰[13/20]
# 顾客2：铁[30/30]，冰[5/10]
# 顾客3：火[2/35]
# 预期观测：[7, 0, 20, 5, 0, 0, 0, 0, 33]

print('\n示例：如果有如下顾客需求：')
print('  顾客1: 火[0/20], 冰[13/20]')
print('  顾客2: 铁[30/30], 冰[5/10]')
print('  顾客3: 火[2/35]')
print('\n则顾客观测应为：')
print('  [冰1, 铁1, 火1, 冰2, 铁2, 火2, 冰3, 铁3, 火3]')
print('  = [7, 0, 20, 5, 0, 0, 0, 0, 33] (未归一化)')
print('\n实际观测（归一化后）：')
cust_obs_start = 17
cust_obs_values = obs[cust_obs_start:cust_obs_start+9]
cust_obs_raw = [v * 100 for v in cust_obs_values]
print(f'  {[f"{v:.0f}" for v in cust_obs_raw]}')
