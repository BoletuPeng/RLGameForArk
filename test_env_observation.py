"""
测试 Gymnasium 环境的29维观测空间
"""
import sys
sys.path.append('backend')

from rl_env.game_env import ResourceGameEnv
import numpy as np

print('=' * 60)
print('测试 Gymnasium 环境的观测空间')
print('=' * 60)

# 创建环境
env = ResourceGameEnv(rounds=10, seed=42)

print(f'\n观测空间: {env.observation_space}')
print(f'动作空间: {env.action_space}')

# 重置环境
obs, info = env.reset(seed=42)

print(f'\n初始观测维度: {len(obs)}')
print(f'观测空间检查: {env.observation_space.contains(obs)}')

# 显示初始观测
print('\n初始观测详情：')
print(f'  手牌: {env.game.hand}')
print(f'  位置: {env.game.position} ({env.game.tile_type()})')
print(f'  资源系数: {env.game.resource_coef}')
print(f'  当前回合: {env.game.current_round}/{env.game.rounds}')
print(f'  可普通收集: {env.game.collectable}')
print(f'  可连击: {len(env.game.can_combo_values()) > 0}')
print(f'  代币: {env.game.tokens}')

# 执行几步动作
print('\n' + '=' * 60)
print('执行一些动作来测试观测变化')
print('=' * 60)

# 获取有效动作
valid_actions = env.game.get_valid_actions()
print(f'\n有效动作掩码: {valid_actions}')

# 找到第一个有效动作
action = np.argmax(valid_actions)
print(f'\n执行动作 {action}')

obs, reward, terminated, truncated, info = env.step(action)

print(f'\n动作后观测维度: {len(obs)}')
print(f'奖励: {reward}')
print(f'游戏结束: {terminated}')
print(f'观测空间检查: {env.observation_space.contains(obs)}')

print(f'\n当前状态：')
print(f'  手牌: {env.game.hand}')
print(f'  位置: {env.game.position} ({env.game.tile_type()})')
print(f'  可普通收集: {env.game.collectable}')
print(f'  可连击: {len(env.game.can_combo_values()) > 0}')

# 测试多个episode
print('\n' + '=' * 60)
print('运行完整的episode测试')
print('=' * 60)

total_episodes = 3
for ep in range(total_episodes):
    obs, info = env.reset(seed=42 + ep)
    episode_reward = 0
    step_count = 0

    while True:
        # 检查观测维度
        assert len(obs) == 29, f"观测维度错误：{len(obs)}"
        assert env.observation_space.contains(obs), "观测不在观测空间范围内"

        # 随机选择有效动作
        valid_actions = env.game.get_valid_actions()
        valid_indices = np.where(valid_actions > 0)[0]

        if len(valid_indices) == 0:
            break

        action = np.random.choice(valid_indices)
        obs, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward
        step_count += 1

        if terminated or truncated:
            break

    print(f'Episode {ep+1}: 步数={step_count}, 总奖励={episode_reward:.2f}, 代币={info.get("final_tokens", env.game.tokens)}')

print('\n' + '=' * 60)
print('✓ 所有测试通过！')
print('✓ 观测空间从36维成功修改为29维')
print('=' * 60)

env.close()
