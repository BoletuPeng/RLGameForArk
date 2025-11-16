"""
测试环境是否正常工作
"""
import sys
import os

# 添加backend路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from rl_env.game_env import ResourceGameEnv
from rl_env.parallel_env import make_parallel_env
import numpy as np


def test_single_env():
    """测试单个环境"""
    print("=" * 60)
    print("测试单个环境")
    print("=" * 60)

    env = ResourceGameEnv(rounds=10, seed=42, render_mode='human')

    # 重置环境
    obs, info = env.reset()
    print(f"\n初始观测形状: {obs.shape}")
    print(f"初始信息: {info}")

    # 运行几步
    for step in range(10):
        print(f"\n--- 步骤 {step + 1} ---")

        # 获取有效动作
        valid_actions = info['action_mask']
        valid_indices = np.where(valid_actions > 0)[0]

        if len(valid_indices) == 0:
            print("没有有效动作，回合结束")
            break

        # 随机选择动作
        action = np.random.choice(valid_indices)
        print(f"选择动作: {action} (有效动作: {valid_indices})")

        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"奖励: {reward}")
        print(f"代币: {info.get('tokens', 0)}")
        print(f"回合: {info.get('round', 0)}")

        env.render()

        if terminated or truncated:
            print(f"\n回合结束！最终代币: {info.get('final_tokens', 0)}")
            break

    env.close()
    print("\n单环境测试完成！")


def test_parallel_env():
    """测试并行环境"""
    print("\n" + "=" * 60)
    print("测试并行环境")
    print("=" * 60)

    n_envs = 4
    envs = make_parallel_env(n_envs=n_envs, rounds=10, seed=42, use_multiprocessing=False)

    # 重置环境
    obs, infos = envs.reset()
    print(f"\n观测形状: {obs.shape} (应该是 {n_envs} x 38)")

    # 运行几步
    for step in range(5):
        print(f"\n--- 步骤 {step + 1} ---")

        # 为每个环境随机选择动作
        actions = []
        for i in range(n_envs):
            valid_actions = infos[i]['action_mask']
            valid_indices = np.where(valid_actions > 0)[0]
            if len(valid_indices) > 0:
                actions.append(np.random.choice(valid_indices))
            else:
                actions.append(0)

        actions = np.array(actions)
        print(f"动作: {actions}")

        # 执行动作
        obs, rewards, dones, infos = envs.step(actions)

        print(f"奖励: {rewards}")
        print(f"完成: {dones}")

        tokens = [info.get('tokens', 0) for info in infos]
        print(f"代币: {tokens}")

    envs.close()
    print("\n并行环境测试完成！")


def test_observation_space():
    """测试观测空间"""
    print("\n" + "=" * 60)
    print("测试观测空间")
    print("=" * 60)

    env = ResourceGameEnv(rounds=10, seed=42)
    obs, info = env.reset()

    print(f"\n观测空间: {env.observation_space}")
    print(f"动作空间: {env.action_space}")
    print(f"\n观测向量形状: {obs.shape}")
    print(f"观测向量范围: [{obs.min():.2f}, {obs.max():.2f}]")
    print(f"\n观测向量前10维: {obs[:10]}")
    print(f"有效动作掩码: {info['action_mask']}")

    env.close()
    print("\n观测空间测试完成！")


def test_reward_structure():
    """测试奖励结构"""
    print("\n" + "=" * 60)
    print("测试奖励结构")
    print("=" * 60)

    env = ResourceGameEnv(rounds=10, seed=42)
    obs, info = env.reset()

    total_reward = 0
    step_count = 0
    max_steps = 100

    while step_count < max_steps:
        valid_actions = info['action_mask']
        valid_indices = np.where(valid_actions > 0)[0]

        if len(valid_indices) == 0:
            break

        action = np.random.choice(valid_indices)
        obs, reward, terminated, truncated, info = env.step(action)

        if reward != 0:
            print(f"步骤 {step_count}: 动作 {action}, 奖励 {reward:.2f}, 代币 {info.get('tokens', 0)}")

        total_reward += reward
        step_count += 1

        if terminated or truncated:
            print(f"\n游戏结束！")
            print(f"总步数: {step_count}")
            print(f"总奖励: {total_reward:.2f}")
            print(f"最终代币: {info.get('final_tokens', 0)}")
            break

    env.close()
    print("\n奖励结构测试完成！")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='测试环境')
    parser.add_argument('--test', type=str, default='all',
                        choices=['all', 'single', 'parallel', 'observation', 'reward'],
                        help='测试类型')

    args = parser.parse_args()

    if args.test in ['all', 'single']:
        test_single_env()

    if args.test in ['all', 'parallel']:
        test_parallel_env()

    if args.test in ['all', 'observation']:
        test_observation_space()

    if args.test in ['all', 'reward']:
        test_reward_structure()

    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)
