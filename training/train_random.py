"""
随机策略训练脚本 - 用于测试环境
"""
import sys
import os

# 添加backend路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from rl_env.game_env import ResourceGameEnv
import numpy as np


def train_random_agent(num_episodes=100):
    """使用随机策略进行训练"""
    env = ResourceGameEnv(rounds=10, seed=42)

    total_rewards = []
    total_tokens = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        step_count = 0

        while not done:
            # 获取有效动作
            valid_actions = info['action_mask']
            valid_indices = np.where(valid_actions > 0)[0]

            if len(valid_indices) == 0:
                print(f"警告：第 {episode} 轮没有有效动作")
                break

            # 随机选择一个有效动作
            action = np.random.choice(valid_indices)

            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1

            done = terminated or truncated

        final_tokens = info.get('final_tokens', info.get('tokens', 0))
        total_rewards.append(episode_reward)
        total_tokens.append(final_tokens)

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(total_rewards[-10:])
            avg_tokens = np.mean(total_tokens[-10:])
            print(f"Episode {episode + 1}/{num_episodes} - "
                  f"平均奖励: {avg_reward:.2f}, 平均代币: {avg_tokens:.2f}")

    env.close()

    # 统计结果
    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"总轮数: {num_episodes}")
    print(f"平均奖励: {np.mean(total_rewards):.2f} (±{np.std(total_rewards):.2f})")
    print(f"平均代币: {np.mean(total_tokens):.2f} (±{np.std(total_tokens):.2f})")
    print(f"最高代币: {np.max(total_tokens)}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='随机策略训练')
    parser.add_argument('--episodes', type=int, default=100, help='训练轮数')
    args = parser.parse_args()

    train_random_agent(num_episodes=args.episodes)
