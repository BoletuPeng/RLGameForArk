"""
从对局记录进行行为克隆训练
使用保存的对局记录来预训练或微调模型
"""
import sys
import os
import json
import glob
import numpy as np

# 添加backend路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from rl_env.game_env import ResourceGameEnv
from game_core import ResourceGame

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("警告：需要安装 stable-baselines3 和 torch")


def load_replay_files(replay_dir):
    """加载所有replay文件"""
    replay_files = glob.glob(os.path.join(replay_dir, "*.json"))

    if not replay_files:
        print(f"警告：在 {replay_dir} 中没有找到replay文件")
        return []

    replays = []
    for filepath in replay_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                replay = json.load(f)
                replays.append(replay)
        except Exception as e:
            print(f"加载 {filepath} 时出错: {e}")

    print(f"成功加载 {len(replays)} 个replay文件")
    return replays


def replay_to_training_data(replays):
    """
    将replay转换为训练数据

    由于我们没有保存observation，这里我们通过重放游戏来重建observation
    """
    observations = []
    actions = []

    for replay_idx, replay in enumerate(replays):
        print(f"处理replay {replay_idx + 1}/{len(replays)}...")

        # 创建游戏环境，使用相同的seed（如果有）
        seed = replay.get('seed', None)
        total_rounds = replay.get('total_rounds', 10)
        env = ResourceGameEnv(rounds=total_rounds, seed=seed)

        obs, info = env.reset(seed=seed)
        action_history = replay.get('action_history', [])

        for action_data in action_history:
            action_type = action_data.get('type')

            if action_type == 'move':
                # 找到对应的手牌索引
                card_value = action_data.get('card_value')
                # 在当前手牌中找到这张牌
                card_index = None
                for i, val in enumerate(env.game.hand):
                    if val == card_value:
                        card_index = i
                        break

                if card_index is not None:
                    action = card_index  # 移动动作 0-4
                    observations.append(obs.copy())
                    actions.append(action)

                    # 执行动作
                    obs, reward, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        break

            elif action_type == 'collect':
                # 找到对应的手牌索引
                card_value = action_data.get('card_value')
                card_index = None
                for i, val in enumerate(env.game.hand):
                    if val == card_value:
                        card_index = i
                        break

                if card_index is not None:
                    action = 5 + card_index  # 收集动作 5-9
                    observations.append(obs.copy())
                    actions.append(action)

                    # 执行动作
                    obs, reward, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        break

        env.close()

    print(f"生成了 {len(observations)} 个训练样本")
    return np.array(observations), np.array(actions)


def train_with_behavior_cloning(
    replay_dir="replays",
    model_path=None,
    save_path="models/ppo_from_replay",
    epochs=50,
    batch_size=32,
    learning_rate=3e-4
):
    """
    使用行为克隆方法从replay训练模型

    Args:
        replay_dir: replay文件目录
        model_path: 可选，从已有模型开始（续训）
        save_path: 保存路径
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
    """
    if not HAS_SB3:
        print("错误：需要安装 stable-baselines3 和 torch")
        return

    # 加载replay文件
    replays = load_replay_files(replay_dir)
    if not replays:
        print("没有可用的replay文件，退出训练")
        return

    # 转换为训练数据
    print("\n转换replay为训练数据...")
    observations, actions = replay_to_training_data(replays)

    if len(observations) == 0:
        print("没有生成任何训练样本，退出训练")
        return

    # 创建或加载模型
    env = DummyVecEnv([lambda: ResourceGameEnv(rounds=10)])

    if model_path and os.path.exists(model_path):
        print(f"\n从已有模型加载: {model_path}")
        model = PPO.load(model_path, env=env, device='cpu')
    else:
        print("\n创建新模型")
        model = PPO("MlpPolicy", env, learning_rate=learning_rate, verbose=0, device='cpu')

    # 行为克隆训练
    print(f"\n开始行为克隆训练 ({epochs} epochs)...")
    print(f"训练样本数: {len(observations)}")
    print(f"批次大小: {batch_size}")

    # 获取策略网络
    policy = model.policy
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # 训练循环
    dataset_size = len(observations)
    for epoch in range(epochs):
        # 打乱数据
        indices = np.random.permutation(dataset_size)
        epoch_loss = 0
        num_batches = 0

        for i in range(0, dataset_size, batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_obs = torch.FloatTensor(observations[batch_indices])
            batch_actions = torch.LongTensor(actions[batch_indices])

            # 前向传播
            # 获取动作logits
            with torch.no_grad():
                features = policy.extract_features(batch_obs)

            latent_pi = policy.mlp_extractor.forward_actor(features)
            action_logits = policy.action_net(latent_pi)

            # 计算损失
            loss = loss_fn(action_logits, batch_actions)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    # 保存模型
    os.makedirs(save_path, exist_ok=True)
    final_model_path = os.path.join(save_path, "bc_model")
    model.save(final_model_path)
    print(f"\n行为克隆模型已保存到: {final_model_path}")

    env.close()
    return model


def evaluate_bc_model(model_path, num_episodes=10):
    """评估行为克隆训练的模型"""
    if not HAS_SB3:
        print("错误：需要安装 stable-baselines3")
        return

    print(f"加载模型: {model_path}")
    model = PPO.load(model_path, device='cpu')

    env = ResourceGameEnv(rounds=10, seed=42)

    total_tokens = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        final_tokens = info.get('final_tokens', info.get('tokens', 0))
        total_tokens.append(final_tokens)

        print(f"Episode {episode + 1}: 代币 = {final_tokens}, 奖励 = {episode_reward:.2f}")

    env.close()

    print("\n" + "=" * 60)
    print(f"评估完成！")
    print(f"平均代币: {np.mean(total_tokens):.2f} (±{np.std(total_tokens):.2f})")
    print(f"最高代币: {np.max(total_tokens)}")
    print(f"最低代币: {np.min(total_tokens)}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='从replay进行行为克隆训练')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'],
                        help='运行模式：train 或 eval')
    parser.add_argument('--replay-dir', type=str, default='replays',
                        help='replay文件目录')
    parser.add_argument('--resume', type=str, default=None,
                        help='从已有模型继续训练（模型路径）')
    parser.add_argument('--save-path', type=str, default='models/ppo_from_replay',
                        help='模型保存路径')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='学习率')
    parser.add_argument('--model-path', type=str, default='models/ppo_from_replay/bc_model',
                        help='评估时使用的模型路径')
    parser.add_argument('--episodes', type=int, default=10,
                        help='评估轮数')

    args = parser.parse_args()

    if args.mode == 'train':
        print("\n" + "=" * 60)
        print("行为克隆训练配置:")
        print(f"  Replay目录: {args.replay_dir}")
        print(f"  训练轮数: {args.epochs}")
        print(f"  批次大小: {args.batch_size}")
        print(f"  学习率: {args.learning_rate}")
        if args.resume:
            print(f"  续训模型: {args.resume}")
        print("=" * 60 + "\n")

        train_with_behavior_cloning(
            replay_dir=args.replay_dir,
            model_path=args.resume,
            save_path=args.save_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
    else:
        evaluate_bc_model(args.model_path, num_episodes=args.episodes)
