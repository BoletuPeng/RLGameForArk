"""
使用 PPO 算法进行训练
需要安装：pip install stable-baselines3
"""
import sys
import os

# 添加backend路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from rl_env.game_env import ResourceGameEnv
from rl_env.parallel_env import make_parallel_env
import numpy as np

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("警告：未安装 stable-baselines3，请运行：pip install stable-baselines3")


class InvalidActionMaskingCallback(BaseCallback):
    """
    自定义回调：记录训练进度
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_tokens = []

    def _on_step(self) -> bool:
        # 检查是否有episode结束
        if 'final_tokens' in self.locals.get('infos', [{}])[0]:
            for info in self.locals['infos']:
                if 'final_tokens' in info:
                    self.episode_tokens.append(info['final_tokens'])

        return True

    def _on_rollout_end(self) -> None:
        if len(self.episode_tokens) > 0:
            avg_tokens = np.mean(self.episode_tokens[-10:]) if len(self.episode_tokens) >= 10 else np.mean(self.episode_tokens)
            print(f"  最近平均代币: {avg_tokens:.2f} (共 {len(self.episode_tokens)} 场游戏)")


def make_env(rank, seed=0):
    """创建环境的工厂函数"""
    def _init():
        env = ResourceGameEnv(rounds=10, seed=seed + rank)
        env = Monitor(env)
        return env
    return _init


def train_ppo(
    total_timesteps=100000,
    n_envs=8,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    save_path="models/ppo_resource_game"
):
    """使用PPO训练智能体"""
    if not HAS_SB3:
        print("错误：需要安装 stable-baselines3")
        return

    # 创建并行环境
    print(f"创建 {n_envs} 个并行环境...")
    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])

    # 创建评估环境
    eval_env = DummyVecEnv([make_env(n_envs)])

    # 创建模型
    print("创建PPO模型...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        verbose=1,
        tensorboard_log="./logs/ppo_resource_game"
    )

    # 创建回调
    callback = InvalidActionMaskingCallback()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path='./logs/eval',
        eval_freq=10000,
        deterministic=True,
        render=False
    )

    # 训练
    print(f"开始训练 {total_timesteps} 步...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[callback, eval_callback],
        progress_bar=True
    )

    # 保存最终模型
    os.makedirs(save_path, exist_ok=True)
    final_model_path = os.path.join(save_path, "final_model")
    model.save(final_model_path)
    print(f"模型已保存到: {final_model_path}")

    env.close()
    eval_env.close()

    return model


def evaluate_model(model_path, num_episodes=10):
    """评估训练好的模型"""
    if not HAS_SB3:
        print("错误：需要安装 stable-baselines3")
        return

    print(f"加载模型: {model_path}")
    model = PPO.load(model_path)

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

    parser = argparse.ArgumentParser(description='PPO训练脚本')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'],
                        help='运行模式：train 或 eval')
    parser.add_argument('--timesteps', type=int, default=100000, help='训练步数')
    parser.add_argument('--n-envs', type=int, default=8, help='并行环境数量')
    parser.add_argument('--model-path', type=str, default='models/ppo_resource_game/final_model',
                        help='模型路径（用于评估）')
    parser.add_argument('--episodes', type=int, default=10, help='评估轮数')

    args = parser.parse_args()

    if args.mode == 'train':
        train_ppo(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs
        )
    else:
        evaluate_model(args.model_path, num_episodes=args.episodes)
