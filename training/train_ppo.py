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
        self.last_episode_count = 0  # 记录上次rollout的场次数

    def _on_step(self) -> bool:
        # 检查是否有episode结束
        if 'final_tokens' in self.locals.get('infos', [{}])[0]:
            for info in self.locals['infos']:
                if 'final_tokens' in info:
                    self.episode_tokens.append(info['final_tokens'])

        return True

    def _on_rollout_end(self) -> None:
        if len(self.episode_tokens) > 0:
            # 计算本次rollout新增的场次
            current_count = len(self.episode_tokens)
            new_episodes = current_count - self.last_episode_count

            # 计算平均代币
            avg_tokens = np.mean(self.episode_tokens[-10:]) if len(self.episode_tokens) >= 10 else np.mean(self.episode_tokens)

            # 显示本次rollout的场次和累计场次
            print(f"  最近平均代币: {avg_tokens:.2f} (本次rollout: {new_episodes} 场, 累计: {current_count} 场)")

            # 更新上次的场次数
            self.last_episode_count = current_count


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
    save_path="models/ppo_resource_game",
    network_arch="medium"
):
    """使用PPO训练智能体

    Args:
        network_arch: 网络架构规模
            - "small": [64, 64] (~7.4K参数) - 默认小网络
            - "medium": [128, 128] (~17K参数) - 中等网络，推荐
            - "large": [256, 256] (~54K参数) - 大网络，更强表达能力
            - "xlarge": [256, 256, 128] (~86K参数) - 超大网络，3层
    """
    if not HAS_SB3:
        print("错误：需要安装 stable-baselines3")
        return

    # 定义网络架构
    network_configs = {
        "small": [64, 64],
        "medium": [128, 128],
        "large": [256, 256],
        "xlarge": [256, 256, 128]
    }

    if network_arch not in network_configs:
        print(f"警告：未知的网络架构 '{network_arch}'，使用 'medium'")
        network_arch = "medium"

    net_arch = network_configs[network_arch]

    # 配置策略网络
    policy_kwargs = dict(
        net_arch=dict(pi=net_arch, vf=net_arch)
    )

    # 估算参数量
    def estimate_params(arch):
        """估算网络参数量"""
        input_dim = 38  # 观测空间维度
        actor_output = 10  # 动作空间维度
        critic_output = 1  # 价值输出

        # 共享层参数
        params = 0
        prev_dim = input_dim
        for hidden_dim in arch:
            params += prev_dim * hidden_dim + hidden_dim  # 权重 + 偏置
            prev_dim = hidden_dim

        # Actor和Critic各自的输出层
        params += prev_dim * actor_output + actor_output
        params += prev_dim * critic_output + critic_output

        # 因为是独立的pi和vf网络，参数量翻倍
        return params * 2

    estimated_params = estimate_params(net_arch)

    # 创建并行环境
    print(f"创建 {n_envs} 个并行环境...")
    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])

    # 创建评估环境
    eval_env = DummyVecEnv([make_env(n_envs)])

    # 创建模型
    print("=" * 60)
    print(f"创建PPO模型 - 网络架构: {network_arch}")
    print(f"  网络结构: {net_arch}")
    print(f"  估算参数量: ~{estimated_params:,} 个")
    print("=" * 60)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./logs/ppo_resource_game",
        device='cpu'  # 强制使用CPU，MlpPolicy在CPU上更高效
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
    parser.add_argument('--network', type=str, default='medium',
                        choices=['small', 'medium', 'large', 'xlarge'],
                        help='网络架构规模：small (~7.4K), medium (~17K), large (~54K), xlarge (~86K)')
    parser.add_argument('--model-path', type=str, default='models/ppo_resource_game/final_model',
                        help='模型路径（用于评估）')
    parser.add_argument('--episodes', type=int, default=10, help='评估轮数')

    args = parser.parse_args()

    if args.mode == 'train':
        print("\n" + "=" * 60)
        print("PPO 训练配置:")
        print(f"  训练步数: {args.timesteps:,}")
        print(f"  并行环境: {args.n_envs}")
        print(f"  网络架构: {args.network}")
        print("=" * 60 + "\n")

        train_ppo(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            network_arch=args.network
        )
    else:
        evaluate_model(args.model_path, num_episodes=args.episodes)
