"""
使用 PPO 算法进行训练
需要安装：pip install stable-baselines3
"""
import sys
import os

# 优化多核CPU性能：设置线程数
# 这些环境变量控制底层计算库（OpenBLAS/MKL/OpenMP）的线程数
# 需要在导入 numpy/torch 之前设置
def setup_cpu_threads(num_threads=None):
    """设置CPU线程数以优化性能

    Args:
        num_threads: 线程数，如果为None则自动设置为CPU核心数的80%
    """
    if num_threads is None:
        import multiprocessing
        num_cores = multiprocessing.cpu_count()
        # 使用80%的核心数，为系统和其他进程留一些余量
        num_threads = max(1, int(num_cores * 0.8))

    num_threads = str(num_threads)
    os.environ['OMP_NUM_THREADS'] = num_threads
    os.environ['MKL_NUM_THREADS'] = num_threads
    os.environ['OPENBLAS_NUM_THREADS'] = num_threads
    os.environ['NUMEXPR_NUM_THREADS'] = num_threads

    print(f"CPU优化: 设置线程数为 {num_threads}")

    return int(num_threads)

# 在导入其他库之前设置线程数
setup_cpu_threads()

# 添加backend路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from rl_env.game_env import ResourceGameEnv
from rl_env.parallel_env import make_parallel_env
import numpy as np

try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("警告：未安装 sb3-contrib，请运行：pip install sb3-contrib")


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


def mask_fn(env):
    """提取action mask的函数,用于ActionMasker包装器"""
    return env.game.get_valid_actions()


def make_env(rank, seed=0, auxiliary_reward_coef=1.0):
    """创建环境的工厂函数"""
    def _init():
        env = ResourceGameEnv(rounds=10, seed=seed + rank, auxiliary_reward_coef=auxiliary_reward_coef)
        env = ActionMasker(env, mask_fn)  # 使用ActionMasker包装器自动提供action masks
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
    network_arch="medium",
    enable_eval=True,
    auxiliary_reward_coef=1.0,
    resume_from=None
):
    """使用MaskablePPO训练智能体(自动使用action masking)

    Args:
        network_arch: 网络架构规模
            - "small": [64, 64] (~7.4K参数) - 默认小网络
            - "medium": [128, 128] (~17K参数) - 中等网络，推荐
            - "large": [256, 256] (~54K参数) - 大网络，更强表达能力
            - "xlarge": [256, 256, 128] (~86K参数) - 超大网络，3层
        auxiliary_reward_coef: 辅助奖励系数，默认1.0
            - 1.0: 完整辅助奖励（默认模式）
            - 0.0: 纯token奖励模式
            - 0~1之间: 混合模式
        resume_from: 从指定模型继续训练（模型路径）
            - None: 从头开始训练
            - "best": 从最佳模型继续训练 (best_model.zip)
            - 其他路径: 从指定路径加载模型
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
        input_dim = 29  # 观测空间维度
        actor_output = 6  # 动作空间维度
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
    print(f"创建 {n_envs} 个并行训练环境...")
    print(f"辅助奖励系数: {auxiliary_reward_coef}")
    env = SubprocVecEnv([make_env(i, auxiliary_reward_coef=auxiliary_reward_coef) for i in range(n_envs)])

    # 创建评估环境（仅在启用评估时）
    # 使用SubprocVecEnv而不是DummyVecEnv，避免Windows上的进程切换问题
    if enable_eval:
        print("创建评估环境（使用SubprocVecEnv避免Windows进程同步问题）...")
        eval_env = SubprocVecEnv([make_env(n_envs + i, auxiliary_reward_coef=auxiliary_reward_coef) for i in range(2)])  # 仅使用2个并行评估环境
    else:
        eval_env = None

    # 创建或加载模型
    if resume_from is not None:
        # 从已有模型继续训练
        if resume_from == "best":
            model_path = os.path.join(save_path, "best_model.zip")
        else:
            model_path = resume_from

        if not os.path.exists(model_path):
            print(f"警告：指定的模型路径不存在: {model_path}")
            print("将从头开始训练...")
            resume_from = None
        else:
            print("=" * 60)
            print(f"从已有模型继续训练: {model_path}")
            print("=" * 60)
            model = MaskablePPO.load(model_path, env=env, device='cpu')
            # 更新学习率和其他参数
            model.learning_rate = learning_rate
            model.n_steps = n_steps
            model.batch_size = batch_size
            model.n_epochs = n_epochs
            model.gamma = gamma

    if resume_from is None:
        # 从头开始训练
        print("=" * 60)
        print(f"创建MaskablePPO模型 (自动action masking) - 网络架构: {network_arch}")
        print(f"  网络结构: {net_arch}")
        print(f"  估算参数量: ~{estimated_params:,} 个")
        print("=" * 60)
        model = MaskablePPO(
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
    callbacks = [callback]

    # 如果启用评估，添加评估回调
    if enable_eval:
        # 计算合理的评估频率：大幅提高频率以避免频繁评估导致的性能问题
        # 在Windows上，频繁的进程切换会导致性能下降和潜在的死锁
        eval_freq = max(500000, n_steps * n_envs * 20)  # 至少每20个rollout评估一次，或500k步

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=save_path,
            log_path='./logs/eval',
            eval_freq=eval_freq,
            n_eval_episodes=10,  # 每个评估环境5个episodes，总共10个
            deterministic=True,
            render=False,
            verbose=1  # 显示评估进度
        )
        callbacks.append(eval_callback)

        print(f"评估配置: 每 {eval_freq:,} 步评估一次 (2个并行环境, 每个5个episodes)")
        print(f"  注意: 使用SubprocVecEnv避免Windows上的进程同步问题")
    else:
        print("评估已禁用 - 训练将更快完成")

    # 训练
    print(f"开始训练 {total_timesteps} 步...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True
    )

    # 保存最终模型
    os.makedirs(save_path, exist_ok=True)
    final_model_path = os.path.join(save_path, "final_model")
    model.save(final_model_path)
    print(f"模型已保存到: {final_model_path}")

    env.close()
    if eval_env is not None:
        eval_env.close()

    return model


def evaluate_model(model_path, num_episodes=10, n_eval_envs=4):
    """评估训练好的模型

    Args:
        model_path: 模型路径
        num_episodes: 评估的总episode数
        n_eval_envs: 并行评估环境数（默认4个）
    """
    if not HAS_SB3:
        print("错误：需要安装 sb3-contrib")
        return

    print(f"加载模型: {model_path}")
    model = MaskablePPO.load(model_path, device='cpu')

    # 使用并行环境加速评估
    print(f"创建 {n_eval_envs} 个并行评估环境...")
    eval_env = SubprocVecEnv([make_env(i, seed=1000 + i) for i in range(n_eval_envs)])

    total_tokens = []
    total_rewards = []
    episodes_done = 0

    # 重置所有环境
    obs = eval_env.reset()

    print(f"开始评估 {num_episodes} 个episodes...")

    while episodes_done < num_episodes:
        # 批量预测
        actions, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = eval_env.step(actions)

        # 检查是否有episode完成
        for i, (done, info) in enumerate(zip(dones, infos)):
            if done and episodes_done < num_episodes:
                final_tokens = info.get('final_tokens', info.get('tokens', 0))
                episode_reward = info.get('episode_reward', 0)
                total_tokens.append(final_tokens)
                total_rewards.append(episode_reward)
                episodes_done += 1
                print(f"Episode {episodes_done}/{num_episodes}: 代币 = {final_tokens}, 奖励 = {episode_reward:.2f}")

    eval_env.close()

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
    parser.add_argument('--no-eval', action='store_true',
                        help='禁用训练期间的评估（可以显著加快训练速度）')
    parser.add_argument('--auxiliary-reward-coef', type=float, default=1.0,
                        help='辅助奖励系数（默认1.0，设为0则为纯token奖励模式）')
    parser.add_argument('--resume', type=str, default=None,
                        help='从指定模型继续训练（使用"best"加载最佳模型，或指定模型路径）')
    parser.add_argument('--model-path', type=str, default='models/ppo_resource_game/final_model',
                        help='模型路径（用于评估）')
    parser.add_argument('--episodes', type=int, default=10, help='评估轮数')
    parser.add_argument('--n-eval-envs', type=int, default=4,
                        help='评估时的并行环境数（默认4，增加可以加快评估速度）')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='折扣因子gamma（默认0.99，范围0-1，越大越重视长期回报）')

    args = parser.parse_args()

    if args.mode == 'train':
        print("\n" + "=" * 60)
        print("PPO 训练配置:")
        print(f"  训练步数: {args.timesteps:,}")
        print(f"  并行环境: {args.n_envs}")
        print(f"  网络架构: {args.network}")
        print(f"  折扣因子gamma: {args.gamma}")
        print(f"  辅助奖励系数: {args.auxiliary_reward_coef}")
        print(f"  启用评估: {not args.no_eval}")
        if args.resume:
            print(f"  续训模型: {args.resume}")
        if not args.no_eval:
            # 计算实际的eval_freq以告知用户
            n_steps = 2048  # 默认值
            eval_freq = max(500000, n_steps * args.n_envs * 20)
            print(f"  评估频率: 每 {eval_freq:,} 步")
            print(f"  提示: 使用 --no-eval 可以完全禁用评估以获得最快速度")
        print("=" * 60 + "\n")

        train_ppo(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            network_arch=args.network,
            gamma=args.gamma,
            enable_eval=not args.no_eval,
            auxiliary_reward_coef=args.auxiliary_reward_coef,
            resume_from=args.resume
        )
    else:
        print("\n" + "=" * 60)
        print("PPO 评估配置:")
        print(f"  评估episodes: {args.episodes}")
        print(f"  并行环境数: {args.n_eval_envs}")
        print("=" * 60 + "\n")
        evaluate_model(args.model_path, num_episodes=args.episodes, n_eval_envs=args.n_eval_envs)
