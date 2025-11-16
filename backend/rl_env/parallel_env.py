"""
并行环境支持 - 用于高效的强化学习训练
"""
import numpy as np
from typing import List, Optional, Tuple, Callable
import multiprocessing as mp
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_env.game_env import ResourceGameEnv


def worker(remote: Connection, parent_remote: Connection, env_fn: Callable):
    """
    工作进程：运行环境实例
    """
    parent_remote.close()
    env = env_fn()

    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                obs, reward, terminated, truncated, info = env.step(data)
                if terminated or truncated:
                    # 自动重置
                    final_obs = obs
                    final_info = info
                    obs, reset_info = env.reset()
                    final_info['final_observation'] = final_obs
                    remote.send((obs, reward, terminated, truncated, final_info))
                else:
                    remote.send((obs, reward, terminated, truncated, info))
            elif cmd == 'reset':
                obs, info = env.reset(seed=data)
                remote.send((obs, info))
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            else:
                raise NotImplementedError(f"Unknown command: {cmd}")
        except EOFError:
            break


class ParallelEnv:
    """
    并行环境：同时运行多个环境实例

    用法：
        envs = ParallelEnv([lambda: ResourceGameEnv() for _ in range(8)])
        obs = envs.reset()
        actions = model.predict(obs)  # shape: (8,)
        obs, rewards, dones, infos = envs.step(actions)
    """

    def __init__(self, env_fns: List[Callable], start_method: Optional[str] = None):
        """
        参数：
            env_fns: 环境工厂函数列表
            start_method: multiprocessing start method ('fork', 'spawn', 'forkserver')
        """
        self.waiting = False
        self.closed = False
        self.n_envs = len(env_fns)

        if start_method is None:
            # 使用默认方法，但在某些系统上可能需要 'spawn'
            forkserver_available = 'forkserver' in mp.get_all_start_methods()
            start_method = 'forkserver' if forkserver_available else 'spawn'

        ctx = mp.get_context(start_method)

        # 创建管道和进程
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.n_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, env_fn)
            process = ctx.Process(target=worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

        # 获取空间信息
        self.remotes[0].send(('get_spaces', None))
        self.observation_space, self.action_space = self.remotes[0].recv()

    def reset(self, seeds: Optional[List[int]] = None):
        """
        重置所有环境

        参数：
            seeds: 可选的种子列表

        返回：
            observations: shape (n_envs, obs_dim)
            infos: 信息字典列表
        """
        if seeds is None:
            seeds = [None] * self.n_envs

        for remote, seed in zip(self.remotes, seeds):
            remote.send(('reset', seed))

        results = [remote.recv() for remote in self.remotes]
        obs, infos = zip(*results)

        return np.array(obs), list(infos)

    def step(self, actions: np.ndarray):
        """
        在所有环境中执行动作

        参数：
            actions: shape (n_envs,)

        返回：
            observations: shape (n_envs, obs_dim)
            rewards: shape (n_envs,)
            dones: shape (n_envs,)
            infos: 信息字典列表
        """
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', int(action)))

        results = [remote.recv() for remote in self.remotes]
        obs, rewards, terminateds, truncateds, infos = zip(*results)

        # 合并 terminated 和 truncated 为 dones
        dones = [t or tr for t, tr in zip(terminateds, truncateds)]

        return np.array(obs), np.array(rewards), np.array(dones), list(infos)

    def close(self):
        """关闭所有环境"""
        if self.closed:
            return

        if self.waiting:
            for remote in self.remotes:
                remote.recv()

        for remote in self.remotes:
            remote.send(('close', None))

        for process in self.processes:
            process.join()

        self.closed = True

    def __del__(self):
        if not self.closed:
            self.close()

    def __len__(self):
        return self.n_envs


class DummyParallelEnv:
    """
    虚拟并行环境：用于调试，不使用多进程，顺序执行

    接口与 ParallelEnv 相同
    """

    def __init__(self, env_fns: List[Callable]):
        self.envs = [env_fn() for env_fn in env_fns]
        self.n_envs = len(self.envs)
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self, seeds: Optional[List[int]] = None):
        if seeds is None:
            seeds = [None] * self.n_envs

        results = [env.reset(seed=seed) for env, seed in zip(self.envs, seeds)]
        obs, infos = zip(*results)

        return np.array(obs), list(infos)

    def step(self, actions: np.ndarray):
        results = [env.step(int(action)) for env, action in zip(self.envs, actions)]
        obs, rewards, terminateds, truncateds, infos = zip(*results)

        # 自动重置已结束的环境
        for i, (terminated, truncated) in enumerate(zip(terminateds, truncateds)):
            if terminated or truncated:
                final_obs = obs[i]
                new_obs, reset_info = self.envs[i].reset()
                obs = list(obs)
                obs[i] = new_obs
                obs = tuple(obs)
                infos[i]['final_observation'] = final_obs

        dones = [t or tr for t, tr in zip(terminateds, truncateds)]

        return np.array(obs), np.array(rewards), np.array(dones), list(infos)

    def close(self):
        for env in self.envs:
            env.close()

    def __len__(self):
        return self.n_envs


def make_parallel_env(
    n_envs: int = 8,
    rounds: int = 10,
    seed: int = 0,
    use_multiprocessing: bool = True,
    auxiliary_reward_coef: float = 1.0
) -> ParallelEnv:
    """
    创建并行环境的便捷函数

    参数：
        n_envs: 并行环境数量
        rounds: 每个游戏的回合数
        seed: 基础随机种子
        use_multiprocessing: 是否使用多进程（False时用DummyParallelEnv）
        auxiliary_reward_coef: 辅助奖励系数（默认1.0）

    返回：
        ParallelEnv 或 DummyParallelEnv 实例
    """
    env_fns = [
        lambda i=i: ResourceGameEnv(rounds=rounds, seed=seed + i, auxiliary_reward_coef=auxiliary_reward_coef)
        for i in range(n_envs)
    ]

    if use_multiprocessing:
        return ParallelEnv(env_fns)
    else:
        return DummyParallelEnv(env_fns)
