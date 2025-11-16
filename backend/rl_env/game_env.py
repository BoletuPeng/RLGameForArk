"""
符合 Gymnasium 标准的强化学习环境
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any
import sys
import os

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from game_core import ResourceGame


class ResourceGameEnv(gym.Env):
    """
    资源收集游戏的 Gymnasium 环境

    观测空间：38维向量（详见 game_core.py 中的 get_observation）
    动作空间：10个离散动作
        - 0-4: 使用手牌索引0-4进行移动
        - 5-9: 使用手牌索引0-4进行收集

    奖励设计：
        - Token获得：每个token对应+1奖励
        - 顾客获得资源辅助奖励：
            * 普通顾客：有效资源量 * 0.01
            * 高价值顾客：有效资源量 * 0.013
        - 高价值目标点奖励：移动到位置时，该位置资源的总需求量 * 0.0002
        - 无效动作：无惩罚
        - 游戏结束：最终代币数 * 0.1 的额外奖励
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, rounds: int = 10, seed: Optional[int] = None, render_mode: Optional[str] = None):
        super().__init__()

        self.rounds = rounds
        self.render_mode = render_mode
        self.game = ResourceGame(rounds=rounds, seed=seed)

        # 定义观测空间和动作空间
        # 观测：38维连续向量
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(38,),
            dtype=np.float32
        )

        # 动作：10个离散动作
        self.action_space = spaces.Discrete(10)

        # 用于渲染
        self.window = None
        self.clock = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """重置环境"""
        super().reset(seed=seed)

        # 重置游戏
        obs = self.game.reset(seed=seed)

        info = self._get_info()

        return obs, info

    def _calculate_resource_need(self, resource_type: str) -> int:
        """计算所有顾客对某个资源的总需求量（需求量 - 已拥有量）"""
        total_need = 0
        for cust in self.game.customers:
            if resource_type in cust.needs:
                need = cust.needs[resource_type]
                have = cust.have.get(resource_type, 0)
                total_need += max(0, need - have)
        return total_need

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        执行一个动作

        返回：
            observation: 新的观测
            reward: 奖励
            terminated: 游戏是否正常结束
            truncated: 游戏是否被截断（此环境中不使用）
            info: 额外信息
        """
        # 检查回合是否结束，如果结束则自动开始新回合
        if self.game.is_round_over() and not self.game.is_game_over():
            self.game.start_round()

        reward = 0.0
        info = {}

        # 解析动作
        if action < 5:
            # 移动动作
            card_index = action
            success, msg = self.game.move(card_index)
            if success:
                info['action_type'] = 'move'
                info['message'] = msg

                # 高价值目标点奖励：移动到的位置对应资源的总需求量 * 0.0002
                resource_type = self.game.tile_type()
                total_need = self._calculate_resource_need(resource_type)
                target_reward = total_need * 0.0002
                reward += target_reward
                info['target_reward'] = target_reward
            else:
                # 无效动作，不再惩罚
                info['action_type'] = 'invalid_move'
                info['message'] = msg
        else:
            # 收集动作
            card_index = action - 5
            success, msg, tokens_earned, customer_gains = self.game.collect(card_index)
            if success:
                # 1. Token奖励：每个token获得对应1的奖励
                reward = tokens_earned

                # 2. 顾客获得资源的辅助奖励
                resource_gain_reward = 0.0
                for is_vip, gain in customer_gains:
                    if is_vip:
                        # 高价值顾客：获得资源量 * 0.013
                        resource_gain_reward += gain * 0.013
                    else:
                        # 普通顾客：获得资源量 * 0.01
                        resource_gain_reward += gain * 0.01
                reward += resource_gain_reward

                info['action_type'] = 'collect'
                info['message'] = msg
                info['tokens_earned'] = tokens_earned
                info['resource_gain_reward'] = resource_gain_reward
            else:
                # 无效动作，不再惩罚
                info['action_type'] = 'invalid_collect'
                info['message'] = msg

        # 获取新观测
        obs = self.game.get_observation()

        # 检查游戏是否结束
        terminated = self.game.is_game_over()

        if terminated:
            # 游戏结束，根据最终代币数给予额外奖励
            # 这里使用代币数作为最终奖励的一部分
            reward += self.game.tokens * 0.1  # 额外奖励
            info['final_tokens'] = self.game.tokens
            info['game_over'] = True

        truncated = False  # 此环境不使用截断

        # 添加有效动作掩码到info中
        info['action_mask'] = self.game.get_valid_actions()

        return obs, reward, terminated, truncated, info

    def _get_info(self) -> Dict[str, Any]:
        """获取当前环境信息"""
        return {
            'round': self.game.current_round,
            'tokens': self.game.tokens,
            'position': self.game.position,
            'resource_coef': self.game.resource_coef,
            'action_mask': self.game.get_valid_actions()
        }

    def render(self):
        """渲染当前环境状态"""
        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()

    def _render_human(self):
        """在终端渲染"""
        state = self.game.get_state()
        print("\n" + "=" * 60)
        print(f"回合 {state['current_round']}/{state['total_rounds']} | "
              f"位置: [{state['position']}] {state['resource_type']} | "
              f"系数: {state['resource_coef']} | "
              f"代币: {state['tokens']}")
        print(f"手牌: {state['hand']}")
        print(f"可收集: {state['collectable']} | "
              f"可连击索引: {state['can_combo_indices']}")
        print("\n顾客状态:")
        for i, cust in enumerate(state['customers'], 1):
            print(f"  顾客{i}: {cust}")
        print("=" * 60)

    def _render_rgb_array(self):
        """返回RGB数组（用于视频录制等）"""
        # TODO: 实现图形化渲染
        pass

    def close(self):
        """关闭环境"""
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()


def make_env(rounds: int = 10, seed: Optional[int] = None):
    """创建环境的工厂函数"""
    def _init():
        env = ResourceGameEnv(rounds=rounds, seed=seed)
        return env
    return _init
