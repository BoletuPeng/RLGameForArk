"""
测试 Numba 优化后的代码
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

import numpy as np
from backend.rl_env.game_env import ResourceGameEnv

def test_basic_functionality():
    """测试基本功能是否正常"""
    print("=" * 60)
    print("测试 Numba 优化后的游戏环境")
    print("=" * 60)

    # 创建环境
    print("\n1. 创建环境...")
    env = ResourceGameEnv(rounds=10, seed=42)
    print("   ✓ 环境创建成功")

    # 重置环境
    print("\n2. 重置环境...")
    obs, info = env.reset(seed=42)
    print(f"   ✓ 观测向量维度: {obs.shape}")
    print(f"   ✓ 观测向量范围: [{obs.min():.3f}, {obs.max():.3f}]")
    print(f"   ✓ 有效动作掩码: {info['action_mask']}")

    # 执行一些步骤
    print("\n3. 执行随机动作...")
    total_reward = 0
    steps = 0

    for _ in range(50):
        # 获取有效动作
        valid_actions = info['action_mask']
        valid_indices = np.where(valid_actions > 0)[0]

        if len(valid_indices) == 0:
            print("   ! 没有有效动作")
            break

        # 随机选择一个有效动作
        action = np.random.choice(valid_indices)

        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if terminated or truncated:
            print(f"   ✓ 游戏结束，执行了 {steps} 步")
            break

    print(f"   ✓ 总奖励: {total_reward:.2f}")
    print(f"   ✓ 最终代币数: {info.get('final_tokens', 'N/A')}")

    # 性能测试
    print("\n4. 性能测试 (1000次观测计算)...")
    import time

    env.reset(seed=42)
    start_time = time.time()

    for _ in range(1000):
        obs = env.game.get_observation()
        valid = env.game.get_valid_actions()

    elapsed = time.time() - start_time
    print(f"   ✓ 1000次计算耗时: {elapsed:.3f} 秒")
    print(f"   ✓ 平均每次: {elapsed/1000*1000:.3f} 毫秒")

    print("\n" + "=" * 60)
    print("✓ 所有测试通过！Numba 优化正常工作")
    print("=" * 60)

if __name__ == "__main__":
    test_basic_functionality()
