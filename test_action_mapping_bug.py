"""
测试动作映射是否正确

这个测试验证动作空间的映射是否与实际执行一致
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from rl_env.game_env import ResourceGameEnv
import numpy as np


def test_action_mapping():
    """测试每个动作是否正确映射到对应的卡牌点数"""
    print("=" * 80)
    print("测试动作映射")
    print("=" * 80)

    env = ResourceGameEnv(rounds=10, seed=42)

    # 重置环境
    obs, info = env.reset()

    # 根据注释,动作空间应该是:
    # 0: move(1)
    # 1: move(2)
    # 2: move(3)
    # 3: collect(1)
    # 4: collect(2)
    # 5: collect(3)

    print("\n根据代码注释,动作空间定义为:")
    print("  动作 0-2: 使用1点、2点、3点牌进行移动")
    print("  动作 3-5: 使用1点、2点、3点牌进行收集")
    print()

    # 测试每个移动动作
    test_cases = [
        ("动作0 (应该是move with 1点卡)", 0, "move", 1),
        ("动作1 (应该是move with 2点卡)", 1, "move", 2),
        ("动作2 (应该是move with 3点卡)", 2, "move", 3),
        ("动作3 (应该是collect with 1点卡)", 3, "collect", 1),
        ("动作4 (应该是collect with 2点卡)", 4, "collect", 2),
        ("动作5 (应该是collect with 3点卡)", 5, "collect", 3),
    ]

    bugs_found = []

    for test_name, action, expected_type, expected_card_value in test_cases:
        # 重置环境确保有所有卡牌
        obs, info = env.reset(seed=42)

        # 手动设置游戏状态,确保有所有点数的卡牌
        env.game.hand = {1: 2, 2: 2, 3: 1}
        env.game.collectable = True  # 允许收集

        print(f"\n测试: {test_name}")
        print(f"  手牌: {env.game.hand}")

        # 记录执行前的手牌
        hand_before = env.game.hand.copy()

        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)

        # 记录执行后的手牌
        hand_after = env.game.hand.copy()

        # 检查哪张卡被消耗了
        consumed_card = None
        for card_value in [1, 2, 3]:
            if hand_before.get(card_value, 0) > hand_after.get(card_value, 0):
                consumed_card = card_value
                break

        # 检查动作类型
        action_type = info.get('action_type', 'unknown')

        print(f"  执行结果: {action_type}")
        if consumed_card:
            print(f"  消耗的卡牌: {consumed_card}点")
        else:
            print(f"  未消耗卡牌 (动作失败)")
        print(f"  消息: {info.get('message', 'N/A')}")

        # 验证结果
        is_correct = True
        error_msg = []

        # 检查动作是否成功
        if action_type.startswith('invalid'):
            is_correct = False
            error_msg.append(f"动作失败: {action_type}")

        # 检查动作类型是否正确
        if not action_type.startswith(expected_type):
            is_correct = False
            error_msg.append(f"动作类型错误: 期望{expected_type}, 实际{action_type}")

        # 检查消耗的卡牌是否正确
        if consumed_card != expected_card_value and consumed_card is not None:
            is_correct = False
            error_msg.append(f"卡牌点数错误: 期望{expected_card_value}点, 实际{consumed_card}点")

        if is_correct:
            print(f"  ✓ 测试通过")
        else:
            print(f"  ✗ 测试失败:")
            for msg in error_msg:
                print(f"    - {msg}")
            bugs_found.append((test_name, error_msg))

    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)

    if bugs_found:
        print(f"\n发现 {len(bugs_found)} 个bug:")
        for test_name, errors in bugs_found:
            print(f"\n  {test_name}:")
            for error in errors:
                print(f"    - {error}")

        print("\n" + "=" * 80)
        print("根本原因分析:")
        print("=" * 80)
        print("""
在 backend/rl_env/game_env.py 的 step() 函数中:

当前错误的实现 (第111-114行):
    if action < 5:
        card_index = action
        success, msg = self.game.move(card_index)

这导致:
  - action 0 -> move(0) -> 无效 (卡牌点数必须是1-3)
  - action 1 -> move(1) -> 正确
  - action 2 -> move(2) -> 正确
  - action 3 -> move(3) -> 正确
  - action 4 -> move(4) -> 无效 (卡牌点数必须是1-3)
  - action 5 -> collect(5-5=0) -> 无效 (卡牌点数必须是1-3)

正确的实现应该是:
    if action < 3:
        card_value = action + 1  # 0->1, 1->2, 2->3
        success, msg = self.game.move(card_value)
    else:
        card_value = (action - 3) + 1  # 3->1, 4->2, 5->3
        success, msg, tokens_earned, customer_gains = self.game.collect(card_value)
""")
    else:
        print("\n所有测试通过!")

    env.close()


if __name__ == "__main__":
    test_action_mapping()
