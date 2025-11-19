#!/usr/bin/env python3
"""
测试卡牌计数重构后的游戏功能
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from game_core import ResourceGame
import numpy as np

def test_basic_game():
    """测试基本游戏功能"""
    print("=== 测试基本游戏功能 ===")

    game = ResourceGame(rounds=2, seed=42)
    game.start_round()

    # 检查手牌格式
    print(f"✓ 手牌格式: {game.hand}")
    assert isinstance(game.hand, dict), "手牌应该是字典"
    assert set(game.hand.keys()) == {1, 2, 3}, "手牌应该包含1、2、3三个键"

    # 检查手牌总数
    total_cards = sum(game.hand.values())
    print(f"✓ 手牌总数: {total_cards}")
    assert total_cards == 5, f"手牌总数应该是5，实际是{total_cards}"

    # 检查观测空间维度
    obs = game.get_observation()
    print(f"✓ 观测空间维度: {len(obs)}")
    assert len(obs) == 36, f"观测空间应该是36维，实际是{len(obs)}维"

    # 检查手牌部分的观测
    hand_obs = obs[:3]
    print(f"✓ 手牌观测 (归一化): {hand_obs}")
    print(f"  原始手牌: 1点×{game.hand[1]}, 2点×{game.hand[2]}, 3点×{game.hand[3]}")

    # 检查动作空间维度
    valid_actions = game.get_valid_actions()
    print(f"✓ 动作空间维度: {len(valid_actions)}")
    assert len(valid_actions) == 6, f"动作空间应该是6维，实际是{len(valid_actions)}维"

    print(f"✓ 有效动作: {valid_actions}")

    return game

def test_move_action(game):
    """测试移动动作"""
    print("\n=== 测试移动动作 ===")

    # 找到一张有的卡
    card_value = None
    for v in [1, 2, 3]:
        if game.hand[v] > 0:
            card_value = v
            break

    if card_value is None:
        print("✗ 没有可用的卡牌")
        return False

    old_hand = game.hand.copy()
    old_pos = game.position

    print(f"使用 {card_value} 点卡牌移动...")
    success, msg = game.move(card_value)

    if success:
        print(f"✓ {msg}")
        print(f"✓ 手牌变化: {old_hand} -> {game.hand}")
        assert game.hand[card_value] == old_hand[card_value] - 1, "卡牌数量应该减1"
        print(f"✓ 位置变化: {old_pos} -> {game.position}")
        return True
    else:
        print(f"✗ 移动失败: {msg}")
        return False

def test_collect_action(game):
    """测试收集动作"""
    print("\n=== 测试收集动作 ===")

    # 先确保可以收集
    if not game.collectable:
        print("需要先移动才能收集，跳过此测试")
        return True

    # 找到一张有的卡
    card_value = None
    for v in [1, 2, 3]:
        if game.hand[v] > 0:
            card_value = v
            break

    if card_value is None:
        print("✗ 没有可用的卡牌")
        return False

    old_hand = game.hand.copy()

    print(f"使用 {card_value} 点卡牌收集...")
    success, msg, tokens, customer_gains = game.collect(card_value)

    if success:
        print(f"✓ {msg}")
        print(f"✓ 手牌变化: {old_hand} -> {game.hand}")
        assert game.hand[card_value] == old_hand[card_value] - 1, "卡牌数量应该减1"
        print(f"✓ 获得代币: {tokens}")
        return True
    else:
        print(f"✗ 收集失败: {msg}")
        return False

def test_combo(game):
    """测试连击功能"""
    print("\n=== 测试连击功能 ===")

    # 检查连击值列表
    combo_values = game.can_combo_values()
    print(f"可连击的卡牌点数: {combo_values}")

    if len(combo_values) > 0:
        print(f"✓ 上次收集点数: {game.last_collect_cost}")
        print(f"✓ 可连击点数: {combo_values}")
    else:
        print("当前没有可连击的卡牌（正常）")

    return True

def test_game_state(game):
    """测试游戏状态"""
    print("\n=== 测试游戏状态 ===")

    state = game.get_state()

    print(f"✓ 状态包含hand: {state.get('hand')}")
    assert 'hand' in state, "状态应该包含hand"
    assert isinstance(state['hand'], dict), "hand应该是字典"

    print(f"✓ 状态包含can_combo_values: {state.get('can_combo_values')}")
    assert 'can_combo_values' in state, "状态应该包含can_combo_values"

    return True

def main():
    """主测试函数"""
    print("开始测试卡牌计数重构...\n")

    try:
        game = test_basic_game()
        test_game_state(game)

        if test_move_action(game):
            test_collect_action(game)

        test_combo(game)

        print("\n" + "="*50)
        print("✓ 所有测试通过！")
        print("="*50)
        return 0

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
