"""
完整的游戏逻辑测试
验证移动、收集、资源分配等核心逻辑
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from game_core import ResourceGame
import numpy as np


def test_movement_logic():
    """测试移动逻辑:消耗卡牌,移动对应距离"""
    print("=" * 80)
    print("测试移动逻辑")
    print("=" * 80)

    game = ResourceGame(rounds=10, seed=42)
    game.start_round()

    # 设置已知状态
    game.position = 0
    game.hand = {1: 1, 2: 1, 3: 1}

    print(f"\n初始位置: {game.position} ({game.tile_type()})")
    print(f"初始手牌: {game.hand}")

    # 测试1:使用1点卡移动
    print("\n测试1:使用1点卡移动")
    success, msg = game.move(1)
    print(f"  结果: {success}")
    print(f"  消息: {msg}")
    print(f"  新位置: {game.position} (期望:1)")
    print(f"  剩余手牌: {game.hand} (期望: {{1: 0, 2: 1, 3: 1}})")
    assert success, "移动应该成功"
    assert game.position == 1, f"位置错误:期望1,实际{game.position}"
    assert game.hand[1] == 0, f"1点卡应该被消耗"

    # 测试2:使用2点卡移动
    print("\n测试2:使用2点卡移动")
    success, msg = game.move(2)
    print(f"  结果: {success}")
    print(f"  消息: {msg}")
    print(f"  新位置: {game.position} (期望:3)")
    assert game.position == 3, f"位置错误:期望3,实际{game.position}"
    assert game.hand[2] == 0, f"2点卡应该被消耗"

    # 测试3:使用3点卡移动
    print("\n测试3:使用3点卡移动")
    success, msg = game.move(3)
    print(f"  结果: {success}")
    print(f"  消息: {msg}")
    print(f"  新位置: {game.position} (期望:6)")
    assert game.position == 6, f"位置错误:期望6,实际{game.position}"
    assert game.hand[3] == 0, f"3点卡应该被消耗"

    # 测试4:环形移动
    print("\n测试4:环形移动(从位置8移动3步)")
    game.position = 8
    game.hand = {1: 0, 2: 0, 3: 1}
    old_coef = game.resource_coef
    success, msg = game.move(3)
    print(f"  旧位置: 8")
    print(f"  新位置: {game.position} (期望:1,因为(8+3)%10=1)")
    print(f"  资源系数: {game.resource_coef} (期望:{old_coef+2},因为绕过起点)")
    assert game.position == 1, f"环形移动位置错误"
    assert game.resource_coef == old_coef + 2, f"绕过起点应该+2资源系数"

    print("\n✓ 移动逻辑测试通过!")


def test_collection_logic():
    """测试收集逻辑:产出=资源系数×卡牌点数"""
    print("\n" + "=" * 80)
    print("测试收集逻辑")
    print("=" * 80)

    game = ResourceGame(rounds=10, seed=42)
    game.start_round()

    # 设置已知状态
    game.position = 0  # 冰
    game.resource_coef = 3
    game.hand = {1: 1, 2: 1, 3: 1}
    game.collectable = True

    print(f"\n初始状态:")
    print(f"  位置: {game.position} ({game.tile_type()})")
    print(f"  资源系数: {game.resource_coef}")
    print(f"  手牌: {game.hand}")

    # 测试1:使用1点卡收集
    print("\n测试1:使用1点卡收集")
    print(f"  期望产出: {game.resource_coef} × 1 = {game.resource_coef * 1} 个{game.tile_type()}")

    # 记录顾客的初始状态
    initial_customer_have = [c.have.copy() for c in game.customers]

    success, msg, tokens, customer_gains = game.collect(1)
    print(f"  结果: {success}")
    print(f"  消息: {msg}")
    print(f"  剩余手牌: {game.hand}")

    # 验证资源分配
    total_gained = 0
    for idx, cust in enumerate(game.customers):
        if game.tile_type() in cust.needs:
            gained = cust.have.get(game.tile_type(), 0) - initial_customer_have[idx].get(game.tile_type(), 0)
            total_gained += gained

    print(f"  顾客总共获得的资源: {total_gained} (所有需要冰的顾客)")
    assert success, "收集应该成功"
    assert game.hand[1] == 0, "1点卡应该被消耗"

    # 测试2:移动到新位置,使用3点卡收集
    print("\n测试2:移动到新位置,使用3点卡收集")
    game.move(2)  # 使用2点卡移动到位置2
    print(f"  新位置: {game.position} ({game.tile_type()})")
    print(f"  资源系数: {game.resource_coef}")
    print(f"  剩余手牌: {game.hand}")
    print(f"  期望产出: {game.resource_coef} × 3 = {game.resource_coef * 3} 个{game.tile_type()}")

    initial_customer_have = [c.have.copy() for c in game.customers]
    success, msg, tokens, customer_gains = game.collect(3)
    print(f"  结果: {success}")
    print(f"  消息: {msg}")

    total_gained = 0
    for idx, cust in enumerate(game.customers):
        if game.tile_type() in cust.needs:
            gained = cust.have.get(game.tile_type(), 0) - initial_customer_have[idx].get(game.tile_type(), 0)
            total_gained += gained

    print(f"  顾客总共获得的资源: {total_gained}")
    assert success, "收集应该成功"

    # 测试3:连击收集
    print("\n测试3:连击收集")
    # 重置手牌以测试连击
    game.hand = {1: 2, 2: 2, 3: 0}
    game.last_collect_cost = 1
    game.last_action_was_move = False
    print(f"  设置上次收集用1点卡")
    print(f"  手牌: {game.hand}")
    print(f"  可以连击的卡: {game.can_combo_values()} (期望: [2])")

    if 2 in game.can_combo_values() and game.hand[2] > 0:
        initial_customer_have = [c.have.copy() for c in game.customers]
        success, msg, tokens, customer_gains = game.collect(2)
        print(f"  结果: {success}")
        print(f"  消息: {msg}")

        if "连击" in msg:
            print(f"  ✓ 连击成功!应该额外获得+2全资源")
        else:
            print(f"  ✗ 应该是连击,但没有触发")
            print(f"     实际消息: {msg}")
        assert "连击" in msg, "应该触发连击"
    else:
        print(f"  ✗ 无法测试连击")

    print("\n✓ 收集逻辑测试通过!")


def test_resource_distribution():
    """测试资源分配逻辑"""
    print("\n" + "=" * 80)
    print("测试资源分配逻辑")
    print("=" * 80)

    game = ResourceGame(rounds=10, seed=42)
    game.start_round()

    # 手动设置顾客需求,便于测试
    from game_core import Customer
    game.customers = [
        Customer(is_vip=True, needs={'冰': 10, '铁': 5}),
        Customer(is_vip=False, needs={'冰': 5}),
        Customer(is_vip=False, needs={'火': 8})
    ]

    print("\n顾客需求:")
    for idx, cust in enumerate(game.customers):
        print(f"  顾客{idx+1}: {cust.progress_str()}")

    # 测试:收集冰
    game.position = 0  # 冰
    game.resource_coef = 10
    game.hand = {1: 1, 2: 1, 3: 1}
    game.collectable = True

    print(f"\n在{game.tile_type()}位置收集,资源系数={game.resource_coef}")
    print(f"使用1点卡收集,期望产出: {game.resource_coef * 1} 个冰")

    success, msg, tokens, customer_gains = game.collect(1)
    print(f"\n收集结果: {msg}")
    print("\n收集后顾客状态:")
    for idx, cust in enumerate(game.customers):
        print(f"  顾客{idx+1}: {cust.progress_str()}")
        if idx < len(customer_gains):
            is_vip, gain = customer_gains[idx]
            print(f"    本次获得有效资源: {gain}")

    # 验证:资源分配是正确的
    # 注意:完成订单的顾客会被替换成新顾客,所以我们检查customer_gains
    total_effective_gain = sum(gain for is_vip, gain in customer_gains)
    print(f"\n所有顾客的有效资源获得总和: {total_effective_gain}")
    # 产出了10个冰,顾客1需要10个,顾客2需要5个(已满足后被替换),总共最多15个有效
    assert total_effective_gain == 15, f"总有效资源应该是15,实际是{total_effective_gain}"

    # 检查是否有顾客完成订单
    print(f"\n代币数: {game.tokens}")
    if tokens > 0:
        print(f"✓ 有顾客完成订单,获得 {tokens} 代币")

    print("\n✓ 资源分配逻辑测试通过!")


def test_full_game_round():
    """测试完整回合"""
    print("\n" + "=" * 80)
    print("测试完整回合")
    print("=" * 80)

    game = ResourceGame(rounds=10, seed=42)
    game.start_round()

    print(f"\n回合 {game.current_round}")
    print(f"手牌: {game.hand}")
    print(f"总手牌数: {sum(game.hand.values())} (应该是5张)")

    assert sum(game.hand.values()) == 5, "每回合应该发5张牌"

    # 验证手牌都是1-3点
    for card_value, count in game.hand.items():
        assert card_value in [1, 2, 3], f"卡牌点数应该在1-3之间,但得到{card_value}"
        assert count >= 0, f"卡牌数量不应该为负"

    print(f"\n✓ 手牌验证通过:共{sum(game.hand.values())}张,点数都在1-3之间")

    # 模拟玩一个回合
    actions_taken = 0
    max_actions = 10  # 每回合最多5张牌,但可能有移动+收集

    print("\n开始模拟回合:")
    while not game.is_round_over() and actions_taken < max_actions:
        # 简单策略:有牌就移动,移动后收集
        if game.can_move():
            # 选择第一个有的卡牌点数
            for card_value in [1, 2, 3]:
                if game.hand[card_value] > 0:
                    success, msg = game.move(card_value)
                    if success:
                        print(f"  {msg}")
                        actions_taken += 1

                        # 移动后尝试收集
                        if game.collectable and game.hand[card_value] > 0:
                            success2, msg2, tokens, gains = game.collect(card_value)
                            if success2:
                                print(f"  {msg2}")
                                actions_taken += 1
                        break
        else:
            break

    print(f"\n回合结束")
    print(f"总移动次数: {game.total_moves}")
    print(f"总收集次数: {game.total_collections}")
    print(f"当前代币: {game.tokens}")
    print(f"是否回合结束: {game.is_round_over()}")

    assert game.is_round_over() or actions_taken >= max_actions, "回合应该结束"

    print("\n✓ 完整回合测试通过!")


if __name__ == "__main__":
    test_movement_logic()
    test_collection_logic()
    test_resource_distribution()
    test_full_game_round()

    print("\n" + "=" * 80)
    print("所有游戏逻辑测试通过!")
    print("=" * 80)
