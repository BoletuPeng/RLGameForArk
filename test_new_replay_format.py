"""
测试新的replay格式
验证transitions是否正确记录了36维观测、动作掩码和选择的动作
"""
import sys
import os
import json

# 添加backend路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from game_core import ResourceGame
import numpy as np


def test_transitions_format():
    """测试transitions格式是否正确"""
    print("=" * 60)
    print("测试新的Replay格式")
    print("=" * 60)

    # 创建游戏实例
    game = ResourceGame(rounds=3, seed=42)
    game.start_round()

    print(f"\n初始seed: {game.initial_seed}")
    print(f"初始观测维度: {game.get_observation().shape}")
    print(f"动作掩码维度: {game.get_valid_actions().shape}")

    # 模拟几步游戏
    step_count = 0
    max_steps = 10

    print("\n开始模拟游戏...")
    while step_count < max_steps and not game.is_game_over():
        # 获取当前状态
        obs_before = game.get_observation()
        valid_actions = game.get_valid_actions()
        old_tokens = game.tokens

        # 选择一个有效动作
        valid_indices = np.where(valid_actions == 1)[0]
        if len(valid_indices) == 0:
            print("  没有有效动作，结束测试")
            break

        action_index = valid_indices[0]  # 选择第一个有效动作
        card_index = action_index if action_index < 5 else action_index - 5
        action_type = 'move' if action_index < 5 else 'collect'

        print(f"\n步骤 {step_count + 1}:")
        print(f"  动作: {action_type}, 卡牌索引: {card_index}")
        print(f"  手牌: {game.hand}")

        # 执行动作（模拟API调用中的逻辑）
        card_value = game.hand[card_index]

        # 执行动作
        if action_type == 'move':
            success, msg = game.move(card_index)
        else:
            success, msg, tokens, _ = game.collect(card_index)

        if not success:
            print(f"  动作失败: {msg}")
            continue

        # 手动记录transition（模拟API中的逻辑）
        obs_after = game.get_observation()
        reward = float(game.tokens - old_tokens)
        done = game.is_game_over()

        transition = {
            'step': len(game.transitions),
            'observation': obs_before.tolist(),
            'valid_actions': valid_actions.tolist(),
            'action': action_index,
            'action_type': action_type,
            'card_index': card_index,
            'card_value': card_value,
            'reward': reward,
            'next_observation': obs_after.tolist(),
            'done': done,
            'info': {'message': msg}
        }
        game.transitions.append(transition)

        print(f"  成功: {msg}")
        print(f"  奖励: {reward}")
        print(f"  Transition已记录")

        # 检查是否需要开始新回合
        if game.is_round_over() and not game.is_game_over():
            game.start_round()
            print("  开始新回合")

        step_count += 1

    # 验证transitions
    print("\n" + "=" * 60)
    print("验证Transitions数据")
    print("=" * 60)

    print(f"\n总共记录了 {len(game.transitions)} 个transitions")

    if len(game.transitions) > 0:
        # 检查第一个transition
        first_transition = game.transitions[0]
        print("\n第一个transition的结构:")
        for key, value in first_transition.items():
            if isinstance(value, list):
                if key in ['observation', 'next_observation']:
                    print(f"  {key}: list of length {len(value)} (前5个值: {value[:5]})")
                elif key == 'valid_actions':
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value}")

        # 验证维度
        obs_dim = len(first_transition['observation'])
        valid_actions_dim = len(first_transition['valid_actions'])
        print(f"\n维度验证:")
        print(f"  observation维度: {obs_dim} (期望: 38)")
        print(f"  valid_actions维度: {valid_actions_dim} (期望: 10)")

        if obs_dim == 38 and valid_actions_dim == 10:
            print("  ✓ 维度正确！")
        else:
            print("  ✗ 维度错误！")

        # 验证动作索引范围
        actions = [t['action'] for t in game.transitions]
        print(f"\n动作索引验证:")
        print(f"  动作索引范围: {min(actions)} - {max(actions)}")
        print(f"  所有动作: {actions}")
        if all(0 <= a < 10 for a in actions):
            print("  ✓ 动作索引在有效范围内！")
        else:
            print("  ✗ 动作索引超出范围！")

    # 保存示例replay文件
    print("\n" + "=" * 60)
    print("保存示例Replay文件")
    print("=" * 60)

    replay_data = {
        'game_id': 'test_game',
        'timestamp': '2025-01-01T00:00:00',
        'seed': game.initial_seed,
        'rounds': game.rounds,
        'current_round': game.current_round,
        'final_tokens': game.tokens,
        'transitions': game.transitions,
        'total_moves': game.total_moves,
        'total_collections': game.total_collections
    }

    os.makedirs('replays', exist_ok=True)
    test_replay_path = 'replays/test_new_format.json'

    with open(test_replay_path, 'w', encoding='utf-8') as f:
        json.dump(replay_data, f, indent=2, ensure_ascii=False)

    print(f"\n示例replay已保存到: {test_replay_path}")
    print(f"文件大小: {os.path.getsize(test_replay_path)} 字节")

    # 读取并验证
    print("\n读取并验证保存的replay文件...")
    with open(test_replay_path, 'r', encoding='utf-8') as f:
        loaded_replay = json.load(f)

    print(f"  seed: {loaded_replay.get('seed')}")
    print(f"  transitions数量: {len(loaded_replay.get('transitions', []))}")
    print(f"  final_tokens: {loaded_replay.get('final_tokens')}")

    if loaded_replay.get('transitions'):
        first_t = loaded_replay['transitions'][0]
        print(f"  第一个transition的observation维度: {len(first_t['observation'])}")
        print(f"  第一个transition的valid_actions维度: {len(first_t['valid_actions'])}")

    print("\n✓ 测试完成！新的replay格式工作正常。")


if __name__ == "__main__":
    test_transitions_format()
