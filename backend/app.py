"""
Flask 后端API服务器
"""
from flask import Flask, jsonify, request, render_template, session
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import uuid
import numpy as np
from typing import Dict, Optional
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from game_core import ResourceGame

# 尝试导入sb3_contrib以支持MaskablePPO模型
try:
    from sb3_contrib import MaskablePPO
    import torch
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("警告: 未安装 sb3-contrib，MaskablePPO模型功能将不可用")

app = Flask(__name__,
            template_folder='../frontend/templates',
            static_folder='../frontend/static')
app.config['SECRET_KEY'] = 'your-secret-key-here'  # 在生产环境中应使用环境变量
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# 存储游戏会话
game_sessions: Dict[str, ResourceGame] = {}

# 全局PPO模型缓存
ppo_models = {
    'best': None,
    'final': None
}

def load_ppo_model(model_name='best'):
    """
    加载MaskablePPO模型

    Args:
        model_name: 'best' 或 'final'

    Returns:
        加载的MaskablePPO模型，如果失败返回None
    """
    if not HAS_SB3:
        print("错误: 未安装 sb3-contrib")
        return None

    # 检查缓存
    if ppo_models.get(model_name) is not None:
        return ppo_models[model_name]

    # 确定模型路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if model_name == 'best':
        model_path = os.path.join(project_root, 'models', 'ppo_resource_game', 'best_model.zip')
    elif model_name == 'final':
        model_path = os.path.join(project_root, 'models', 'ppo_resource_game', 'final_model.zip')
    else:
        print(f"未知的模型名称: {model_name}")
        return None

    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return None

    try:
        print(f"正在加载MaskablePPO模型: {model_path}")
        model = MaskablePPO.load(model_path)
        ppo_models[model_name] = model
        print(f"MaskablePPO模型加载成功: {model_name}")
        return model
    except Exception as e:
        print(f"加载MaskablePPO模型失败: {e}")
        import traceback
        traceback.print_exc()
        return None


# ==================== 辅助函数 ====================

def get_or_create_game(game_id: Optional[str] = None) -> tuple[str, ResourceGame]:
    """获取或创建游戏会话"""
    if game_id and game_id in game_sessions:
        return game_id, game_sessions[game_id]

    # 创建新游戏
    new_id = str(uuid.uuid4())
    game = ResourceGame(rounds=10)
    game.start_round()
    game_sessions[new_id] = game

    return new_id, game


def serialize_game_state(game: ResourceGame) -> dict:
    """序列化游戏状态（用于API返回）"""
    state = game.get_state()

    # 添加额外的AI相关信息
    state['observation'] = game.get_observation().tolist()
    state['valid_actions'] = game.get_valid_actions().tolist()
    state['action_history'] = game.action_history[-10:]  # 最近10个动作

    return state


# ==================== 全局错误处理 ====================

@app.errorhandler(500)
def internal_error(error):
    """处理500错误，返回JSON格式"""
    return jsonify({
        'error': 'Internal server error',
        'message': str(error)
    }), 500


@app.errorhandler(404)
def not_found(error):
    """处理404错误，返回JSON格式"""
    return jsonify({
        'error': 'Not found',
        'message': str(error)
    }), 404


# ==================== Web Routes ====================

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')


@app.route('/api/game/new', methods=['POST'])
def new_game():
    """创建新游戏"""
    data = request.json or {}
    rounds = data.get('rounds', 10)
    seed = data.get('seed', None)

    game_id = str(uuid.uuid4())
    game = ResourceGame(rounds=rounds, seed=seed)
    game.start_round()
    game_sessions[game_id] = game

    return jsonify({
        'game_id': game_id,
        'state': serialize_game_state(game)
    })


@app.route('/api/game/<game_id>/state', methods=['GET'])
def get_state(game_id: str):
    """获取游戏状态"""
    if game_id not in game_sessions:
        return jsonify({'error': 'Game not found'}), 404

    game = game_sessions[game_id]
    return jsonify(serialize_game_state(game))


@app.route('/api/game/<game_id>/action', methods=['POST'])
def perform_action(game_id: str):
    """执行动作"""
    if game_id not in game_sessions:
        return jsonify({'error': 'Game not found'}), 404

    game = game_sessions[game_id]
    data = request.json

    action_type = data.get('type')  # 'move' or 'collect'
    card_value = data.get('card_value')

    if action_type not in ['move', 'collect']:
        return jsonify({'error': 'Invalid action type'}), 400

    if card_value is None or card_value not in [1, 2, 3]:
        return jsonify({'error': 'Invalid card value'}), 400

    # 检查是否有该点数的牌
    if game.hand.get(card_value, 0) == 0:
        return jsonify({'error': f'No cards with value {card_value}'}), 400

    # ==================== 记录transition（用于replay） ====================
    # 1. 记录执行动作前的状态
    observation = game.get_observation()
    valid_actions = game.get_valid_actions()
    old_tokens = game.tokens

    # 2. 计算动作索引（0-5）
    # 动作空间：[move_1, move_2, move_3, collect_1, collect_2, collect_3]
    if action_type == 'move':
        action_index = card_value - 1  # 0-2
    else:  # 'collect'
        action_index = 3 + card_value - 1  # 3-5

    result = {'success': False}
    transition_info = {}

    if action_type == 'move':
        success, msg = game.move(card_value)
        result = {
            'success': success,
            'message': msg,
            'type': 'move'
        }
        transition_info['message'] = msg
    elif action_type == 'collect':
        success, msg, tokens, customer_gains = game.collect(card_value)
        result = {
            'success': success,
            'message': msg,
            'type': 'collect',
            'tokens_earned': tokens if success else 0
        }
        transition_info['message'] = msg
        transition_info['tokens_earned'] = tokens if success else 0
        transition_info['customer_gains'] = customer_gains

    # 3. 计算奖励（简化版本：主要基于tokens变化）
    reward = float(game.tokens - old_tokens)

    # 执行动作后检查是否需要开始新回合
    if game.is_round_over() and not game.is_game_over():
        game.start_round()

    # 4. 记录执行动作后的状态
    next_observation = game.get_observation()
    done = game.is_game_over()

    # 5. 保存完整的transition
    if success:  # 只记录成功的动作
        transition = {
            'step': len(game.transitions),
            'observation': observation.tolist(),
            'valid_actions': valid_actions.tolist(),
            'action': action_index,
            'action_type': action_type,
            'card_value': card_value,
            'reward': reward,
            'next_observation': next_observation.tolist(),
            'done': done,
            'info': transition_info
        }
        game.transitions.append(transition)

    # 返回结果和新状态
    return jsonify({
        'result': result,
        'state': serialize_game_state(game)
    })


@app.route('/api/game/<game_id>/next_round', methods=['POST'])
def next_round(game_id: str):
    """开始下一回合"""
    if game_id not in game_sessions:
        return jsonify({'error': 'Game not found'}), 404

    game = game_sessions[game_id]

    if game.is_game_over():
        return jsonify({'error': 'Game is over'}), 400

    if not game.is_round_over():
        return jsonify({'error': 'Current round is not over'}), 400

    game.start_round()

    return jsonify({
        'state': serialize_game_state(game)
    })


@app.route('/api/game/<game_id>/reset', methods=['POST'])
def reset_game(game_id: str):
    """重置游戏"""
    if game_id not in game_sessions:
        return jsonify({'error': 'Game not found'}), 404

    data = request.json or {}
    seed = data.get('seed', None)

    game = game_sessions[game_id]
    game.reset(seed=seed)

    return jsonify({
        'state': serialize_game_state(game)
    })


@app.route('/api/game/<game_id>/delete', methods=['DELETE'])
def delete_game(game_id: str):
    """删除游戏会话"""
    if game_id in game_sessions:
        del game_sessions[game_id]
        return jsonify({'message': 'Game deleted'})
    return jsonify({'error': 'Game not found'}), 404


@app.route('/api/game/<game_id>/save_replay', methods=['POST'])
def save_replay(game_id: str):
    """
    保存对局记录

    新格式包含完整的transitions数据，每个transition包含：
    - observation: 29维观测向量
    - valid_actions: 6维动作掩码
    - action: 选择的动作索引（0-5）
    - reward, next_observation, done等

    请求格式：
    {
        "name": "replay_name" (可选，默认使用时间戳)
    }
    """
    try:
        if game_id not in game_sessions:
            return jsonify({'error': 'Game not found'}), 404

        game = game_sessions[game_id]

        # 只能保存已结束的游戏
        if not game.is_game_over():
            return jsonify({'error': 'Game is not over yet'}), 400

        data = request.json or {}
        replay_name = data.get('name', None)

        # 创建replay目录
        replay_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'replays')
        os.makedirs(replay_dir, exist_ok=True)

        # 生成文件名
        import datetime
        if replay_name:
            # 清理文件名，移除非法字符
            replay_name = "".join(c for c in replay_name if c.isalnum() or c in (' ', '-', '_')).strip()
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{replay_name}_{timestamp}.json"
        else:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"replay_{timestamp}.json"

        filepath = os.path.join(replay_dir, filename)

        # 保存对局记录（新格式）
        import json
        replay_data = {
            'game_id': game_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'seed': game.initial_seed,  # 保存初始seed，用于重现游戏
            'rounds': game.rounds,
            'current_round': game.current_round,
            'final_tokens': game.tokens,
            'transitions': game.transitions,  # 完整的transition数据
            'total_moves': game.total_moves,
            'total_collections': game.total_collections,
            # 保留旧的action_history以保持向后兼容性（可选）
            'action_history': game.action_history
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(replay_data, f, indent=2, ensure_ascii=False)

        return jsonify({
            'message': 'Replay saved successfully',
            'filename': filename,
            'filepath': filepath,
            'transitions_count': len(game.transitions),
            'final_tokens': game.tokens,
            'seed': game.initial_seed
        })
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error saving replay: {error_details}")
        return jsonify({
            'error': f'Failed to save replay: {str(e)}'
        }), 500


# ==================== AI 相关接口 ====================

@app.route('/api/models/status', methods=['GET'])
def models_status():
    """
    获取模型状态信息

    返回格式：
    {
        "has_sb3": true/false,
        "models": {
            "ppo_best": {"available": true/false, "path": "..."},
            "ppo_final": {"available": true/false, "path": "..."}
        }
    }
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    best_model_path = os.path.join(project_root, 'models', 'ppo_resource_game', 'best_model.zip')
    final_model_path = os.path.join(project_root, 'models', 'ppo_resource_game', 'final_model.zip')

    status = {
        'has_sb3': HAS_SB3,
        'models': {
            'ppo_best': {
                'available': os.path.exists(best_model_path),
                'path': best_model_path,
                'loaded': ppo_models.get('best') is not None
            },
            'ppo_final': {
                'available': os.path.exists(final_model_path),
                'path': final_model_path,
                'loaded': ppo_models.get('final') is not None
            }
        }
    }

    return jsonify(status)


@app.route('/api/models/list', methods=['GET'])
def list_models():
    """
    列出所有可用的模型文件

    返回格式：
    {
        "models": [
            {
                "name": "best_model.zip",
                "path": "models/ppo_resource_game/best_model.zip",
                "relative_path": "models/ppo_resource_game/best_model.zip",
                "size": 12345,
                "modified": "2024-01-01T12:00:00"
            },
            ...
        ]
    }
    """
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(project_root, 'models')

        if not os.path.exists(models_dir):
            return jsonify({'models': []})

        model_files = []

        # 递归查找所有.zip文件
        for root, dirs, files in os.walk(models_dir):
            for file in files:
                if file.endswith('.zip'):
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, project_root)

                    # 获取文件信息
                    stat = os.stat(full_path)
                    import datetime
                    modified_time = datetime.datetime.fromtimestamp(stat.st_mtime).isoformat()

                    model_files.append({
                        'name': file,
                        'path': full_path,
                        'relative_path': relative_path,
                        'directory': os.path.relpath(root, models_dir),
                        'size': stat.st_size,
                        'modified': modified_time
                    })

        # 按修改时间倒序排序
        model_files.sort(key=lambda x: x['modified'], reverse=True)

        return jsonify({'models': model_files})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'models': []}), 500


@app.route('/api/game/<game_id>/ai/predict', methods=['POST'])
def ai_predict(game_id: str):
    """
    AI预测接口
    接收模型预测并返回动作建议

    请求格式：
    {
        "model_type": "random" | "rule_based" | "ppo_best" | "ppo_final" | "ppo_custom" | "custom",
        "model_path": "..." (可选，用于ppo_custom类型),
        "probabilities": [...] (可选，用于custom类型)
    }
    """
    if game_id not in game_sessions:
        return jsonify({'error': 'Game not found'}), 404

    game = game_sessions[game_id]
    data = request.json or {}
    model_type = data.get('model_type', 'random')
    model_path = data.get('model_path', None)

    obs = game.get_observation()
    valid_actions = game.get_valid_actions()

    if model_type == 'random':
        # 随机策略（只选择有效动作）
        valid_indices = np.where(valid_actions > 0)[0]
        if len(valid_indices) == 0:
            return jsonify({'error': 'No valid actions'}), 400

        action = np.random.choice(valid_indices)
        probs = np.zeros(6)
        probs[valid_indices] = 1.0 / len(valid_indices)

    elif model_type == 'rule_based':
        # 简单的基于规则的策略
        probs, action = rule_based_policy(game, valid_actions)

    elif model_type in ['ppo_best', 'ppo_final', 'ppo_custom']:
        # MaskablePPO模型推理
        if not HAS_SB3:
            return jsonify({'error': 'sb3-contrib not installed'}), 500

        # 加载对应的模型
        if model_type == 'ppo_custom':
            # 使用自定义模型路径
            if not model_path:
                return jsonify({'error': 'model_path is required for ppo_custom'}), 400

            # 验证模型路径安全性（防止路径遍历攻击）
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

            # 如果是相对路径，转换为绝对路径
            if not os.path.isabs(model_path):
                abs_model_path = os.path.join(project_root, model_path)
            else:
                abs_model_path = model_path

            # 确保路径在项目目录内
            abs_model_path = os.path.abspath(abs_model_path)
            if not abs_model_path.startswith(project_root):
                return jsonify({'error': 'Invalid model path: must be within project directory'}), 400

            if not os.path.exists(abs_model_path):
                return jsonify({'error': f'Model file not found: {model_path}'}), 404

            try:
                print(f"正在加载自定义MaskablePPO模型: {abs_model_path}")
                ppo_model = MaskablePPO.load(abs_model_path)
                print(f"自定义MaskablePPO模型加载成功")
            except Exception as e:
                print(f"加载自定义MaskablePPO模型失败: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({'error': f'Failed to load custom model: {str(e)}'}), 500
        else:
            # 使用预设模型
            model_name = 'best' if model_type == 'ppo_best' else 'final'
            ppo_model = load_ppo_model(model_name)

            if ppo_model is None:
                return jsonify({'error': f'Failed to load MaskablePPO model: {model_name}'}), 500

        try:
            # 使用MaskablePPO模型进行预测
            # 重要：为了确保前端显示的概率分布与选择的动作一致，
            # 我们需要手动计算masked概率分布，然后从中选择最优动作

            # 1. 将观测转换为张量
            obs_tensor = torch.FloatTensor(obs.reshape(1, -1))
            device = next(ppo_model.policy.parameters()).device
            obs_tensor = obs_tensor.to(device)

            # 2. 获取原始动作logits（未应用softmax前的值）
            # MaskablePPO使用特殊的mask方式：将无效动作的logits设为-inf
            with torch.no_grad():
                # 获取actor网络的输出（logits）
                features = ppo_model.policy.extract_features(obs_tensor)
                if hasattr(ppo_model.policy, 'mlp_extractor'):
                    latent_pi = ppo_model.policy.mlp_extractor.forward_actor(features)
                else:
                    latent_pi = features
                logits = ppo_model.policy.action_net(latent_pi)

                # 3. 应用action mask（将无效动作的logits设为-inf）
                # 这与MaskablePPO内部的做法一致
                mask_tensor = torch.FloatTensor(valid_actions).to(device)
                # 对于mask=0的位置，设置为-inf；mask=1的位置保持不变
                masked_logits = torch.where(
                    mask_tensor.bool(),
                    logits[0],
                    torch.tensor(float('-inf')).to(device)
                )

                # 4. 计算masked后的概率分布
                action_probs_tensor = torch.nn.functional.softmax(masked_logits, dim=-1)
                action_probs = action_probs_tensor.cpu().numpy()

                # 5. 选择概率最高的动作（deterministic）
                # 这样前端显示的建议动作就是概率最高的那个
                action = int(np.argmax(action_probs))
                probs = action_probs

            # 验证概率分布的有效性
            if probs.sum() == 0 or not np.isfinite(probs).all():
                return jsonify({'error': 'No valid actions from MaskablePPO model'}), 400

        except Exception as e:
            print(f"MaskablePPO模型推理失败: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'MaskablePPO inference failed: {str(e)}'}), 500

    elif model_type == 'custom':
        # 自定义模型提供的概率分布
        probs = np.array(data.get('probabilities', []))
        if len(probs) != 6:
            return jsonify({'error': 'Invalid probabilities length'}), 400

        # 应用有效动作掩码
        probs = probs * valid_actions
        if probs.sum() == 0:
            return jsonify({'error': 'No valid actions in probabilities'}), 400

        probs = probs / probs.sum()
        action = int(np.argmax(probs))

    else:
        return jsonify({'error': 'Unknown model type'}), 400

    # 解析动作
    # 动作空间：[move_1, move_2, move_3, collect_1, collect_2, collect_3]
    if action < 3:
        card_value = action + 1
        action_info = {
            'type': 'move',
            'card_value': int(card_value)
        }
    else:
        card_value = action - 3 + 1
        action_info = {
            'type': 'collect',
            'card_value': int(card_value),
            'is_combo': card_value in game.can_combo_values()
        }

    return jsonify({
        'action': int(action),
        'action_info': action_info,
        'probabilities': probs.tolist(),
        'observation': obs.tolist(),
        'valid_actions': valid_actions.tolist()
    })


def rule_based_policy(game: ResourceGame, valid_actions: np.ndarray) -> tuple:
    """
    简单的基于规则的策略
    优先级：
    1. 连击收集（如果可以）
    2. 普通收集（如果资源系数高）
    3. 移动到下一个有价值的位置

    动作空间：[move_1, move_2, move_3, collect_1, collect_2, collect_3]
    """
    probs = np.zeros(6)
    valid_indices = np.where(valid_actions > 0)[0]

    if len(valid_indices) == 0:
        return probs, 0

    # 检查是否有连击机会
    combo_values = game.can_combo_values()
    if len(combo_values) > 0:
        # 优先连击
        card_value = combo_values[0]
        action = 3 + card_value - 1
        probs[action] = 1.0
        return probs, action

    # 检查是否应该收集
    if game.collectable and game.resource_coef >= 5:
        # 资源系数高时优先收集
        collect_actions = [i for i in range(3, 6) if valid_actions[i] > 0]
        if len(collect_actions) > 0:
            # 选择点数最大的卡牌收集（索引越大点数越大）
            best_collect = max(collect_actions)
            probs[best_collect] = 1.0
            return probs, best_collect

    # 否则移动
    move_actions = [i for i in range(3) if valid_actions[i] > 0]
    if len(move_actions) > 0:
        # 选择点数最小的卡牌移动（保留高点数用于收集）
        action = min(move_actions)
        probs[action] = 1.0
        return probs, action

    # 默认：均匀选择有效动作
    probs[valid_indices] = 1.0 / len(valid_indices)
    return probs, valid_indices[0]


# ==================== WebSocket 支持 ====================

@socketio.on('connect')
def handle_connect():
    """客户端连接"""
    print('Client connected')
    emit('connected', {'message': 'Connected to game server'})


@socketio.on('disconnect')
def handle_disconnect():
    """客户端断开"""
    print('Client disconnected')


@socketio.on('join_game')
def handle_join_game(data):
    """加入游戏"""
    game_id = data.get('game_id')
    if not game_id:
        emit('error', {'message': 'No game_id provided'})
        return

    if game_id not in game_sessions:
        emit('error', {'message': 'Game not found'})
        return

    # 发送当前游戏状态
    game = game_sessions[game_id]
    emit('game_state', serialize_game_state(game))


@socketio.on('perform_action')
def handle_perform_action(data):
    """执行动作（通过WebSocket）"""
    game_id = data.get('game_id')
    action_type = data.get('type')
    card_value = data.get('card_value')

    if not game_id or game_id not in game_sessions:
        emit('error', {'message': 'Invalid game_id'})
        return

    game = game_sessions[game_id]

    result = {}
    if action_type == 'move':
        success, msg = game.move(card_value)
        result = {'success': success, 'message': msg, 'type': 'move'}
    elif action_type == 'collect':
        success, msg, tokens, _ = game.collect(card_value)
        result = {'success': success, 'message': msg, 'type': 'collect', 'tokens_earned': tokens}

    # 执行动作后检查是否需要开始新回合
    if game.is_round_over() and not game.is_game_over():
        game.start_round()

    # 广播新状态
    emit('action_result', {
        'result': result,
        'state': serialize_game_state(game)
    }, broadcast=False)


# ==================== 主程序 ====================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Resource Game Backend Server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()

    print(f"Starting server on {args.host}:{args.port}")
    print(f"Access the game at: http://localhost:{args.port}")

    socketio.run(app, host=args.host, port=args.port, debug=args.debug, allow_unsafe_werkzeug=True)
