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

app = Flask(__name__,
            template_folder='../frontend/templates',
            static_folder='../frontend/static')
app.config['SECRET_KEY'] = 'your-secret-key-here'  # 在生产环境中应使用环境变量
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# 存储游戏会话
game_sessions: Dict[str, ResourceGame] = {}


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
    card_index = data.get('card_index')

    if action_type not in ['move', 'collect']:
        return jsonify({'error': 'Invalid action type'}), 400

    if card_index is None or card_index < 0 or card_index >= len(game.hand):
        return jsonify({'error': 'Invalid card index'}), 400

    result = {'success': False}

    if action_type == 'move':
        success, msg = game.move(card_index)
        result = {
            'success': success,
            'message': msg,
            'type': 'move'
        }
    elif action_type == 'collect':
        success, msg, tokens, _ = game.collect(card_index)
        result = {
            'success': success,
            'message': msg,
            'type': 'collect',
            'tokens_earned': tokens if success else 0
        }

    # 执行动作后检查是否需要开始新回合
    if game.is_round_over() and not game.is_game_over():
        game.start_round()

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


# ==================== AI 相关接口 ====================

@app.route('/api/game/<game_id>/ai/predict', methods=['POST'])
def ai_predict(game_id: str):
    """
    AI预测接口
    接收模型预测并返回动作建议

    请求格式：
    {
        "model_type": "random" | "rule_based" | "custom",
        "probabilities": [...] (可选，用于自定义模型)
    }
    """
    if game_id not in game_sessions:
        return jsonify({'error': 'Game not found'}), 404

    game = game_sessions[game_id]
    data = request.json or {}
    model_type = data.get('model_type', 'random')

    obs = game.get_observation()
    valid_actions = game.get_valid_actions()

    if model_type == 'random':
        # 随机策略（只选择有效动作）
        valid_indices = np.where(valid_actions > 0)[0]
        if len(valid_indices) == 0:
            return jsonify({'error': 'No valid actions'}), 400

        action = np.random.choice(valid_indices)
        probs = np.zeros(10)
        probs[valid_indices] = 1.0 / len(valid_indices)

    elif model_type == 'rule_based':
        # 简单的基于规则的策略
        probs, action = rule_based_policy(game, valid_actions)

    elif model_type == 'custom':
        # 自定义模型提供的概率分布
        probs = np.array(data.get('probabilities', []))
        if len(probs) != 10:
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
    if action < 5:
        action_info = {
            'type': 'move',
            'card_index': int(action),
            'card_value': game.hand[action] if action < len(game.hand) else None
        }
    else:
        card_idx = action - 5
        action_info = {
            'type': 'collect',
            'card_index': int(card_idx),
            'card_value': game.hand[card_idx] if card_idx < len(game.hand) else None,
            'is_combo': card_idx in game.can_combo_indices()
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
    """
    probs = np.zeros(10)
    valid_indices = np.where(valid_actions > 0)[0]

    if len(valid_indices) == 0:
        return probs, 0

    # 检查是否有连击机会
    combo_indices = game.can_combo_indices()
    if len(combo_indices) > 0:
        # 优先连击
        action = 5 + combo_indices[0]
        probs[action] = 1.0
        return probs, action

    # 检查是否应该收集
    if game.collectable and game.resource_coef >= 5:
        # 资源系数高时优先收集
        collect_actions = [i for i in range(5, 10) if valid_actions[i] > 0]
        if len(collect_actions) > 0:
            # 选择点数最大的卡牌收集
            best_collect = max(collect_actions, key=lambda i: game.hand[i-5] if i-5 < len(game.hand) else 0)
            probs[best_collect] = 1.0
            return probs, best_collect

    # 否则移动
    move_actions = [i for i in range(5) if valid_actions[i] > 0]
    if len(move_actions) > 0:
        # 选择点数适中的卡牌移动（保留高点数用于收集）
        action = min(move_actions, key=lambda i: game.hand[i] if i < len(game.hand) else 0)
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
    card_index = data.get('card_index')

    if not game_id or game_id not in game_sessions:
        emit('error', {'message': 'Invalid game_id'})
        return

    game = game_sessions[game_id]

    result = {}
    if action_type == 'move':
        success, msg = game.move(card_index)
        result = {'success': success, 'message': msg, 'type': 'move'}
    elif action_type == 'collect':
        success, msg, tokens, _ = game.collect(card_index)
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
