// 游戏前端逻辑
class GameClient {
    constructor() {
        this.gameId = null;
        this.gameState = null;
        this.selectedCard = null;
        this.aiEnabled = false;
        this.aiPrediction = null;

        this.init();
    }

    async init() {
        // 初始化UI事件监听
        this.setupEventListeners();

        // 创建新游戏
        await this.newGame();
    }

    setupEventListeners() {
        // 新游戏按钮
        document.getElementById('new-game-btn').addEventListener('click', () => {
            this.newGame();
        });

        // 重新开始按钮
        document.getElementById('restart-btn').addEventListener('click', () => {
            this.newGame();
        });

        // AI 切换按钮
        document.getElementById('toggle-ai-btn').addEventListener('click', () => {
            this.toggleAI();
        });

        // AI 决策按钮
        document.getElementById('ai-action-btn').addEventListener('click', () => {
            this.getAIPrediction();
        });
    }

    async newGame() {
        try {
            const response = await fetch('/api/game/new', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ rounds: 10 })
            });

            const data = await response.json();
            this.gameId = data.game_id;
            this.gameState = data.state;

            this.addLog('新游戏开始！', 'info');
            this.render();

            // 隐藏游戏结束面板
            document.getElementById('game-over-panel').style.display = 'none';

            // 如果AI已启用，自动获取预测
            if (this.aiEnabled) {
                await this.getAIPrediction();
            }
        } catch (error) {
            console.error('创建游戏失败:', error);
            this.addLog('创建游戏失败', 'error');
        }
    }

    async performAction(type, cardIndex) {
        if (!this.gameId) return;

        try {
            const response = await fetch(`/api/game/${this.gameId}/action`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    type: type,
                    card_index: cardIndex
                })
            });

            const data = await response.json();

            if (data.result.success) {
                this.gameState = data.state;
                this.addLog(data.result.message, 'success');

                // 如果获得代币，显示特殊消息
                if (data.result.tokens_earned > 0) {
                    this.addLog(`🎉 获得 ${data.result.tokens_earned} 代币！`, 'success');
                }

                this.render();

                // 检查游戏是否结束
                if (this.gameState.is_game_over) {
                    this.showGameOver();
                } else if (this.aiEnabled) {
                    // AI模式下，自动获取下一步预测
                    setTimeout(() => this.getAIPrediction(), 500);
                }
            } else {
                this.addLog(data.result.message, 'error');
            }

            this.selectedCard = null;
        } catch (error) {
            console.error('执行动作失败:', error);
            this.addLog('执行动作失败', 'error');
        }
    }

    async getAIPrediction() {
        if (!this.gameId || this.gameState.is_game_over) return;

        try {
            const aiMode = document.getElementById('ai-mode-select').value;

            const response = await fetch(`/api/game/${this.gameId}/ai/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model_type: aiMode
                })
            });

            const data = await response.json();
            this.aiPrediction = data;

            this.renderAIPanel();
        } catch (error) {
            console.error('获取AI预测失败:', error);
        }
    }

    toggleAI() {
        this.aiEnabled = !this.aiEnabled;
        const btn = document.getElementById('toggle-ai-btn');
        const aiPanel = document.getElementById('ai-panel');
        const aiModeSelector = document.querySelector('.ai-mode-selector');

        if (this.aiEnabled) {
            btn.textContent = 'AI 已启用';
            btn.style.background = '#ed8936';
            aiPanel.style.display = 'block';
            aiModeSelector.style.display = 'flex';
            this.getAIPrediction();
        } else {
            btn.textContent = 'AI 辅助';
            btn.style.background = '#48bb78';
            aiPanel.style.display = 'none';
            aiModeSelector.style.display = 'none';
            this.aiPrediction = null;
        }
    }

    render() {
        if (!this.gameState) return;

        this.renderGameInfo();
        this.renderMap();
        this.renderHand();
        this.renderCustomers();
    }

    renderGameInfo() {
        const state = this.gameState;

        document.getElementById('round-info').textContent =
            `${state.current_round} / ${state.total_rounds}`;
        document.getElementById('tokens-info').textContent = state.tokens;
        document.getElementById('coef-info').textContent = state.resource_coef;
        document.getElementById('position-info').textContent =
            `[${state.position}] ${state.resource_type}`;
        document.getElementById('collectable-info').textContent =
            state.collectable ? '是' : '否';
    }

    renderMap() {
        const state = this.gameState;
        const mapContainer = document.getElementById('map');
        mapContainer.innerHTML = '';

        const resourceIcons = { '冰': '❄️', '铁': '⚙️', '火': '🔥' };
        const resourceClasses = { '冰': 'ice', '铁': 'iron', '火': 'fire' };

        state.map.forEach((resource, index) => {
            const tile = document.createElement('div');
            tile.className = `map-tile ${resourceClasses[resource]}`;
            if (index === state.position) {
                tile.classList.add('current');
            }
            tile.innerHTML = `
                <div>${resourceIcons[resource]} ${resource}</div>
                <div style="font-size: 10px; margin-top: 4px;">[${index}]</div>
            `;
            mapContainer.appendChild(tile);
        });
    }

    renderHand() {
        const state = this.gameState;
        const handContainer = document.getElementById('hand-cards');
        handContainer.innerHTML = '';

        const canComboIndices = state.can_combo_indices || [];

        state.hand.forEach((card, index) => {
            const cardElem = document.createElement('div');
            cardElem.className = 'card';

            if (canComboIndices.includes(index)) {
                cardElem.classList.add('can-combo');
            }

            cardElem.innerHTML = `
                <div>${card}</div>
                <div class="card-actions">
                    <button class="card-btn move" data-index="${index}" data-type="move">
                        移动
                    </button>
                    <button class="card-btn collect" data-index="${index}" data-type="collect">
                        ${canComboIndices.includes(index) ? '连击' : '收集'}
                    </button>
                </div>
            `;

            // 添加点击事件
            const moveBtn = cardElem.querySelector('.move');
            const collectBtn = cardElem.querySelector('.collect');

            moveBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.performAction('move', index);
            });

            collectBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.performAction('collect', index);
            });

            handContainer.appendChild(cardElem);
        });

        // 更新提示
        const hint = document.getElementById('hand-hint');
        if (state.hand.length === 0) {
            hint.textContent = '本回合已结束，点击按钮开始下一回合';
        } else if (canComboIndices.length > 0) {
            hint.textContent = `可以使用 ${canComboIndices.map(i => state.hand[i]).join(', ')} 点牌进行连击！`;
        } else if (state.collectable) {
            hint.textContent = '可以进行收集或继续移动';
        } else {
            hint.textContent = '选择一张牌进行移动';
        }
    }

    renderCustomers() {
        const state = this.gameState;
        const container = document.getElementById('customers-container');
        container.innerHTML = '';

        state.customers.forEach((customer, index) => {
            const customerElem = document.createElement('div');
            customerElem.className = 'customer';
            if (customer.is_vip) {
                customerElem.classList.add('vip');
            }

            const header = document.createElement('div');
            header.className = 'customer-header';
            header.textContent = `顾客 ${index + 1} ${customer.is_vip ? '(VIP)' : ''} - ${customer.reward} 代币`;

            const needsContainer = document.createElement('div');
            needsContainer.className = 'customer-needs';

            const resourceIcons = { '冰': '❄️', '铁': '⚙️', '火': '🔥' };

            Object.entries(customer.needs).forEach(([resource, need]) => {
                const have = customer.have[resource] || 0;
                const progress = (have / need) * 100;

                const needItem = document.createElement('div');
                needItem.className = 'need-item';

                needItem.innerHTML = `
                    <div class="need-label">${resourceIcons[resource]} ${resource}</div>
                    <div class="need-progress">
                        <div class="need-progress-bar ${progress >= 100 ? 'complete' : ''}"
                             style="width: ${Math.min(progress, 100)}%">
                        </div>
                    </div>
                    <div style="font-size: 11px; color: #666; min-width: 50px;">
                        ${have}/${need}
                    </div>
                `;

                needsContainer.appendChild(needItem);
            });

            customerElem.appendChild(header);
            customerElem.appendChild(needsContainer);
            container.appendChild(customerElem);
        });
    }

    renderAIPanel() {
        if (!this.aiPrediction) return;

        const pred = this.aiPrediction;

        // 渲染建议动作
        const suggestionElem = document.getElementById('ai-suggestion');
        const actionInfo = pred.action_info;
        let suggestionText = '';

        if (actionInfo.type === 'move') {
            suggestionText = `建议移动：使用序号 ${actionInfo.card_index} 的牌（${actionInfo.card_value} 点）`;
        } else {
            suggestionText = `建议收集：使用序号 ${actionInfo.card_index} 的牌（${actionInfo.card_value} 点）`;
            if (actionInfo.is_combo) {
                suggestionText += ' [连击]';
            }
        }

        suggestionElem.textContent = suggestionText;

        // 渲染动作概率分布
        const probsContainer = document.getElementById('action-probs');
        probsContainer.innerHTML = '';

        const actionNames = [
            'M0', 'M1', 'M2', 'M3', 'M4',
            'C0', 'C1', 'C2', 'C3', 'C4'
        ];

        pred.probabilities.forEach((prob, index) => {
            const probBar = document.createElement('div');
            probBar.className = 'prob-bar';

            const percentage = (prob * 100).toFixed(1);

            probBar.innerHTML = `
                <div class="prob-label">${actionNames[index]}</div>
                <div class="prob-fill">
                    <div class="prob-fill-inner" style="width: ${percentage}%"></div>
                </div>
                <div class="prob-value">${percentage}%</div>
            `;

            probsContainer.appendChild(probBar);
        });

        // 渲染观测向量
        const obsPreview = document.getElementById('obs-preview');
        const obsArray = pred.observation;
        const obsText = `[${obsArray.slice(0, 10).map(v => v.toFixed(2)).join(', ')}...] (共${obsArray.length}维)`;
        obsPreview.textContent = obsText;

        // 渲染有效动作
        const validActionsContainer = document.getElementById('valid-actions-info');
        validActionsContainer.innerHTML = '';

        pred.valid_actions.forEach((valid, index) => {
            const indicator = document.createElement('div');
            indicator.className = `valid-action-indicator ${valid ? 'valid' : 'invalid'}`;
            indicator.textContent = actionNames[index];
            validActionsContainer.appendChild(indicator);
        });
    }

    showGameOver() {
        const panel = document.getElementById('game-over-panel');
        const finalTokens = document.getElementById('final-tokens');

        finalTokens.textContent = this.gameState.tokens;
        panel.style.display = 'block';

        this.addLog(`游戏结束！最终获得 ${this.gameState.tokens} 代币`, 'info');
    }

    addLog(message, type = 'info') {
        const logContainer = document.getElementById('action-log');
        const entry = document.createElement('div');
        entry.className = `log-entry ${type}`;
        entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;

        logContainer.insertBefore(entry, logContainer.firstChild);

        // 限制日志条目数量
        while (logContainer.children.length > 50) {
            logContainer.removeChild(logContainer.lastChild);
        }
    }
}

// 初始化游戏
let gameClient;
document.addEventListener('DOMContentLoaded', () => {
    gameClient = new GameClient();
});
