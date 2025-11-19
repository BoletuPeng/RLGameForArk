// 游戏前端逻辑
class GameClient {
    constructor() {
        this.gameId = null;
        this.gameState = null;
        this.selectedCard = null;
        this.aiEnabled = false;
        this.aiPrediction = null;
        this.previousRound = 0;  // 记录上一个回合号，用于检测回合切换
        this.selectedModelPath = null;  // 选中的自定义模型路径
        this.modelsList = [];  // 可用的模型列表
        this.endgameMode = false;  // 残局模式标志

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

        // 保存对局记录按钮
        document.getElementById('save-replay-btn').addEventListener('click', () => {
            this.saveReplay();
        });

        // 残局模式按钮
        document.getElementById('toggle-endgame-btn').addEventListener('click', () => {
            this.toggleEndgameMode();
        });

        // AI 切换按钮
        document.getElementById('toggle-ai-btn').addEventListener('click', () => {
            this.toggleAI();
        });

        // AI 决策按钮
        document.getElementById('ai-action-btn').addEventListener('click', () => {
            this.getAIPrediction();
        });

        // AI 模式选择器变化
        document.getElementById('ai-mode-select').addEventListener('change', (e) => {
            this.onAIModeChange(e.target.value);
        });

        // 浏览模型按钮
        document.getElementById('browse-model-btn').addEventListener('click', () => {
            this.openModelBrowser();
        });

        // 模型浏览器关闭按钮
        document.getElementById('close-modal-btn').addEventListener('click', () => {
            this.closeModelBrowser();
        });

        // 取消按钮
        document.getElementById('cancel-model-btn').addEventListener('click', () => {
            this.closeModelBrowser();
        });

        // 确认选择按钮
        document.getElementById('confirm-model-btn').addEventListener('click', () => {
            this.confirmModelSelection();
        });

        // 点击模态框外部关闭
        document.getElementById('model-browser-modal').addEventListener('click', (e) => {
            if (e.target.id === 'model-browser-modal') {
                this.closeModelBrowser();
            }
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
            this.previousRound = data.state.current_round;  // 初始化回合号

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

    async performAction(type, cardValue) {
        if (!this.gameId) return;

        try {
            const response = await fetch(`/api/game/${this.gameId}/action`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    type: type,
                    card_value: cardValue
                })
            });

            const data = await response.json();

            if (data.result.success) {
                // 检测回合是否切换
                const oldRound = this.previousRound;
                const newRound = data.state.current_round;

                this.gameState = data.state;
                this.addLog(data.result.message, 'success');

                // 如果回合切换了，显示提示
                if (oldRound > 0 && newRound > oldRound) {
                    this.addLog(`📢 回合 ${oldRound} 结束，回合 ${newRound} 开始！`, 'info');
                }

                // 更新回合号
                this.previousRound = newRound;

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

            const requestBody = {
                model_type: aiMode
            };

            // 如果是自定义模型，添加模型路径
            if (aiMode === 'ppo_custom' && this.selectedModelPath) {
                requestBody.model_path = this.selectedModelPath;
            }

            const response = await fetch(`/api/game/${this.gameId}/ai/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestBody)
            });

            const data = await response.json();

            if (!response.ok) {
                this.addLog(`AI预测失败: ${data.error}`, 'error');
                return;
            }

            this.aiPrediction = data;

            this.renderAIPanel();
        } catch (error) {
            console.error('获取AI预测失败:', error);
            this.addLog('获取AI预测失败', 'error');
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

        // 残局模式下添加点击事件
        if (this.endgameMode) {
            const infoCard = document.querySelector('.info-card');
            const h2 = infoCard.querySelector('h2');

            // 移除之前的事件监听器（如果有）
            const newH2 = h2.cloneNode(true);
            h2.parentNode.replaceChild(newH2, h2);

            newH2.addEventListener('click', () => {
                this.editGameState();
            });
        }
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

            // 残局模式下添加点击事件
            if (this.endgameMode) {
                tile.style.cursor = 'pointer';
                tile.addEventListener('click', () => {
                    this.moveToPosition(index);
                });
            }

            mapContainer.appendChild(tile);
        });
    }

    renderHand() {
        const state = this.gameState;
        const handContainer = document.getElementById('hand-cards');
        handContainer.innerHTML = '';

        const canComboValues = state.can_combo_values || [];

        // hand 现在是一个字典：{1: count1, 2: count2, 3: count3}
        // 按点数从小到大显示
        for (let cardValue = 1; cardValue <= 3; cardValue++) {
            const count = state.hand[cardValue] || 0;

            if (count === 0) continue; // 跳过没有的卡牌

            const cardElem = document.createElement('div');
            cardElem.className = 'card';

            if (canComboValues.includes(cardValue)) {
                cardElem.classList.add('can-combo');
            }

            cardElem.innerHTML = `
                <div>${cardValue} 点</div>
                <div style="font-size: 14px; color: #666;">×${count}</div>
                <div class="card-actions">
                    <button class="card-btn move" data-value="${cardValue}" data-type="move">
                        移动
                    </button>
                    <button class="card-btn collect" data-value="${cardValue}" data-type="collect">
                        ${canComboValues.includes(cardValue) ? '连击' : '收集'}
                    </button>
                </div>
            `;

            // 添加点击事件
            const moveBtn = cardElem.querySelector('.move');
            const collectBtn = cardElem.querySelector('.collect');

            moveBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.performAction('move', cardValue);
            });

            collectBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.performAction('collect', cardValue);
            });

            handContainer.appendChild(cardElem);
        }

        // 更新提示
        const hint = document.getElementById('hand-hint');
        if (canComboValues.length > 0) {
            hint.textContent = `可以使用 ${canComboValues.join(', ')} 点牌进行连击！`;
        } else if (state.collectable) {
            hint.textContent = '可以进行收集或继续移动';
        } else {
            hint.textContent = '选择一张牌进行移动';
        }

        // 残局模式下添加点击事件
        if (this.endgameMode) {
            const handContainer = document.querySelector('.hand-container');
            const h2 = handContainer.querySelector('h2');

            // 移除之前的事件监听器（如果有）
            const newH2 = h2.cloneNode(true);
            h2.parentNode.replaceChild(newH2, h2);

            newH2.addEventListener('click', () => {
                this.editHand();
            });
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

            // 残局模式下添加点击事件
            if (this.endgameMode) {
                customerElem.style.cursor = 'pointer';
                customerElem.addEventListener('click', (e) => {
                    // 如果点击的是编辑表单内的元素，不触发编辑
                    if (e.target.closest('.edit-form')) {
                        return;
                    }
                    this.editCustomer(index);
                });
            }

            container.appendChild(customerElem);
        });
    }

    renderAIPanel() {
        if (!this.aiPrediction) return;

        const pred = this.aiPrediction;

        // 渲染建议动作
        const suggestionElem = document.getElementById('ai-suggestion');
        const actionInfo = pred.action_info;
        const probabilities = pred.probabilities;
        const selectedAction = pred.action;

        // 获取选中动作的概率（置信度）
        const confidence = probabilities[selectedAction];
        const confidencePercent = (confidence * 100).toFixed(1);

        // 找出概率最高的前3个动作进行对比
        // 动作空间：[move_1, move_2, move_3, collect_1, collect_2, collect_3]
        const actionNames = [
            'M1', 'M2', 'M3',
            'C1', 'C2', 'C3'
        ];
        const sortedActions = probabilities
            .map((prob, idx) => ({ action: idx, prob: prob, name: actionNames[idx] }))
            .filter(item => pred.valid_actions[item.action] > 0)  // 只考虑有效动作
            .sort((a, b) => b.prob - a.prob);

        // 构建基于AI预测的建议文本
        let suggestionText = '';

        if (actionInfo.type === 'move') {
            suggestionText = `AI建议移动：使用 ${actionInfo.card_value} 点牌`;
        } else {
            suggestionText = `AI建议收集：使用 ${actionInfo.card_value} 点牌`;
            if (actionInfo.is_combo) {
                suggestionText += ' [连击]';
            }
        }

        // 添加置信度信息
        suggestionText += ` | 置信度: ${confidencePercent}%`;

        // 如果有次优选择，显示对比信息
        if (sortedActions.length > 1) {
            const secondBest = sortedActions[1];
            const secondProb = (secondBest.prob * 100).toFixed(1);
            suggestionText += ` | 次选: ${secondBest.name} (${secondProb}%)`;
        }

        suggestionElem.textContent = suggestionText;

        // 渲染动作概率分布
        const probsContainer = document.getElementById('action-probs');
        probsContainer.innerHTML = '';

        // 只渲染前6个动作，避免undefined
        actionNames.forEach((name, index) => {
            const prob = pred.probabilities[index];
            const isValid = pred.valid_actions[index] > 0;

            const probBar = document.createElement('div');
            probBar.className = `prob-bar ${isValid ? 'valid' : 'invalid'}`;

            const percentage = (prob * 100).toFixed(1);

            probBar.innerHTML = `
                <div class="prob-label">${name}</div>
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
    }

    showGameOver() {
        const panel = document.getElementById('game-over-panel');
        const finalTokens = document.getElementById('final-tokens');
        const saveReplayBtn = document.getElementById('save-replay-btn');

        finalTokens.textContent = this.gameState.tokens;
        panel.style.display = 'block';

        // 残局模式下隐藏保存按钮
        if (this.endgameMode) {
            saveReplayBtn.style.display = 'none';
        } else {
            saveReplayBtn.style.display = 'block';
        }

        this.addLog(`游戏结束！最终获得 ${this.gameState.tokens} 代币`, 'info');
    }

    async saveReplay() {
        if (!this.gameId) {
            this.addLog('没有可保存的游戏记录', 'error');
            return;
        }

        // 可选：提示用户输入记录名称
        const replayName = prompt('请输入对局记录名称（可选）：', '');

        try {
            const response = await fetch(`/api/game/${this.gameId}/save_replay`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    name: replayName || undefined
                })
            });

            const data = await response.json();

            if (response.ok) {
                this.addLog(`对局记录已保存：${data.filename}`, 'success');
                this.addLog(`动作数量：${data.actions_count}，最终代币：${data.final_tokens}`, 'info');
                alert(`对局记录已成功保存！\n文件名：${data.filename}\n动作数量：${data.actions_count}`);
            } else {
                this.addLog(`保存失败：${data.error}`, 'error');
                alert(`保存失败：${data.error}`);
            }
        } catch (error) {
            this.addLog(`保存对局记录时出错：${error.message}`, 'error');
            alert(`保存失败：${error.message}`);
        }
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

    // 模型浏览器相关方法

    onAIModeChange(mode) {
        const browseBtn = document.getElementById('browse-model-btn');

        if (mode === 'ppo_custom') {
            browseBtn.style.display = 'block';
            // 如果没有选择模型，提示用户
            if (!this.selectedModelPath) {
                this.addLog('请点击"浏览模型"选择一个模型文件', 'info');
            }
        } else {
            browseBtn.style.display = 'none';
        }
    }

    async openModelBrowser() {
        const modal = document.getElementById('model-browser-modal');
        const listContainer = document.getElementById('model-list-container');

        modal.style.display = 'flex';
        listContainer.innerHTML = '<div class="loading">正在加载模型列表...</div>';

        try {
            const response = await fetch('/api/models/list');
            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || '加载模型列表失败');
            }

            this.modelsList = data.models || [];
            this.renderModelList();
        } catch (error) {
            console.error('加载模型列表失败:', error);
            listContainer.innerHTML = `
                <div class="model-error">
                    加载模型列表失败: ${error.message}
                </div>
            `;
        }
    }

    renderModelList() {
        const listContainer = document.getElementById('model-list-container');

        if (this.modelsList.length === 0) {
            listContainer.innerHTML = `
                <div class="no-models">
                    <div class="no-models-icon">📁</div>
                    <div>没有找到可用的模型文件</div>
                    <div style="font-size: 12px; margin-top: 10px; color: #999;">
                        请将.zip格式的模型文件放在 models/ 目录下
                    </div>
                </div>
            `;
            return;
        }

        listContainer.innerHTML = '';

        this.modelsList.forEach((model, index) => {
            const modelItem = document.createElement('div');
            modelItem.className = 'model-item';
            modelItem.dataset.index = index;

            // 检查是否是当前选中的模型
            if (this.selectedModelPath === model.relative_path) {
                modelItem.classList.add('selected');
            }

            // 格式化文件大小
            const sizeKB = (model.size / 1024).toFixed(2);
            const sizeMB = (model.size / 1024 / 1024).toFixed(2);
            const sizeStr = model.size > 1024 * 1024 ? `${sizeMB} MB` : `${sizeKB} KB`;

            // 格式化修改时间
            const modifiedDate = new Date(model.modified);
            const modifiedStr = modifiedDate.toLocaleString('zh-CN');

            modelItem.innerHTML = `
                <div class="model-item-header">
                    <div class="model-item-name">${model.name}</div>
                    <div class="model-item-size">${sizeStr}</div>
                </div>
                <div class="model-item-details">
                    <div class="model-item-path">${model.relative_path}</div>
                </div>
                <div class="model-item-modified">修改时间: ${modifiedStr}</div>
            `;

            modelItem.addEventListener('click', () => {
                this.selectModel(index);
            });

            listContainer.appendChild(modelItem);
        });
    }

    selectModel(index) {
        // 移除之前的选中状态
        document.querySelectorAll('.model-item').forEach(item => {
            item.classList.remove('selected');
        });

        // 添加新的选中状态
        const selectedItem = document.querySelector(`.model-item[data-index="${index}"]`);
        if (selectedItem) {
            selectedItem.classList.add('selected');
        }

        // 更新选中的模型信息
        const model = this.modelsList[index];
        const selectedInfo = document.getElementById('selected-model-info');
        const selectedName = document.getElementById('selected-model-name');
        const confirmBtn = document.getElementById('confirm-model-btn');

        selectedName.textContent = model.relative_path;
        selectedInfo.style.display = 'block';
        confirmBtn.disabled = false;

        // 临时存储选中的模型索引
        this.tempSelectedModelIndex = index;
    }

    confirmModelSelection() {
        if (this.tempSelectedModelIndex !== undefined) {
            const model = this.modelsList[this.tempSelectedModelIndex];
            this.selectedModelPath = model.relative_path;

            this.addLog(`已选择模型: ${model.name}`, 'info');
            this.closeModelBrowser();

            // 自动获取AI预测
            if (this.aiEnabled && !this.gameState.is_game_over) {
                setTimeout(() => this.getAIPrediction(), 300);
            }
        }
    }

    closeModelBrowser() {
        const modal = document.getElementById('model-browser-modal');
        const selectedInfo = document.getElementById('selected-model-info');
        const confirmBtn = document.getElementById('confirm-model-btn');

        modal.style.display = 'none';
        selectedInfo.style.display = 'none';
        confirmBtn.disabled = true;
        this.tempSelectedModelIndex = undefined;
    }

    // ===== 残局模式相关方法 =====

    toggleEndgameMode() {
        this.endgameMode = !this.endgameMode;
        const btn = document.getElementById('toggle-endgame-btn');
        const gameArea = document.querySelector('.game-area');

        if (this.endgameMode) {
            btn.textContent = '退出残局模式';
            btn.classList.add('active');
            gameArea.classList.add('endgame-mode');
            this.addLog('已进入残局模式 - 可以编辑游戏状态', 'info');
        } else {
            btn.textContent = '残局模式';
            btn.classList.remove('active');
            gameArea.classList.remove('endgame-mode');
            this.addLog('已退出残局模式', 'info');
        }

        // 重新渲染以应用残局模式的交互
        this.render();
    }

    // 编辑游戏状态
    editGameState() {
        const infoCard = document.querySelector('.info-card');

        // 检查是否已经有编辑表单
        const existingForm = infoCard.querySelector('.edit-form');
        if (existingForm) {
            existingForm.remove();
            return;
        }

        const form = document.createElement('div');
        form.className = 'edit-form';
        form.innerHTML = `
            <div class="edit-form-row">
                <label class="edit-form-label">当前回合：</label>
                <input type="number" class="edit-form-input" id="edit-round"
                       value="${this.gameState.current_round}" min="1" max="${this.gameState.total_rounds}">
            </div>
            <div class="edit-form-row">
                <label class="edit-form-label">代币：</label>
                <input type="number" class="edit-form-input" id="edit-tokens"
                       value="${this.gameState.tokens}" min="0">
            </div>
            <div class="edit-form-row">
                <label class="edit-form-label">资源系数：</label>
                <input type="number" class="edit-form-input" id="edit-coef"
                       value="${this.gameState.resource_coef}" min="1">
            </div>
            <div class="edit-form-buttons">
                <button class="btn btn-primary" id="save-state-btn">保存</button>
                <button class="btn btn-secondary" id="cancel-state-btn">取消</button>
            </div>
        `;

        infoCard.appendChild(form);

        document.getElementById('save-state-btn').addEventListener('click', () => {
            const newRound = parseInt(document.getElementById('edit-round').value);
            const newTokens = parseInt(document.getElementById('edit-tokens').value);
            const newCoef = parseInt(document.getElementById('edit-coef').value);

            this.gameState.current_round = newRound;
            this.gameState.tokens = newTokens;
            this.gameState.resource_coef = newCoef;

            this.addLog(`已更新游戏状态: 回合=${newRound}, 代币=${newTokens}, 系数=${newCoef}`, 'success');
            form.remove();
            this.render();

            // 如果AI已启用,重新获取预测
            if (this.aiEnabled) {
                this.getAIPrediction();
            }
        });

        document.getElementById('cancel-state-btn').addEventListener('click', () => {
            form.remove();
        });
    }

    // 编辑手牌
    editHand() {
        const handContainer = document.querySelector('.hand-container');

        // 检查是否已经有编辑表单
        const existingForm = handContainer.querySelector('.edit-form');
        if (existingForm) {
            existingForm.remove();
            return;
        }

        const form = document.createElement('div');
        form.className = 'edit-form';
        form.innerHTML = `
            <div class="edit-form-row">
                <label class="edit-form-label">1点牌数量：</label>
                <input type="number" class="edit-form-input" id="edit-card-1"
                       value="${this.gameState.hand[1] || 0}" min="0">
            </div>
            <div class="edit-form-row">
                <label class="edit-form-label">2点牌数量：</label>
                <input type="number" class="edit-form-input" id="edit-card-2"
                       value="${this.gameState.hand[2] || 0}" min="0">
            </div>
            <div class="edit-form-row">
                <label class="edit-form-label">3点牌数量：</label>
                <input type="number" class="edit-form-input" id="edit-card-3"
                       value="${this.gameState.hand[3] || 0}" min="0">
            </div>
            <div class="edit-form-buttons">
                <button class="btn btn-primary" id="save-hand-btn">保存</button>
                <button class="btn btn-secondary" id="cancel-hand-btn">取消</button>
            </div>
        `;

        handContainer.appendChild(form);

        document.getElementById('save-hand-btn').addEventListener('click', () => {
            const count1 = parseInt(document.getElementById('edit-card-1').value);
            const count2 = parseInt(document.getElementById('edit-card-2').value);
            const count3 = parseInt(document.getElementById('edit-card-3').value);

            this.gameState.hand = { 1: count1, 2: count2, 3: count3 };

            this.addLog(`已更新手牌: 1点×${count1}, 2点×${count2}, 3点×${count3}`, 'success');
            form.remove();
            this.render();

            // 如果AI已启用,重新获取预测
            if (this.aiEnabled) {
                this.getAIPrediction();
            }
        });

        document.getElementById('cancel-hand-btn').addEventListener('click', () => {
            form.remove();
        });
    }

    // 编辑顾客订单
    editCustomer(customerIndex) {
        const customer = this.gameState.customers[customerIndex];
        const customerElem = document.querySelectorAll('.customer')[customerIndex];

        // 检查是否已经有编辑表单
        const existingForm = customerElem.querySelector('.edit-form');
        if (existingForm) {
            existingForm.remove();
            return;
        }

        const resources = ['冰', '铁', '火'];
        let formHTML = '<div class="edit-form">';

        resources.forEach(resource => {
            const need = customer.needs[resource] || 0;
            const have = customer.have[resource] || 0;

            formHTML += `
                <div class="edit-form-grid" style="margin-bottom: 15px;">
                    <div class="edit-form-grid-item">
                        <label class="edit-form-grid-label">${resource} 已满足量：</label>
                        <input type="number" class="edit-form-input"
                               id="edit-have-${resource}-${customerIndex}"
                               value="${have}" min="0">
                    </div>
                    <div class="edit-form-grid-item">
                        <label class="edit-form-grid-label">${resource} 总需求量：</label>
                        <input type="number" class="edit-form-input"
                               id="edit-need-${resource}-${customerIndex}"
                               value="${need}" min="0">
                    </div>
                </div>
            `;
        });

        formHTML += `
            <div class="edit-form-buttons">
                <button class="btn btn-primary" id="save-customer-${customerIndex}-btn">保存</button>
                <button class="btn btn-secondary" id="cancel-customer-${customerIndex}-btn">取消</button>
            </div>
        </div>`;

        customerElem.insertAdjacentHTML('beforeend', formHTML);

        document.getElementById(`save-customer-${customerIndex}-btn`).addEventListener('click', () => {
            resources.forEach(resource => {
                const have = parseInt(document.getElementById(`edit-have-${resource}-${customerIndex}`).value);
                const need = parseInt(document.getElementById(`edit-need-${resource}-${customerIndex}`).value);

                this.gameState.customers[customerIndex].have[resource] = have;
                this.gameState.customers[customerIndex].needs[resource] = need;
            });

            this.addLog(`已更新顾客 ${customerIndex + 1} 的订单信息`, 'success');
            customerElem.querySelector('.edit-form').remove();
            this.render();

            // 如果AI已启用,重新获取预测
            if (this.aiEnabled) {
                this.getAIPrediction();
            }
        });

        document.getElementById(`cancel-customer-${customerIndex}-btn`).addEventListener('click', () => {
            customerElem.querySelector('.edit-form').remove();
        });
    }

    // 点击地图移动到指定位置
    moveToPosition(targetPosition) {
        if (!this.endgameMode) return;

        const currentPos = this.gameState.position;
        const steps = (targetPosition - currentPos + 10) % 10;

        if (steps === 0) {
            this.addLog('已经在该位置', 'info');
            return;
        }

        // 更新位置
        this.gameState.position = targetPosition;
        this.gameState.resource_type = this.gameState.map[targetPosition];
        this.gameState.collectable = true;

        this.addLog(`已移动到位置 [${targetPosition}] ${this.gameState.resource_type}`, 'success');
        this.render();

        // 如果AI已启用,重新获取预测
        if (this.aiEnabled) {
            this.getAIPrediction();
        }
    }
}

// 初始化游戏
let gameClient;
document.addEventListener('DOMContentLoaded', () => {
    gameClient = new GameClient();
});
