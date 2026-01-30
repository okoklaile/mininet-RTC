# PPO 算法核心机制分析报告

您好！根据您的需求，我分析了 `BandwidthEstimator.py`、`ppo_agent.py` 和 `actor_critic.py` 的代码。以下是目前 PPO 算法的输入、输出、奖励函数和损失函数的详细说明。

## 1. PPO 的输入 (Input / State Space)
PPO 接收一个 **32 维特征向量**，包含网络状态和应用层 QoE 状态。
相比于 Base Model，PPO 的最大优势是能看到最后 3 个 QoE 特征。

| 索引 | 特征名称 | 描述 |
| :--- | :--- | :--- |
| 0 | `delay` | 当前平均网络延迟 |
| 1 | `loss_ratio` | 丢包率 |
| 2 | `receiving_rate` | 接收速率 |
| 3 | `prev_bandwidth` | 上一次的带宽预测值 |
| 4 | `delay_gradient` | 延迟变化率 |
| 5 | `throughput_effective` | 有效吞吐量 (排除丢包) |
| 6-11 | 延迟统计 | 均值、方差、最小值、排队延迟、二阶导数、趋势 |
| 12 | `loss_change` | 丢包率变化 |
| 13-15 | 带宽利用率 | 利用率、接收速率均值/方差 |
| 16-23 | (Reserved) | 预留位，目前为 0.0 |
| **24** | **`render_fps`** | **渲染帧率** (QoE 关键指标) |
| **25** | **`freeze_rate`** | **卡顿率** (QoE 关键指标) |
| **26** | **`e2e_delay_ms`** | **端到端延迟** (QoE 关键指标) |
| 27-31 | (Reserved) | 预留位，目前为 0.0 |

## 2. PPO 的输出 (Output / Action Space)
PPO 输出一个 **标量值 (Scalar)**，经过 Sigmoid 激活函数限制在 `[0, 1]` 区间。

*   **输出层激活函数**: `Sigmoid` (在 `ActorCritic.forward` 中 `action_mean = torch.sigmoid(...)`)
*   **动作映射逻辑**:
    $$ \text{调整系数} = 0.5 + 1.5 \times \text{Action} $$
    *   如果 Action = 0.0 $\rightarrow$ 系数 = 0.5 (带宽减半)
    *   如果 Action = 0.5 $\rightarrow$ 系数 = 1.25 (小幅增加)
    *   如果 Action = 1.0 $\rightarrow$ 系数 = 2.0 (带宽翻倍)
*   **最终应用**:
    $$ \text{最终带宽} = \text{BaseModel预测值} \times \text{调整系数} $$

## 3. 奖励函数 (Reward Function)
奖励函数是一个**加权求和**，包含 QoE 指标（权重约 50%）和网络指标（权重约 50%）。目标是最大化这个值。

| 组件 | 目标 | 权重 | 计算公式 (简化) |
| :--- | :--- | :--- | :--- |
| **QoE: FPS** | 目标 30 FPS | 0.20 | $\min(FPS/30, 1.0)$ |
| **QoE: Freeze** | 越低越好 | 0.15 | $\max(0, 1.0 - Rate/10\%)$ |
| **QoE: E2E Delay** | < 500ms | 0.15 | $\max(0, 1.0 - Delay/500)$ |
| **Net: Delay** | < 300ms | 0.15 | $\max(0, 1.0 - Delay/300)$ |
| **Net: Loss** | < 5% | 0.15 | $\max(0, 1.0 - Loss/5\%)$ |
| **Net: Throughput** | 目标 10Mbps | 0.20 | $\min(TP/10M, 1.0)$ |

## 4. 损失函数 (Loss Function)
PPO 使用标准的 **Actor-Critic 联合损失函数**，并在 `ppo_agent.py` 中实现。

$$ Loss = L_{policy} + L_{value} + L_{entropy} $$

1.  **Policy Loss (策略损失)**:
    使用 **Clipped Surrogate Objective**，防止策略更新幅度过大。
    $$ L_{policy} = -\min(\text{ratio} \cdot A, \text{clamp}(\text{ratio}, 1-\epsilon, 1+\epsilon) \cdot A) $$
    *   `ratio`: 新旧策略的概率比率
    *   `A`: 优势函数 (Advantage)，衡量当前动作比平均好多少
    *   `epsilon`: 裁剪阈值 (`ppo_clip=0.2`)

2.  **Value Loss (价值损失)**:
    Critic 网络预测值与真实回报之间的均方误差 (MSE)。
    $$ L_{value} = 0.5 \times (V_{pred} - V_{target})^2 $$

3.  **Entropy Regularization (熵正则化)**:
    鼓励探索，防止过早收敛到局部最优。
    $$ L_{entropy} = -0.01 \times \text{Entropy} $$
