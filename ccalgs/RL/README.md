# 在线强化学习带宽估计器 (Online RL Bandwidth Estimator with PPO)

基于 BC-GCC trial3 模型的在线强化学习带宽估计算法，使用 **PPO (Proximal Policy Optimization)** 进行在线学习，并通过 QoE 特征（render_fps, freeze_rate, e2e_delay）计算奖励。

## 核心特性

- **基础模型**: 使用 BC-GCC trial3.pt 作为预训练基础模型
- **PPO 在线学习**: 使用 PPO 算法在线调整基础模型的带宽预测
- **QoE 特征集成**: 在模型输入的保留字段中填充 QoE 特征：
  - `render_fps`: 渲染帧率 (FPS)
  - `freeze_rate`: 卡顿率 (%)
  - `e2e_delay_ms`: 端到端延迟 (ms)
- **奖励函数**: 基于 QoE 指标计算奖励，优化用户体验

## 架构设计

```
BC-GCC 基础模型 (trial3.pt)
    ↓
32维特征提取（包含 QoE 特征）
    ↓
PPO Agent (Actor-Critic)
    ↓
动作：带宽调整系数 [0.5, 2.0]
    ↓
最终带宽预测 = 基础预测 × 调整系数
```

## 文件结构

```
RL/
├── BandwidthEstimator.py    # 主估计器类（集成 PPO）
├── config.py                # 配置文件（32维特征）
├── model.py                 # 模型定义（继承自 BC-GCC）
├── packet_info.py           # 数据包信息类
├── packet_record.py         # 数据包记录类
├── deep_rl/
│   ├── __init__.py
│   ├── actor_critic.py      # Actor-Critic 网络（基于 LSTM）
│   └── ppo_agent.py         # PPO Agent 实现
├── receiver_pyinfer.json    # 接收端配置
├── sender_pyinfer.json      # 发送端配置
├── trial3.pt                # 预训练基础模型权重
└── README.md                # 本文件
```

## PPO 算法参数

- **状态维度**: 32（包括 QoE 特征）
- **动作维度**: 1（带宽调整系数）
- **学习率**: 3e-4
- **折扣因子 (γ)**: 0.99
- **PPO 裁剪参数**: 0.2
- **PPO 更新迭代次数**: 10
- **更新频率**: 每 4 步更新一次（可配置）

## 奖励函数设计

奖励函数基于 QoE 指标：

```python
reward = fps_reward + freeze_penalty + delay_reward

其中：
- fps_reward = min(render_fps / 30.0, 1.0) * 0.4      # 帧率奖励（目标 30 FPS）
- freeze_penalty = max(0, 1.0 - freeze_rate / 10.0) * 0.3  # 卡顿惩罚（10% 时惩罚为0）
- delay_reward = max(0, 1.0 - e2e_delay_ms / 500.0) * 0.3  # 延迟奖励（500ms 时奖励为0）
```

## 模型输入格式

模型输入为 32 维特征向量：

- **索引 0-15**: 核心网络特征（延迟、丢包率、接收速率等）
- **索引 16-23**: 原有保留字段（全为0）
- **索引 24**: render_fps (渲染帧率)
- **索引 25**: freeze_rate (卡顿率)
- **索引 26**: e2e_delay_ms (端到端延迟)
- **索引 27-31**: 其他保留字段（全为0）

## QoE 数据接收

QoE 数据通过 `report_states()` 方法接收，格式如下：

```python
qoe_stats = {
    "type": "qoe",
    "render_fps": 30.0,      # 渲染帧率
    "freeze_rate": 2.5,      # 卡顿率 (%)
    "e2e_delay_ms": 150.0    # 端到端延迟 (ms)
}
estimator.report_states(qoe_stats)
```

## 使用方法

1. **初始化估计器**:
```python
from BandwidthEstimator import Estimator

estimator = Estimator(
    model_path="/path/to/trial3.pt",
    step_time=200,           # 时间步长(ms)
    use_rl=True,             # 启用在线强化学习
    update_frequency=4       # 每4步更新一次PPO策略
)
```

2. **接收数据包信息**:
```python
packet_stats = {
    "send_time_ms": 1000,
    "arrival_time_ms": 1150,
    "payload_type": 98,
    "sequence_number": 1,
    "ssrc": 12345,
    "padding_length": 0,
    "header_length": 12,
    "payload_size": 1200
}
estimator.report_states(packet_stats)
```

3. **接收 QoE 数据**:
```python
qoe_stats = {
    "type": "qoe",
    "render_fps": 30.0,
    "freeze_rate": 2.5,
    "e2e_delay_ms": 150.0
}
estimator.report_states(qoe_stats)
```

4. **获取带宽估计**:
```python
bandwidth_bps = estimator.get_estimated_bandwidth()
# 返回的带宽 = BC-GCC基础预测 × PPO调整系数
```

## 工作流程

1. **特征提取**: 从数据包统计和 QoE 数据中提取 32 维特征
2. **基础预测**: 使用 BC-GCC trial3 模型预测基础带宽
3. **PPO 动作选择**: 使用 PPO Actor 网络选择带宽调整系数
4. **应用调整**: 最终带宽 = 基础预测 × 调整系数（范围 [0.5, 2.0]）
5. **奖励计算**: 基于当前 QoE 指标计算奖励
6. **在线更新**: 每 N 步收集经验后，使用 PPO 更新策略网络

## 配置说明

- **归一化范围**:
  - `render_fps`: [0, 60] FPS
  - `freeze_rate`: [0, 100] %
  - `e2e_delay_ms`: [0, 10000] ms

- **带宽范围**: [50kbps, 10Mbps]

- **动作空间**: 调整系数 [0.5, 2.0]
  - 0.5: 将基础预测减半
  - 1.0: 保持基础预测不变
  - 2.0: 将基础预测加倍

## 注意事项

1. **QoE 数据平滑**: QoE 数据会通过历史窗口进行平滑处理（默认窗口大小为5）
2. **默认值**: 如果 QoE 数据不可用，将使用默认值：
   - render_fps: 30.0 FPS
   - freeze_rate: 0.0 %
   - e2e_delay_ms: 0.0 ms
3. **模型自动归一化**: 模型会自动对 QoE 特征进行归一化处理
4. **在线学习**: PPO 策略会在运行时在线更新，无需预训练
5. **禁用 RL**: 可以通过 `use_rl=False` 禁用在线学习，仅使用基础模型

## 与 HRCC 的区别

- **基础模型**: 使用 BC-GCC LSTM 模型而非 GCC 启发式算法
- **状态空间**: 32维特征（包括 QoE）而非 6 维
- **网络架构**: 使用 LSTM 处理时序特征，而非 Conv1D
- **奖励函数**: 基于 QoE 指标（render_fps, freeze_rate, e2e_delay）而非网络指标
