# BC-GCC (Behavior Cloning - Google Congestion Control)

BC-GCC 是一个基于深度学习的带宽估计算法，使用 LSTM 网络进行带宽预测。

## 📁 文件结构

```
BC-GCC/
├── BandwidthEstimator.py    # 核心带宽估计器
├── model.py                  # LSTM 模型定义
├── config.py                 # 训练配置和归一化参数
├── packet_info.py            # 数据包信息类
├── packet_record.py          # 数据包统计工具
├── trial1.pt                 # 预训练模型 (Epoch 43, Val Loss: 0.023419)
├── receiver_pyinfer.json     # 接收端配置
├── sender_pyinfer.json       # 发送端配置
├── test_estimator.py         # 测试脚本
└── README.md                 # 本文件
```

## 🎯 模型架构

- **输入**: 14维特征向量
  - 核心特征 (6维): delay, loss_ratio, receiving_rate, prev_bandwidth, delay_gradient, throughput
  - 保留特征 (8维): 用于未来扩展（当前填充0）
  
- **网络结构**:
  - LSTM: 2层 × 256隐藏单元
  - 全连接: [128, 64] + ReLU + Dropout(0.2)
  - 输出层: Linear(64, 1) + ReLU
  
- **参数量**: 846,081

## 🔧 使用方法

### 1. 基本使用

```python
from BandwidthEstimator import Estimator

# 初始化估计器
estimator = Estimator()

# 接收数据包统计信息
stats = {
    "send_time_ms": 1000,
    "arrival_time_ms": 1100,
    "payload_type": 125,        # 视频包
    "sequence_number": 1000,
    "ssrc": 12345,
    "padding_length": 0,
    "header_length": 12,
    "payload_size": 1200,
}
estimator.report_states(stats)

# 获取带宽估计（每200ms调用一次）
bandwidth = estimator.get_estimated_bandwidth()  # 返回 bps
print(f"带宽估计: {bandwidth/1e6:.3f} Mbps")
```

### 2. 在项目中使用

```bash
# 使用 BC-GCC 算法运行测试
python multi_cc_test.py --algorithm BC-GCC --trace trace/4G_700kbps.json
```

## 📊 模型输入输出

### 输入特征 (14维)

| 索引 | 特征名 | 说明 | 单位 | 归一化范围 |
|------|--------|------|------|------------|
| 0 | delay | 当前延迟 | ms | [0, 10000] |
| 1 | loss_ratio | 丢包率 | - | [0, 1] |
| 2 | receiving_rate | 接收速率 | bps | [0, 10e6] |
| 3 | prev_bandwidth | 上一次带宽预测 | bps | [0, 10e6] |
| 4 | delay_gradient | 延迟梯度 | ms | [-2000, 2000] |
| 5 | throughput | 吞吐量 | bps | [0, 10e6] |
| 6-13 | reserved | 保留字段 | - | 0 |

### 输出

- **范围**: [0, 1] (归一化值)
- **反归一化**: `actual_bw = output × 10e6`  
- **实际输出**: [50 kbps, 10 Mbps]

## ⚙️ 配置文件

### receiver_pyinfer.json

```json
{
  "bwe_feedback_duration": 200,  // 带宽估计反馈间隔 (ms)
  "logging": {
    "enabled": true,
    "log_output_path": "/path/to/output/BC-GCC_receiver.log"
  }
}
```

### sender_pyinfer.json

```json
{
  "bwe_feedback_duration": 200,
  "autoclose": 60  // 自动关闭时间 (秒)
}
```

## 🧪 测试

运行测试脚本验证模型：

```bash
cd /home/wyq/桌面/mininet-RTC/ccalgs/BC-GCC
python test_estimator.py
```

预期输出：
```
场景: 良好网络 (1 Mbps, 低延迟)
✅ 最终带宽估计: 1.408 Mbps

场景: 中等网络 (500 kbps, 中延迟)
✅ 最终带宽估计: 1.306 Mbps

场景: 拥塞网络 (200 kbps, 高延迟)
✅ 最终带宽估计: 1.217 Mbps
```

## 📝 关键实现细节

### 1. 特征归一化

使用 Min-Max 归一化：
```python
normalized = (value - min) / (max - min)
```

### 2. 输出反归一化

如果模型输出 < 1.0，视为归一化值，需要反归一化：
```python
if output < 1.0:
    bandwidth = output * 10e6
```

### 3. 统计时间窗口

- 默认时间窗口: 200ms
- 只统计视频包 (`payload_type == 125`)
- 自动处理乱序包（静默忽略）

## 🚀 性能指标

- **模型参数**: 846,081
- **推理时间**: ~5ms (CPU)
- **内存占用**: ~10MB (模型文件)
- **训练集**: ghent, norway, NY, opennetlab
- **验证损失**: 0.023419

## 📚 依赖项

- Python >= 3.8
- PyTorch >= 1.8
- NumPy

## 🔍 调试

如需启用调试输出，在 `BandwidthEstimator.py` 中添加：

```python
# 在 get_estimated_bandwidth() 方法中
print(f"delay={delay:.1f}ms, loss={loss_ratio:.4f}, rate={receiving_rate/1e6:.3f}Mbps")
print(f"模型输出: {bandwidth_pred/1e6:.3f} Mbps")
```

## ⚠️ 注意事项

1. **模型文件**: 确保 `trial1.pt` 在正确路径
2. **数据包顺序**: 乱序包会被自动忽略
3. **最小带宽**: 输出限制在 [50kbps, 10Mbps] 范围
4. **时间同步**: 使用 timer_delta 抵消时钟偏移

## 📖 参考

- 模型训练: Behavior Cloning from GCC expert traces
- 数据集: Multi-location network traces (4G, 5G, WiFi)
- 优化: Sample weighting for loss/delay scenarios
