#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BC-GCC 带宽估计器
基于深度学习模型的带宽预测
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from collections import deque
from packet_info import PacketInfo
from packet_record import PacketRecord
from model import GCCBC_LSTM
from config import Config


class Estimator(object):
    """BC-GCC 带宽估计器"""
    
    def __init__(self, model_path="/home/wyq/桌面/mininet-RTC/ccalgs/BC-GCC/trial2.pt", step_time=200):
        """
        初始化估计器
        Args:
            model_path: PyTorch模型路径
            step_time: 时间步长(毫秒)，默认200ms
        """
        # 1. 加载配置和模型检查点
        self.config = Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 2. 创建模型实例
        self.model = GCCBC_LSTM(self.config)
        
        # 3. 加载模型权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()  # 设置为评估模式
        
        print(f"✅ BC-GCC 模型加载成功 (Epoch {checkpoint['epoch']}, Val Loss: {checkpoint['best_val_loss']:.6f})")
        print(f"   模型参数量: {self.model.count_parameters():,}")
        
        # 4. 初始化packet_record用于统计网络指标
        self.packet_record = PacketRecord()
        self.packet_record.reset()
        self.step_time = step_time  # 时间步长(ms)
        
        # 5. 初始化特征历史（用于计算梯度和模型输入）
        self.prev_delay = 0.0  # 上一次的延迟(ms)
        self.prev_delay_gradient = 0.0  # 上一次的延迟梯度(ms)
        self.prev_loss_ratio = 0.0  # 上一次的丢包率
        self.prev_bandwidth = 300000.0  # 上一次的带宽预测(bps)，初始300kbps
        self.bandwidth_prediction = 300000.0  # 当前带宽预测(bps)
        
        # 6. 历史数据窗口（用于计算统计特征）
        self.delay_history = deque(maxlen=Config.WINDOW_SIZE)  # 延迟历史
        self.recv_rate_history = deque(maxlen=Config.WINDOW_SIZE)  # 接收速率历史
        self.min_delay_seen = float('inf')  # 观察到的最小延迟（基线RTT）

    def reset(self):
        """重置估计器状态"""
        self.packet_record.reset()
        self.prev_delay = 0.0
        self.prev_delay_gradient = 0.0
        self.prev_loss_ratio = 0.0
        self.prev_bandwidth = 300000.0
        self.bandwidth_prediction = 300000.0
        self.delay_history.clear()
        self.recv_rate_history.clear()
        self.min_delay_seen = float('inf')

    def report_states(self, stats: dict):
        """
        接收数据包信息
        Args:
            stats: 数据包统计信息字典，包含：
                - send_time_ms: 发送时间戳(毫秒)
                - arrival_time_ms: 到达时间戳(毫秒)
                - payload_type: 载荷类型
                - sequence_number: 序列号
                - ssrc: 同步源标识符
                - padding_length: 填充长度(字节)
                - header_length: 头部长度(字节)
                - payload_size: 载荷大小(字节)
        """
        if stats.get("type") == "qoe":
            return
            
        # 构造PacketInfo对象
        packet_info = PacketInfo()
        packet_info.payload_type = stats["payload_type"]
        packet_info.ssrc = stats["ssrc"]
        packet_info.sequence_number = stats["sequence_number"]
        packet_info.send_timestamp = stats["send_time_ms"]
        packet_info.receive_timestamp = stats["arrival_time_ms"]
        packet_info.padding_length = stats["padding_length"]
        packet_info.header_length = stats["header_length"]
        packet_info.payload_size = stats["payload_size"]
        packet_info.bandwidth_prediction = self.bandwidth_prediction
        
        # 更新packet_record用于统计网络指标
        self.packet_record.on_receive(packet_info)

    def get_estimated_bandwidth(self) -> int:
        """
        计算并返回带宽估计值
        
        模型输入格式 (24维特征向量):
        核心特征 (16维):
        - 索引0-5: 基础特征 (delay, loss_ratio, receiving_rate, prev_bandwidth, delay_gradient, throughput_effective)
        - 索引6-11: 延迟统计 (delay_mean, delay_std, delay_min, queue_delay, delay_accel, delay_trend)
        - 索引12: 丢包变化 (loss_change)
        - 索引13-15: 带宽利用率 (bw_utilization, recv_rate_mean, recv_rate_std)
        保留字段 (8维):
        - 索引16-23: reserved (全为0)
        
        模型输出格式:
        - bandwidth_prediction (bps) - 归一化值，需要反归一化
        
        Returns:
            bandwidth_prediction: 带宽预测值(bps)
        """
        # 视频包的 payload_type 为 125
        VIDEO_PAYLOAD_TYPE = 125
        
        # 1. 计算基础网络指标
        # 延迟 (ms)
        delay = self.packet_record.calculate_average_delay(
            interval=self.step_time, 
            filter_payload_type=VIDEO_PAYLOAD_TYPE
        )
        
        # 丢包率 [0, 1]
        loss_ratio = self.packet_record.calculate_loss_ratio(
            interval=self.step_time, 
            filter_payload_type=VIDEO_PAYLOAD_TYPE
        )
        
        # 接收速率 (bps)
        receiving_rate = self.packet_record.calculate_receiving_rate(
            interval=self.step_time, 
            filter_payload_type=VIDEO_PAYLOAD_TYPE
        )
        
        # 2. 计算衍生特征
        # 延迟梯度 (1st order) = delay[t] - delay[t-1]
        delay_gradient = delay - self.prev_delay
        
        # 有效吞吐量 (考虑丢包)
        throughput_effective = receiving_rate * (1.0 - loss_ratio)
        
        # 3. 更新历史窗口
        self.delay_history.append(delay)
        self.recv_rate_history.append(receiving_rate)
        if delay > 0:
            self.min_delay_seen = min(self.min_delay_seen, delay)
        
        # 4. 计算延迟统计特征
        delay_mean = np.mean(self.delay_history) if len(self.delay_history) > 0 else delay
        delay_std = np.std(self.delay_history) if len(self.delay_history) > 1 else 0.0
        delay_min = self.min_delay_seen if self.min_delay_seen != float('inf') else delay
        queue_delay = max(0, delay - delay_min)  # 排队延迟
        
        # 延迟加速度 (2nd order) = gradient[t] - gradient[t-1]
        delay_accel = delay_gradient - self.prev_delay_gradient
        
        # 延迟趋势 (线性回归斜率)
        delay_trend = self._calculate_trend(self.delay_history)
        
        # 5. 计算丢包变化
        loss_change = loss_ratio - self.prev_loss_ratio
        
        # 6. 计算带宽利用率特征
        bw_utilization = receiving_rate / self.prev_bandwidth if self.prev_bandwidth > 0 else 0.0
        recv_rate_mean = np.mean(self.recv_rate_history) if len(self.recv_rate_history) > 0 else receiving_rate
        recv_rate_std = np.std(self.recv_rate_history) if len(self.recv_rate_history) > 1 else 0.0
        
        # 7. 构造模型输入特征向量 [24维] - 原始值
        features_raw = np.array([
            # 基础特征 (0-5)
            delay,                  # 0: 延迟 (ms)
            loss_ratio,             # 1: 丢包率 [0, 1]
            receiving_rate,         # 2: 接收速率 (bps)
            self.prev_bandwidth,    # 3: 上一次的带宽预测 (bps)
            delay_gradient,         # 4: 延迟梯度 (ms)
            throughput_effective,   # 5: 有效吞吐量 (bps)
            
            # 延迟统计特征 (6-11)
            delay_mean,             # 6: 平均延迟 (ms)
            delay_std,              # 7: 延迟标准差 (ms)
            delay_min,              # 8: 最小延迟 (ms)
            queue_delay,            # 9: 排队延迟 (ms)
            delay_accel,            # 10: 延迟加速度 (ms)
            delay_trend,            # 11: 延迟趋势 (ms/step)
            
            # 丢包变化 (12)
            loss_change,            # 12: 丢包率变化
            
            # 带宽利用率 (13-15)
            bw_utilization,         # 13: 带宽利用率
            recv_rate_mean,         # 14: 平均接收速率 (bps)
            recv_rate_std,          # 15: 接收速率标准差 (bps)
            
            # 保留字段 (16-23)
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ], dtype=np.float32)
        
        # 8. 归一化特征（根据训练时使用的归一化参数）
        features_norm = self._normalize_features(features_raw)
        
        # 9. 转换为PyTorch张量并添加序列维度 [1, 1, 24]
        # 注意：模型期望 [batch, seq_len, feature_dim] 格式
        input_tensor = torch.from_numpy(features_norm).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # 10. 模型推理
        with torch.no_grad():
            # 使用模型的 predict 方法
            output, _ = self.model.predict(input_tensor)
            bandwidth_pred_raw = output.cpu().item()  # 提取标量值
        
        # 11. 反归一化输出
        # 根据Config.NORM_STATS，模型输出可能是归一化的值，需要反归一化
        # bandwidth范围: [0, 10e6] bps
        # 如果模型输出很小（<1），假设是归一化的
        if bandwidth_pred_raw < 1.0:
            # 输出在[0, 1]范围，需要反归一化
            bandwidth_pred = bandwidth_pred_raw * self.config.NORM_STATS['bandwidth_prediction']['max']
        else:
            # 输出已经是实际带宽值
            bandwidth_pred = bandwidth_pred_raw
        
        
        # 12. 限制带宽范围 [50kbps, 10Mbps]
        bandwidth_pred = np.clip(bandwidth_pred, 50000, 10e6)
        
        # 13. 更新历史状态
        self.prev_delay = delay
        self.prev_delay_gradient = delay_gradient
        self.prev_loss_ratio = loss_ratio
        self.prev_bandwidth = bandwidth_pred
        self.bandwidth_prediction = int(bandwidth_pred)
        
        return self.bandwidth_prediction
    
    def _normalize_features(self, features_raw):
        """
        归一化输入特征（根据Config中定义的归一化参数）
        
        Args:
            features_raw: 原始特征向量 [24]
        Returns:
            features_norm: 归一化后的特征向量 [24]
        """
        features_norm = features_raw.copy()
        
        # 归一化核心特征（索引0-15）
        # 使用 min-max 归一化: (x - min) / (max - min)
        norm_stats = self.config.NORM_STATS
        feature_names = [
            # 基础特征 (0-5)
            'delay', 'loss_ratio', 'receiving_rate', 'prev_bandwidth', 'delay_gradient', 'throughput_effective',
            # 延迟统计 (6-11)
            'delay_mean', 'delay_std', 'delay_min', 'queue_delay', 'delay_accel', 'delay_trend',
            # 丢包变化 (12)
            'loss_change',
            # 带宽利用率 (13-15)
            'bw_utilization', 'recv_rate_mean', 'recv_rate_std'
        ]
        
        for i, name in enumerate(feature_names):
            if name in norm_stats:
                min_val = norm_stats[name]['min']
                max_val = norm_stats[name]['max']
                
                # Clip to range (if USE_CLIPPING is enabled)
                if self.config.USE_CLIPPING:
                    features_norm[i] = np.clip(features_raw[i], min_val, max_val)
                
                # Normalize to [0, 1]
                if max_val > min_val:
                    features_norm[i] = (features_norm[i] - min_val) / (max_val - min_val)
                else:
                    features_norm[i] = 0.0
        
        # 保留字段（索引16-23）已经是0，不需要归一化
        
        return features_norm
    
    def _calculate_trend(self, data_history):
        """
        计算数据的线性回归趋势（斜率）
        
        Args:
            data_history: deque 历史数据
        Returns:
            trend: 线性回归斜率
        """
        if len(data_history) < 2:
            return 0.0
        
        # 转换为numpy数组
        y = np.array(data_history)
        x = np.arange(len(y))
        
        # 简单线性回归: y = ax + b
        # a = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x * x)
        
        denominator = n * sum_x2 - sum_x * sum_x
        if abs(denominator) < 1e-10:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope
