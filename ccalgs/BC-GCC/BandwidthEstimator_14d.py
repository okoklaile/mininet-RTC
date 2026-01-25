#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BC-GCC 带宽估计器 (14维版本 - 兼容旧模型)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from packet_info import PacketInfo
from packet_record import PacketRecord
from model import GCCBC_LSTM


class Config14D:
    """14维特征配置（用于加载旧模型）"""
    TOTAL_FEATURE_DIM = 14
    LSTM_HIDDEN_SIZE = 256
    LSTM_NUM_LAYERS = 2
    DROPOUT = 0.2
    FC_HIDDEN_SIZES = [128, 64]
    USE_CLIPPING = True
    
    NORM_STATS = {
        'delay': {'min': 0, 'max': 10000},
        'loss_ratio': {'min': 0, 'max': 1},
        'receiving_rate': {'min': 0, 'max': 10e6},
        'prev_bandwidth': {'min': 0, 'max': 10e6},
        'delay_gradient': {'min': -2000, 'max': 2000},
        'throughput': {'min': 0, 'max': 10e6},
        'bandwidth_prediction': {'min': 0, 'max': 10e6},
    }


class Estimator(object):
    """BC-GCC 带宽估计器 (14维版本)"""
    
    def __init__(self, model_path="/home/wyq/桌面/mininet-RTC/ccalgs/BC-GCC/trial1.pt", step_time=200):
        # 使用14维配置
        self.config = Config14D()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 创建14维模型
        self.model = GCCBC_LSTM(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ BC-GCC 模型加载成功 (14维, Epoch {checkpoint['epoch']}, Val Loss: {checkpoint['best_val_loss']:.6f})")
        print(f"   模型参数量: {self.model.count_parameters():,}")
        
        # 初始化packet_record
        self.packet_record = PacketRecord()
        self.packet_record.reset()
        self.step_time = step_time
        
        # 初始化历史状态
        self.prev_delay = 0.0
        self.prev_bandwidth = 300000.0
        self.bandwidth_prediction = 300000.0

    def reset(self):
        self.packet_record.reset()
        self.prev_delay = 0.0
        self.prev_bandwidth = 300000.0
        self.bandwidth_prediction = 300000.0

    def report_states(self, stats: dict):
        if stats.get("type") == "qoe":
            return
        
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
        
        self.packet_record.on_receive(packet_info)

    def get_estimated_bandwidth(self) -> int:
        VIDEO_PAYLOAD_TYPE = 125
        
        # 计算基础特征（14维）
        delay = self.packet_record.calculate_average_delay(
            interval=self.step_time, filter_payload_type=VIDEO_PAYLOAD_TYPE
        )
        loss_ratio = self.packet_record.calculate_loss_ratio(
            interval=self.step_time, filter_payload_type=VIDEO_PAYLOAD_TYPE
        )
        receiving_rate = self.packet_record.calculate_receiving_rate(
            interval=self.step_time, filter_payload_type=VIDEO_PAYLOAD_TYPE
        )
        delay_gradient = delay - self.prev_delay
        throughput = receiving_rate
        
        # 构造14维特征
        features_raw = np.array([
            delay, loss_ratio, receiving_rate, self.prev_bandwidth,
            delay_gradient, throughput,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # 保留字段
        ], dtype=np.float32)
        
        # 归一化
        features_norm = self._normalize_features(features_raw)
        
        # 推理
        input_tensor = torch.from_numpy(features_norm).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output, _ = self.model.predict(input_tensor)
            bandwidth_pred_raw = output.cpu().item()
        
        # 反归一化
        if bandwidth_pred_raw < 1.0:
            bandwidth_pred = bandwidth_pred_raw * self.config.NORM_STATS['bandwidth_prediction']['max']
        else:
            bandwidth_pred = bandwidth_pred_raw
        
        # 限制范围
        bandwidth_pred = np.clip(bandwidth_pred, 50000, 10e6)
        
        # 更新状态
        self.prev_delay = delay
        self.prev_bandwidth = bandwidth_pred
        self.bandwidth_prediction = int(bandwidth_pred)
        
        return self.bandwidth_prediction
    
    def _normalize_features(self, features_raw):
        features_norm = features_raw.copy()
        norm_stats = self.config.NORM_STATS
        feature_names = ['delay', 'loss_ratio', 'receiving_rate', 'prev_bandwidth', 'delay_gradient', 'throughput']
        
        for i, name in enumerate(feature_names):
            if name in norm_stats:
                min_val = norm_stats[name]['min']
                max_val = norm_stats[name]['max']
                
                if self.config.USE_CLIPPING:
                    features_norm[i] = np.clip(features_raw[i], min_val, max_val)
                
                if max_val > min_val:
                    features_norm[i] = (features_norm[i] - min_val) / (max_val - min_val)
                else:
                    features_norm[i] = 0.0
        
        return features_norm
