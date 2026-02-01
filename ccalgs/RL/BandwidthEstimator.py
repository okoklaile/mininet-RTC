#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在线强化学习带宽估计器
基于 BC-GCC 的 trial3 模型，使用 PPO 进行在线强化学习
使用 QoE 特征（render_fps, freeze_rate, e2e_delay）计算奖励并更新策略
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import time
from collections import deque
from packet_info import PacketInfo
from packet_record import PacketRecord
from model import GCCBC_LSTM
from config import Config32D
from deep_rl.ppo_agent import PPO, RolloutStorage


class Estimator(object):
    """在线强化学习带宽估计器（基于 PPO）"""
    
    def __init__(self, model_path="/home/wyq/桌面/mininet-RTC/ccalgs/RL/trial3.pt", 
                 step_time=200, use_rl=True, update_frequency=4):
        """
        初始化估计器
        Args:
            model_path: PyTorch模型路径（trial3.pt）
            step_time: 时间步长(毫秒)，默认200ms
            use_rl: 是否启用在线强化学习
            update_frequency: PPO 更新频率（每 N 步更新一次）
        """
        # 1. 加载配置和基础模型（BC-GCC trial3）
        self.config = Config32D()  # 使用32维配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 2. 创建基础模型实例（BC-GCC）
        self.base_model = GCCBC_LSTM(self.config)
        self.base_model.load_state_dict(checkpoint['model_state_dict'])
        self.base_model.to(self.device)
        self.base_model.eval()
        
        print(f"✅ 基础模型（BC-GCC trial3）加载成功")
        print(f"   模型参数量: {self.base_model.count_parameters():,}")
        
        # 3. 初始化 PPO Agent（在线强化学习）
        self.use_rl = use_rl
        self.update_frequency = update_frequency
        self.step_counter = 0
        
        if self.use_rl:
            # PPO 超参数
            exploration_param = 0.1  # 探索参数
            lr = 3e-4  # 学习率
            betas = (0.9, 0.999)
            gamma = 0.99  # 折扣因子
            ppo_epoch = 10  # PPO 更新迭代次数
            ppo_clip = 0.2  # PPO 裁剪参数
            
            self.ppo = PPO(
                state_dim=32,  # 32维特征（包括 QoE）
                action_dim=1,  # 动作：带宽调整系数
                exploration_param=exploration_param,
                lr=lr,
                betas=betas,
                gamma=gamma,
                ppo_epoch=ppo_epoch,
                ppo_clip=ppo_clip
            )
            
            # 经验存储
            self.storage = RolloutStorage()
            
            print(f"✅ PPO Agent 初始化成功")
            print(f"   更新频率: 每 {update_frequency} 步")
        
        # 4. 初始化packet_record用于统计网络指标
        self.packet_record = PacketRecord()
        self.packet_record.reset()
        self.step_time = step_time
        
        # 5. 初始化特征历史
        self.prev_delay = 0.0
        self.prev_delay_gradient = 0.0
        self.prev_loss_ratio = 0.0
        self.prev_bandwidth = 300000.0
        self.bandwidth_prediction = 300000.0
        
        # 模型保存路径
        self.save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
        if self.use_rl:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            else:
                # 尝试加载最新的 checkpoint
                self._load_latest_checkpoint()
            
        # 6. 历史数据窗口
        self.delay_history = deque(maxlen=self.config.WINDOW_SIZE)
        self.recv_rate_history = deque(maxlen=self.config.WINDOW_SIZE)
        self.min_delay_seen = float('inf')
        
        # 7. QoE 特征存储
        self.render_fps = 30.0
        self.freeze_rate = 0.0
        self.e2e_delay_ms = 0.0
        
        # QoE 历史（用于平滑，QoE 数据 1 秒更新一次）
        self.render_fps_history = deque(maxlen=5)
        self.freeze_rate_history = deque(maxlen=5)
        self.e2e_delay_history = deque(maxlen=5)
        
        # QoE 数据更新时间戳（用于判断数据是否过期）
        self.qoe_update_time = None  # 最后一次 QoE 更新的时间戳（毫秒）
        self.qoe_update_interval = 1000  # QoE 更新间隔：1000ms (1秒)
        
        # 8. 奖励相关
        self.prev_render_fps = 30.0
        self.prev_freeze_rate = 0.0
        self.prev_e2e_delay = 0.0
        self.base_bandwidth_prediction = 300000.0  # BC-GCC 的基础预测

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
        
        # 重置 QoE 特征
        self.render_fps = 30.0
        self.freeze_rate = 0.0
        self.e2e_delay_ms = 0.0
        self.render_fps_history.clear()
        self.freeze_rate_history.clear()
        self.e2e_delay_history.clear()
        self.qoe_update_time = None
        
        # 重置 PPO
        if self.use_rl:
            self.storage.clear()
            self.step_counter = 0
            self.prev_render_fps = 30.0
            self.prev_freeze_rate = 0.0
            self.prev_e2e_delay = 0.0
            self.base_bandwidth_prediction = 300000.0

    def report_states(self, stats: dict):
        """
        接收数据包信息或 QoE 统计信息
        Args:
            stats: 数据包统计信息字典或 QoE 统计信息字典
        """
        # 处理 QoE 统计信息（1 秒更新一次）
        if stats.get("type") == "qoe":
            # 记录 QoE 更新时间戳（使用当前时间或数据包时间戳）
            current_time = stats.get("timestamp_ms") or stats.get("arrival_time_ms") or time.time() * 1000
            self.qoe_update_time = current_time
            
            if "render_fps" in stats:
                self.render_fps_history.append(stats["render_fps"])
                if len(self.render_fps_history) > 0:
                    self.render_fps = np.mean(self.render_fps_history)
            
            if "freeze_rate" in stats:
                self.freeze_rate_history.append(stats["freeze_rate"])
                if len(self.freeze_rate_history) > 0:
                    self.freeze_rate = np.mean(self.freeze_rate_history)
            
            if "e2e_delay_ms" in stats:
                self.e2e_delay_history.append(stats["e2e_delay_ms"])
                if len(self.e2e_delay_history) > 0:
                    self.e2e_delay_ms = np.mean(self.e2e_delay_history)
            
            return
        
        # 处理数据包信息
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

    def _compute_reward(self):
        """
        计算奖励函数（基于 QoE 指标和网络指标）
        注意：QoE 数据 1 秒更新一次，网络指标 200ms 更新一次
        Returns:
            reward: 奖励值（越高越好）
        """
        VIDEO_PAYLOAD_TYPE = 125
        
        # 获取网络指标（200ms 更新一次）
        delay = self.packet_record.calculate_average_delay(
            interval=self.step_time, 
            filter_payload_type=VIDEO_PAYLOAD_TYPE
        )
        loss_ratio = self.packet_record.calculate_loss_ratio(
            interval=self.step_time, 
            filter_payload_type=VIDEO_PAYLOAD_TYPE
        )
        receiving_rate = self.packet_record.calculate_receiving_rate(
            interval=self.step_time, 
            filter_payload_type=VIDEO_PAYLOAD_TYPE
        )
        throughput_effective = receiving_rate * (1.0 - loss_ratio)  # 有效吞吐量
        
        # 检查 QoE 数据是否过期（QoE 数据 1 秒更新一次）
        # 如果 QoE 数据超过 1.5 秒未更新，使用上一次的值
        current_time_ms = time.time() * 1000
        qoe_data_valid = True
        if self.qoe_update_time is not None:
            time_since_qoe_update = current_time_ms - self.qoe_update_time
            if time_since_qoe_update > self.qoe_update_interval * 1.5:  # 超过 1.5 秒未更新
                qoe_data_valid = False
                # 使用上一次的 QoE 值（保持不变）
                # self.render_fps, self.freeze_rate, self.e2e_delay_ms 保持当前值
        
        # QoE 指标奖励（权重：0.5）
        # 注意：QoE 数据可能不是最新的（1 秒更新一次），但这是正常的
        # 1. render_fps: 越高越好，目标 30 FPS
        fps_reward = min(self.render_fps / 30.0, 1.0) * 0.15
        
        # 2. freeze_rate: 越低越好，惩罚卡顿
        #freeze_penalty = max(0, 1.0 - self.freeze_rate / 10.0) * 0.15  # 10% 卡顿率时惩罚为0
        freeze_penalty = (1.0 - self.freeze_rate / 10.0) * 0.05
        # 3. e2e_delay: 越低越好，目标 < 200ms
        #e2e_delay_reward = max(0, 1.0 - self.e2e_delay_ms / 500.0) * 0.15  # 500ms 时奖励为0
        e2e_delay_reward = ( 1.0 - self.e2e_delay_ms / 500.0) * 0.3
        # 网络指标奖励（权重：0.5）
        # 4. network_delay: 越低越好，目标 < 100ms
        #network_delay_reward = max(0, 1.0 - delay / 300.0) * 0.15  # 300ms 时奖励为0
        network_delay_reward = (1.0 - delay / 200.0) * 0.3 
        # 5. loss_ratio: 越低越好，惩罚丢包
        loss_penalty = max(0, 1.0 - loss_ratio / 0.05) * 0.15  # 5% 丢包率时惩罚为0
        
        # 6. throughput: 越高越好，奖励高吞吐量（归一化到 10Mbps）
        throughput_reward = min(throughput_effective / 10e6, 1.0) * 0.05  # 10Mbps 时奖励为1
        
        # 总奖励
        reward = (fps_reward + freeze_penalty + e2e_delay_reward + 
                 network_delay_reward + loss_penalty + throughput_reward)
        
        return reward

    def _extract_features(self):
        """
        提取32维特征向量（包括 QoE 特征）
        Returns:
            features: [32] 特征向量
        """
        VIDEO_PAYLOAD_TYPE = 125
        
        # 1. 计算基础网络指标
        delay = self.packet_record.calculate_average_delay(
            interval=self.step_time, 
            filter_payload_type=VIDEO_PAYLOAD_TYPE
        )
        loss_ratio = self.packet_record.calculate_loss_ratio(
            interval=self.step_time, 
            filter_payload_type=VIDEO_PAYLOAD_TYPE
        )
        receiving_rate = self.packet_record.calculate_receiving_rate(
            interval=self.step_time, 
            filter_payload_type=VIDEO_PAYLOAD_TYPE
        )
        
        # 2. 计算衍生特征
        delay_gradient = delay - self.prev_delay
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
        queue_delay = max(0, delay - delay_min)
        delay_accel = delay_gradient - self.prev_delay_gradient
        delay_trend = self._calculate_trend(self.delay_history)
        
        # 5. 其他特征
        loss_change = loss_ratio - self.prev_loss_ratio
        bw_utilization = receiving_rate / self.prev_bandwidth if self.prev_bandwidth > 0 else 0.0
        recv_rate_mean = np.mean(self.recv_rate_history) if len(self.recv_rate_history) > 0 else receiving_rate
        recv_rate_std = np.std(self.recv_rate_history) if len(self.recv_rate_history) > 1 else 0.0
        
        # 6. 构造32维特征向量
        features_raw = np.array([
            # 基础特征 (0-5)
            delay, loss_ratio, receiving_rate, self.prev_bandwidth,
            delay_gradient, throughput_effective,
            # 延迟统计 (6-11)
            delay_mean, delay_std, delay_min, queue_delay,
            delay_accel, delay_trend,
            # 丢包变化 (12)
            loss_change,
            # 带宽利用率 (13-15)
            bw_utilization, recv_rate_mean, recv_rate_std,
            # 原有保留字段 (16-23)
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            # QoE 特征 (24-26)
            self.render_fps, self.freeze_rate, self.e2e_delay_ms,
            # 其他保留字段 (27-31)
            0.0, 0.0, 0.0, 0.0, 0.0
        ], dtype=np.float32)
        
        # 7. 归一化特征
        features_norm = self._normalize_features(features_raw)
        
        return features_norm

    def get_estimated_bandwidth(self) -> int:
        """
        计算并返回带宽估计值
        使用 BC-GCC 基础模型 + PPO 调整
        Returns:
            bandwidth_prediction: 带宽预测值(bps)
        """
        # 1. 提取特征
        features_norm = self._extract_features()
        
        # 2. 准备特征
        # features_norm 包含完整的特征（包括 QoE），用于 PPO
        # features_base 将 QoE 部分强制置零，用于 Base Model（模仿学习时未见过 QoE）
        features_base = features_norm.copy()
        
        # QoE 特征在 features_norm 中的索引（参考 config.py 和 _normalize_features）
        # 24: render_fps, 25: freeze_rate, 26: e2e_delay_ms
        qoe_indices = [24, 25, 26]
        features_base[qoe_indices] = 0.0
        
        # 3. 使用 BC-GCC 基础模型预测 (使用 masked features)
        input_tensor = torch.from_numpy(features_base).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output, _ = self.base_model.predict(input_tensor)
            base_bandwidth_raw = output.cpu().item()
        
        # 反归一化
        if base_bandwidth_raw < 1.0:
            base_bandwidth = base_bandwidth_raw * self.config.NORM_STATS['bandwidth_prediction']['max']
        else:
            base_bandwidth = base_bandwidth_raw
        
        base_bandwidth = np.clip(base_bandwidth, 50000, 10e6)
        self.base_bandwidth_prediction = base_bandwidth
        
        # 3. 使用 PPO 调整（如果启用）
        if self.use_rl:
            # 选择动作（带宽调整系数）
            action = self.ppo.select_action(features_norm, self.storage)
            
            # 动作范围 [0, 1]，映射到调整系数 [0.5, 2.0]
            # action = 0 -> 系数 = 0.5 (减半)
            # action = 1 -> 系数 = 2.0 (加倍)
            if isinstance(action, np.ndarray):
                action_value = action[0] if action.size > 0 else action.item()
            else:
                action_value = float(action)
            adjustment_factor = 0.5 + 1.5 * action_value
            
            # 应用调整
            adjusted_bandwidth = base_bandwidth * adjustment_factor
            adjusted_bandwidth = np.clip(adjusted_bandwidth, 50000, 10e6)
            
            # 计算奖励
            reward = self._compute_reward()
            
            # 存储奖励和完成标志
            self.storage.rewards.append(reward)
            self.storage.dones.append(False)  # 非终止状态
            
            # 更新计数器
            self.step_counter += 1
            
            # 定期更新 PPO 策略
            if self.step_counter >= self.update_frequency:
                policy_loss, value_loss = self.ppo.update(self.storage)
                self.storage.clear()
                self.step_counter = 0
                
                # 更新 QoE 历史（用于下次奖励计算）
                self.prev_render_fps = self.render_fps
                self.prev_freeze_rate = self.freeze_rate
                self.prev_e2e_delay = self.e2e_delay_ms
                
                # 保存模型（每 100 次更新保存一次，或者根据需要调整）
                # 这里使用一个简单的计数器来控制保存频率
                if not hasattr(self, 'update_count'):
                    self.update_count = 0
                self.update_count += 1
                
                if self.update_count % 100 == 0:
                    self.save_model(f"ppo_checkpoint_{self.update_count}.pth")
            
            self.bandwidth_prediction = int(adjusted_bandwidth)
        else:
            # 不使用 RL，直接使用基础模型预测
            self.bandwidth_prediction = int(base_bandwidth)
        
        # 4. 更新历史状态
        VIDEO_PAYLOAD_TYPE = 125
        delay = self.packet_record.calculate_average_delay(
            interval=self.step_time, 
            filter_payload_type=VIDEO_PAYLOAD_TYPE
        )
        delay_gradient = delay - self.prev_delay
        
        self.prev_delay = delay
        self.prev_delay_gradient = delay_gradient
        self.prev_loss_ratio = self.packet_record.calculate_loss_ratio(
            interval=self.step_time, 
            filter_payload_type=VIDEO_PAYLOAD_TYPE
        )
        self.prev_bandwidth = self.bandwidth_prediction
        
        return self.bandwidth_prediction
    
    def _normalize_features(self, features_raw):
        """归一化输入特征"""
        features_norm = features_raw.copy()
        norm_stats = self.config.NORM_STATS
        
        feature_names = [
            'delay', 'loss_ratio', 'receiving_rate', 'prev_bandwidth', 
            'delay_gradient', 'throughput_effective',
            'delay_mean', 'delay_std', 'delay_min', 'queue_delay', 
            'delay_accel', 'delay_trend',
            'loss_change',
            'bw_utilization', 'recv_rate_mean', 'recv_rate_std'
        ]
        
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
        
        # 归一化 QoE 特征
        qoe_features = [
            ('render_fps', 24),
            ('freeze_rate', 25),
            ('e2e_delay_ms', 26)
        ]
        
        for name, idx in qoe_features:
            if name in norm_stats:
                min_val = norm_stats[name]['min']
                max_val = norm_stats[name]['max']
                if self.config.USE_CLIPPING:
                    features_norm[idx] = np.clip(features_raw[idx], min_val, max_val)
                if max_val > min_val:
                    features_norm[idx] = (features_norm[idx] - min_val) / (max_val - min_val)
                else:
                    features_norm[idx] = 0.0
        
        return features_norm
    
    def _calculate_trend(self, data_history):
        """计算数据的线性回归趋势（斜率）"""
        if len(data_history) < 2:
            return 0.0
        
        y = np.array(data_history)
        x = np.arange(len(y))
        
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

    def save_model(self, filename):
        """保存 PPO 模型"""
        if not self.use_rl:
            return
            
        save_path = os.path.join(self.save_dir, filename)
        try:
            self.ppo.save(save_path)
            # print(f"✅ 模型已保存: {save_path}") # 避免过多日志输出
        except Exception as e:
            print(f"❌ 模型保存失败: {e}")

    def _load_latest_checkpoint(self):
        """加载最新的 PPO checkpoint"""
        try:
            checkpoints = [f for f in os.listdir(self.save_dir) if f.startswith('ppo_checkpoint_') and f.endswith('.pth')]
            if not checkpoints:
                return

            # 解析文件名中的索引
            # 格式: ppo_checkpoint_{index}.pth
            latest_checkpoint = None
            max_index = -1
            
            for cp in checkpoints:
                try:
                    idx = int(cp.split('_')[-1].split('.')[0])
                    if idx > max_index:
                        max_index = idx
                        latest_checkpoint = cp
                except ValueError:
                    continue
            
            if latest_checkpoint:
                checkpoint_path = os.path.join(self.save_dir, latest_checkpoint)
                self.ppo.load(checkpoint_path)
                
                # 恢复计数器
                self.update_count = max_index
                print(f"✅ 已加载最新 PPO 模型: {latest_checkpoint}")
                
        except Exception as e:
            print(f"⚠️ 加载 Checkpoint 失败: {e}")


