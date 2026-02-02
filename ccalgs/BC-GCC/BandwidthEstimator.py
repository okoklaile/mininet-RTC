#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BC-GCC 带宽估计器
基于深度学习模型的带宽预测，集成了GCC的慢启动逻辑
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from collections import deque
import collections
from packet_info import PacketInfo
from packet_record import PacketRecord
from model import GCCBC_LSTM
from config import Config

# --- GCC Constants ---
kMinNumDeltas = 60
threshold_gain_ = 4
kBurstIntervalMs = 5
kTrendlineWindowSize = 20
kTrendlineSmoothingCoeff = 0.9
kOverUsingTimeThreshold = 10
kMaxAdaptOffsetMs = 15.0
k_up_ = 0.0087
k_down_ = 0.039
Time_Interval = 200

class Config32D(Config):
    """32维特征配置（用于trial3模型）"""
    TOTAL_FEATURE_DIM = 32
    RESERVED_FEATURES = Config.RESERVED_FEATURES + [
        'custom_5', 'custom_6', 'custom_7', 'custom_8',
        'custom_9', 'custom_10', 'custom_11', 'custom_12'
    ]

class PacketGroup:
    """
    数据包组类 (来自GCC)
    """
    def __init__(self, pkt_group):
        self.pkts = pkt_group
        self.arrival_time_list = [pkt.receive_timestamp for pkt in pkt_group]
        self.send_time_list = [pkt.send_timestamp for pkt in pkt_group]
        self.pkt_group_size = sum([pkt.size for pkt in pkt_group])
        self.pkt_num_in_group = len(pkt_group)
        self.complete_time = self.arrival_time_list[-1]
        self.transfer_duration = self.arrival_time_list[-1] - self.arrival_time_list[0]

class Estimator(object):
    """BC-GCC 带宽估计器"""
    
    def __init__(self, model_path="/home/wyq/桌面/mininet-RTC/ccalgs/BC-GCC/trial3.pt", step_time=200, use_slow_start=False):
        """
        初始化估计器
        """
        # --- Model Init ---
        self.config = Config32D()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = GCCBC_LSTM(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ BC-GCC 模型加载成功 (Epoch {checkpoint['epoch']}, Val Loss: {checkpoint['best_val_loss']:.6f})")
        
        self.packet_record = PacketRecord()
        self.packet_record.reset()
        self.step_time = step_time
        
        # Feature History
        self.prev_delay = 0.0
        self.prev_delay_gradient = 0.0
        self.prev_loss_ratio = 0.0
        self.prev_bandwidth = 300000.0  # 300kbps
        self.bandwidth_prediction = 300000.0
        
        self.delay_history = deque(maxlen=self.config.WINDOW_SIZE)
        self.recv_rate_history = deque(maxlen=self.config.WINDOW_SIZE)
        self.feature_history = deque(maxlen=self.config.WINDOW_SIZE)
        self.min_delay_seen = float('inf')
        
        # --- Slow Start / GCC Init ---
        self.use_slow_start = use_slow_start
        self.in_slow_start = use_slow_start
        
        # GCC State Variables
        self.packets_list = []  # For GCC logic
        self.first_group_complete_time = -1
        
        self.acc_delay = 0
        self.smoothed_delay = 0
        self.acc_delay_list = collections.deque([])
        self.smoothed_delay_list = collections.deque([])
        
        self.state = 'Hold'
        self.last_bandwidth_estimation = 300 * 1000 # GCC tracks this separately, synced with self.bandwidth_prediction
        self.avg_max_bitrate_kbps_ = -1
        self.var_max_bitrate_kbps_ = -1
        self.rate_control_region_ = "kRcMaxUnknown"
        self.time_last_bitrate_change_ = -1
        
        self.gamma1 = 12.5
        self.num_of_deltas_ = 0
        self.time_over_using = -1
        self.prev_trend = 0.0
        self.overuse_counter = 0
        self.overuse_flag = 'NORMAL'
        self.last_update_ms = -1
        self.last_update_threshold_ms = -1
        self.now_ms = -1
        self.timer_delta = None

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
        self.feature_history.clear()
        self.min_delay_seen = float('inf')
        
        # Reset GCC state
        self.in_slow_start = self.use_slow_start
        self.packets_list = []
        self.first_group_complete_time = -1
        self.acc_delay = 0
        self.smoothed_delay = 0
        self.acc_delay_list = collections.deque([])
        self.smoothed_delay_list = collections.deque([])
        self.state = 'Hold'
        self.last_bandwidth_estimation = 300 * 1000
        self.avg_max_bitrate_kbps_ = -1
        self.var_max_bitrate_kbps_ = -1
        self.rate_control_region_ = "kRcMaxUnknown"
        self.time_last_bitrate_change_ = -1
        self.gamma1 = 12.5
        self.num_of_deltas_ = 0
        self.time_over_using = -1
        self.prev_trend = 0.0
        self.overuse_counter = 0
        self.overuse_flag = 'NORMAL'
        self.last_update_ms = -1
        self.last_update_threshold_ms = -1
        self.now_ms = -1
        self.timer_delta = None

    def report_states(self, stats: dict):
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
        
        # Calculate size for GCC
        packet_info.size = stats["header_length"] + stats["payload_size"] + stats["padding_length"]
        
        self.now_ms = packet_info.receive_timestamp
        
        # 更新packet_record (Model用)
        self.packet_record.on_receive(packet_info)
        
        # 更新packets_list (GCC用)
        if self.in_slow_start:
            self.packets_list.append(packet_info)

    def get_estimated_bandwidth(self) -> int:
        # Sync bandwidth state
        self.last_bandwidth_estimation = self.bandwidth_prediction

        if self.in_slow_start:
            # Run GCC Slow Start Logic
            bw, keep_running = self._run_gcc_logic()
            
            if keep_running:
                self.bandwidth_prediction = int(bw)
                self.prev_bandwidth = self.bandwidth_prediction
                # Also update history for model to avoid jump when switching
                self._update_model_history_in_background()
                return self.bandwidth_prediction
            else:
                print(f"🐌 Exiting Slow Start. Switching to BC-GCC Model.")
                self.in_slow_start = False
                # Fall through to model logic
        
        # --- Model Logic ---
        VIDEO_PAYLOAD_TYPE = 98
        
        # 1. Calculate stats
        delay = self.packet_record.calculate_average_delay(self.step_time, VIDEO_PAYLOAD_TYPE)
        loss_ratio = self.packet_record.calculate_loss_ratio(self.step_time, VIDEO_PAYLOAD_TYPE)
        receiving_rate = self.packet_record.calculate_receiving_rate(self.step_time, VIDEO_PAYLOAD_TYPE)
        
        # 2. Derivative features
        delay_gradient = delay - self.prev_delay
        throughput_effective = receiving_rate * (1.0 - loss_ratio)
        
        # 3. Update history
        self.delay_history.append(delay)
        self.recv_rate_history.append(receiving_rate)
        if delay > 0:
            self.min_delay_seen = min(self.min_delay_seen, delay)
            
        # 4. Statistical features
        delay_mean = np.mean(self.delay_history) if len(self.delay_history) > 0 else delay
        delay_std = np.std(self.delay_history) if len(self.delay_history) > 1 else 0.0
        delay_min = self.min_delay_seen if self.min_delay_seen != float('inf') else delay
        queue_delay = max(0, delay - delay_min)
        delay_accel = delay_gradient - self.prev_delay_gradient
        delay_trend = self._calculate_trend(self.delay_history)
        
        loss_change = loss_ratio - self.prev_loss_ratio
        
        bw_utilization = receiving_rate / self.prev_bandwidth if self.prev_bandwidth > 0 else 0.0
        recv_rate_mean = np.mean(self.recv_rate_history) if len(self.recv_rate_history) > 0 else receiving_rate
        recv_rate_std = np.std(self.recv_rate_history) if len(self.recv_rate_history) > 1 else 0.0
        
        # 5. Model Input
        features_raw = np.array([
            delay, loss_ratio, receiving_rate, self.prev_bandwidth, delay_gradient, throughput_effective,
            delay_mean, delay_std, delay_min, queue_delay, delay_accel, delay_trend,
            loss_change,
            bw_utilization, recv_rate_mean, recv_rate_std,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ], dtype=np.float32)
        
        features_norm = self._normalize_features(features_raw)
        
        # 更新特征历史并构造 10-step 序列
        self.feature_history.append(features_norm)
        
        # 构造序列输入 (如果不足 10 step 则进行补零)
        seq_len = self.config.WINDOW_SIZE
        current_features = list(self.feature_history)
        if len(current_features) < seq_len:
            # 补零
            padding = [np.zeros_like(features_norm) for _ in range(seq_len - len(current_features))]
            state_seq = np.array(padding + current_features, dtype=np.float32)
        else:
            state_seq = np.array(current_features, dtype=np.float32)

        input_tensor = torch.from_numpy(state_seq).unsqueeze(0).to(self.device) # [1, 10, 32]
        
        with torch.no_grad():
            output, _ = self.model.predict(input_tensor)
            bandwidth_pred_raw = output.cpu().item()
            
        if bandwidth_pred_raw < 1.0:
            bandwidth_pred = bandwidth_pred_raw * self.config.NORM_STATS['bandwidth_prediction']['max']
        else:
            bandwidth_pred = bandwidth_pred_raw
            
        bandwidth_pred = np.clip(bandwidth_pred, 50000, 10e6)
        
        # Update state
        self.prev_delay = delay
        self.prev_delay_gradient = delay_gradient
        self.prev_loss_ratio = loss_ratio
        self.prev_bandwidth = bandwidth_pred
        self.bandwidth_prediction = int(bandwidth_pred)
        
        return self.bandwidth_prediction

    def _update_model_history_in_background(self):
        """When in slow start, we still need to update model feature history"""
        VIDEO_PAYLOAD_TYPE = 98
        delay = self.packet_record.calculate_average_delay(self.step_time, VIDEO_PAYLOAD_TYPE)
        loss_ratio = self.packet_record.calculate_loss_ratio(self.step_time, VIDEO_PAYLOAD_TYPE)
        receiving_rate = self.packet_record.calculate_receiving_rate(self.step_time, VIDEO_PAYLOAD_TYPE)
        
        delay_gradient = delay - self.prev_delay
        throughput_effective = receiving_rate * (1.0 - loss_ratio)
        
        self.delay_history.append(delay)
        self.recv_rate_history.append(receiving_rate)
        if delay > 0: self.min_delay_seen = min(self.min_delay_seen, delay)
        
        # 计算特征向量用于更新 feature_history
        delay_mean = np.mean(self.delay_history) if len(self.delay_history) > 0 else delay
        delay_std = np.std(self.delay_history) if len(self.delay_history) > 1 else 0.0
        delay_min = self.min_delay_seen if self.min_delay_seen != float('inf') else delay
        queue_delay = max(0, delay - delay_min)
        delay_accel = delay_gradient - self.prev_delay_gradient
        delay_trend = self._calculate_trend(self.delay_history)
        loss_change = loss_ratio - self.prev_loss_ratio
        bw_utilization = receiving_rate / self.prev_bandwidth if self.prev_bandwidth > 0 else 0.0
        recv_rate_mean = np.mean(self.recv_rate_history) if len(self.recv_rate_history) > 0 else receiving_rate
        recv_rate_std = np.std(self.recv_rate_history) if len(self.recv_rate_history) > 1 else 0.0
        
        features_raw = np.array([
            delay, loss_ratio, receiving_rate, self.prev_bandwidth, delay_gradient, throughput_effective,
            delay_mean, delay_std, delay_min, queue_delay, delay_accel, delay_trend,
            loss_change,
            bw_utilization, recv_rate_mean, recv_rate_std,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ], dtype=np.float32)
        
        features_norm = self._normalize_features(features_raw)
        self.feature_history.append(features_norm)
        
        self.prev_delay = delay
        self.prev_delay_gradient = delay_gradient
        self.prev_loss_ratio = loss_ratio

    # --- GCC Logic Implementation ---
    def _run_gcc_logic(self):
        """
        Run one step of GCC logic.
        Returns: (bandwidth, keep_running)
        """
        # Check for loss-based exit first
        loss_rate = self.caculate_loss_rate()
        if loss_rate > 0.02: # 2% loss -> Exit Slow Start
             self.packets_list = [] # Clear buffer
             return self.last_bandwidth_estimation, False

        # Delay-based estimation
        if len(self.packets_list) == 0:
            return self.last_bandwidth_estimation, True

        pkt_group_list = self.divide_packet_group()
        if len(pkt_group_list) < 2:
            return self.last_bandwidth_estimation, True

        send_time_delta_list, _, _, delay_gradient_list = self.compute_deltas_for_pkt_group(pkt_group_list)
        trendline = self.trendline_filter(delay_gradient_list, pkt_group_list)
        
        if trendline is None:
             return self.last_bandwidth_estimation, True

        self.overuse_detector(trendline, sum(send_time_delta_list))
        
        # Check exit condition: OVERUSE
        if self.overuse_flag == 'OVERUSE':
             self.packets_list = [] # Clear buffer
             return self.last_bandwidth_estimation, False
             
        state = self.ChangeState()
        
        # Calculate bandwidth
        bandwidth_estimation = self.rate_adaptation_by_delay(state)
        
        # Clear processed packets
        self.packets_list = []
        
        return bandwidth_estimation, True

    def caculate_loss_rate(self):
        flag = False
        valid_packets_num = 0
        min_sequence_number, max_sequence_number = 0, 0
        if len(self.packets_list) == 0:
            return -1
        for i in range(len(self.packets_list)):
            if self.packets_list[i].payload_type == 98:
                if not flag:
                    min_sequence_number = self.packets_list[i].sequence_number
                    max_sequence_number = self.packets_list[i].sequence_number
                    flag = True
                valid_packets_num += 1
                min_sequence_number = min(min_sequence_number, self.packets_list[i].sequence_number)
                max_sequence_number = max(max_sequence_number, self.packets_list[i].sequence_number)
        if (max_sequence_number - min_sequence_number) == 0:
            return -1
        receive_rate = valid_packets_num / (max_sequence_number - min_sequence_number)
        loss_rate = 1 - receive_rate
        return loss_rate

    def divide_packet_group(self):
        pkt_group_list = []
        if not self.packets_list:
            return []
        first_send_time_in_group = self.packets_list[0].send_timestamp
        pkt_group = [self.packets_list[0]]
        for pkt in self.packets_list[1:]:
            if pkt.send_timestamp - first_send_time_in_group <= kBurstIntervalMs:
                pkt_group.append(pkt)
            else:
                pkt_group_list.append(PacketGroup(pkt_group))
                if self.first_group_complete_time == -1:
                    self.first_group_complete_time = pkt_group[-1].receive_timestamp
                first_send_time_in_group = pkt.send_timestamp
                pkt_group = [pkt]
        return pkt_group_list

    def compute_deltas_for_pkt_group(self, pkt_group_list):
        send_time_delta_list, arrival_time_delta_list, group_size_delta_list, delay_gradient_list = [], [], [], []
        for idx in range(1, len(pkt_group_list)): 
            send_time_delta = pkt_group_list[idx].send_time_list[-1] - pkt_group_list[idx - 1].send_time_list[-1]
            arrival_time_delta = pkt_group_list[idx].arrival_time_list[-1] - pkt_group_list[idx - 1].arrival_time_list[-1]
            group_size_delta = pkt_group_list[idx].pkt_group_size - pkt_group_list[idx - 1].pkt_group_size
            delay = arrival_time_delta - send_time_delta
            self.num_of_deltas_ += 1
            send_time_delta_list.append(send_time_delta)
            arrival_time_delta_list.append(arrival_time_delta)
            group_size_delta_list.append(group_size_delta)
            delay_gradient_list.append(delay)
        return send_time_delta_list, arrival_time_delta_list, group_size_delta_list, delay_gradient_list

    def trendline_filter(self, delay_gradient_list, pkt_group_list):
        trendline = None
        for i, delay_gradient in enumerate(delay_gradient_list):
            accumulated_delay = self.acc_delay + delay_gradient
            smoothed_delay = kTrendlineSmoothingCoeff * self.smoothed_delay + (1 - kTrendlineSmoothingCoeff) * accumulated_delay
            self.acc_delay = accumulated_delay
            self.smoothed_delay = smoothed_delay
            arrival_time_ms = pkt_group_list[i + 1].complete_time
            self.acc_delay_list.append(arrival_time_ms - self.first_group_complete_time)
            self.smoothed_delay_list.append(smoothed_delay)
            if len(self.acc_delay_list) > kTrendlineWindowSize:
                self.acc_delay_list.popleft()
                self.smoothed_delay_list.popleft()
        if len(self.acc_delay_list) == kTrendlineWindowSize:
            avg_acc_delay = sum(self.acc_delay_list) / len(self.acc_delay_list)
            avg_smoothed_delay = sum(self.smoothed_delay_list) / len(self.smoothed_delay_list)
            numerator = 0
            denominator = 0
            for i in range(kTrendlineWindowSize):
                numerator += (self.acc_delay_list[i] - avg_acc_delay) * (self.smoothed_delay_list[i] - avg_smoothed_delay)
                denominator += (self.acc_delay_list[i] - avg_acc_delay) * (self.acc_delay_list[i] - avg_acc_delay)
            trendline = numerator / (denominator + 1e-05)
        else:
            trendline = None
            self.acc_delay_list.clear()
            self.smoothed_delay_list.clear()
            self.acc_delay = 0
            self.smoothed_delay = 0
        return trendline

    def overuse_detector(self, trendline, ts_delta):
        now_ms = self.now_ms
        if self.num_of_deltas_ < 2:
            return
        modified_trend = trendline * min(self.num_of_deltas_, kMinNumDeltas) * threshold_gain_
        if modified_trend > self.gamma1:
            if self.time_over_using == -1:
                self.time_over_using = ts_delta / 2
            else:
                self.time_over_using += ts_delta
            self.overuse_counter += 1
            if self.time_over_using > kOverUsingTimeThreshold and self.overuse_counter > 1:
                if trendline > self.prev_trend:
                    self.time_over_using = 0
                    self.overuse_counter = 0
                    self.overuse_flag = 'OVERUSE'
        elif modified_trend < -self.gamma1:
            self.time_over_using = -1
            self.overuse_counter = 0
            self.overuse_flag = 'UNDERUSE'
        else:
            self.time_over_using = -1
            self.overuse_counter = 0
            self.overuse_flag = 'NORMAL'
        self.prev_trend = trendline
        self.update_threthold(modified_trend, now_ms)

    def update_threthold(self, modified_trend, now_ms):
        if self.last_update_threshold_ms == -1:
            self.last_update_threshold_ms = now_ms
        if abs(modified_trend) > self.gamma1 + kMaxAdaptOffsetMs:
            self.last_update_threshold_ms = now_ms
            return
        if abs(modified_trend) < self.gamma1:
            k = k_down_
        else:
            k = k_up_
        kMaxTimeDeltaMs = 100
        time_delta_ms = min(now_ms - self.last_update_threshold_ms, kMaxTimeDeltaMs)
        self.gamma1 += k * (abs(modified_trend) - self.gamma1) * time_delta_ms
        if (self.gamma1 < 6):
            self.gamma1 = 6
        elif (self.gamma1 > 600):
            self.gamma1 = 600
        self.last_update_threshold_ms = now_ms

    def ChangeState(self):
        overuse_flag = self.overuse_flag
        if overuse_flag == 'NORMAL':
            if self.state == 'Hold':
                self.state = 'Increase'
        elif overuse_flag == 'OVERUSE':
            if self.state != 'Decrease':
                self.state = 'Decrease'
        elif overuse_flag == 'UNDERUSE':
            self.state = 'Hold'
        return self.state

    def rate_adaptation_by_delay(self, state):
        # Calculate throughput
        estimated_throughput = 0
        for pkt in self.packets_list:
            estimated_throughput += pkt.size
        if len(self.packets_list) == 0:
            estimated_throughput_bps = 0
        else:
            time_delta = self.now_ms - self.packets_list[0].receive_timestamp
            time_delta = max(time_delta , Time_Interval)
            estimated_throughput_bps = 1000 * 8 * estimated_throughput / time_delta
        estimated_throughput_kbps = estimated_throughput_bps / 1000
        
        troughput_based_limit = 3 * estimated_throughput_bps + 10
        self.UpdateMaxThroughputEstimate(estimated_throughput_kbps)
        std_max_bit_rate = pow(self.var_max_bitrate_kbps_ * self.avg_max_bitrate_kbps_, 0.5)

        bandwidth_estimation = self.last_bandwidth_estimation
        
        if state == 'Increase':
            if self.avg_max_bitrate_kbps_ >= 0 and estimated_throughput_kbps > self.avg_max_bitrate_kbps_ + 3 * std_max_bit_rate:
                self.avg_max_bitrate_kbps_ = -1.0
                self.rate_control_region_ = "kRcMaxUnknown"

            if self.rate_control_region_ == "kRcNearMax":
                additive_increase_bps = self.AdditiveRateIncrease(self.now_ms, self.time_last_bitrate_change_)
                bandwidth_estimation = self.last_bandwidth_estimation + additive_increase_bps
            elif self.rate_control_region_ == "kRcMaxUnknown":
                multiplicative_increase_bps = self.MultiplicativeRateIncrease(self.now_ms, self.time_last_bitrate_change_)
                bandwidth_estimation = self.last_bandwidth_estimation + multiplicative_increase_bps
            
            bandwidth_estimation = min(bandwidth_estimation, troughput_based_limit)
            self.time_last_bitrate_change_ = self.now_ms
            
        elif state == 'Decrease':
            beta = 0.85
            bandwidth_estimation = beta * estimated_throughput_bps + 0.5
            if bandwidth_estimation > self.last_bandwidth_estimation:
                if self.rate_control_region_ != "kRcMaxUnknown":
                    bandwidth_estimation = (beta * self.avg_max_bitrate_kbps_ * 1000 + 0.5)
                bandwidth_estimation = min(bandwidth_estimation, self.last_bandwidth_estimation)
            self.rate_control_region_ = "kRcNearMax"
            if estimated_throughput_kbps < self.avg_max_bitrate_kbps_-3*std_max_bit_rate:
                self.avg_max_bitrate_kbps_ = -1
            self.UpdateMaxThroughputEstimate(estimated_throughput_kbps)
            self.state='Hold'
            self.time_last_bitrate_change_ = self.now_ms
            
        elif state == 'Hold':
            bandwidth_estimation = self.last_bandwidth_estimation
            
        return bandwidth_estimation

    def AdditiveRateIncrease(self, now_ms, last_ms):
        sum_packet_size = 0
        for pkt in self.packets_list:
            sum_packet_size += pkt.size
        avg_packet_size = 8 * sum_packet_size / len(self.packets_list)
        beta = 0.0
        if self.timer_delta is None and len(self.packets_list) > 0:
            pkt = self.packets_list[-1]
            self.timer_delta = -(pkt.receive_timestamp - pkt.send_timestamp)
        response_time = 200
        if last_ms > 0:
            beta = min(((now_ms - last_ms) / response_time), 1.0)
        additive_increase_bps = max(800, beta * avg_packet_size)
        return additive_increase_bps

    def MultiplicativeRateIncrease(self, now_ms, last_ms):
        alpha = 1.08
        if last_ms > -1:
            time_since_last_update_ms = min(now_ms - last_ms, 1000)
            alpha = pow(alpha, time_since_last_update_ms / 1000)
        multiplicative_increase_bps = max(self.last_bandwidth_estimation * (alpha - 1.0), 1000.0)
        return multiplicative_increase_bps

    def UpdateMaxThroughputEstimate(self, estimated_throughput_kbps):
        alpha = 0.05
        if self.avg_max_bitrate_kbps_ == -1:
            self.avg_max_bitrate_kbps_ = estimated_throughput_kbps
        else:
            self.avg_max_bitrate_kbps_ = (1 - alpha) * self.avg_max_bitrate_kbps_ + alpha * estimated_throughput_kbps
        norm = max(self.avg_max_bitrate_kbps_, 1.0)
        var_value = pow((self.avg_max_bitrate_kbps_ - estimated_throughput_kbps), 2) / norm
        self.var_max_bitrate_kbps_ = (1 - alpha) * self.var_max_bitrate_kbps_ + alpha * var_value
        if self.var_max_bitrate_kbps_ < 0.4:
            self.var_max_bitrate_kbps_ = 0.4
        if self.var_max_bitrate_kbps_ > 2.5:
            self.var_max_bitrate_kbps_ = 2.5

    def _normalize_features(self, features_raw):
        features_norm = features_raw.copy()
        norm_stats = self.config.NORM_STATS
        feature_names = [
            'delay', 'loss_ratio', 'receiving_rate', 'prev_bandwidth', 'delay_gradient', 'throughput_effective',
            'delay_mean', 'delay_std', 'delay_min', 'queue_delay', 'delay_accel', 'delay_trend',
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
        return features_norm
    
    def _calculate_trend(self, data_history):
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
