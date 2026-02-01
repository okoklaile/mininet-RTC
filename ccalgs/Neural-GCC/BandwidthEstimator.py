#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural-GCC 带宽估计器
基于深度学习模型 (BC-GCC) + PPO 在线强化学习
"""
import sys
import os
import torch
import numpy as np
from collections import deque
import collections
import logging

# 将当前目录添加到路径，以便导入本地模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from packet_info import PacketInfo
from packet_record import PacketRecord
from model import GCCBC_LSTM
from config import Config
from deep_rl.ppo_agent import PPO

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    """Neural-GCC 带宽估计器 (BC-GCC + PPO)"""
    
    def __init__(self, model_path="/home/wyq/桌面/mininet-RTC/ccalgs/Neural-GCC/trial3.pt", 
                 step_time=200, use_rl=True, update_frequency=4, use_slow_start=True):
        """
        初始化估计器
        Args:
            model_path: PyTorch模型路径 (Base Model)
            step_time: 时间步长(毫秒)，默认200ms
            use_rl: 是否启用强化学习 (PPO)
            update_frequency: PPO 更新频率 (多少个step更新一次)
            use_slow_start: 是否启用GCC慢启动
        """
        # 1. 加载 Base Model (BC-GCC)
        self.config = Config32D()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载 Base Model 权重
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.base_model = GCCBC_LSTM(self.config)
            self.base_model.load_state_dict(checkpoint['model_state_dict'])
            self.base_model.to(self.device)
            self.base_model.eval()
            # 冻结 Base Model 参数
            for param in self.base_model.parameters():
                param.requires_grad = False
            logger.info(f"✅ Base Model 加载成功 (Epoch {checkpoint['epoch']})")
        except Exception as e:
            logger.error(f"❌ Base Model 加载失败: {e}")
            raise e

        # 2. 初始化 PPO (用于 Fine-tuning)
        self.use_rl = use_rl
        if self.use_rl:
            # 在 Fine-tuning 模式下，我们直接更新 self.base_model 的参数
            # PPO Agent 负责计算 Policy Gradient，但直接作用于 base_model
            
            # 开启 Base Model 的梯度
            for param in self.base_model.parameters():
                param.requires_grad = True
                
            # 定义 Optimizer (针对 base_model)
            self.optimizer = torch.optim.Adam(self.base_model.parameters(), lr=1e-5) # 使用较小的学习率进行微调
            
            # PPO 超参数
            self.ppo_clip = 0.2
            self.gamma = 0.99
            self.update_frequency = update_frequency
            self.step_counter = 0
            self.storage = [] # 存储 (s, a, r, s', done, log_prob)
            
            # 模型保存路径
            self.save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            else:
                self._load_latest_checkpoint()

        # 3. 初始化状态和历史
        self.packet_record = PacketRecord()
        self.packet_record.reset()
        self.step_time = step_time
        
        self.prev_delay = 0.0
        self.prev_delay_gradient = 0.0
        self.prev_loss_ratio = 0.0
        self.prev_bandwidth = 300000.0
        self.bandwidth_prediction = 300000.0
        self.prev_prev_bandwidth = 300000.0
        
        self.delay_history = deque(maxlen=self.config.WINDOW_SIZE)
        self.recv_rate_history = deque(maxlen=self.config.WINDOW_SIZE)
        self.feature_buffer = deque(maxlen=self.config.WINDOW_SIZE)
        self.min_delay_seen = float('inf')
        
        # 4. RL 状态变量
        self.last_state = None
        self.last_action = None
        self.last_log_prob = None
        self.last_value = None
        
        # QoE 指标缓存 (用于计算奖励)
        self.prev_throughput = 0.0
        self.prev_loss = 0.0
        self.prev_rtt = 0.0

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

    def reset(self):
        """重置状态"""
        self.packet_record.reset()
        self.prev_delay = 0.0
        self.prev_delay_gradient = 0.0
        self.prev_loss_ratio = 0.0
        self.prev_bandwidth = 300000.0
        self.bandwidth_prediction = 300000.0
        self.prev_prev_bandwidth = 300000.0
        self.delay_history.clear()
        self.recv_rate_history.clear()
        self.feature_buffer.clear()
        self.min_delay_seen = float('inf')
        
        if self.use_rl:
            self.last_state = None
            self.last_action = None
            self.storage = []
            self.step_counter = 0
            
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
        """接收数据包报告"""
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
        
        # Calculate size for GCC
        packet_info.size = stats["header_length"] + stats["payload_size"] + stats["padding_length"]
        
        self.now_ms = packet_info.receive_timestamp
        
        self.packet_record.on_receive(packet_info)
        
        # 更新packets_list (GCC用)
        if self.in_slow_start:
            self.packets_list.append(packet_info)

    def get_estimated_bandwidth(self) -> int:
        """核心逻辑：GCC Slow Start -> Fine-tuning PPO"""
        # Sync bandwidth state
        self.prev_prev_bandwidth = self.last_bandwidth_estimation
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
                logger.info(f"🐌 Exiting Slow Start. Switching to Neural-GCC Model.")
                self.in_slow_start = False
                # Fall through to model logic

        VIDEO_PAYLOAD_TYPE = 98 
        
        # 1. 计算特征
        delay = self.packet_record.calculate_average_delay(self.step_time, VIDEO_PAYLOAD_TYPE)
        loss_ratio = self.packet_record.calculate_loss_ratio(self.step_time, VIDEO_PAYLOAD_TYPE)
        receiving_rate = self.packet_record.calculate_receiving_rate(self.step_time, VIDEO_PAYLOAD_TYPE)
        
        delay_gradient = delay - self.prev_delay
        throughput_effective = receiving_rate * (1.0 - loss_ratio)
        
        self.delay_history.append(delay)
        self.recv_rate_history.append(receiving_rate)
        if delay > 0: self.min_delay_seen = min(self.min_delay_seen, delay)
        
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
        
        # 构造特征向量
        features_raw = np.array([
            delay, loss_ratio, receiving_rate, self.prev_bandwidth, delay_gradient, throughput_effective,
            delay_mean, delay_std, delay_min, queue_delay, delay_accel, delay_trend,
            loss_change,
            bw_utilization, recv_rate_mean, recv_rate_std,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ], dtype=np.float32)
        
        features_norm = self._normalize_features(features_raw)
        
        # 2. 更新序列缓冲区 (解决推理与 BC 训练序列长度不一致导致的震荡)
        self.feature_buffer.append(features_norm)
        
        # 构造序列输入 [WINDOW_SIZE, feature_dim]
        if len(self.feature_buffer) < self.config.WINDOW_SIZE:
            # 填充: 用第一帧补齐
            padding = [self.feature_buffer[0]] * (self.config.WINDOW_SIZE - len(self.feature_buffer))
            features_seq = np.array(padding + list(self.feature_buffer), dtype=np.float32)
        else:
            features_seq = np.array(self.feature_buffer, dtype=np.float32)

        # 3. PPO Fine-tuning 逻辑
        if self.use_rl:
            # 计算上一步的奖励并存储经验
            if self.last_state is not None:
                reward = self._calculate_reward(throughput_effective, loss_ratio, delay, 
                                              self.last_bandwidth_estimation, self.prev_prev_bandwidth)
                
                # 存储: (state, action, reward, next_state, done, log_prob)
                # 注意：这里的 state 和 next_state 都是 [WINDOW_SIZE, dim] 的序列
                self.storage.append((self.last_state, self.last_action, reward, features_seq, False, self.last_log_prob))
                self.step_counter += 1
                
                # 执行 PPO 更新
                if self.step_counter >= self.update_frequency:
                    self._update_policy()
                    self.storage = []
                    self.step_counter = 0
                    self._check_and_save_model()

        # 4. 前向传播 (Base Model 作为 Actor)
        # 即使在训练，也使用 Masked Input (为了保持与预训练时的一致性)
        features_input_seq = features_seq.copy()
        features_input_seq[:, 16:] = 0.0 
        
        input_tensor = torch.from_numpy(features_input_seq).unsqueeze(0).to(self.device) # [1, 10, 32]
        
        # 采样动作 (Sampling for exploration)
        if self.use_rl:
            # 在训练模式下，我们需要从分布中采样
            mu, _ = self.base_model.forward(input_tensor) # [1, 1]
            
            # 使用一个固定或可学习的 log_std
            if not hasattr(self, 'log_std'):
                # 进一步减小初始 sigma 到 0.05 (log(0.05) ≈ -3.0)
                self.log_std = torch.full((1, 1), -3.0).to(self.device)
            
            std = torch.exp(self.log_std)
            dist = torch.distributions.Normal(mu, std)
            
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            bw_norm = action.item()
            
            # 保存状态用于下一步 (保存整个序列)
            self.last_state = features_seq
            self.last_action = action.detach() 
            self.last_log_prob = log_prob.detach()
            
        else:
            # 推理模式
            with torch.no_grad():
                output, _ = self.base_model.predict(input_tensor)
                bw_norm = output.cpu().item()

        # 5. 反归一化
        if bw_norm < 10.0: # 简单的阈值判断是否为归一化值
            final_bw = bw_norm * self.config.NORM_STATS['bandwidth_prediction']['max']
        else:
            final_bw = bw_norm
            
        # 5. 后处理
        final_bw = np.clip(final_bw, 50000, 10e6)
        
        # 更新历史
        self.prev_delay = delay
        self.prev_delay_gradient = delay_gradient
        self.prev_loss_ratio = loss_ratio
        self.prev_bandwidth = final_bw
        self.bandwidth_prediction = int(final_bw)
        
        return self.bandwidth_prediction

    def _update_model_history_in_background(self):
        """When in slow start, we still need to update model feature history"""
        VIDEO_PAYLOAD_TYPE = 98
        delay = self.packet_record.calculate_average_delay(self.step_time, VIDEO_PAYLOAD_TYPE)
        loss_ratio = self.packet_record.calculate_loss_ratio(self.step_time, VIDEO_PAYLOAD_TYPE)
        receiving_rate = self.packet_record.calculate_receiving_rate(self.step_time, VIDEO_PAYLOAD_TYPE)
        delay_gradient = delay - self.prev_delay
        
        self.delay_history.append(delay)
        self.recv_rate_history.append(receiving_rate)
        if delay > 0:
            self.min_delay_seen = min(self.min_delay_seen, delay)
            
        self.prev_delay = delay
        self.prev_delay_gradient = delay_gradient
        self.prev_loss_ratio = loss_ratio

    def _update_policy(self):
        """执行 PPO 更新"""
        if len(self.storage) == 0: return
        
        # 整理数据
        states = torch.FloatTensor([x[0] for x in self.storage]).to(self.device)
        # actions = torch.stack([x[1] for x in self.storage]).squeeze()
        actions = torch.cat([x[1] for x in self.storage]).view(-1, 1)
        rewards = torch.FloatTensor([x[2] for x in self.storage]).to(self.device).view(-1, 1)
        # next_states = torch.FloatTensor([x[3] for x in self.storage]).to(self.device)
        old_log_probs = torch.cat([x[5] for x in self.storage]).view(-1, 1)
        
        # Mask input for states (consistent with forward)
        # states shape: [batch, WINDOW_SIZE, feature_dim]
        states[:, :, 16:] = 0.0
        # states = states.unsqueeze(1) # 不再需要，因为已经是 [batch, 10, 32]
        
        # 计算优势函数 (Advantage) - 简化版：Advantage = Reward (假设 baseline=0)
        # 更严谨的做法需要一个 Critic Network 来估计 Value Function
        # 这里为了微调简单，我们可以直接用 Reward 作为 Advantage，或者用 Normalize 后的 Reward
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # PPO Epochs
        # 必须切换到 train 模式才能进行 backward
        self.base_model.train()
        
        for _ in range(5):
            # 新的分布
            mu, _ = self.base_model.forward(states)
            # mu = mu.squeeze(1) # [batch, 1]
            
            std = torch.exp(self.log_std)
            dist = torch.distributions.Normal(mu, std)
            
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Surrogate Loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Total Loss
            loss = actor_loss - 0.01 * entropy
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        # 更新完毕后切回 eval 模式，以免影响 inference 时的行为 (如 dropout, batchnorm)
        self.base_model.eval()
        
        avg_reward = rewards.mean().item()
        logger.info(f"🔄 PPO Update: Loss={loss.item():.4f}, Avg Reward={avg_reward:.2f}")

        # 保存历史最佳模型
        if not hasattr(self, 'best_reward'):
            self.best_reward = -float('inf')

        if avg_reward > self.best_reward:
            self.best_reward = avg_reward
            save_path = os.path.join(self.save_dir, "best_checkpoint.pth")
            try:
                torch.save({
                    'model_state_dict': self.base_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'log_std': self.log_std,
                    'best_reward': self.best_reward
                }, save_path)
                logger.info(f"🌟 New Best Reward: {self.best_reward:.2f}, saved to best_checkpoint.pth")
            except Exception as e:
                logger.error(f"❌ Failed to save best model: {e}")


    def _calculate_reward(self, receiving_rate, loss_ratio, delay, current_prediction, last_prediction):
        """
        在线强化学习奖励函数 (平滑版：解决剧烈震荡和过度下调问题)
        """
        
        # 1. 吞吐量收益归一化
        min_bw = 80000.0
        max_bw = self.config.NORM_STATS['receiving_rate']['max']
        
        def liner_to_log(val):
            val_mbps = np.clip(val / 1000000.0, min_bw / 1000000.0, max_bw / 1000000.0)
            log_val = np.log(val_mbps)
            log_min, log_max = np.log(min_bw / 1000000.0), np.log(max_bw / 1000000.0)
            return (log_val - log_min) / (log_max - log_min)

        r_tp = liner_to_log(receiving_rate)
        
        # 2. 延迟惩罚 (引入容忍窗口)
        # 150ms 以内不惩罚，500ms 达到最大惩罚，避免在操作点附近反复横跳
        if delay < 150.0:
            p_delay = 0.0
        else:
            p_delay = min((delay - 150.0) / 350.0, 1.0)
        
        # 3. 丢包惩罚 (稍微回调权重)
        p_loss = loss_ratio
        
        # 4. 过估计惩罚 (增加容忍度)
        # 给模型 15% 的波动空间，避免微小的测量误差导致 BWE 暴跌
        p_overestimation = 0.0
        if current_prediction > receiving_rate * 1.15 and receiving_rate > 0:
            over_ratio = (current_prediction - receiving_rate) / receiving_rate
            p_overestimation = min(over_ratio, 1.0)
        
        # 5. 稳定性惩罚 (显著加重权重)
        # 震荡太厉害，必须强力要求模型保持 BWE 连续性
        delta_prediction = abs(current_prediction - last_prediction)
        p_stability = liner_to_log(delta_prediction)
        
        # --- 最终奖励组合 ---
        # 权重配比：吞吐量(1.0) vs 延迟(1.5) vs 丢包(4.0) vs 过估计(0.3) vs 稳定性(0.5)
        reward = r_tp - 1.5 * p_delay - 4.0 * p_loss - 0.3 * p_overestimation - 0.5 * p_stability
        
        return round(float(reward), 4)

    def _check_and_save_model(self):
        """定期保存模型"""
        if not hasattr(self, 'update_count'):
            self.update_count = 0
        self.update_count += 1
        
        if self.update_count % 100 == 0:
            filename = f"ppo_checkpoint_{self.update_count}.pth"
            save_path = os.path.join(self.save_dir, filename)
            try:
                # Save both model state dict and optimizer state
                torch.save({
                    'model_state_dict': self.base_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'log_std': self.log_std,
                    'update_count': self.update_count
                }, save_path)
                logger.info(f"💾 Model saved: {filename}")
            except Exception as e:
                logger.error(f"❌ Failed to save model: {e}")

    def _load_latest_checkpoint(self):
        """加载最新的 Checkpoint"""
        try:
            checkpoints = [f for f in os.listdir(self.save_dir) if f.startswith('ppo_checkpoint_') and f.endswith('.pth')]
            if not checkpoints: return
            
            latest_cp = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            cp_path = os.path.join(self.save_dir, latest_cp)
            
            checkpoint = torch.load(cp_path, map_location=self.device)
            self.base_model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint and hasattr(self, 'optimizer'):
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'log_std' in checkpoint:
                self.log_std = checkpoint['log_std'].to(self.device)
            if 'update_count' in checkpoint:
                self.update_count = checkpoint['update_count']
                
            logger.info(f"📂 Loaded checkpoint: {latest_cp}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load checkpoint: {e}")

    def _normalize_features(self, features_raw):
        """与 BC-GCC 保持一致的归一化逻辑"""
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
        if len(data_history) < 2: return 0.0
        y = np.array(data_history)
        x = np.arange(len(y))
        n = len(x)
        denominator = n * np.sum(x*x) - np.sum(x)**2
        if abs(denominator) < 1e-10: return 0.0
        slope = (n * np.sum(x*y) - np.sum(x) * np.sum(y)) / denominator
        return slope

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
