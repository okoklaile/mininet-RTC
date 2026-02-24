#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural-GCC-NoBC 带宽估计器
消融实验变体1：纯 PPO 在线学习（无 BC-GCC 预训练，无 KL 约束）
- 不使用预训练的 BC-GCC 模型，随机初始化
- 完全依赖 PPO 在线学习
- 无 KL 散度约束
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
from model import GCCBC_LSTM, Critic
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
    """Neural-GCC-NoBC 带宽估计器 (纯 PPO，无 BC-GCC 预训练)"""
    
    def __init__(self, model_path="/home/wyq/桌面/mininet-RTC/ccalgs/BC-GCC/trial3.pt", 
                 step_time=200, 
                 use_slow_start=False,
                 use_rl=True,
                 update_frequency=32,
                 inference_only=False,
                 use_bc_model=False,
                 kl_weight=0.0):
        """
        初始化估计器
        Args:
            inference_only (bool): 如果为True，只加载最佳模型进行推理，不进行训练
            use_bc_model (bool): 是否使用 BC-GCC 预训练模型（NoBC 版本固定为 False）
            kl_weight (float): KL 约束权重（NoBC 版本固定为 0）
        """
        self.inference_only = inference_only
        self.use_bc_model = use_bc_model
        self.kl_weight = kl_weight
        
        # ... Logging init ...
        logging.basicConfig(level=logging.INFO)
        global logger
        logger = logging.getLogger("NeuralGCC-NoBC")

        # 1. 初始化模型（随机初始化，不加载预训练权重）
        self.config = Config32D()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 随机初始化 Base Model
        self.base_model = GCCBC_LSTM(self.config)
        self.base_model.to(self.device)
        self.base_model.train()  # 训练模式
        
        # 无参考模型（不使用 KL 约束）
        self.ref_model = None
        
        logger.info(f"✅ Base Model 随机初始化成功（无预训练）")

        # 2. 初始化 PPO (纯在线学习，无预训练)
        self.use_rl = use_rl
        if self.use_rl:
            # 纯 PPO 模式：直接更新 self.base_model 的参数
            # 由于没有预训练，使用更高的学习率
            
            # 开启 Base Model 的梯度
            for param in self.base_model.parameters():
                param.requires_grad = True
                
            # 初始化 Critic
            self.critic = Critic(self.config).to(self.device)
            
            # 定义 Optimizer (针对 base_model 和 critic)
            # 使用更高的学习率（因为没有预训练）
            self.optimizer = torch.optim.Adam([
                {'params': self.base_model.parameters(), 'lr': 1e-4},  # 更高的学习率
                {'params': self.critic.parameters(), 'lr': 1e-3}
            ])
            
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
            
            if self.inference_only:
                # 推理模式：尝试加载 best_checkpoint.pth
                best_path = os.path.join(self.save_dir, "best_checkpoint.pth")
                if os.path.exists(best_path):
                    logger.info(f"🚀 Loading BEST model for Inference: {best_path}")
                    checkpoint = torch.load(best_path, map_location=self.device)
                    self.base_model.load_state_dict(checkpoint['model_state_dict'])
                    self.base_model.eval()
                    # 确保参数冻结
                    for param in self.base_model.parameters():
                        param.requires_grad = False
                    # 关闭 RL 训练标志，防止触发 update
                    self.use_rl = False 
                else:
                    logger.warning("⚠️ No best_checkpoint.pth found! Falling back to Base Model.")
                    self.use_rl = False # 依然关闭训练
            else:
                # 训练模式：加载最新的 checkpoint
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
        self.feature_history = deque(maxlen=self.config.WINDOW_SIZE)
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
        
        # 新增 QoE 状态存储
        self.last_qoe_stats = {}
        self.last_freeze_duration = 0
        self.cached_qoe_penalty = 0.0 # 缓存的 QoE 惩罚，用于填补 1s 的空窗期
        self.last_render_fps = 0.0  # 缓存渲染帧率

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
        self.feature_history.clear()
        self.min_delay_seen = float('inf')
        
        # Reset QoE state
        self.last_qoe_stats = {}
        self.last_freeze_duration = 0
        self.cached_qoe_penalty = 0.0
        self.last_render_fps = 0.0
        
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
            self.last_qoe_stats = stats
            
            # 缓存 render_fps
            self.last_render_fps = stats.get("renderFps", 0.0)
            
            # --- 处理 QoE 稀疏性问题 (1s 一次) ---
            # 我们在收到报告时立即计算“惩罚强度”，并在接下来的 1s 内持续生效，
            # 直到下一次报告到来更新这个强度。
            
            # 1. 计算卡顿增量
            total_freeze = stats.get("totalFreezesDurationMs", 0)
            freeze_delta = max(0, total_freeze - self.last_freeze_duration)
            self.last_freeze_duration = total_freeze 
            
            # 2. 计算惩罚值 (归一化)
            # 如果这一秒内发生了卡顿，我们认为网络状态极差，
            # 这个惩罚应该持续作用于接下来的每一步，直到收到“不再卡顿”的消息。
            p_freeze = 0.0
            if freeze_delta > 0:
                p_freeze = min(freeze_delta / 100.0, 1.0) # 100ms 卡顿即满罚
                
            # 3. 计算端到端延迟惩罚 (重点惩罚非网络延迟部分，如编解码、渲染排队)
            e2e_delay = stats.get("e2eDelayMs", 0)
            network_delay = stats.get("networkDelayMs", 0)
            
            # system_delay = e2e - network (代表系统内部处理耗时，如 JitterBuffer + 解码 + 渲染)
            # 如果 system_delay 过大，说明接收端处理不过来了，应该降低码率
            system_delay = max(0, e2e_delay - network_delay)
            
            p_e2e = 0.0
            # 设定阈值：例如系统处理延迟超过 100ms 开始惩罚
            if system_delay > 100:
                p_e2e = min((system_delay - 100) / 200.0, 1.0)
                
            # 4. 存储总惩罚 (权重已包含在内)
            # reward公式里是: -2.0 * p_e2e - 5.0 * p_freeze
            self.cached_qoe_penalty = 10 * p_e2e + 10 * p_freeze
            
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
        
        # 获取 QoE 特征: Jitter Buffer
        # 归一化: 假设最大 1000ms, 映射到 0-1
        jitter_buffer_ms = self.last_qoe_stats.get("jitterBufferMs", 0.0)
        jitter_buffer_norm = min(jitter_buffer_ms / 1000.0, 1.0)
        
        # 获取 QoE 特征: Render FPS
        # 归一化: 假设最大 60fps, 映射到 0-1
        render_fps_norm = min(self.last_render_fps / 60.0, 1.0)
        
        # 构造特征向量
        features_raw = np.array([
            delay, loss_ratio, receiving_rate, self.prev_bandwidth, delay_gradient, throughput_effective,
            delay_mean, delay_std, delay_min, queue_delay, delay_accel, delay_trend,
            loss_change,
            bw_utilization, recv_rate_mean, recv_rate_std,
            jitter_buffer_norm, render_fps_norm, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ], dtype=np.float32)
        
        features_norm = self._normalize_features(features_raw)
        
        # 手动覆盖 jitter 和 render_fps (因为 _normalize_features 可能会跳过它们或把它们清零)
        features_norm[16] = jitter_buffer_norm
        features_norm[17] = render_fps_norm
        
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

        # 2. PPO Fine-tuning 逻辑 (仅当 use_rl=True 且不是 inference_only 时)
        if self.use_rl and not self.inference_only:
            # 计算上一步的奖励并存储经验
            if self.last_state is not None:
                reward = self._calculate_reward(throughput_effective, loss_ratio, delay, 
                                              self.last_bandwidth_estimation, self.prev_prev_bandwidth)
                
                # 存储: (state, action, reward, next_state, done, log_prob)
                # 注意：这里的 state 是 10-step 序列
                self.storage.append((self.last_state, self.last_action, reward, state_seq, False, self.last_log_prob))
                self.step_counter += 1
                
                # 执行 PPO 更新
                if self.step_counter >= self.update_frequency:
                    self._update_policy()
                    self.storage = []
                    self.step_counter = 0
                    self._check_and_save_model()

        # 3. 前向传播 (Base Model 作为 Actor)
        # 对序列进行 Mask (只保留前 18 维特征: 16 原有 + 1 Jitter + 1 RenderFPS)
        input_seq = state_seq.copy()
        input_seq[:, 18:] = 0.0 
        
        input_tensor = torch.from_numpy(input_seq).unsqueeze(0).to(self.device) # [1, 10, 32]
        
        # 采样动作 (Sampling for exploration)
        if self.use_rl and not self.inference_only:
            # 在训练模式下，我们需要从分布中采样，而不是直接取均值
            mu, _ = self.base_model.forward(input_tensor) # [1, 1]
            
            # 使用一个固定或可学习的 log_std
            if not hasattr(self, 'log_std'):
                # 降低初始噪声 (std ≈ 0.05)，减少盲目探索
                self.log_std = torch.full((1, 1), -3.0).to(self.device)
            
            std = torch.exp(self.log_std)
            dist = torch.distributions.Normal(mu, std)
            
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            bw_norm = action.item()
            
            # 保存状态用于下一步
            self.last_state = state_seq
            self.last_action = action.detach() # 存储 Tensor
            self.last_log_prob = log_prob.detach()
            
        else:
            # 推理模式，直接取确定性输出 (包括 inference_only 模式)
            with torch.no_grad():
                output, _ = self.base_model.predict(input_tensor)
                bw_norm = output.cpu().item()

        # 4. 反归一化
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
        
        # 获取 QoE 特征: Jitter Buffer
        jitter_buffer_ms = self.last_qoe_stats.get("jitterBufferMs", 0.0)
        jitter_buffer_norm = min(jitter_buffer_ms / 1000.0, 1.0)
        
        # 获取 QoE 特征: Render FPS
        render_fps_norm = min(self.last_render_fps / 60.0, 1.0)
        
        features_raw = np.array([
            delay, loss_ratio, receiving_rate, self.prev_bandwidth, delay_gradient, throughput_effective,
            delay_mean, delay_std, delay_min, queue_delay, delay_accel, delay_trend,
            loss_change,
            bw_utilization, recv_rate_mean, recv_rate_std,
            jitter_buffer_norm, render_fps_norm, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ], dtype=np.float32)
        
        features_norm = self._normalize_features(features_raw)
        features_norm[16] = jitter_buffer_norm
        features_norm[17] = render_fps_norm
        
        self.feature_history.append(features_norm)
        
        self.prev_delay = delay
        self.prev_delay_gradient = delay_gradient
        self.prev_loss_ratio = loss_ratio

    def _update_policy(self):
        """执行 PPO 更新"""
        if len(self.storage) == 0: return
        
        # 整理数据
        states = torch.FloatTensor([x[0] for x in self.storage]).to(self.device)
        actions = torch.cat([x[1] for x in self.storage]).view(-1, 1)
        rewards = torch.FloatTensor([x[2] for x in self.storage]).to(self.device).view(-1, 1)
        next_states = torch.FloatTensor([x[3] for x in self.storage]).to(self.device)
        dones = torch.FloatTensor([x[4] for x in self.storage]).to(self.device).view(-1, 1)
        old_log_probs = torch.cat([x[5] for x in self.storage]).view(-1, 1)
        
        # Mask input for states (consistent with forward)
        # 允许前 18 维 (0-17)
        states[:, :, 18:] = 0.0
        next_states[:, :, 18:] = 0.0
        
        # --- GAE (Generalized Advantage Estimation) 计算 ---
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)
            
            deltas = rewards + self.gamma * next_values * (1 - dones) - values
            
            advantages = torch.zeros_like(rewards)
            gae = 0
            for t in reversed(range(len(rewards))):
                gae = deltas[t] + self.gamma * 0.95 * gae * (1 - dones[t]) # lambda=0.95
                advantages[t] = gae
            
            returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO Epochs
        # 必须切换到 train 模式才能进行 backward
        self.base_model.train()
        self.critic.train()
        
        for _ in range(5):
            # 新的分布
            mu, _ = self.base_model.forward(states)
            
            std = torch.exp(self.log_std)
            dist = torch.distributions.Normal(mu, std)
            
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # 无 KL 约束（NoBC 版本）
            
            # Ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Surrogate Loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic Loss (Value Loss)
            new_values = self.critic(states)
            critic_loss = 0.5 * ((new_values - returns) ** 2).mean()
            
            # Total Loss: 无 KL 约束，增加熵正则化鼓励探索
            loss = actor_loss + 0.5 * critic_loss - 0.05 * entropy
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        # 更新完毕后切回 eval 模式，以免影响 inference 时的行为 (如 dropout, batchnorm)
        self.base_model.eval()
        self.critic.eval()
        
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
                    'critic_state_dict': self.critic.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'log_std': self.log_std,
                    'best_reward': self.best_reward
                }, save_path)
                logger.info(f"🌟 New Best Reward: {self.best_reward:.2f}, saved to best_checkpoint.pth")
            except Exception as e:
                logger.error(f"❌ Failed to save best model: {e}")


    def _calculate_reward(self, receiving_rate, loss_ratio, delay, current_prediction, last_prediction):
        """
        参考简洁版奖励函数重构
        核心思想：全局归一化 + 线性组合
        """
        
        # 1. 吞吐量 (Log-scale, 0~1)
        # 调整策略：降低高带宽带来的奖励收益，抑制模型盲目追求高带宽
        # 方法：增加 Log 函数的底数或直接降低权重，让边际收益递减得更快
        min_bw = 80000.0
        max_bw = self.config.NORM_STATS['receiving_rate']['max']
        
        def liner_to_log(val):
            val_mbps = np.clip(val / 1e6, min_bw / 1e6, max_bw / 1e6)
            # 使用 Log10 代替 Loge，使得高带宽的奖励增长更平缓
            log_val = np.log10(val_mbps)
            log_min, log_max = np.log10(min_bw / 1e6), np.log10(max_bw / 1e6)
            return (log_val - log_min) / (log_max - log_min)

        # 降低吞吐量奖励的权重：从 1.0 (隐式) 降为 0.5
        # 这样即使带宽翻倍，奖励也只增加一点点，不值得为此冒险增加延迟
        r_tp = 0.5 * liner_to_log(receiving_rate)
        
        # 2. 延迟 (Linear, 0~1)
        # 目标：让延迟接近最小延迟 -> 重点惩罚排队延迟 (Queue Delay)
        # queue_delay = current_delay - min_delay_seen
        min_delay = self.min_delay_seen if self.min_delay_seen != float('inf') else 0
        queue_delay = max(0, delay - min_delay)
        
        # 归一化：我们设置一个 100ms 的排队容忍上限。
        # 意味着如果排队延迟超过 100ms，惩罚项 p_delay 就会达到最大值 1.0
        # 这样会强迫模型将排队延迟控制在 0~100ms 之间，尽可能接近 0
        p_delay = min(queue_delay / 30.0, 1.0)
        
        # 3. 丢包 (Linear, 0~1)
        # 参考逻辑：直接使用 loss_ratio
        p_loss = loss_ratio
        
        # 4. 稳定性 (Log-scale, 0~1)
        # 参考逻辑：liner_to_log(delta_prediction)
        delta_prediction = abs(current_prediction - last_prediction)
        p_stability = liner_to_log(delta_prediction)
        
        # 5. QoE 惩罚 (使用缓存的持续惩罚值)
        # 解决了 QoE 报告稀疏 (1s) 而决策频繁 (200ms) 的问题。
        # 如果上一秒报告了卡顿，这个惩罚会一直存在，迫使模型在这一整秒内都保持谨慎。
        penalty_qoe = self.cached_qoe_penalty

        # --- 最终奖励组合 ---
        # reward = r_tp - 10*p_delay - 5*p_loss - 0.1*p_stability - penalty_qoe
        
        reward = r_tp - 10.0 * p_delay - 10 * p_loss - 10 * p_stability - penalty_qoe
        
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
                    'critic_state_dict': self.critic.state_dict(),
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
            
            if 'critic_state_dict' in checkpoint and hasattr(self, 'critic'):
                self.critic.load_state_dict(checkpoint['critic_state_dict'])
                
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
