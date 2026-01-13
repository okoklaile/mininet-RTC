#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RL-DelayGCC 带宽估计器
结合GCC基线算法和PPO强化学习在线训练
- 使用PPO+KL+MLP进行在线学习
- 支持BC（Behavior Cloning）模型作为参考策略
- 每步都进行RL决策，定期更新策略
"""

import torch
import numpy as np
import os
from packet_info import PacketInfo
from packet_record import PacketRecord
from BandwidthEstimator_gcc import GCCEstimator
from deep_rl import PPO, ActorCriticMLP, Storage


class Estimator(object):
    """
    RL-DelayGCC带宽估计器
    结合GCC基线和PPO在线强化学习
    """
    def __init__(self, model_path=None, bc_model_path=None, step_time=60, 
                 mode='train', save_model_path=None, save_frequency=500):
        """
        初始化RL-DelayGCC估计器
        Args:
            model_path: PPO模型路径（可选，用于加载预训练模型）
            bc_model_path: BC模型路径（可选，用于KL约束的参考策略）
            step_time: 时间步长(毫秒)，默认60ms
            mode: 运行模式，'train'（训练模式）或 'inference'（推理模式），默认'train'
            save_model_path: 模型保存路径（可选），如果提供则定期保存模型
            save_frequency: 模型保存频率（每多少步保存一次），默认500步
        """
        # 1. 定义PPO强化学习模型相关参数
        exploration_param = 0.1  # 动作分布的标准差（探索参数）
        K_epochs = 10  # 策略更新的迭代次数
        ppo_clip = 0.2  # PPO的裁剪参数
        gamma = 0.99  # 折扣因子
        lr = 3e-4  # 学习率（在线学习可以稍大）
        betas = (0.9, 0.999)  # Adam优化器参数
        self.state_dim = 6  # 状态维度：接收率、延迟、丢包率、带宽预测、过载距离、上次过载容量
        self.state_length = 10  # 状态历史长度
        action_dim = 1  # 动作维度：带宽调整系数
        
        # KL散度约束参数
        kl_coef = 0.1  # KL散度惩罚系数
        kl_target = 0.01  # KL散度目标值
        
        # 2. 初始化PPO Agent
        self.device = torch.device("cpu")
        self.ppo = PPO(
            self.state_dim, self.state_length, action_dim, 
            exploration_param, lr, betas, gamma, K_epochs, ppo_clip,
            kl_coef=kl_coef, kl_target=kl_target, device=self.device
        )
        
        # 加载PPO预训练模型（如果提供）
        if model_path is not None:
            self.ppo.policy.load_state_dict(torch.load(model_path, map_location=self.device))
            self.ppo.policy_old.load_state_dict(self.ppo.policy.state_dict())
            print(f"Loaded PPO model from {model_path}")
        
        # 3. 初始化参考策略（用于KL约束）
        # 可以是BC模型，暂时初始化为与policy相同
        self.ref_policy = ActorCriticMLP(
            self.state_dim, self.state_length, action_dim, 
            exploration_param, self.device
        ).to(self.device)
        self.ref_policy.load_state_dict(self.ppo.policy.state_dict())
        self.ref_policy.eval()
        
        # 加载BC模型作为参考策略（如果提供）
        if bc_model_path is not None:
            self.ref_policy.load_state_dict(torch.load(bc_model_path, map_location=self.device))
            self.ref_policy.eval()
            print(f"Loaded BC model from {bc_model_path} as reference policy")
        
        # 4. 初始化经验存储Buffer
        self.storage = Storage()
        
        # 5. 初始化数据包记录器和GCC估计器
        self.packet_record = PacketRecord()
        self.packet_record.reset()
        self.step_time = step_time  # 时间步长(ms)
        
        # 6. 初始化状态和控制变量
        self.state = torch.zeros((1, self.state_dim, self.state_length))  # 状态张量
        self.counter = 0  # 时间步计数器
        self.bandwidth_prediction = 300000  # 带宽预测值(bps)，初始300kbps
        
        # 初始化GCC基线估计器
        self.gcc_estimator = GCCEstimator()
        
        # 网络指标历史列表
        self.receiving_rate_list = []  # 接收率历史
        self.delay_list = []  # 延迟历史
        self.loss_ratio_list = []  # 丢包率历史
        
        # 过载检测相关
        self.overuse_flag = 'NORMAL'  # 过载标志
        self.overuse_distance = 5  # 距离上次过载的时间步数
        self.last_overuse_cap = 1000000  # 上次发生过载时的接收率(bps)
        
        # 奖励计算相关
        self.last_receiving_rate = 0
        self.last_delay = 0
        self.last_loss_ratio = 0
        
        # 更新频率控制
        self.update_frequency = 100  # 每100步更新一次策略
        
        # 训练/推理模式设置
        self.mode = mode  # 'train' 或 'inference'
        if self.mode == 'inference':
            self.ppo.policy.eval()  # 推理模式下设置为评估模式
            self.ppo.policy.random_action = False  # 推理时使用确定性动作
            print("Running in INFERENCE mode - no training, deterministic actions")
        else:
            self.ppo.policy.train()  # 训练模式下设置为训练模式
            print("Running in TRAIN mode - online learning enabled")
        
        # 模型保存设置
        self.save_model_path = save_model_path
        self.save_frequency = save_frequency
        self.save_counter = 0  # 保存计数器
        
    def report_states(self, stats: dict):
        """
        接收并记录数据包信息
        Args:
            stats: 数据包统计信息字典
        """
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

        # 更新packet_record和gcc_estimator
        self.packet_record.on_receive(packet_info)
        self.gcc_estimator.report_states(stats)
    
    def calculate_reward(self, receiving_rate, delay, loss_ratio):
        """
        计算奖励函数
        奖励设计：鼓励高吞吐量、低延迟、低丢包率
        Args:
            receiving_rate: 接收率(bps)
            delay: 延迟(ms)
            loss_ratio: 丢包率(0-1)
        Returns:
            reward: 奖励值
        """
        # 归一化参数
        max_rate = 6000000  # 6Mbps
        max_delay = 1000  # 1000ms
        
        # 吞吐量奖励（归一化到[0,1]）
        throughput_reward = min(receiving_rate / max_rate, 1.0)
        
        # 延迟惩罚（归一化到[0,1]）
        delay_penalty = min(delay / max_delay, 1.0)
        
        # 丢包惩罚
        loss_penalty = loss_ratio
        
        # 综合奖励（权重可调）
        reward = 0.5 * throughput_reward - 0.3 * delay_penalty - 0.2 * loss_penalty
        
        return reward
    
    def get_estimated_bandwidth(self) -> int:
        """
        计算并返回最终的带宽估计值
        在线训练流程：
        1. 计算当前网络状态
        2. 使用PPO选择动作
        3. 应用动作调整GCC估计
        4. 计算奖励并存储
        5. 定期更新策略
        
        Returns:
            bandwidth_prediction: 最终的带宽预测值(bps)
        """
        # 1. 计算当前时间窗口的网络状态指标
        self.receiving_rate = self.packet_record.calculate_receiving_rate(interval=self.step_time)
        self.receiving_rate_list.append(self.receiving_rate)
        
        self.delay = self.packet_record.calculate_average_delay(interval=self.step_time)
        self.delay_list.append(self.delay)

        self.loss_ratio = self.packet_record.calculate_loss_ratio(interval=self.step_time)
        self.loss_ratio_list.append(self.loss_ratio)

        # 获取GCC估计器的带宽估计和过载状态
        self.gcc_decision, self.overuse_flag = self.gcc_estimator.get_estimated_bandwidth()
        
        # 更新过载距离和上次过载容量
        if self.overuse_flag == 'OVERUSE':
            self.overuse_distance = 0
            self.last_overuse_cap = self.receiving_rate
        else:
            self.overuse_distance += 1
        
        # 更新状态张量（滑动窗口）
        self.state = self.state.clone().detach()
        self.state = torch.roll(self.state, -1, dims=-1)

        # 填充最新的状态（归一化到[0,1]范围）
        self.state[0, 0, -1] = self.receiving_rate / 6000000.0  # 接收率归一化
        self.state[0, 1, -1] = self.delay / 1000.0  # 延迟归一化
        self.state[0, 2, -1] = self.loss_ratio  # 丢包率
        self.state[0, 3, -1] = self.bandwidth_prediction / 6000000.0  # 带宽预测归一化
        self.state[0, 4, -1] = self.overuse_distance / 100.0  # 过载距离归一化
        self.state[0, 5, -1] = self.last_overuse_cap / 6000000.0  # 上次过载容量归一化

        # 维护历史列表长度
        if len(self.receiving_rate_list) == self.state_length:
            self.receiving_rate_list.pop(0)
            self.delay_list.pop(0)
            self.loss_ratio_list.pop(0)

        # 2. RL决策逻辑
        # 转换状态为Tensor
        state_tensor = self.state  # [1, state_dim, state_length]
        
        # 选择动作
        if self.mode == 'train':
            # 训练模式：存储经验用于后续更新
            action = self.ppo.select_action(state_tensor, self.storage)
        else:
            # 推理模式：只进行决策，不存储经验
            action = self.ppo.select_action_inference(state_tensor)
        action_value = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        
        # 3. 执行动作（应用到GCC估计）
        # action范围[0,1]，映射到调整系数 2^(2*action-1)，范围约为[0.5, 2]
        multiplier = pow(2, (2 * action_value - 1))
        # RL以GCC为基线的带宽估计
        rl_adjusted_bwe = self.gcc_decision * multiplier
        
        # 获取GCC基于丢包率的带宽估计
        gcc_loss_bwe = self.gcc_estimator.get_estimated_bandwidth_by_loss()
        
        # 最终带宽 = min{RL调整后的带宽, GCC的loss_base带宽}
        self.bandwidth_prediction = int(min(rl_adjusted_bwe, gcc_loss_bwe))
        
        # 更新GCC估计器的带宽值，保持一致性
        self.gcc_estimator.change_bandwidth_estimation(self.bandwidth_prediction)
        
        # 4. 训练模式下的经验收集和策略更新
        if self.mode == 'train':
            # 计算并存储奖励（上一帧动作产生的后果）
            # 注意：第一个动作没有奖励，需要等待下一帧才能计算
            if len(self.storage.actions) > 1:
                # 计算上一帧的奖励（基于当前网络状态，这是上一帧动作的结果）
                r = self.calculate_reward(self.receiving_rate, self.delay, self.loss_ratio)
                self.storage.rewards.append(r)
            
            # 保存当前状态用于下一帧计算奖励
            self.last_receiving_rate = self.receiving_rate
            self.last_delay = self.delay
            self.last_loss_ratio = self.loss_ratio
            
            # 5. 触发策略更新
            self.counter += 1
            if len(self.storage.actions) >= self.update_frequency:
                # 为最后一个动作添加奖励（使用当前网络状态）
                if len(self.storage.rewards) < len(self.storage.actions):
                    final_reward = self.calculate_reward(self.receiving_rate, self.delay, self.loss_ratio)
                    self.storage.rewards.append(final_reward)
                
                # 补充最后一个状态的值（用于计算Advantage）
                next_value = self.ppo.get_value(state_tensor)
                
                # 调用PPO更新（支持KL散度约束）
                policy_loss, value_loss, kl_loss = self.ppo.update(
                    self.storage, next_value, self.ref_policy
                )
                
                # 清空Buffer
                self.storage.clear()
                
                # 打印更新日志
                print(f"PPO Updated! Policy Loss: {policy_loss:.4f}, "
                      f"Value Loss: {value_loss:.4f}, KL Loss: {kl_loss:.4f}")
                
                # 模型保存（如果设置了保存路径）
                if self.save_model_path is not None:
                    self.save_counter += 1
                    if self.save_counter >= self.save_frequency:
                        self.save_model()
                        self.save_counter = 0
        
        return self.bandwidth_prediction
    
    def save_model(self, path=None):
        """
        保存当前PPO模型
        Args:
            path: 保存路径，如果为None则使用初始化时设置的save_model_path
        """
        import time
        if path is None:
            if self.save_model_path is None:
                print("Warning: No save path specified, model not saved")
                return
            # 确保目录存在
            os.makedirs(os.path.dirname(self.save_model_path) if os.path.dirname(self.save_model_path) else '.', exist_ok=True)
            # 生成带时间戳的文件名
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            base_name = os.path.splitext(self.save_model_path)[0]
            path = f"{base_name}_step{self.counter}_{timestamp}.pth"
        
        torch.save(self.ppo.policy.state_dict(), path)
        print(f"Model saved to {path}")
    
    def set_mode(self, mode):
        """
        切换训练/推理模式
        Args:
            mode: 'train' 或 'inference'
        """
        if mode not in ['train', 'inference']:
            raise ValueError(f"Mode must be 'train' or 'inference', got {mode}")
        
        self.mode = mode
        if mode == 'inference':
            self.ppo.policy.eval()
            self.ppo.policy.random_action = False
            print("Switched to INFERENCE mode")
        else:
            self.ppo.policy.train()
            self.ppo.policy.random_action = True
            print("Switched to TRAIN mode")
