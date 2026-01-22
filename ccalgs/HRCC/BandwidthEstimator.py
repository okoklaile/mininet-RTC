#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HRCC (Hybrid Reinforcement learning and rule-based Congestion Control) 带宽估计器
结合GCC启发式算法和PPO强化学习模型的混合带宽估计方案
- 基础层：使用GCC算法进行带宽估计
- 优化层：使用PPO强化学习模型调整GCC的估计结果
"""
from deep_rl.ppo_agent import PPO
import torch
from packet_info import PacketInfo
from packet_record import PacketRecord
from BandwidthEstimator_gcc import GCCEstimator


class Estimator(object):
    """
    混合带宽估计器
    结合GCC基线算法和PPO强化学习模型，实现更智能的带宽估计
    """
    def __init__(self, model_path="./model/pretrained_model.pth", step_time=200):
        """
        初始化混合带宽估计器
        Args:
            model_path: 预训练模型路径
            step_time: 时间步长(毫秒)，默认200ms
        """
        # 1. 定义PPO强化学习模型相关参数
        exploration_param = 0.1  # 动作分布的标准差（探索参数）
        K_epochs = 37  # 策略更新的迭代次数
        ppo_clip = 0.1  # PPO的裁剪参数，限制策略更新幅度
        gamma = 0.99  # 折扣因子，用于计算未来奖励的权重
        lr = 3e-5  # Adam优化器的学习率
        betas = (0.9, 0.999)  # Adam优化器的动量参数
        self.state_dim = 6  # 状态维度：接收率、延迟、丢包率、带宽预测、过载距离、上次过载容量
        self.state_length = 10  # 状态历史长度，保留最近10个时间步的状态
        action_dim = 1  # 动作维度：带宽调整系数
        
        # 2. 加载预训练的PPO模型
        self.device = torch.device("cpu")
        self.ppo = PPO(self.state_dim, self.state_length, action_dim, exploration_param, lr, betas, gamma, K_epochs, ppo_clip)
        self.ppo.policy.load_state_dict(torch.load('/home/wyq/桌面/mininet-RTC/ccalgs/HRCC/hrcc.pth'))
        
        # 初始化数据包记录器（用于统计网络指标）
        self.packet_record = PacketRecord()
        self.packet_record.reset()
        self.step_time = step_time  # 时间步长(ms)
        
        # 3. 初始化状态和控制变量
        self.state = torch.zeros((1, self.state_dim, self.state_length))  # 状态张量 [batch, features, time_steps]
        self.time_to_guide = False  # 是否到达RL指导时机
        self.counter = 0  # 时间步计数器
        self.bandwidth_prediction = 300000  # 带宽预测值(bps)，初始300kbps
        
        # 初始化GCC基线估计器
        self.gcc_estimator = GCCEstimator()
        
        # 网络指标历史列表
        self.receiving_rate_list = []  # 接收率历史
        self.delay_list = []  # 延迟历史
        self.loss_ratio_list = []  # 丢包率历史
        self.bandwidth_prediction_list = []  # 带宽预测历史
        
        # 过载检测相关
        self.overuse_flag = 'NORMAL'  # 过载标志：'NORMAL', 'OVERUSE', 'UNDERUSE'
        self.overuse_distance = 5  # 距离上次过载的时间步数
        self.last_overuse_cap = 1000000  # 上次发生过载时的接收率(bps)

    def report_states(self, stats: dict):
        """
        接收并记录数据包信息
        将数据包信息同时传递给packet_record和gcc_estimator进行处理
        
        Args:
            stats: 数据包统计信息字典，包含以下字段：
        {
                "send_time_ms": uint,        # 发送时间戳(毫秒)
                "arrival_time_ms": uint,     # 到达时间戳(毫秒)
                "payload_type": int,         # 载荷类型
                "sequence_number": uint,     # 序列号
                "ssrc": int,                 # 同步源标识符
                "padding_length": uint,      # 填充长度(字节)
                "header_length": uint,       # 头部长度(字节)
                "payload_size": uint         # 载荷大小(字节)
        }
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
        # 更新gcc_estimator用于基线带宽估计
        self.gcc_estimator.report_states(stats)

    def get_estimated_bandwidth(self)->int:
        """
        计算并返回最终的带宽估计值
        混合方案：每4个时间步使用一次RL调整，其余时间使用GCC估计
        
        工作流程：
        1. 计算当前网络状态（接收率、延迟、丢包率等）
        2. 获取GCC的基线带宽估计
        3. 更新状态张量
        4. 每4步使用PPO模型调整GCC估计（其余时间直接使用GCC估计）
        
        Returns:
            bandwidth_prediction: 最终的带宽预测值(bps)
        """
        # 1. 计算当前时间窗口的网络状态指标
        # 视频包的 payload_type 为 125
        VIDEO_PAYLOAD_TYPE = 125
        
        # 计算接收率(bps) - 只统计视频包
        self.receiving_rate = self.packet_record.calculate_receiving_rate(interval=self.step_time, filter_payload_type=VIDEO_PAYLOAD_TYPE)
        self.receiving_rate_list.append(self.receiving_rate)
        
        # 计算平均延迟(ms) - 只统计视频包
        self.delay = self.packet_record.calculate_average_delay(interval=self.step_time, filter_payload_type=VIDEO_PAYLOAD_TYPE)
        self.delay_list.append(self.delay)

        # 计算丢包率(0.0-1.0) - 只统计视频包
        self.loss_ratio = self.packet_record.calculate_loss_ratio(interval=self.step_time, filter_payload_type=VIDEO_PAYLOAD_TYPE)
        self.loss_ratio_list.append(self.loss_ratio)

        # 获取GCC估计器的带宽估计和过载状态
        self.gcc_decision, self.overuse_flag = self.gcc_estimator.get_estimated_bandwidth()
        
        # 更新过载距离和上次过载容量
        if self.overuse_flag == 'OVERUSE':
            self.overuse_distance = 0  # 刚发生过载，距离重置为0
            self.last_overuse_cap = self.receiving_rate  # 记录过载时的接收率
        else:
            self.overuse_distance += 1  # 距离上次过载又过了一步
        
        # 更新状态张量（滑动窗口）
        self.state = self.state.clone().detach()
        self.state = torch.roll(self.state, -1, dims=-1)  # 向左滚动，丢弃最旧的状态

        # 填充最新的状态（归一化到[0,1]范围）
        self.state[0, 0, -1] = self.receiving_rate / 6000000.0  # 接收率归一化（假设最大6Mbps）
        self.state[0, 1, -1] = self.delay / 1000.0  # 延迟归一化（假设最大1000ms）
        self.state[0, 2, -1] = self.loss_ratio  # 丢包率已经在[0,1]范围
        self.state[0, 3, -1] = self.bandwidth_prediction / 6000000.0  # 带宽预测归一化
        self.state[0, 4, -1] = self.overuse_distance / 100.0  # 过载距离归一化
        self.state[0, 5, -1] = self.last_overuse_cap / 6000000.0  # 上次过载容量归一化

        # 维护历史列表长度
        if len(self.receiving_rate_list) == self.state_length:
            self.receiving_rate_list.pop(0)
            self.delay_list.pop(0)
            self.loss_ratio_list.pop(0)

        # 更新计数器
        self.counter += 1
        
        # 每4步触发一次RL指导
        if self.counter % 4 == 0:
            self.time_to_guide = True
            self.counter = 0

        # 2. 使用RL智能体调整GCC的带宽估计
        # 临时禁用 RL，只使用 GCC 基线（调试用）
        USE_RL = True # 设置为 True 启用 RL，False 只使用 GCC
        
        if USE_RL and self.time_to_guide == True:
            # 使用PPO策略网络预测动作
            action, _, _, _ = self.ppo.policy.forward(self.state)
            # action范围约为[0,1]，映射到调整系数 2^(2*action-1)，范围约为[0.5, 2]
            # 这样可以实现对GCC估计的缩放调整
            self.bandwidth_prediction = self.gcc_decision * pow(2, (2 * action - 1))
            # 更新GCC估计器的带宽值，保持一致性
            self.gcc_estimator.change_bandwidth_estimation(self.bandwidth_prediction)
            self.time_to_guide = False
        else:
            # 非指导时间步或禁用RL时，直接使用GCC估计
            self.bandwidth_prediction = self.gcc_decision

        return self.bandwidth_prediction
