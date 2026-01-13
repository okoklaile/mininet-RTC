#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLP版本的Actor-Critic网络
使用全连接层（MLP）处理状态，适用于序列状态输入
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class ActorCriticMLP(nn.Module):
    """
    MLP版本的Actor-Critic网络
    使用多层感知机处理状态序列
    """
    def __init__(self, state_dim, state_length, action_dim, exploration_param=0.1, device="cpu"):
        """
        初始化MLP Actor-Critic网络
        Args:
            state_dim: 状态特征维度
            state_length: 状态序列长度（历史时间步数）
            action_dim: 动作维度
            exploration_param: 探索参数（动作分布的标准差）
            device: 计算设备
        """
        super(ActorCriticMLP, self).__init__()
        
        self.state_dim = state_dim
        self.state_length = state_length
        self.action_dim = action_dim
        self.device = device
        
        # 输入维度：state_dim * state_length（展平后的状态）
        input_dim = state_dim * state_length
        
        # Actor网络（策略网络）
        self.actor_fc1 = nn.Linear(input_dim, 256)
        self.actor_fc2 = nn.Linear(256, 256)
        self.actor_fc3 = nn.Linear(256, 128)
        self.actor_output = nn.Linear(128, action_dim)
        
        # Critic网络（价值网络）
        self.critic_fc1 = nn.Linear(input_dim, 256)
        self.critic_fc2 = nn.Linear(256, 256)
        self.critic_fc3 = nn.Linear(256, 128)
        self.critic_output = nn.Linear(128, 1)
        
        # 动作分布参数
        self.action_var = torch.full((action_dim,), exploration_param**2).to(self.device)
        self.random_action = False  # 是否使用随机采样（训练时True，推理时False）
        
    def forward(self, state):
        """
        前向传播
        Args:
            state: 状态张量 [batch_size, state_dim, state_length] 或 [batch_size, state_dim * state_length]
        Returns:
            action: 动作值
            action_logprobs: 动作对数概率
            value: 状态值估计
            action_mean: 动作均值
        """
        # 展平状态
        if len(state.shape) == 3:
            # [batch, state_dim, state_length] -> [batch, state_dim * state_length]
            state_flat = state.view(state.shape[0], -1)
        else:
            state_flat = state
            
        # Actor网络
        actor_x = F.relu(self.actor_fc1(state_flat))
        actor_x = F.relu(self.actor_fc2(actor_x))
        actor_x = F.relu(self.actor_fc3(actor_x))
        action_mean = torch.sigmoid(self.actor_output(actor_x))  # 输出[0,1]范围
        
        # 创建动作分布
        action_std = torch.sqrt(self.action_var).to(self.device)
        dist = Normal(action_mean, action_std)
        
        # 采样动作
        if self.random_action:
            action = dist.sample()
        else:
            action = action_mean  # 确定性动作（推理时）
            
        action_logprobs = dist.log_prob(action).sum(dim=-1)  # 对数概率
        
        # Critic网络
        critic_x = F.relu(self.critic_fc1(state_flat))
        critic_x = F.relu(self.critic_fc2(critic_x))
        critic_x = F.relu(self.critic_fc3(critic_x))
        value = self.critic_output(critic_x)
        
        return action.detach(), action_logprobs.detach(), value, action_mean.detach()
    
    def evaluate(self, state, action):
        """
        评估给定状态和动作的对数概率和熵
        Args:
            state: 状态张量
            action: 动作张量
        Returns:
            action_logprobs: 动作对数概率
            value: 状态值估计
            dist_entropy: 分布熵
        """
        # 展平状态
        if len(state.shape) == 3:
            state_flat = state.view(state.shape[0], -1)
        else:
            state_flat = state
            
        # Actor网络
        actor_x = F.relu(self.actor_fc1(state_flat))
        actor_x = F.relu(self.actor_fc2(actor_x))
        actor_x = F.relu(self.actor_fc3(actor_x))
        action_mean = torch.sigmoid(self.actor_output(actor_x))
        
        # 创建动作分布
        action_std = torch.sqrt(self.action_var).to(self.device)
        dist = Normal(action_mean, action_std)
        
        action_logprobs = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1).mean()
        
        # Critic网络
        critic_x = F.relu(self.critic_fc1(state_flat))
        critic_x = F.relu(self.critic_fc2(critic_x))
        critic_x = F.relu(self.critic_fc3(critic_x))
        value = self.critic_output(critic_x)
        
        return action_logprobs, torch.squeeze(value), dist_entropy
