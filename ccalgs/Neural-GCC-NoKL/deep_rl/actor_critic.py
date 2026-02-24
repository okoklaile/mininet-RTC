#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Actor-Critic 网络架构
基于 LSTM 的 Actor-Critic，适配 32 维特征输入（包括 QoE 特征）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


class ActorCritic(nn.Module):
    """
    Actor-Critic 网络
    使用 LSTM 处理时序特征，输出动作（带宽调整系数）和状态价值
    """
    def __init__(self, state_dim, lstm_hidden_size=128, lstm_num_layers=2, 
                 action_dim=1, exploration_param=0.1, device="cpu"):
        super(ActorCritic, self).__init__()
        
        self.state_dim = state_dim  # 32维特征
        self.action_dim = action_dim
        self.device = device
        
        # LSTM 层（共享特征提取）
        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=0.1 if lstm_num_layers > 1 else 0
        )
        
        # Actor 网络（策略网络）
        self.actor_fc1 = nn.Linear(lstm_hidden_size, 64)
        self.actor_fc2 = nn.Linear(64, 32)
        self.actor_output = nn.Linear(32, action_dim)
        
        # Critic 网络（价值网络）
        self.critic_fc1 = nn.Linear(lstm_hidden_size, 64)
        self.critic_fc2 = nn.Linear(64, 32)
        self.critic_output = nn.Linear(32, 1)
        
        # 动作分布的标准差（探索参数）
        self.action_var = torch.full((action_dim,), exploration_param**2).to(device)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0.0)
    
    def forward(self, state):
        """
        前向传播
        Args:
            state: [batch, seq_len, state_dim] 或 [batch, state_dim]
        Returns:
            action: 动作（带宽调整系数）
            action_logprobs: 动作的对数概率
            value: 状态价值
            action_mean: 动作均值
        """
        # 处理输入维度
        if state.dim() == 2:
            state = state.unsqueeze(1)  # [batch, 1, state_dim]
        
        # LSTM 特征提取
        lstm_out, _ = self.lstm(state)  # [batch, seq_len, hidden_size]
        lstm_features = lstm_out[:, -1, :]  # 取最后一个时间步 [batch, hidden_size]
        
        # Actor 分支（策略网络）
        actor_x = F.relu(self.actor_fc1(lstm_features))
        actor_x = F.relu(self.actor_fc2(actor_x))
        action_mean = torch.sigmoid(self.actor_output(actor_x))  # [batch, action_dim]，范围 [0, 1]
        
        # 动作分布
        action_std = torch.sqrt(self.action_var).expand_as(action_mean)
        dist = Normal(action_mean, action_std)
        
        # 采样动作（训练时）或使用均值（推理时）
        action = action_mean  # 在线学习时使用确定性动作
        action_logprobs = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        # Critic 分支（价值网络）
        critic_x = F.relu(self.critic_fc1(lstm_features))
        critic_x = F.relu(self.critic_fc2(critic_x))
        value = self.critic_output(critic_x)  # [batch, 1]
        
        return action.detach(), action_logprobs, value, action_mean
    
    def evaluate(self, state, action):
        """
        评估给定状态和动作的对数概率和价值
        Args:
            state: [batch, seq_len, state_dim] 或 [batch, state_dim]
            action: [batch, action_dim]
        Returns:
            action_logprobs: 动作的对数概率
            state_values: 状态价值
            dist_entropy: 分布熵（用于正则化）
        """
        # 处理输入维度
        if state.dim() == 2:
            state = state.unsqueeze(1)
        
        # LSTM 特征提取
        lstm_out, _ = self.lstm(state)
        lstm_features = lstm_out[:, -1, :]
        
        # Actor 分支
        actor_x = F.relu(self.actor_fc1(lstm_features))
        actor_x = F.relu(self.actor_fc2(actor_x))
        action_mean = torch.sigmoid(self.actor_output(actor_x))
        
        # 动作分布
        action_std = torch.sqrt(self.action_var).expand_as(action_mean)
        dist = Normal(action_mean, action_std)
        
        action_logprobs = dist.log_prob(action).sum(dim=-1, keepdim=True)
        dist_entropy = dist.entropy().sum(dim=-1).mean()
        
        # Critic 分支
        critic_x = F.relu(self.critic_fc1(lstm_features))
        critic_x = F.relu(self.critic_fc2(critic_x))
        state_values = self.critic_output(critic_x)
        
        return action_logprobs, state_values, dist_entropy
