#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPO (Proximal Policy Optimization) Agent
用于在线强化学习的 PPO 算法实现
"""
import torch
import numpy as np
import time
from .actor_critic import ActorCritic


class RolloutStorage:
    """经验回放存储"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.logprobs = []
        self.values = []
        self.returns = []
        self.dones = []
    
    def clear(self):
        """清空存储"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.logprobs = []
        self.values = []
        self.returns = []
        self.dones = []
    
    def compute_returns(self, gamma=0.99):
        """计算折扣回报"""
        returns = []
        discounted_reward = 0
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + gamma * discounted_reward
            returns.insert(0, discounted_reward)
        self.returns = returns


class PPO:
    """
    PPO Agent
    实现 Proximal Policy Optimization 算法用于在线学习
    """
    def __init__(self, state_dim, action_dim=1, exploration_param=0.1, 
                 lr=3e-4, betas=(0.9, 0.999), gamma=0.99, 
                 ppo_epoch=10, ppo_clip=0.2, use_gae=False):
        """
        初始化 PPO Agent
        Args:
            state_dim: 状态维度（32）
            action_dim: 动作维度（1，带宽调整系数）
            exploration_param: 探索参数（动作分布标准差）
            lr: 学习率
            betas: Adam 优化器的动量参数
            gamma: 折扣因子
            ppo_epoch: PPO 更新迭代次数
            ppo_clip: PPO 裁剪参数
            use_gae: 是否使用 GAE（Generalized Advantage Estimation）
        """
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.ppo_clip = ppo_clip
        self.ppo_epoch = ppo_epoch
        self.use_gae = use_gae
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建策略网络
        self.policy = ActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            exploration_param=exploration_param,
            device=self.device
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), 
            lr=lr, 
            betas=betas
        )
        
        # 旧策略网络（用于计算重要性采样比率）
        self.policy_old = ActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            exploration_param=exploration_param,
            device=self.device
        ).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
    
    def select_action(self, state, storage):
        """
        选择动作
        Args:
            state: 当前状态 [state_dim] 或 [1, state_dim]
            storage: RolloutStorage 对象
        Returns:
            action: 选择的动作（标量值）
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)  # [1, state_dim]
        
        # 使用旧策略网络选择动作
        action, action_logprobs, value, action_mean = self.policy_old.forward(state)
        
        # 存储经验
        storage.states.append(state)
        storage.actions.append(action)
        storage.logprobs.append(action_logprobs)
        storage.values.append(value)
        
        # 返回动作值（标量）
        if isinstance(action, torch.Tensor):
            return action.cpu().numpy()[0] if action.dim() > 0 else action.cpu().item()
        return action
    
    def get_value(self, state):
        """
        获取状态价值
        Args:
            state: 当前状态
        Returns:
            value: 状态价值
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        _, _, value, _ = self.policy_old.forward(state)
        return value.cpu().item()
    
    def update(self, storage):
        """
        更新策略网络
        Args:
            storage: RolloutStorage 对象，包含收集的经验
        Returns:
            policy_loss: 策略损失
            value_loss: 价值损失
        """
        if len(storage.states) == 0:
            return 0.0, 0.0
        
        # 计算回报
        storage.compute_returns(self.gamma)
        
        # 转换为张量
        old_states = torch.cat(storage.states, dim=0).to(self.device).detach()
        old_actions = torch.cat(storage.actions, dim=0).to(self.device).detach()
        old_logprobs = torch.cat(storage.logprobs, dim=0).to(self.device).detach()
        old_values = torch.cat(storage.values, dim=0).to(self.device).detach()
        returns = torch.FloatTensor(storage.returns).to(self.device).unsqueeze(1).detach()
        
        # 计算优势
        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO 更新
        total_policy_loss = 0.0
        total_value_loss = 0.0
        
        for epoch in range(self.ppo_epoch):
            # 评估当前策略
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # 计算重要性采样比率
            ratios = torch.exp(logprobs - old_logprobs)
            
            # PPO 裁剪目标
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.ppo_clip, 1 + self.ppo_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean() - 0.01 * dist_entropy  # 添加熵正则化
            
            # 价值损失
            value_loss = 0.5 * (state_values - returns).pow(2).mean()
            
            # 总损失
            loss = policy_loss + value_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)  # 梯度裁剪
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
        
        # 更新旧策略网络
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        return total_policy_loss / self.ppo_epoch, total_value_loss / self.ppo_epoch

    def save(self, checkpoint_path):
        """保存模型"""
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        """加载模型"""
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    
    def save_model(self, filepath):
        """保存模型"""
        torch.save(self.policy.state_dict(), filepath)
    
    def load_model(self, filepath):
        """加载模型"""
        self.policy.load_state_dict(torch.load(filepath, map_location=self.device))
        self.policy_old.load_state_dict(self.policy.state_dict())
