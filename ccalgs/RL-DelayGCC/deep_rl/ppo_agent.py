#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPO Agent with KL Divergence Constraint
支持KL散度约束的PPO算法实现
"""

import torch
import torch.nn.functional as F
from torch.distributions import kl_divergence, Normal
from .actor_critic_mlp import ActorCriticMLP


class PPO:
    """
    PPO算法实现，支持KL散度约束
    用于防止策略更新过大，保持训练稳定性
    """
    def __init__(self, state_dim, state_length, action_dim, exploration_param, lr, betas, gamma, 
                 ppo_epoch, ppo_clip, kl_coef=0.1, kl_target=0.01, use_gae=False, device="cpu"):
        """
        初始化PPO Agent
        Args:
            state_dim: 状态维度
            state_length: 状态序列长度
            action_dim: 动作维度
            exploration_param: 探索参数
            lr: 学习率
            betas: Adam优化器的beta参数
            gamma: 折扣因子
            ppo_epoch: PPO更新迭代次数
            ppo_clip: PPO裁剪参数
            kl_coef: KL散度惩罚系数
            kl_target: KL散度目标值
            use_gae: 是否使用GAE
            device: 计算设备
        """
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.ppo_clip = ppo_clip
        self.ppo_epoch = ppo_epoch
        self.use_gae = use_gae
        self.kl_coef = kl_coef  # KL散度惩罚系数
        self.kl_target = kl_target  # KL散度目标值
        
        self.device = torch.device(device)
        
        # 创建策略网络
        self.policy = ActorCriticMLP(state_dim, state_length, action_dim, exploration_param, self.device).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        
        # 旧策略网络（用于计算重要性采样比率）
        self.policy_old = ActorCriticMLP(state_dim, state_length, action_dim, exploration_param, self.device).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.policy_old.eval()
        
    def select_action(self, state, storage):
        """
        选择动作并存储相关信息
        Args:
            state: 当前状态
            storage: 经验存储对象
        Returns:
            action: 选择的动作
        """
        state = torch.FloatTensor(state).to(self.device)
        self.policy_old.random_action = True  # 训练时使用随机采样
        action, action_logprobs, value, action_mean = self.policy_old.forward(state)
        
        # 存储到buffer
        storage.logprobs.append(action_logprobs)
        storage.values.append(value)
        storage.states.append(state)
        storage.actions.append(action)
        
        return action.cpu().numpy()
    
    def get_value(self, state):
        """
        获取状态值估计
        Args:
            state: 状态
        Returns:
            value: 状态值
        """
        state = torch.FloatTensor(state).to(self.device)
        self.policy_old.random_action = False
        _, _, value, _ = self.policy_old.forward(state)
        return value
    
    def compute_kl_divergence(self, old_mean, old_std, new_mean, new_std):
        """
        计算KL散度 D_KL(new || old)
        Args:
            old_mean: 旧策略的动作均值
            old_std: 旧策略的动作标准差
            new_mean: 新策略的动作均值
            new_std: 新策略的动作标准差
        Returns:
            kl: KL散度值
        """
        # 确保标准差为正
        old_std = torch.clamp(old_std, min=1e-6)
        new_std = torch.clamp(new_std, min=1e-6)
        
        old_dist = Normal(old_mean, old_std)
        new_dist = Normal(new_mean, new_std)
        # kl_divergence(new_dist, old_dist) = D_KL(new || old)
        kl = kl_divergence(new_dist, old_dist)
        # 如果是多维动作，需要sum
        if len(kl.shape) > 1:
            kl = kl.sum(dim=-1)
        return kl.mean()
    
    def update(self, storage, next_value, ref_policy=None):
        """
        更新策略网络
        Args:
            storage: 经验存储对象
            next_value: 下一个状态的值估计
            ref_policy: 参考策略（用于KL约束，可以是BC模型）
        Returns:
            policy_loss: 策略损失
            value_loss: 价值损失
            kl_loss: KL散度损失
        """
        # 计算回报
        storage.compute_returns(next_value, self.gamma, self.use_gae)
        
        # 转换为张量
        old_states = torch.squeeze(torch.stack(storage.states).to(self.device), 1).detach()
        old_actions = torch.squeeze(torch.stack(storage.actions).to(self.device), 1).detach()
        old_action_logprobs = torch.squeeze(torch.stack(storage.logprobs), 1).to(self.device).detach()
        old_returns = torch.squeeze(torch.stack(storage.returns), 1).to(self.device).detach()
        old_values = torch.squeeze(torch.stack(storage.values), 1).to(self.device).detach()
        
        # 计算优势
        advantages = old_returns - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_kl_loss = 0
        
        # PPO更新循环
        for epoch in range(self.ppo_epoch):
            # 评估当前策略
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # 计算重要性采样比率
            ratios = torch.exp(logprobs - old_action_logprobs)
            
            # PPO裁剪目标
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.ppo_clip, 1 + self.ppo_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 价值函数损失
            value_loss = 0.5 * (state_values - old_returns).pow(2).mean()
            
            # KL散度约束
            kl_loss = torch.tensor(0.0).to(self.device)
            if ref_policy is not None:
                # 计算与参考策略的KL散度（用于约束策略不要偏离BC模型太远）
                with torch.no_grad():
                    _, _, _, ref_action_mean = ref_policy.forward(old_states)
                _, _, _, new_action_mean = self.policy.forward(old_states)
                
                # 获取标准差
                action_std = torch.sqrt(self.policy.action_var).to(self.device)
                ref_action_std = torch.sqrt(ref_policy.action_var).to(self.device) if hasattr(ref_policy, 'action_var') else action_std
                
                # 计算KL散度 D_KL(new || ref)
                kl = self.compute_kl_divergence(
                    ref_action_mean, ref_action_std,  # old (ref policy)
                    new_action_mean, action_std      # new (current policy)
                )
                # KL散度惩罚：鼓励KL散度接近目标值
                kl_loss = self.kl_coef * (kl - self.kl_target).pow(2)
            else:
                # 计算新旧策略之间的KL散度（用于早停）
                with torch.no_grad():
                    _, _, _, old_action_mean = self.policy_old.forward(old_states)
                _, _, _, new_action_mean = self.policy.forward(old_states)
                
                action_std = torch.sqrt(self.policy.action_var).to(self.device)
                old_action_std = torch.sqrt(self.policy_old.action_var).to(self.device)
                
                # 计算KL散度 D_KL(new || old)
                kl = self.compute_kl_divergence(
                    old_action_mean, old_action_std,  # old (policy_old)
                    new_action_mean, action_std       # new (current policy)
                )
                # 如果KL散度太大，提前停止更新（防止策略更新过大）
                if kl > 2 * self.kl_target:
                    break
                    
            # 总损失
            loss = policy_loss + value_loss + kl_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)  # 梯度裁剪
            self.optimizer.step()
            
            total_policy_loss += policy_loss.detach()
            total_value_loss += value_loss.detach()
            if isinstance(kl_loss, torch.Tensor):
                total_kl_loss += kl_loss.detach()
            else:
                total_kl_loss += torch.tensor(kl_loss).to(self.device)
        
        # 更新旧策略
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        avg_kl_loss = total_kl_loss / self.ppo_epoch if isinstance(total_kl_loss, torch.Tensor) else total_kl_loss / self.ppo_epoch
        if isinstance(avg_kl_loss, torch.Tensor):
            avg_kl_loss = avg_kl_loss.item()
        
        return (total_policy_loss.item() / self.ppo_epoch, 
                total_value_loss.item() / self.ppo_epoch, 
                avg_kl_loss)
