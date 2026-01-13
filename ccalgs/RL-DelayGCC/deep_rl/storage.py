#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
经验回放存储类
用于PPO算法存储状态、动作、奖励等经验数据
"""


class Storage:
    """
    经验回放存储类
    存储PPO训练所需的所有经验数据
    """
    def __init__(self):
        self.states = []  # 状态列表
        self.actions = []  # 动作列表
        self.logprobs = []  # 动作对数概率列表
        self.rewards = []  # 奖励列表
        self.values = []  # 状态值列表
        self.returns = []  # 回报列表（用于计算优势）
        self.dones = []  # 是否结束标志列表
        
    def clear(self):
        """清空所有存储的数据"""
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.returns = []
        self.dones = []
        
    def compute_returns(self, next_value, gamma=0.99, use_gae=False, lam=0.95):
        """
        计算回报（Returns）和优势（Advantages）
        Args:
            next_value: 下一个状态的值估计
            gamma: 折扣因子
            use_gae: 是否使用GAE（Generalized Advantage Estimation）
            lam: GAE的lambda参数
        """
        if not self.rewards:
            return
            
        self.returns = []
        if use_gae:
            # 使用GAE计算优势
            advantages = []
            gae = 0
            for step in reversed(range(len(self.rewards))):
                if step == len(self.rewards) - 1:
                    next_value_step = next_value
                else:
                    next_value_step = self.values[step + 1]
                    
                delta = self.rewards[step] + gamma * next_value_step - self.values[step]
                gae = delta + gamma * lam * gae
                advantages.insert(0, gae)
                self.returns.insert(0, gae + self.values[step])
        else:
            # 标准回报计算
            returns = next_value
            for step in reversed(range(len(self.rewards))):
                returns = self.rewards[step] + gamma * returns
                self.returns.insert(0, returns)
