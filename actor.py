# -*- coding: utf-8 -*-
"""
Actor模块
实现动作生成（连续值）和量化策略
包含三种量化方法：OP、KNN、OPN
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class ActorNetwork(nn.Module):
    """
    Actor网络：生成连续动作（卸载偏好）

    输入: 用户状态 [N*3] = [Q_i, Y_i, A_i] for each user
    输出: 连续动作 [N] = 卸载偏好 p_i (0-1之间)
    """

    def __init__(self, N, hidden_dims=[256, 128]):
        super(ActorNetwork, self).__init__()

        # 构建网络
        layers = []
        input_dim = N * 3  # Q, Y, A for each user

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim

        # 输出层：使用Sigmoid确保输出在[0,1]
        layers.append(nn.Linear(input_dim, N))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, state):
        """
        前向传播

        参数:
            state: 输入状态 [batch_size, N*3]

        返回:
            action: 卸载偏好 [batch_size, N]
        """
        return self.network(state)


class Quantizer:
    """
    量化器基类
    """

    def quantize(self, continuous_action):
        raise NotImplementedError


class OPQuantizer(Quantizer):
    """
    Order-Preserving (OP) 量化器

    保持动作的相对顺序，将连续动作映射到离散集合
    """

    def __init__(self, N, num_levels=10):
        self.N = N
        self.num_levels = num_levels
        self.levels = np.linspace(0, 1, num_levels)

    def quantize(self, continuous_action):
        """
        OP量化 - 基于实际值量化，保留相对差异

        参数:
            continuous_action: 连续动作 [N]

        返回:
            quantized_action: 量化后动作 [N]
        """
        # 归一化到 [0, 1] 范围
        action_min = continuous_action.min()
        action_max = continuous_action.max()

        if action_max - action_min < 1e-6:
            # 所有值相同，直接返回
            return continuous_action.copy()

        # 线性映射到离散级别
        normalized = (continuous_action - action_min) / (action_max - action_min)
        quantized = np.round(normalized * (self.num_levels - 1)) / (self.num_levels - 1)

        # 恢复到原始范围
        quantized_action = quantized * (action_max - action_min) + action_min

        return quantized_action


class KNNQuantizer(Quantizer):
    """
    K-Nearest Neighbors (KNN) 量化器

    基于聚类的K近邻量化
    """

    def __init__(self, N, num_levels=10):
        self.N = N
        self.num_levels = num_levels
        self.centers = np.linspace(0, 1, num_levels)

    def quantize(self, continuous_action):
        """
        KNN量化

        参数:
            continuous_action: 连续动作 [N]

        返回:
            quantized_action: 量化后动作 [N]
        """
        quantized_action = np.zeros_like(continuous_action)

        for i in range(self.N):
            # 找到最近的中心
            distances = np.abs(self.centers - continuous_action[i])
            nearest_idx = np.argmin(distances)
            quantized_action[i] = self.centers[nearest_idx]

        return quantized_action


class OPNQuantizer(OPQuantizer):
    """
    Order-Preserving with Noise (OPN) 量化器

    在OP基础上添加随机噪声以增加探索
    """

    def __init__(self, N, num_levels=10, noise_std=0.02):
        super().__init__(N, num_levels)
        self.noise_std = noise_std

    def quantize(self, continuous_action):
        """
        OPN量化

        参数:
            continuous_action: 连续动作 [N]

        返回:
            quantized_action: 量化后动作 [N]
        """
        # 添加噪声
        noise = np.random.normal(0, self.noise_std, self.N)
        noisy_action = np.clip(continuous_action + noise, 0, 1)

        # 使用OP量化
        sorted_indices = np.argsort(noisy_action)
        sorted_action = noisy_action[sorted_indices]

        quantized_sorted = np.zeros_like(sorted_action)
        for i in range(self.N):
            level_idx = int(i * (self.num_levels - 1) / (self.N - 1))
            quantized_sorted[i] = self.levels[level_idx]

        quantized_action = np.zeros_like(continuous_action)
        quantized_action[sorted_indices] = quantized_sorted

        return quantized_action


class Actor:
    """
    Actor模块：整合网络和量化器
    """

    def __init__(self, N, config):
        self.N = N
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 创建网络
        self.network = ActorNetwork(N, hidden_dims=[256, 128]).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.actor_lr)

        # 创建量化器
        if config.quantization == 'OP':
            self.quantizer = OPQuantizer(N, config.num_levels)
        elif config.quantization == 'KNN':
            self.quantizer = KNNQuantizer(N, config.num_levels)
        elif config.quantization == 'OPN':
            self.quantizer = OPNQuantizer(N, config.num_levels, config.noise_std)
        else:
            self.quantizer = OPQuantizer(N, config.num_levels)

        # 经验回放
        self.memory = []
        self.memory_size = config.memory_size
        self.batch_size = config.batch_size

    def get_action(self, state, deterministic=False):
        """
        获取动作

        参数:
            state: 状态 [N*3]
            deterministic: 是否使用确定性策略

        返回:
            continuous_action: 连续动作
            quantized_action: 量化后动作
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            continuous_action = self.network(state_tensor).cpu().numpy()[0]

        # 添加探索噪声 (epsilon-greedy)
        if not deterministic and np.random.rand() < 0.1:
            noise = np.random.randn(self.N) * 0.1
            continuous_action = np.clip(continuous_action + noise, 0, 1)

        # 量化
        quantized_action = self.quantizer.quantize(continuous_action)

        return continuous_action, quantized_action

    def store_transition(self, state, action, reward):
        """
        存储经验

        参数:
            state: 状态
            action: 动作
            reward: 奖励
        """
        self.memory.append((state, action, reward))

        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

    def update(self, critic):
        """
        更新Actor网络

        使用策略梯度：直接最大化奖励

        参数:
            critic: Critic网络 (未使用，简化为直接用reward)

        返回:
            loss: 策略梯度损失
        """
        if len(self.memory) < self.batch_size:
            return 0.0

        # 采样批数据
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states = torch.FloatTensor(np.array([self.memory[i][0] for i in batch])).to(self.device)
        actions = torch.FloatTensor(np.array([self.memory[i][1] for i in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([self.memory[i][2] for i in batch])).unsqueeze(1).to(self.device)

        # 获取当前策略生成的动作
        current_actions = self.network(states)

        # 策略梯度：直接最大化奖励
        # 使用 rewards 作为目标
        policy_loss = -torch.mean(current_actions * rewards.abs() + rewards)

        # 更新网络
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()

        return policy_loss.item()

    def save(self, path):
        """保存模型"""
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


if __name__ == "__main__":
    # 测试代码
    class Config:
        actor_lr = 0.001
        quantization = 'OP'
        num_levels = 10
        noise_std = 0.02
        memory_size = 10000
        batch_size = 64

    N = 10
    actor = Actor(N, Config())

    # 测试动作生成
    state = np.random.randn(N * 3)
    continuous, quantized = actor.get_action(state)

    print(f"Continuous action: {continuous[:5]}")
    print(f"Quantized action: {quantized[:5]}")