# -*- coding: utf-8 -*-
"""
Critic模块
实现资源分配和价值评估
包含本地执行、边缘计算的资源分配算法
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class CriticNetwork(nn.Module):
    """
    Critic网络：评估状态-动作对的价值

    输入: 状态-动作对 [N*3 + N] = [Q_i, Y_i, A_i, p_i]
    输出: Q值
    """

    def __init__(self, N, hidden_dims=[256, 128]):
        super(CriticNetwork, self).__init__()

        # 状态和动作分别处理
        self.state_fc = nn.Sequential(
            nn.Linear(N * 3, 128),
            nn.ReLU()
        )

        self.action_fc = nn.Sequential(
            nn.Linear(N, 64),
            nn.ReLU()
        )

        # 合并后
        layers = []
        input_dim = 128 + 64

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))

        self.network = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, state, action):
        """
        前向传播

        参数:
            state: 状态 [batch_size, N*3]
            action: 动作 [batch_size, N]

        返回:
            q_value: Q值 [batch_size, 1]
        """
        state_features = self.state_fc(state)
        action_features = self.action_fc(action)

        combined = torch.cat([state_features, action_features], dim=1)
        q_value = self.network(combined)

        return q_value


class ResourceAllocator:
    """
    资源分配器

    实现本地执行和边缘计算的资源分配
    """

    def __init__(self, config):
        self.config = config
        self.N = config.N
        self.f_local = config.f_local          # 本地计算频率 (cycles/s)
        self.f_mec = config.f_mec              # MEC计算频率 (cycles/s)
        self.bandwidth = config.bandwidth      # 带宽 (Hz)
        self.noise_power = config.noise_power  # 噪声功率 (W)
        self.transmit_power = config.transmit_power  # 发射功率 (W)
        self.task_size = config.task_size      # 任务大小 (bits)
        self.cpu_cycles = config.cpu_cycles    # CPU周期数

    def compute_local_delay(self, p_local):
        """
        计算本地执行时延

        参数:
            p_local: 本地执行比例

        返回:
            delay: 时延 (秒)
        """
        # 本地计算时间 = 任务大小 / 计算速率
        # 计算速率 = f_local / cpu_cycles
        computation_rate = self.f_local / self.cpu_cycles  # bits/s
        local_delay = (p_local * self.task_size) / computation_rate
        return local_delay

    def compute_offload_delay(self, p_offload, channel_gain):
        """
        计算边缘计算时延

        时延 = 传输时延 + MEC处理时延

        参数:
            p_offload: 卸载比例
            channel_gain: 信道增益

        返回:
            delay: 时延 (秒)
        """
        # 传输速率 (Shannon公式)
        snr = self.transmit_power * channel_gain / self.noise_power
        rate = self.bandwidth * np.log2(1 + snr)  # bits/s

        # 传输时延
        transmit_delay = (p_offload * self.task_size) / rate

        # MEC处理时延
        computation_rate = self.f_mec / self.cpu_cycles
        mec_delay = (p_offload * self.task_size) / computation_rate

        return transmit_delay + mec_delay

    def compute_local_energy(self, p_local):
        """
        计算本地执行能耗

        参数:
            p_local: 本地执行比例

        返回:
            energy: 能耗 (J)
        """
        # 能耗 = 计算时间 * 功率
        # 本地功率模型: P = κ * f^3
        kappa = 1e-28  # 芯片特定常数
        power = kappa * (self.f_local ** 3)
        delay = self.compute_local_delay(p_local)
        energy = power * delay
        return energy

    def compute_offload_energy(self, p_offload):
        """
        计算边缘计算传输能耗

        参数:
            p_offload: 卸载比例

        返回:
            energy: 能耗 (J)
        """
        # 传输能耗 = 传输时间 * 发射功率
        delay = p_offload * self.task_size / self.bandwidth  # 简化
        energy = self.transmit_power * delay
        return energy

    def allocate_resources(self, p, channel_gains):
        """
        资源分配主函数

        使用Lambert-W函数求解最优资源分配

        参数:
            p: 卸载决策 [N]
            channel_gains: 信道增益 [N]

        返回:
            delays: 时延 [N]
            energies: 能耗 [N]
        """
        delays = np.zeros(self.N)
        energies = np.zeros(self.N)

        for i in range(self.N):
            p_local = 1 - p[i]
            p_offload = p[i]

            # 本地执行时延和能耗
            local_delay = self.compute_local_delay(1.0)  # 假设全部本地执行
            local_energy = self.compute_local_energy(1.0)

            # 边缘计算时延和能耗
            offload_delay = self.compute_offload_delay(1.0, channel_gains[i])
            offload_energy = self.compute_offload_energy(1.0)

            # 加权平均
            delays[i] = p_local * local_delay + p_offload * offload_delay
            energies[i] = p_local * local_energy + p_offload * offload_energy

        return delays, energies


class Critic:
    """
    Critic模块：整合网络和资源分配器
    """

    def __init__(self, N, config):
        self.N = N
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 创建网络
        self.network = CriticNetwork(N, hidden_dims=[256, 128]).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.critic_lr)

        # 创建资源分配器
        self.allocator = ResourceAllocator(config)

        # 目标网络
        self.target_network = CriticNetwork(N, hidden_dims=[256, 128]).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.tau = 0.001  # 软更新参数

    def get_q_value(self, state, action):
        """
        获取Q值

        参数:
            state: 状态
            action: 动作

        返回:
            q_value: Q值
        """
        return self.network(state, action)

    def compute_reward(self, p, channel_gains, Q, Y, V, w):
        """
        计算奖励

        奖励函数 = -w * delay - V * energy

        同时考虑Lyapunov队列稳定性

        参数:
            p: 卸载决策 [N]
            channel_gains: 信道增益 [N]
            Q: 数据队列 [N]
            Y: 虚拟能量队列 [N]
            V: 控制参数
            w: 权重

        返回:
            reward: 奖励值
            delay: 总时延
            energy: 总能耗
        """
        delays, energies = self.allocator.allocate_resources(p, channel_gains)

        total_delay = np.sum(delays)
        total_energy = np.sum(energies)

        # 奖励 = -w * delay - V * energy + Q相关项
        reward = -w * total_delay - V * total_energy

        # 添加队列稳定性奖励
        reward += np.sum(Q * (delays - 0.01))  # 队列积压惩罚

        return reward, total_delay, total_energy

    def update(self, states, actions, rewards, next_states, next_actions, dones):
        """
        更新Critic网络

        使用TD学习

        参数:
            states: 当前状态
            actions: 当前动作
            rewards: 奖励
            next_states: 下一状态
            next_actions: 下一动作
            dones: 是否结束

        返回:
            loss: 损失值
        """
        # 当前Q值
        current_q = self.network(states, actions)

        # 目标Q值
        with torch.no_grad():
            target_q = rewards + (1 - dones) * self.config.gamma * \
                       self.target_network(next_states, next_actions)

        # TD损失
        loss = nn.MSELoss()(current_q, target_q)

        # 更新网络
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()

        # 软更新目标网络
        self._soft_update()

        return loss.item()

    def _soft_update(self):
        """软更新目标网络"""
        for target_param, param in zip(self.target_network.parameters(),
                                        self.network.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def save(self, path):
        """保存模型"""
        torch.save({
            'network': self.network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


def lambert_w(x):
    """
    Lambert W函数近似

    用于求解资源分配优化问题

    参数:
        x: 输入值

    返回:
        w: W(x)
    """
    # 使用迭代近似
    if isinstance(x, np.ndarray):
        w = np.zeros_like(x)
        for i in range(len(x)):
            w[i] = _lambert_w_scalar(x[i])
        return w
    else:
        return _lambert_w_scalar(x)


def _lambert_w_scalar(x):
    """标量Lambert W函数"""
    if x < 0:
        # 负数情况，使用级数展开
        if x > -1/e:
            w = x
            for _ in range(10):
                w = w - (w * np.exp(w) - x) / (np.exp(w) * (w + 1))
            return w
    else:
        # 正数情况，使用Halley迭代
        w = np.log(x + 1) if x > 0 else x
        for _ in range(10):
            w = w - (w * np.exp(w) - x) / (np.exp(w) * (w + 1) -
                    (w + 2) * (w * np.exp(w) - x) / (2 * w + 2))
        return w
    return 0


if __name__ == "__main__":
    # 测试代码
    class Config:
        N = 10
        f_local = 1e9          # 1 GHz
        f_mec = 10e9           # 10 GHz
        bandwidth = 10e6       # 10 MHz
        noise_power = 1e-10    # -70 dBm
        transmit_power = 0.1   # 100 mW
        task_size = 1e6        # 1 Mbits
        cpu_cycles = 1000      # cycles/bit
        critic_lr = 0.001
        gamma = 0.99

    config = Config()
    critic = Critic(10, config)

    # 测试资源分配
    p = np.random.rand(10)
    channel_gains = np.random.rand(10) * 1e-5

    delays, energies = critic.allocator.allocate_resources(p, channel_gains)
    print(f"Delays: {delays[:5]}")
    print(f"Energies: {energies[:5]}")