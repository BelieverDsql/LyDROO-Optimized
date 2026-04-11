# -*- coding: utf-8 -*-
"""
虚拟队列模块

【论文重点1：基于能量阈值约束】

本模块实现了能量阈值约束机制：
1. 能量阈值E_thresh: 限制每时隙最大能耗
2. 自适应调整: 根据历史能耗动态优化阈值

【论文重点2：虚拟队列的Lyapunov-AC任务卸载算法】

在能量阈值约束基础上，实现基于Lyapunov优化理论的虚拟队列机制：
1. 数据队列Q: 跟踪待处理的数据任务积压情况
2. 虚拟能量队列Y: 将能量阈值约束转化为队列稳定性问题

Lyapunov优化核心思想：
- 通过构造虚拟队列，将长期约束问题转化为瞬时队列稳定性问题
- 队列稳定即意味着长期约束满足
"""

import numpy as np


class VirtualQueues:
    """
    虚拟队列管理类

    【核心创新点2：虚拟队列设计】

    包含：
    - 数据队列 Q: 存储待处理的数据量，反映任务积压情况
    - 虚拟能量队列 Y: 将能量阈值约束E[t] <= E_thresh转化为队列稳定性问题

    虚拟队列Y的构造原理：
    - 原始约束: E[t] <= E_thresh (每时隙能耗不超过阈值)
    - 引入松弛变量: Y[t+1] = max(Y[t] + (E[t] - E_thresh), 0)
    - 当E[t] > E_thresh时，Y队列增长；反之减少
    - Y队列稳定意味着长期平均能耗 <= 阈值
    """

    def __init__(self, N, energy_threshold, nu=1000, enable_queue_constraint=False, queue_threshold=None):
        """
        初始化虚拟队列

        参数:
            N: 用户数量
            energy_threshold: 能量阈值 (J/time_slot)
            nu: 能量队列因子，控制惩罚力度
            enable_queue_constraint: 是否启用队列长度约束
            queue_threshold: 队列长度阈值
        """
        self.N = N
        self.energy_threshold = energy_threshold
        self.nu = nu
        self.enable_queue_constraint = enable_queue_constraint
        self.queue_threshold = queue_threshold if queue_threshold else 1000

        # 队列状态
        self.Q = np.zeros(N)      # 数据队列 (Mbits)
        self.Y = np.zeros(N)      # 虚拟能量队列 (mJ)
        self.Z = np.zeros(N)      # 虚拟队列长度约束队列 (仅当启用队列约束时)

        # 历史记录
        self.Q_history = []
        self.Y_history = []
        self.Z_history = []

    def get_lyapunov_function(self):
        """
        获取Lyapunov函数值

        Lyapunov函数定义:
        L(t) = 0.5 * sum(Q_i^2) + 0.5 * sum(Y_i^2)

        返回:
            L: Lyapunov函数值
        """
        L = 0.5 * np.sum(self.Q ** 2) + 0.5 * np.sum(self.Y ** 2)

        if self.enable_queue_constraint:
            L += 0.5 * np.sum(self.Z ** 2)

        return L

    def compute_drift(self, Q_next, Y_next):
        """
        计算Lyapunov漂移

        参数:
            Q_next: 下一时刻数据队列
            Y_next: 下一时刻能量队列

        返回:
            drift: Lyapunov漂移值 ΔL(t) = L(t+1) - L(t)
        """
        L_current = self.get_lyapunov_function()

        # 临时更新队列计算下一时刻Lyapunov值
        Q_temp = self.Q.copy()
        Y_temp = self.Y.copy()
        self.Q = Q_next
        self.Y = Y_next
        L_next = self.get_lyapunov_function()

        # 恢复队列状态
        self.Q = Q_temp
        self.Y = Y_temp

        drift = L_next - L_current
        return drift

    def update(self, data_arrival, computation_rate, energy_consumption):
        """
        更新虚拟队列

        数据队列更新:
            Q[t+1] = max(Q[t] + A[t] - R[t], 0)

        虚拟能量队列更新:
            Y[t+1] = max(Y[t] + (E[t] - E_thresh) * nu, 0)

        参数:
            data_arrival: 数据到达率 (Mbits)
            computation_rate: 计算速率 (Mbits/time_slot)
            energy_consumption: 能耗 (J/time_slot)

        返回:
            Q, Y: 更新后的队列状态
        """
        # ========== 数据队列更新 ==========
        Q_next = np.maximum(self.Q + data_arrival - computation_rate, 0)

        # ========== 虚拟能量队列更新 ==========
        # 能量超过阈值的部分进入虚拟队列
        energy_excess = energy_consumption - self.energy_threshold
        Y_next = np.maximum(self.Y + energy_excess * self.nu, 0)

        # ========== 队列长度约束队列更新 (可选) ==========
        if self.enable_queue_constraint:
            queue_excess = np.maximum(self.Q - self.queue_threshold, 0)
            Z_next = np.maximum(self.Z + queue_excess, 0)
            self.Z = Z_next
            self.Z_history.append(Z_next.copy())

        # 更新队列状态
        self.Q = Q_next
        self.Y = Y_next

        # 记录历史
        self.Q_history.append(Q_next.copy())
        self.Y_history.append(Y_next.copy())

        return self.Q.copy(), self.Y.copy()

    def get_state(self, Q_scale=10000, Y_scale=10000):
        """
        获取归一化的队列状态用于神经网络输入

        参数:
            Q_scale: 数据队列缩放因子
            Y_scale: 能量队列缩放因子

        返回:
            state: 归一化后的队列状态
        """
        return np.concatenate([
            self.Q / Q_scale,
            self.Y / Y_scale
        ])

    def check_stability(self):
        """
        检查队列稳定性

        返回:
            is_stable: 队列是否稳定
            avg_queue_length: 平均队列长度
        """
        if len(self.Q_history) == 0:
            return True, 0

        avg_queue_length = np.mean([np.sum(q) for q in self.Q_history[-100:]])
        is_stable = avg_queue_length < 1e6  # 队列不发散

        return is_stable, avg_queue_length


class AdaptiveThresholdManager:
    """
    【核心创新点1：能量阈值约束管理器】

    自适应能量阈值管理器
    根据近期能量消耗动态调整能量阈值

    能量阈值约束机制：
    - 设定能量阈值E_thresh，约束每时隙能耗E[t] <= E_thresh
    - 通过Lyapunov虚拟队列Y实现长期能量约束
    - 自适应调整机制：根据实际能耗动态优化阈值
    """

    def __init__(self, initial_threshold, learning_rate=0.01, window_size=50):
        """
        初始化

        参数:
            initial_threshold: 初始能量阈值
            learning_rate: 学习率
            window_size: 滑动窗口大小
        """
        self.threshold = initial_threshold
        self.learning_rate = learning_rate
        self.window_size = window_size
        self.energy_history = []
        self.threshold_history = [initial_threshold]

    def update(self, energy_consumption):
        """
        更新能量阈值

        基于近期平均能量消耗调整阈值

        参数:
            energy_consumption: 当前能耗
        """
        self.energy_history.append(energy_consumption)

        # 保持历史窗口大小
        if len(self.energy_history) > self.window_size:
            self.energy_history.pop(0)

        # 计算近期平均能耗
        if len(self.energy_history) >= 10:
            avg_energy = np.mean(self.energy_history[-10:])

            # 调整阈值：向平均能耗方向移动
            self.threshold += self.learning_rate * (avg_energy - self.threshold)

            # 限制阈值范围
            self.threshold = np.clip(self.threshold, 0.01, 0.5)

        self.threshold_history.append(self.threshold)

        return self.threshold

    def get_threshold(self):
        """获取当前阈值"""
        return self.threshold


def compute_drift_penalty_upper_bound(Q, Y, w, V, data_arrival):
    """
    计算漂移-惩罚上界

    根据Lyapunov优化理论，漂移-惩罚项的上界可表示为:

    ΔL(t) + V * J(t) <= C + Σ (Q_i + V*w_i) * (A_i - r_i)
                           - Σ Y_i * (e_i - E_thresh)

    参数:
        Q: 数据队列长度
        Y: 虚拟能量队列
        w: 用户权重
        V: 控制参数
        data_arrival: 数据到达率

    返回:
        upper_bound: 漂移-惩罚上界
    """
    # 常数项（与决策无关）
    C = 0.5 * np.sum(data_arrival ** 2)

    # 计算权重系数 a_i = Q_i + V * w_i
    a = Q + V * w

    return C, a


if __name__ == "__main__":
    # 测试代码
    queues = VirtualQueues(N=10, energy_threshold=0.08, nu=1000)

    # 模拟更新
    for t in range(10):
        data_arrival = np.random.exponential(3, 10)
        computation_rate = np.random.uniform(1, 5, 10)
        energy_consumption = np.random.uniform(0.05, 0.1, 10)

        Q, Y = queues.update(data_arrival, computation_rate, energy_consumption)

        L = queues.get_lyapunov_function()
        print(f"Time {t}: L = {L:.4f}, Q_sum = {Q.sum():.4f}, Y_sum = {Y.sum():.4f}")