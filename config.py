# -*- coding: utf-8 -*-
"""
配置参数文件
用于定义Lyapunov-AC算法的各种参数
"""

import numpy as np


class Config:
    """算法配置参数类"""

    def __init__(self):
        # ========== 系统参数 ==========
        self.N = 10                      # 用户数量
        self.n_frames = 500              # 时间帧数量

        # ========== Actor网络参数 ==========
        self.hidden_dims = [256, 128]    # 隐藏层维度
        self.learning_rate = 0.01        # 学习率
        self.training_interval = 20      # 训练间隔
        self.batch_size = 32             # 批次大小，减小以便100帧内能更新
        self.memory_size = 1024          # 经验回放池大小

        # ========== 量化策略参数 ==========
        self.K = self.N                  # 候选动作数量
        self.decoder_mode = 'OPN'        # 量化模式: OP/KNN/OPN

        # ========== Lyapunov优化参数 ==========
        self.V = 10                      # 控制参数，减小以让奖励变化更明显
        self.nu = 10                     # 能量队列因子

        # ========== 能量阈值约束参数 ==========
        self.energy_threshold = 0.35     # 能量消耗阈值 (J/time_slot)，提高以匹配实际能耗
        self.enable_adaptive_threshold = True  # 是否启用自适应能量阈值
        self.threshold_update_interval = 50    # 阈值更新间隔
        self.threshold_learning_rate = 0.01    # 阈值学习率

        # ========== 队列长度约束参数 ==========
        self.enable_queue_length_constraint = False  # 是否启用队列长度约束
        self.queue_threshold = 1000                   # 队列长度阈值 (Mbits)

        # ========== 资源分配参数 ==========
        self.f_local = 1e9          # 本地计算频率 (cycles/s)
        self.f_mec = 10e9           # MEC计算频率 (cycles/s)
        self.bandwidth = 10e6       # 带宽 (Hz)
        self.noise_power = 1e-10    # 噪声功率 (W)
        self.transmit_power = 0.1   # 发射功率 (W)
        self.task_size = 1e6        # 任务大小 (bits)
        self.cpu_cycles = 1000      # CPU周期数

        # ========== 奖励函数参数 ==========
        self.w = 1.0                # 时延权重
        self.gamma = 0.99           # 折扣因子

        # ========== 网络参数 ==========
        self.actor_lr = 0.003       # Actor学习率，提高
        self.critic_lr = 0.002      # Critic学习率
        self.quantization = 'OP'    # 量化方法: OP/KNN/OPN
        self.num_levels = 10        # 量化级别数
        self.noise_std = 0.02       # OPN噪声标准差

        # ========== 状态归一化参数 ==========
        self.Q_scale = 10000        # 数据队列缩放因子
        self.Y_scale = 10000        # 能量队列缩放因子
        self.arrival_scale = 10     # 到达率缩放因子

        # ========== 任务参数 ==========
        self.arrival_rate = 2       # 数据到达率 (Mbits)，降低以加快收敛

        # ========== 兼容旧参数 ==========
        self.P_max = self.transmit_power
        self.f_max = self.f_local / 1e6  # MHz
        self.phi = self.cpu_cycles
        self.k_factor = 1e-26
        self.arrival_lambda = self.arrival_rate
        self.CH_FACT = 1e10
        self.Rician_factor = 0.3
        self.Delta = 32
        self.rolling_interval = 50
        self.user_weights = [1.5 if i % 2 == 0 else 1 for i in range(self.N)]

    def __repr__(self):
        return f"Config(N={self.N}, n_frames={self.n_frames}, V={self.V}, nu={self.nu})"


# 默认配置实例
default_config = Config()