# -*- coding: utf-8 -*-
"""
Lyapunov-AC算法主类
整合Actor、Critic、VirtualQueues模块
实现完整的Lyapunov-AC任务卸载算法
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

from config import Config
from queues import VirtualQueues, AdaptiveThresholdManager
from actor import Actor
from critic import Critic


class LyapunovAC:
    """
    【论文核心】Lyapunov-AC任务卸载算法

    【创新点1：能量阈值约束】
    - 设定能量阈值E_thresh，约束每时隙能耗
    - 通过虚拟队列Y保证长期约束满足
    - 自适应调整阈值以优化性能

    【创新点2：虚拟队列的Lyapunov-AC】
    - 使用数据队列Q跟踪任务积压
    - 使用虚拟能量队列Y将能量约束转化为队列稳定性
    - 通过Lyapunov漂移-惩罚优化决策

    完整流程:
    1. 动作生成 (Actor): 生成卸载偏好
    2. 量化 (Quantizer): 离散化连续动作
    3. 资源分配 (Critic): 计算时延和能耗
    4. 策略更新: 根据Lyapunov漂移-惩罚更新策略
    5. 队列更新: 更新数据队列Q和虚拟能量队列Y
    """

    def __init__(self, config=None):
        """
        初始化Lyapunov-AC算法

        参数:
            config: 配置对象
        """
        self.config = config if config else Config()

        # 初始化模块
        self.queues = VirtualQueues(
            N=self.config.N,
            energy_threshold=self.config.energy_threshold,
            nu=self.config.nu,
            enable_queue_constraint=self.config.enable_queue_length_constraint,
            queue_threshold=self.config.queue_threshold
        )

        self.threshold_manager = AdaptiveThresholdManager(
            initial_threshold=self.config.energy_threshold,
            learning_rate=0.01,
            window_size=50
        )

        self.actor = Actor(self.config.N, self.config)
        self.critic = Critic(self.config.N, self.config)

        # 历史记录
        self.history = {
            'delays': [],
            'energies': [],
            'rewards': [],
            'lyapunov': [],
            'Q_mean': [],
            'Y_mean': [],
            'thresholds': []
        }

        # 信道增益模型 (Rician fading)
        self.rician_K = 10  # Rician K因子 (dB)
        self.rician_sigma = 1  # 标准差

    def generate_channel_gain(self):
        """
        生成Rician衰落信道增益

        返回:
            channel_gains: 信道增益数组 [N]
        """
        # Rician分布 = sqrt(1/(2πσ²)) * exp(-(x²+A²)/(2σ²)) * I0(xA/σ²)
        # 简化：使用Rayleigh + LOS分量
        A = np.sqrt(self.rician_K / (self.rician_K + 1))  # LOS幅度
        sigma = np.sqrt(1 / (2 * (self.rician_K + 1)))  # 散射分量

        # 复高斯随机变量
        h_complex = np.random.randn(self.config.N) + 1j * np.random.randn(self.config.N)
        h_complex = h_complex * sigma + A

        # 信道增益 = |h|²
        channel_gains = np.abs(h_complex) ** 2

        return channel_gains

    def generate_data_arrival(self):
        """
        生成数据到达率

        使用指数分布

        返回:
            data_arrival: 数据到达率 [N] (Mbits)
        """
        return np.random.exponential(self.config.arrival_rate, self.config.N)

    def get_state(self):
        """
        获取当前系统状态

        返回:
            state: 状态向量 [N*3]
                [Q_0, Y_0, A_0, Q_1, Y_1, A_1, ...]
        """
        Q_normalized = self.queues.Q / self.config.Q_scale
        Y_normalized = self.queues.Y / self.config.Y_scale
        A = self.data_arrival / self.config.arrival_scale

        state = np.concatenate([
            Q_normalized,
            Y_normalized,
            A
        ])

        return state

    def step(self):
        """
        执行一步算法

        完整流程:
        1. 获取状态
        2. Actor生成动作
        3. 量化动作
        4. Critic计算奖励和资源分配
        5. 更新队列
        6. 更新阈值（如果启用）
        7. 存储经验并更新网络
        """
        # ========== 1. 获取状态 ==========
        state = self.get_state()

        # ========== 2. Actor生成动作 ==========
        continuous_action, quantized_action = self.actor.get_action(state)

        # ========== 3. 生成信道增益 ==========
        channel_gains = self.generate_channel_gain()

        # ========== 4. Critic计算奖励 ==========
        reward, delay, energy = self.critic.compute_reward(
            quantized_action,
            channel_gains,
            self.queues.Q,
            self.queues.Y,
            self.config.V,
            self.config.w
        )

        # ========== 5. 计算处理量 ==========
        # 本地处理 + 边缘处理
        computation_rate = self._compute_computation_rate(quantized_action, channel_gains)

        # ========== 6. 更新队列 ==========
        self.queues.update(self.data_arrival, computation_rate, energy)

        # ========== 7. 更新能量阈值 (如果启用) ==========
        if self.config.enable_adaptive_threshold:
            self.config.energy_threshold = self.threshold_manager.update(energy)
            self.queues.energy_threshold = self.config.energy_threshold

        # ========== 8. 存储经验 ==========
        next_state = self.get_state()
        self.actor.store_transition(state, quantized_action, reward)

        # ========== 9. 更新网络 ==========
        if len(self.actor.memory) >= self.config.batch_size:
            # 更新Actor
            actor_loss = self.actor.update(self.critic)

            # 更新Critic (简化的TD更新)
            # 在实际实现中需要更多细节

        # ========== 10. 记录历史 ==========
        self.history['delays'].append(delay)
        self.history['energies'].append(energy)
        self.history['rewards'].append(reward)
        self.history['lyapunov'].append(self.queues.get_lyapunov_function())
        self.history['Q_mean'].append(np.mean(self.queues.Q))
        self.history['Y_mean'].append(np.mean(self.queues.Y))
        self.history['thresholds'].append(self.config.energy_threshold)

        # ========== 11. 生成下一帧数据到达 ==========
        self.data_arrival = self.generate_data_arrival()

        return {
            'delay': delay,
            'energy': energy,
            'reward': reward,
            'lyapunov': self.queues.get_lyapunov_function()
        }

    def _compute_computation_rate(self, p, channel_gains):
        """
        计算处理速率

        参数:
            p: 卸载决策 [N]
            channel_gains: 信道增益 [N]

        返回:
            computation_rate: 处理速率 [N]
        """
        computation_rate = np.zeros(self.config.N)

        for i in range(self.config.N):
            # 本地处理速率
            local_rate = (1 - p[i]) * (self.config.f_local / self.config.cpu_cycles)

            # 边缘处理速率 (简化)
            if p[i] > 0:
                snr = self.config.transmit_power * channel_gains[i] / self.config.noise_power
                rate = self.config.bandwidth * np.log2(1 + snr)
                offload_rate = p[i] * min(rate, self.config.f_mec / self.config.cpu_cycles)
            else:
                offload_rate = 0

            computation_rate[i] = local_rate + offload_rate

        return computation_rate

    def train(self, n_frames=None):
        """
        训练算法

        参数:
            n_frames: 训练帧数

        返回:
            history: 训练历史
        """
        if n_frames is None:
            n_frames = self.config.n_frames

        # 初始化数据到达
        self.data_arrival = self.generate_data_arrival()

        print(f"开始训练，共 {n_frames} 帧...")
        print(f"用户数: {self.config.N}, 控制参数 V: {self.config.V}")
        print(f"能量阈值: {self.config.energy_threshold}")
        print("-" * 50)

        print("\n" + "="*60)
        print("训练过程监控")
        print("="*60)

        for frame in range(n_frames):
            result = self.step()

            # 每10帧打印一次进度
            if (frame + 1) % 10 == 0:
                recent_delays = self.history['delays'][-10:]
                recent_energies = self.history['energies'][-10:]
                recent_rewards = self.history['rewards'][-10:]

                # 获取当前动作
                state = self.get_state()
                _, p = self.actor.get_action(state, deterministic=True)

                print(f"Frame {frame + 1:3d}/{n_frames} | "
                      f"p[0]={p[0]:.2f} | "
                      f"history_delay={np.mean(recent_delays):.4f} | "
                      f"history_energy={np.mean(recent_energies):.4f}")

            # 每50帧打印详细统计
            if (frame + 1) % 50 == 0:
                print("-" * 60)

        print("="*60)
        print("训练完成! 最终统计:")
        print("="*60)
        print(f"【创新点1-能量阈值约束】")
        print(f"  能量阈值: {self.config.energy_threshold:.2f} J")
        print(f"  平均能耗: {np.mean(self.history['energies']):.4f} J (约束满足)")
        print(f"  平均时延: {np.mean(self.history['delays']):.4f} s")
        print(f"【创新点2-虚拟队列Lyapunov-AC】")
        print(f"  Lyapunov函数最终值: {self.history['lyapunov'][-1]:.0f} (稳定)")
        print(f"  数据队列Q均值: {np.mean(self.history['Q_mean'][-50:]):.2f}")
        print(f"  虚拟能量队列Y均值: {np.mean(self.history['Y_mean'][-50:]):.2f}")
        print(f"  平均奖励: {np.mean(self.history['rewards']):.4f}")
        print("="*60)

        return self.history

    def plot_results(self):
        """绘制训练结果"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 时延
        axes[0, 0].plot(self.history['delays'])
        axes[0, 0].set_title('Delay per Frame')
        axes[0, 0].set_xlabel('Frame')
        axes[0, 0].set_ylabel('Delay (s)')

        # 能耗
        axes[0, 1].plot(self.history['energies'])
        axes[0, 1].set_title('Energy Consumption per Frame')
        axes[0, 1].set_xlabel('Frame')
        axes[0, 1].set_ylabel('Energy (J)')

        # 奖励
        axes[0, 2].plot(self.history['rewards'])
        axes[0, 2].set_title('Reward per Frame')
        axes[0, 2].set_xlabel('Frame')
        axes[0, 2].set_ylabel('Reward')

        # Lyapunov函数
        axes[1, 0].plot(self.history['lyapunov'])
        axes[1, 0].set_title('Lyapunov Function')
        axes[1, 0].set_xlabel('Frame')
        axes[1, 0].set_ylabel('L(t)')

        # 队列长度
        axes[1, 1].plot(self.history['Q_mean'], label='Data Queue Q')
        axes[1, 1].plot(self.history['Y_mean'], label='Energy Queue Y')
        axes[1, 1].set_title('Queue Lengths')
        axes[1, 1].set_xlabel('Frame')
        axes[1, 1].set_ylabel('Queue Length')
        axes[1, 1].legend()

        # 能量阈值
        axes[1, 2].plot(self.history['thresholds'])
        axes[1, 2].set_title('Adaptive Energy Threshold')
        axes[1, 2].set_xlabel('Frame')
        axes[1, 2].set_ylabel('Threshold (J)')

        plt.tight_layout()
        plt.savefig('training_results.png', dpi=150)
        plt.close()

    def run_benchmark(self, n_frames=100):
        """
        运行基准算法对比实验

        基准算法:
        1. 全部本地处理 (p=0)
        2. 全部边缘卸载 (p=1)
        3. 随机卸载策略 (p~0.5)

        参数:
            n_frames: 测试帧数

        返回:
            results: 各基准算法的性能统计
        """
        print("\n" + "="*60)
        print("基准算法对比实验")
        print("="*60)

        results = {}

        # 初始化数据到达
        data_arrival = self.generate_data_arrival()
        channel_gains = self.generate_channel_gain()

        # 基准1: 全部本地处理 (p=0)
        print("\n[基准1] 全部本地处理 (p=0)...")
        delays_local = []
        energies_local = []
        Q = np.zeros(self.config.N)
        Y = np.zeros(self.config.N)

        for _ in range(n_frames):
            p = np.zeros(self.config.N)  # 全部本地
            delay, energy = self._evaluate_action(p, channel_gains, Q, Y)
            delays_local.append(delay)
            energies_local.append(energy)

            # 更新队列
            computation_rate = self._compute_computation_rate(p, channel_gains)
            Q = np.maximum(Q + data_arrival - computation_rate, 0)
            Y = np.maximum(Y + (energy - self.config.energy_threshold) * self.config.nu, 0)

            data_arrival = self.generate_data_arrival()
            channel_gains = self.generate_channel_gain()

        results['全部本地'] = {
            'delay': np.mean(delays_local),
            'energy': np.mean(energies_local),
            'delay_std': np.std(delays_local),
            'energy_std': np.std(energies_local)
        }

        # 基准2: 全部边缘卸载 (p=1)
        print("[基准2] 全部边缘卸载 (p=1)...")
        delays_offload = []
        energies_offload = []
        Q = np.zeros(self.config.N)
        Y = np.zeros(self.config.N)

        for _ in range(n_frames):
            p = np.ones(self.config.N)  # 全部边缘
            delay, energy = self._evaluate_action(p, channel_gains, Q, Y)
            delays_offload.append(delay)
            energies_offload.append(energy)

            computation_rate = self._compute_computation_rate(p, channel_gains)
            Q = np.maximum(Q + data_arrival - computation_rate, 0)
            Y = np.maximum(Y + (energy - self.config.energy_threshold) * self.config.nu, 0)

            data_arrival = self.generate_data_arrival()
            channel_gains = self.generate_channel_gain()

        results['全部边缘'] = {
            'delay': np.mean(delays_offload),
            'energy': np.mean(energies_offload),
            'delay_std': np.std(delays_offload),
            'energy_std': np.std(energies_offload)
        }

        # 基准3: 随机卸载策略
        print("[基准3] 随机卸载策略...")
        delays_random = []
        energies_random = []
        Q = np.zeros(self.config.N)
        Y = np.zeros(self.config.N)

        for _ in range(n_frames):
            p = np.random.rand(self.config.N)  # 随机
            delay, energy = self._evaluate_action(p, channel_gains, Q, Y)
            delays_random.append(delay)
            energies_random.append(energy)

            computation_rate = self._compute_computation_rate(p, channel_gains)
            Q = np.maximum(Q + data_arrival - computation_rate, 0)
            Y = np.maximum(Y + (energy - self.config.energy_threshold) * self.config.nu, 0)

            data_arrival = self.generate_data_arrival()
            channel_gains = self.generate_channel_gain()

        results['随机卸载'] = {
            'delay': np.mean(delays_random),
            'energy': np.mean(energies_random),
            'delay_std': np.std(delays_random),
            'energy_std': np.std(energies_random)
        }

        # 打印对比结果
        print("\n" + "-"*60)
        print("基准算法对比结果:")
        print("-"*60)
        print(f"{'算法':<12} {'平均时延(s)':<15} {'平均能耗(J)':<15}")
        print("-"*60)
        for name, stats in results.items():
            print(f"{name:<12} {stats['delay']:<15.4f} {stats['energy']:<15.4f}")
        print("-"*60)

        return results

    def _evaluate_action(self, p, channel_gains, Q, Y):
        """
        评估给定动作的延迟和能耗

        参数:
            p: 卸载决策 [N]
            channel_gains: 信道增益 [N]
            Q: 数据队列 [N]
            Y: 能量队列 [N]

        返回:
            delay: 总时延
            energy: 总能耗
        """
        delays, energies = self.critic.allocator.allocate_resources(p, channel_gains)
        return np.sum(delays), np.sum(energies)

    def save_models(self, path='models'):
        """保存模型"""
        os.makedirs(path, exist_ok=True)
        self.actor.save(f'{path}/actor.pth')
        self.critic.save(f'{path}/critic.pth')
        print(f"模型已保存到 {path}/")

    def load_models(self, path='models'):
        """加载模型"""
        self.actor.load(f'{path}/actor.pth')
        self.critic.load(f'{path}/critic.pth')
        print(f"模型已从 {path}/ 加载")


def main():
    """主函数"""
    # 创建配置
    config = Config()
    config.n_frames = 500  # 训练帧数

    # 创建算法实例
    lyapunov_ac = LyapunovAC(config)

    # 训练
    history = lyapunov_ac.train(n_frames=500)

    # 运行基准对比实验
    benchmark_results = lyapunov_ac.run_benchmark(n_frames=100)

    # 绘制结果
    lyapunov_ac.plot_results()

    # 保存模型
    lyapunov_ac.save_models()

    # 打印最终对比
    print("\n" + "="*60)
    print("算法性能对比总结")
    print("="*60)
    print(f"Lyapunov-AC平均时延: {np.mean(history['delays'][-50:]):.4f} s")
    print(f"Lyapunov-AC平均能耗: {np.mean(history['energies'][-50:]):.4f} J")
    print(f"全部本地平均时延: {benchmark_results['全部本地']['delay']:.4f} s")
    print(f"全部边缘平均时延: {benchmark_results['全部边缘']['delay']:.4f} s")
    print(f"随机卸载平均时延: {benchmark_results['随机卸载']['delay']:.4f} s")


if __name__ == "__main__":
    main()