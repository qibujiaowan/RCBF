"""
全局超参数配置
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class VehicleConfig:
    # 履带车物理参数
    track_width: float = 0.5       # 轨距 B (m)
    wheel_radius: float = 0.15     # 链轮半径 r (m)
    mass: float = 100.0            # 整车质量 (kg)
    inertia: float = 8.0           # 绕质心转动惯量 (kg·m²)
    max_wheel_speed: float = 10.0  # 最大轮速指令 (rad/s)
    max_delta_speed: float = 2.0   # 每步最大轮速变化量 (rad/s)
    max_yaw_rate: float = 1.0      # 横摆角速度上限 (rad/s)

    # 仿真步长
    dt: float = 0.05               # 控制周期 (s)

    # 滑移参数（用于"真实环境"，控制器不可见）
    # 不同路面工况对应不同滑移率
    slip_config: str = "medium"    # "none", "light", "medium", "heavy", "variable"


@dataclass
class GPConfig:
    # 高斯过程扰动估计
    max_points: int = 500          # 最大训练点数（稀疏GP截断）
    update_interval: int = 10      # 每隔多少步更新一次GP
    confidence_k: float = 2.0      # kc置信度参数（2.0 ≈ 95.5%置信区间）
    noise_var: float = 0.01        # 观测噪声方差
    length_scale: float = 1.0      # RBF核长度尺度初始值
    output_scale: float = 1.0      # RBF核输出方差初始值
    state_dim: int = 4             # 状态维度 [X, Y, psi, v]


@dataclass
class RCBFConfig:
    # RCBF安全层参数
    alpha_yaw: float = 0.5         # 横摆角速度约束的class-K函数系数
    alpha_sat: float = 1.0         # 执行器饱和约束的class-K函数系数
    alpha_rate: float = 0.5        # 控制变化率约束的class-K函数系数
    slack_penalty: float = 1e5     # 松弛变量惩罚系数 l
    qp_solver: str = "OSQP"        # QP求解器


@dataclass
class SACConfig:
    # SAC超参数
    hidden_dim: int = 256
    learning_rate: float = 1e-4
    gamma: float = 0.99
    tau: float = 0.005             # 目标网络软更新系数
    alpha: float = 0.2             # 熵正则化系数（自动调整时为初始值）
    auto_entropy: bool = True
    batch_size: int = 512
    replay_buffer_size: int = 200_000
    warmup_steps: int = 1000       # 随机探索步数（减少warmup以最大化10k步中的实际训练时间）
    update_interval: int = 5       # 每5步更新一次（降低off-policy程度，提高训练稳定性）

    # 可微分安全层
    use_diff_safety_layer: bool = True


@dataclass
class TrainConfig:
    total_steps: int = 300_000
    eval_interval: int = 2000
    eval_episodes: int = 5
    save_interval: int = 20_000
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    seed: int = 42
    bc_pretrain_steps: int = 5000   # MPC数据收集步数
    bc_pretrain_epochs: int = 50    # 行为克隆训练轮数


@dataclass
class RewardConfig:
    # 路径跟踪奖励权重（安全约束不进奖励）
    w_lateral: float = 2.0        # 横向误差权重
    w_heading: float = 2.0        # 航向误差权重
    w_velocity: float = 0.3       # 速度误差权重
    w_smooth: float = 0.5         # 控制平滑性权重


@dataclass
class Config:
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    gp: GPConfig = field(default_factory=GPConfig)
    rcbf: RCBFConfig = field(default_factory=RCBFConfig)
    sac: SACConfig = field(default_factory=SACConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)


DEFAULT_CONFIG = Config()
