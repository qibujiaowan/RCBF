"""
履带车标称运动学模型（不含滑移）
控制器内部使用此模型计算 f(x), g(x)

状态: x = [X, Y, psi, v]
  X, Y   : 世界坐标系位置 (m)
  psi    : 航向角 (rad)
  v      : 纵向速度 (m/s)

控制输入: u = [omega_L, omega_R]  (左/右履带轮速指令, rad/s)

标称运动学（无滑移）:
  vL = omega_L * r
  vR = omega_R * r
  v  = (vL + vR) / 2
  r  = (vR - vL) / B        (横摆角速度，注意与状态v区分)
  Xdot   = v * cos(psi)
  Ydot   = v * sin(psi)
  psidot = (vR - vL) / B
  vdot   = (vR + vL) / 2 / dt  (近似用速度差分，或用加速度动力学)
"""

import numpy as np
import torch
from typing import Tuple


class TrackedVehicleKinematics:
    """
    不含滑移的理想差速运动学，作为标称模型 f(x) + g(x)u
    所有方法同时支持 numpy 和 torch Tensor（用于可微分安全层）
    """

    def __init__(self, track_width: float, wheel_radius: float, dt: float):
        self.B = track_width      # 轨距
        self.r = wheel_radius     # 链轮半径
        self.dt = dt

    # ------------------------------------------------------------------
    # numpy 接口：用于环境仿真和GP数据收集
    # ------------------------------------------------------------------

    def f_np(self, x: np.ndarray) -> np.ndarray:
        """漂移项 f(x)，控制输入为零时的状态导数"""
        X, Y, psi, v = x
        return np.array([
            v * np.cos(psi),
            v * np.sin(psi),
            0.0,   # psi_dot 的漂移项为0（纯差速驱动）
            0.0    # v_dot 的漂移项为0
        ])

    def g_np(self, x: np.ndarray) -> np.ndarray:
        """
        输入矩阵 g(x)，shape (4, 2)
        u = [omega_L, omega_R]
        """
        X, Y, psi, v = x
        r, B = self.r, self.B
        # vL = r * omega_L, vR = r * omega_R
        # Xdot   = (vL + vR)/2 * cos(psi) = r/2*(oL+oR)*cos(psi)
        # Ydot   = (vL + vR)/2 * sin(psi) = r/2*(oL+oR)*sin(psi)
        # psidot = (vR - vL)/B             = r/B*(oR - oL)
        # vdot   = (vR + vL) / (2*dt)      ≈ 用euler: v_{t+1} = (vR+vL)/2
        #          这里 g 对应的是 v 的增量方向，近似为:
        #          vdot ≈ (r*omega_R + r*omega_L)/2  / 1  (单位化)
        # 注: v_dot 的精确动力学需要牵引力模型，此处用运动学近似
        return np.array([
            [r / 2 * np.cos(psi),  r / 2 * np.cos(psi)],
            [r / 2 * np.sin(psi),  r / 2 * np.sin(psi)],
            [-r / B,               r / B              ],
            [r / 2,                r / 2              ]
        ])

    def nominal_step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """欧拉积分一步，返回 x_{t+1} 的标称预测"""
        xdot = self.f_np(x) + self.g_np(x) @ u
        return x + self.dt * xdot

    def compute_yaw_rate(self, u: np.ndarray) -> float:
        """由轮速指令计算横摆角速度（标称）"""
        vL = self.r * u[0]
        vR = self.r * u[1]
        return (vR - vL) / self.B

    # ------------------------------------------------------------------
    # torch 接口：用于可微分安全层中的约束计算
    # ------------------------------------------------------------------

    def f_torch(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (..., 4)
        返回 f(x): shape (..., 4)
        """
        X, Y, psi, v = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
        zeros = torch.zeros_like(v)
        return torch.stack([
            v * torch.cos(psi),
            v * torch.sin(psi),
            zeros,
            zeros
        ], dim=-1)

    def g_torch(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (..., 4)
        返回 g(x): shape (..., 4, 2)
        """
        X, Y, psi, v = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
        r, B = self.r, self.B
        zeros = torch.zeros_like(psi)
        col_L = torch.stack([
            r / 2 * torch.cos(psi),
            r / 2 * torch.sin(psi),
            -r / B * torch.ones_like(psi),
            r / 2 * torch.ones_like(psi)
        ], dim=-1)
        col_R = torch.stack([
            r / 2 * torch.cos(psi),
            r / 2 * torch.sin(psi),
            r / B * torch.ones_like(psi),
            r / 2 * torch.ones_like(psi)
        ], dim=-1)
        return torch.stack([col_L, col_R], dim=-1)  # (..., 4, 2)
