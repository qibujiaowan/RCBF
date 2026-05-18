"""
基于标称运动学的线性化MPC教师控制器

在参考轨迹点处对 f(x)+g(x)u 做 Jacobian 线性化，得到 LTV 系统，
用 scipy SLSQP 求解有限时域最优控制问题。

用途：为行为克隆预训练生成高质量 (obs, u) 示范数据。
"""

import numpy as np
from scipy.optimize import minimize
from typing import Tuple

from config import VehicleConfig
from envs.vehicle_kinematics import TrackedVehicleKinematics
from envs.tracked_vehicle_env import ReferencePath


class MPCTeacher:
    """
    线性化 MPC 控制器

    预测步长 N，在每个参考点处线性化运动学方程。
    代价函数：误差二次型 + 控制量变化率惩罚
    约束：轮速幅值、偏航率、轮速变化率
    """

    def __init__(self, vcfg: VehicleConfig, N: int = 10):
        self.vcfg = vcfg
        self.N = N
        self.kinematics = TrackedVehicleKinematics(
            track_width=vcfg.track_width,
            wheel_radius=vcfg.wheel_radius,
            dt=vcfg.dt,
        )
        self.dt = vcfg.dt
        self.max_u = vcfg.max_wheel_speed
        self.max_du = vcfg.max_delta_speed
        self.max_yaw = vcfg.max_yaw_rate

        # MPC代价权重
        self.Q = np.diag([5.0, 3.0, 1.0, 0.0])  # [X, Y, psi, v] 误差权重
        self.R = np.diag([0.1, 0.1])              # 控制量变化率权重

        # 参考轮速（沿参考路径匀速行驶的标称输入）
        v_ref = 1.5
        self._u_ref = np.array([v_ref / vcfg.wheel_radius, v_ref / vcfg.wheel_radius])

    # ------------------------------------------------------------------

    def get_action(
        self,
        x_state: np.ndarray,
        ref_path: ReferencePath,
        path_idx: int,
        u_prev: np.ndarray,
    ) -> np.ndarray:
        """
        给定当前物理状态和参考路径，返回 MPC 最优控制输入。

        x_state : [X, Y, psi, v]
        ref_path : ReferencePath 对象
        path_idx : 当前路径索引
        u_prev   : 上一步控制输入（用于变化率约束）
        返回     : u = [omega_L, omega_R]，已 clip 到 [-max_u, max_u]
        """
        N = self.N
        n_u = 2

        # 收集 N 个参考轨迹点（超出路径末端则用最后一点）
        refs = []
        for k in range(N):
            idx = min(path_idx + k, len(ref_path._path) - 1)
            row = ref_path._path[idx]
            refs.append(row[:4])  # [X_ref, Y_ref, psi_ref, v_ref]
        refs = np.array(refs)  # (N, 4)

        # 在当前状态处计算线性化矩阵
        A_list, B_list = self._linearize_sequence(x_state, refs)

        # 优化变量：展开的控制序列 U = [u_0, u_1, ..., u_{N-1}]，shape (N*2,)
        u0 = np.tile(self._u_ref, N)  # 初值为参考轮速

        def cost(U):
            U = U.reshape(N, n_u)
            x = x_state.copy()
            J = 0.0
            u_k_prev = u_prev.copy()
            for k in range(N):
                e = x - refs[k]
                e[2] = np.arctan2(np.sin(e[2]), np.cos(e[2]))  # 航向误差归一化
                J += e @ self.Q @ e
                du = U[k] - u_k_prev
                J += du @ self.R @ du
                # 用线性化模型传播状态
                x = A_list[k] @ x + B_list[k] @ U[k]
                u_k_prev = U[k]
            return J

        # 约束
        constraints = []
        r, B = self.vcfg.wheel_radius, self.vcfg.track_width

        for k in range(N):
            k_ = k  # closure capture

            # 偏航率约束：|(omega_R - omega_L)*r/B| <= max_yaw
            constraints.append({
                "type": "ineq",
                "fun": lambda U, k=k_: (
                    self.max_yaw - abs((U[2*k+1] - U[2*k]) * r / B)
                ),
            })

            # 轮速变化率约束（相对上一步）
            if k == 0:
                constraints.append({
                    "type": "ineq",
                    "fun": lambda U, k=k_: (
                        self.max_du - np.max(np.abs(U[2*k:2*k+2] - u_prev))
                    ),
                })
            else:
                constraints.append({
                    "type": "ineq",
                    "fun": lambda U, k=k_: (
                        self.max_du - np.max(np.abs(U[2*k:2*k+2] - U[2*(k-1):2*(k-1)+2]))
                    ),
                })

        bounds = [(-self.max_u, self.max_u)] * (N * n_u)

        result = minimize(
            cost, u0, method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 50, "ftol": 1e-4},
        )

        u_opt = result.x[:2]  # 只取第一步控制量
        return np.clip(u_opt, -self.max_u, self.max_u)

    # ------------------------------------------------------------------

    def _linearize_sequence(
        self,
        x0: np.ndarray,
        refs: np.ndarray,
    ) -> Tuple[list, list]:
        """
        在参考轨迹点序列处线性化，返回 A_k, B_k 列表。
        A_k = I + dt * dF/dx|_{x_ref, u_ref}  （4×4）
        B_k = dt * g(x_ref)                    （4×2）
        """
        dt = self.dt
        A_list, B_list = [], []
        x = x0.copy()
        for k in range(self.N):
            x_ref = refs[k]
            psi_ref = x_ref[2]
            v_ref = x_ref[3]
            r, B_w = self.vcfg.wheel_radius, self.vcfg.track_width

            # dF/dx（在 x_ref, u_ref 处）
            # F = f(x) + g(x)*u_ref
            # df/dx:
            #   d(v*cos(psi))/dpsi = -v*sin(psi)   d(...)/dv = cos(psi)
            #   d(v*sin(psi))/dpsi =  v*cos(psi)   d(...)/dv = sin(psi)
            #   其余为0
            dFdx = np.zeros((4, 4))
            dFdx[0, 2] = -v_ref * np.sin(psi_ref)
            dFdx[0, 3] = np.cos(psi_ref)
            dFdx[1, 2] = v_ref * np.cos(psi_ref)
            dFdx[1, 3] = np.sin(psi_ref)

            A_k = np.eye(4) + dt * dFdx
            B_k = dt * self.kinematics.g_np(x_ref)

            A_list.append(A_k)
            B_list.append(B_k)
            # 用线性化模型传播状态（仅用于预热，影响不大）
            x = A_k @ x + B_k @ self._u_ref

        return A_list, B_list
