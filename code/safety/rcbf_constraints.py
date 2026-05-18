"""
RCBF 约束函数定义

各约束的障碍函数 h(x, u)，梯度 ∇h，以及 Lie 导数 Lf_h, Lg_h

当前版本：
  - h_yaw   : 横摆角速度约束（相对度 1）√
  - h_sat   : 执行器饱和约束（输入约束，相对度 1）√
  - h_rate  : 控制变化率约束（输入约束，相对度 1）√
  - h_slip  : 滑移率约束（相对度 2，TODO: HOCBF）

HOCBF（h_slip）设计思路（待推导后填充）:
  i_k = 1 - v_k_actual / (omega_k * r)
  h_slip = i_max^2 - i_k^2
  ∂h_slip/∂x = ...  (需要通过动力学方程展开)
  相对度分析完成后实现 psi_0, psi_1, RCBF条件
"""

import numpy as np
import torch
from typing import Tuple, Optional
from config import VehicleConfig, RCBFConfig


class RCBFConstraints:
    """
    计算所有约束对应的 RCBF 条件所需量
    为保证 QP 层可微分，核心计算用 torch 实现

    约束均转化为 Lf_h + Lg_h @ u >= -alpha(h) - min_d (nabla_h @ d) + eps
    """

    def __init__(self, vcfg: VehicleConfig, rcbf_cfg: RCBFConfig):
        self.B = vcfg.track_width
        self.r = vcfg.wheel_radius
        self.max_yaw = vcfg.max_yaw_rate
        self.max_u = vcfg.max_wheel_speed
        self.max_du = vcfg.max_delta_speed
        self.alpha_yaw = rcbf_cfg.alpha_yaw
        self.alpha_sat = rcbf_cfg.alpha_sat
        self.alpha_rate = rcbf_cfg.alpha_rate

    # ------------------------------------------------------------------
    # 1. 横摆角速度约束  h_yaw = r_max^2 - yaw_rate^2
    #    yaw_rate = (vR - vL)/B = r/B * (omega_R - omega_L)
    #    相对度 1：Lg_h_yaw != 0
    # ------------------------------------------------------------------

    def h_yaw(self, u: torch.Tensor) -> torch.Tensor:
        """h_yaw = r_max^2 - (r/B*(uR - uL))^2"""
        yaw_rate = self.r / self.B * (u[..., 1] - u[..., 0])
        return self.max_yaw**2 - yaw_rate**2

    def lf_h_yaw(self, x: torch.Tensor) -> torch.Tensor:
        """Lf_h_yaw: f(x) 中不含控制量影响横摆角速度，但横摆角速度是u的直接函数
        注：h_yaw 是纯输入约束（只依赖u），Lf_h_yaw = 0"""
        return torch.zeros(x.shape[:-1], device=x.device)

    def lg_h_yaw(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lg_h_yaw: ∇_u h_yaw, shape (..., 2)
        h_yaw = r_max^2 - (r/B*(uR-uL))^2
        ∂h_yaw/∂uL = 2*(r/B)^2*(uR-uL)
        ∂h_yaw/∂uR = -2*(r/B)^2*(uR-uL)
        注：这是关于u的梯度，需要当前u值
        """
        # 此处返回结构系数，具体值在QP中代入当前u_RL
        # 见 rcbf_qp_layer.py 中的组装逻辑
        # 这里返回 shape (..., 2) 的系数，乘以 (uR - uL)
        coeff = 2.0 * (self.r / self.B) ** 2
        # 方向：[+coeff, -coeff] * (uR - uL)
        # 为线性化处理，在QP中将 h_yaw 的约束直接写成线性形式
        # 详见 build_qp_constraints
        ones = torch.ones(x.shape[:-1], device=x.device)
        return torch.stack([ones, -ones], dim=-1) * coeff  # 待乘 (uR-uL)

    def alpha_h_yaw(self, h_val: torch.Tensor) -> torch.Tensor:
        return self.alpha_yaw * h_val

    # ------------------------------------------------------------------
    # 2. 执行器饱和约束  h_sat_k = u_max^2 - u_k^2  (k = L, R)
    #    直接作用于控制输入，相对度 1
    # ------------------------------------------------------------------

    def h_sat(self, u: torch.Tensor) -> torch.Tensor:
        """h_sat = [u_max^2 - uL^2, u_max^2 - uR^2], shape (..., 2)"""
        return self.max_u**2 - u**2

    def lg_h_sat(self) -> torch.Tensor:
        """∇_u h_sat: diag(-2uL, -2uR)，在QP中代入当前u"""
        return None  # 在 build_qp_constraints 中内联计算

    def alpha_h_sat(self, h_val: torch.Tensor) -> torch.Tensor:
        return self.alpha_sat * h_val

    # ------------------------------------------------------------------
    # 3. 控制变化率约束  h_rate = du_max^2 - ||u - u_prev||^2
    #    扩展状态方式处理，相对度 1
    # ------------------------------------------------------------------

    def h_rate(self, u: torch.Tensor, u_prev: torch.Tensor) -> torch.Tensor:
        """h_rate = du_max^2 - sum((u - u_prev)^2)"""
        delta = u - u_prev
        return self.max_du**2 - torch.sum(delta**2, dim=-1)

    def alpha_h_rate(self, h_val: torch.Tensor) -> torch.Tensor:
        return self.alpha_rate * h_val

    # ------------------------------------------------------------------
    # 4. 滑移率约束（TODO: HOCBF）
    # ------------------------------------------------------------------

    def h_slip(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        TODO: 高阶RCBF实现
        占位符：当前版本跳过此约束

        理论框架：
          i_k = 1 - v_k_actual / (omega_k * r)
          h_slip_k = i_max^2 - i_k^2
          相对度分析后用HOCBF:
            psi_0 = h_slip
            psi_1 = psi_0_dot + gamma_1 * psi_0
          RCBF条件施加在 psi_1 上
        """
        raise NotImplementedError(
            "滑移率RCBF(HOCBF)待理论推导后实现。"
            "需要确定 h_slip 对控制输入的相对度，并推导 Lie 导数表达式。"
        )

    # ------------------------------------------------------------------
    # 辅助：最坏情况扰动项
    # ------------------------------------------------------------------

    def worst_case_disturbance(
        self,
        nabla_h: torch.Tensor,  # (..., state_dim)
        mu_d: torch.Tensor,     # (..., state_dim)
        sigma_d: torch.Tensor,  # (..., state_dim)
        k_c: float = 2.0,
    ) -> torch.Tensor:
        """
        min_{d in D(x)} nabla_h @ d
          = nabla_h @ mu_d - k_c * ||nabla_h|| * sigma_d 的逐元素下确界
          = sum_i (nabla_h_i * mu_d_i - k_c * |nabla_h_i| * sigma_d_i)
        """
        return torch.sum(
            nabla_h * mu_d - k_c * torch.abs(nabla_h) * sigma_d, dim=-1
        )
