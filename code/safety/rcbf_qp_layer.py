"""
RCBF-QP 安全层

求解：
  (u^S, eps) = argmin  ||u^S||^2 + l * eps^2
               s.t.  A_i @ (u^RL + u^S) >= b_i - eps,  for each constraint i

其中 A_i, b_i 由 RCBF 条件线性化得到。

两种模式：
  1. non-differentiable: 用 cvxpy + OSQP 求解（速度快，用于推理/评估）
  2. differentiable    : 用 cvxpylayers 嵌入 PyTorch 计算图（用于训练，梯度反传）
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict

try:
    import cvxpy as cp
    from cvxpylayers.torch import CvxpyLayer
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    print("[WARN] cvxpy/cvxpylayers 未安装，可微分QP层不可用")

from safety.rcbf_constraints import RCBFConstraints
from config import VehicleConfig, RCBFConfig


class RCBFQPLayer(nn.Module):
    """
    可微分 RCBF-QP 安全层

    输入:
      u_rl    : SAC原始动作, shape (batch, 2)
      x       : 物理状态 [X, Y, psi, v], shape (batch, 4)
      u_prev  : 上一步动作, shape (batch, 2)
      mu_d    : GP扰动均值, shape (batch, 4)
      sigma_d : GP扰动标准差, shape (batch, 4)

    输出:
      u_star  : 安全修正后动作 u_rl + u_S, shape (batch, 2)
      u_s     : 安全补偿量, shape (batch, 2)
      info    : 各约束激活情况等
    """

    def __init__(
        self,
        vcfg: VehicleConfig,
        rcbf_cfg: RCBFConfig,
        differentiable: bool = True,
    ):
        super().__init__()
        self.constraints = RCBFConstraints(vcfg, rcbf_cfg)
        self.slack_penalty = rcbf_cfg.slack_penalty
        self.differentiable = differentiable and CVXPY_AVAILABLE
        self.B = vcfg.track_width
        self.r = vcfg.wheel_radius
        self.max_u = vcfg.max_wheel_speed
        self.max_du = vcfg.max_delta_speed
        self.max_yaw = vcfg.max_yaw_rate
        self.alpha_yaw = rcbf_cfg.alpha_yaw
        self.alpha_sat = rcbf_cfg.alpha_sat
        self.alpha_rate = rcbf_cfg.alpha_rate

        if self.differentiable:
            self._build_cvxpy_layer()

    def _build_cvxpy_layer(self):
        """
        构建参数化 QP 的 cvxpylayers 层
        约束数量：2（横摆）+ 4（饱和L,R）+ 4（变化率） = 10行
        变量：[u_S (2), eps (1)]

        DPP形式：A @ u_s >= rhs，其中 rhs = b - A @ u_rl（预先计算后作为参数传入）
        """
        n_u = 2
        n_constraints = 10  # 2横摆 + 4饱和 + 4变化率

        u_s = cp.Variable(n_u)        # 安全补偿量
        eps = cp.Variable(1)          # 松弛变量

        # DPP参数：A (n_constraints, n_u), rhs = b - A @ u_rl (n_constraints,)
        A_param = cp.Parameter((n_constraints, n_u))
        rhs_param = cp.Parameter(n_constraints)

        objective = cp.Minimize(
            cp.sum_squares(u_s) + self.slack_penalty * cp.sum_squares(eps)
        )
        constraints = [
            A_param @ u_s >= rhs_param - eps,
            eps >= 0,
        ]
        problem = cp.Problem(objective, constraints)

        self.cvxpy_layer = CvxpyLayer(
            problem,
            parameters=[A_param, rhs_param],
            variables=[u_s, eps],
        )

    def forward(
        self,
        u_rl: torch.Tensor,
        x: torch.Tensor,
        u_prev: torch.Tensor,
        mu_d: torch.Tensor,
        sigma_d: torch.Tensor,
        k_c: float = 2.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        前向传播，支持批量推理和单步在线控制（batch=1）
        """
        batch_size = u_rl.shape[0]
        results_u_star = []
        results_u_s = []

        for i in range(batch_size):
            u_s_i, info_i = self._solve_single(
                u_rl[i], x[i], u_prev[i], mu_d[i], sigma_d[i], k_c
            )
            results_u_star.append(u_rl[i] + u_s_i)
            results_u_s.append(u_s_i)

        u_star = torch.stack(results_u_star, dim=0)
        u_s = torch.stack(results_u_s, dim=0)
        return u_star, u_s, {}

    def _solve_single(
        self,
        u_rl: torch.Tensor,   # (2,)
        x: torch.Tensor,       # (4,)
        u_prev: torch.Tensor,  # (2,)
        mu_d: torch.Tensor,    # (4,)
        sigma_d: torch.Tensor, # (4,)
        k_c: float,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        构建并求解单步 RCBF-QP

        QP 约束矩阵 A, b 的行对应：
          行0: 横摆角速度约束   A_yaw @ u >= b_yaw
          行1: 执行器饱和 L      h_sat_L: u_max^2 - uL^2 >= 0 → 线性化
          行2: 执行器饱和 R
          行3: 控制变化率约束
          行4: 松弛变量非负（由 QP 结构保证，此处预留）
        """
        A, b = self._build_constraint_matrix(u_rl, x, u_prev, mu_d, sigma_d, k_c)

        if self.differentiable:
            # cvxpylayers 求解（梯度可反传）
            # rhs = b - A @ u_rl，使约束为 A @ u_s >= rhs（DPP形式）
            try:
                rhs = b - A @ u_rl
                u_s, eps = self.cvxpy_layer(A, rhs)
                return u_s.squeeze(0) if u_s.dim() > 1 else u_s, {}
            except Exception as e:
                # QP求解失败时回退到零补偿
                return torch.zeros_like(u_rl), {"qp_failed": True}
        else:
            # numpy 求解（快速推理用）
            return self._solve_numpy(u_rl, A, b)

    def _build_constraint_matrix(
        self,
        u_rl: torch.Tensor,
        x: torch.Tensor,
        u_prev: torch.Tensor,
        mu_d: torch.Tensor,
        sigma_d: torch.Tensor,
        k_c: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将所有 RCBF 条件线性化为 A @ u >= b 的形式

        RCBF条件通用形式:
          Lf_h + Lg_h @ u >= -alpha(h) - min_{d}(nabla_h @ d)
        其中 min 项 = nabla_h @ mu_d - kc * |nabla_h| * sigma_d

        对于输入约束（h直接是u的函数），Lf_h=0，Lg_h=∇_u h
        """
        rows_A = []
        rows_b = []

        # --- 约束1: 横摆角速度 ---
        # h_yaw = r_max^2 - (r/B*(uR-uL))^2
        # 为避免非线性，直接用线性化：
        # 横摆角速度约束等价于: -r_max <= r/B*(uR-uL) <= r_max
        # 即两个线性约束:
        #   r/B*(uR-uL) <= r_max  →  r/B*(-uL+uR) <= r_max  →  A1 @ u <= r_max
        #   r/B*(uR-uL) >= -r_max →  r/B*(-uL+uR) >= -r_max →  A2 @ u >= -r_max
        coeff_yaw = self.r / self.B
        # 第一行：[-coeff, +coeff] @ u >= -r_max   (即 (uR-uL) >= -B/r * r_max)
        # 第二行：[+coeff, -coeff] @ u >= -r_max   (即 (uL-uR) >= -B/r * r_max)
        a_yaw1 = torch.tensor([-coeff_yaw, coeff_yaw], device=u_rl.device)
        b_yaw1 = torch.tensor(-self.max_yaw, device=u_rl.device)
        a_yaw2 = torch.tensor([coeff_yaw, -coeff_yaw], device=u_rl.device)
        b_yaw2 = torch.tensor(-self.max_yaw, device=u_rl.device)
        # 注：这里用的是"直接约束u"的形式，适合输入约束
        # 完整RCBF条件（含扰动项）待后续GP集成后完善
        rows_A.extend([a_yaw1, a_yaw2])
        rows_b.extend([b_yaw1, b_yaw2])

        # --- 约束2: 执行器饱和（L, R）---
        # h_sat = u_max^2 - u_k^2 >= 0
        # 线性化（绕当前u_rl点一阶展开）:
        # ∇_u h_sat @ u^S >= -h_sat(u_rl) - alpha * h_sat(u_rl)
        # ∂h_sat_L/∂uL = -2*uL_rl,  ∂h_sat_L/∂uR = 0
        h_sat = self.max_u**2 - u_rl**2  # (2,)
        # 饱和L：约束 uL <= u_max → 线性: [-1,0]@u >= -u_max
        a_sat_L = torch.tensor([-1.0, 0.0], device=u_rl.device)
        b_sat_L = torch.tensor(-self.max_u, device=u_rl.device)
        a_sat_L2 = torch.tensor([1.0, 0.0], device=u_rl.device)
        b_sat_L2 = torch.tensor(-self.max_u, device=u_rl.device)
        # 饱和R
        a_sat_R = torch.tensor([0.0, -1.0], device=u_rl.device)
        b_sat_R = torch.tensor(-self.max_u, device=u_rl.device)
        a_sat_R2 = torch.tensor([0.0, 1.0], device=u_rl.device)
        b_sat_R2 = torch.tensor(-self.max_u, device=u_rl.device)
        rows_A.extend([a_sat_L, a_sat_L2, a_sat_R, a_sat_R2])
        rows_b.extend([b_sat_L, b_sat_L2, b_sat_R, b_sat_R2])

        # --- 约束3: 控制变化率 ---
        # ||u - u_prev||^2 <= du_max^2
        # 线性化：±[1,0], ±[0,1] @ u >= -(du_max + u_prev_k) 等
        for k in range(2):
            a_r1 = torch.zeros(2, device=u_rl.device); a_r1[k] = 1.0
            a_r2 = torch.zeros(2, device=u_rl.device); a_r2[k] = -1.0
            rows_A.append(a_r1)
            rows_A.append(a_r2)
            rows_b.append(u_prev[k] - self.max_du)
            rows_b.append(-u_prev[k] - self.max_du)

        A = torch.stack(rows_A, dim=0)  # (n_constraints, 2)
        b = torch.stack(rows_b, dim=0)  # (n_constraints,)
        return A, b

    def _solve_numpy(
        self,
        u_rl: torch.Tensor,
        A: torch.Tensor,
        b: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """numpy fallback：用OSQP直接求解，不支持梯度"""
        try:
            import osqp
            import scipy.sparse as sp

            A_np = A.detach().cpu().numpy().astype(np.float64)
            b_np = b.detach().cpu().numpy().astype(np.float64)
            u_rl_np = u_rl.detach().cpu().numpy().astype(np.float64)

            n = 3  # [u_S(2), eps(1)]
            P = sp.eye(n, format="csc") * 1.0
            P[2, 2] = self.slack_penalty
            q = np.zeros(n)

            # A_full @ [u_S, eps] >= b_full
            n_c = A_np.shape[0]
            A_full = np.hstack([A_np, -np.ones((n_c, 1))])
            l_full = b_np - A_np @ u_rl_np
            u_full = np.inf * np.ones(n_c)

            solver = osqp.OSQP()
            solver.setup(
                P=P, q=q,
                A=sp.csc_matrix(A_full),
                l=l_full, u=u_full,
                verbose=False, eps_abs=1e-5, eps_rel=1e-5,
            )
            res = solver.solve()

            if res.info.status == "solved":
                u_s_np = res.x[:2]
                u_s = torch.tensor(u_s_np, dtype=u_rl.dtype, device=u_rl.device)
                return u_s, {}
        except Exception as e:
            pass

        return torch.zeros_like(u_rl), {"qp_failed": True}

    def solve_np(
        self,
        u_rl: np.ndarray,
        x: np.ndarray,
        u_prev: np.ndarray,
        mu_d: np.ndarray,
        sigma_d: np.ndarray,
        k_c: float = 2.0,
    ) -> np.ndarray:
        """纯 numpy 接口，用于在线控制推理（无梯度需求）"""
        u_rl_t = torch.tensor(u_rl, dtype=torch.float32)
        x_t = torch.tensor(x, dtype=torch.float32)
        u_prev_t = torch.tensor(u_prev, dtype=torch.float32)
        mu_t = torch.tensor(mu_d, dtype=torch.float32)
        sig_t = torch.tensor(sigma_d, dtype=torch.float32)

        with torch.no_grad():
            A, b = self._build_constraint_matrix(u_rl_t, x_t, u_prev_t, mu_t, sig_t, k_c)
            u_s, _ = self._solve_numpy(u_rl_t, A, b)

        return (u_rl_t + u_s).numpy()
