"""
履带车路径跟踪 Gymnasium 环境

"真实世界"：含左右履带独立滑移的非线性运动学
控制器可见：仅状态观测 s = [e_y, e_psi, e_v, kappa_ref, de_y, de_psi]

滑移建模：
  vL_actual = vL * (1 - i_L(t))    i_L ∈ [0, slip_max_L]
  vR_actual = vR * (1 - i_R(t))    i_R ∈ [0, slip_max_R]
  i_k 根据路面附着系数和当前轮速动态确定

路面工况（slip_config）:
  "none"     : i_L = i_R = 0
  "light"    : i_max ≈ 0.05
  "medium"   : i_max ≈ 0.15
  "heavy"    : i_max ≈ 0.30
  "variable" : i_max 在训练中随机阶跃变化
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

from envs.vehicle_kinematics import TrackedVehicleKinematics
from config import VehicleConfig, RewardConfig


SLIP_PROFILES = {
    "none":     {"max_slip": 0.00, "asymmetry": 0.0},
    "light":    {"max_slip": 0.05, "asymmetry": 0.3},
    "medium":   {"max_slip": 0.15, "asymmetry": 0.4},
    "heavy":    {"max_slip": 0.30, "asymmetry": 0.5},
}


class TrackedVehicleEnv(gym.Env):
    """
    观测空间 (6,): [e_y, e_psi, e_v, kappa_ref, de_y, de_psi]
    动作空间 (2,): [omega_L, omega_R] ∈ [-max_speed, max_speed]
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        vehicle_cfg: VehicleConfig = None,
        reward_cfg: RewardConfig = None,
        reference_path: str = "sine",   # "straight", "circle", "sine"
        slip_config: str = "medium",
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.vcfg = vehicle_cfg or VehicleConfig()
        self.rcfg = reward_cfg or RewardConfig()
        self.slip_config = slip_config
        self.render_mode = render_mode

        self.kinematics = TrackedVehicleKinematics(
            track_width=self.vcfg.track_width,
            wheel_radius=self.vcfg.wheel_radius,
            dt=self.vcfg.dt,
        )

        # 动作空间：左右轮速指令
        self.action_space = spaces.Box(
            low=-self.vcfg.max_wheel_speed,
            high=self.vcfg.max_wheel_speed,
            shape=(2,),
            dtype=np.float32,
        )

        # 观测空间：[e_y, e_psi, e_v, kappa_ref, de_y, de_psi]
        obs_high = np.array([5.0, np.pi, 5.0, 2.0, 5.0, 5.0], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-obs_high, high=obs_high, dtype=np.float32
        )

        # 参考路径生成器
        self.ref_path = ReferencePath(reference_path)
        self.reference_path_type = reference_path

        # 内部状态
        self._state: np.ndarray = None          # [X, Y, psi, v]
        self._prev_obs: np.ndarray = None
        self._prev_u: np.ndarray = np.zeros(2)
        self._step_count: int = 0
        self._slip_L: float = 0.0
        self._slip_R: float = 0.0
        self._path_idx: int = 0

        # 用于GP残差收集
        self._last_nominal_xdot: np.ndarray = None

    # ------------------------------------------------------------------
    # Gymnasium 接口
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.ref_path.reset(self.np_random)

        # 在参考路径起点附近初始化，加入小扰动
        init_pose = self.ref_path.get_init_pose()
        self._state = init_pose + self.np_random.uniform(-0.05, 0.05, size=4)
        self._state[2] = init_pose[2]  # 航向角不加大扰动

        self._prev_u = np.zeros(2)
        self._step_count = 0
        self._path_idx = 0
        self._slip_L, self._slip_R = self._sample_slip()

        obs = self._compute_obs()
        self._prev_obs = obs.copy()
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = np.clip(action, -self.vcfg.max_wheel_speed, self.vcfg.max_wheel_speed)

        # 记录标称预测（用于GP残差计算）
        self._last_nominal_xdot = (
            self.kinematics.f_np(self._state)
            + self.kinematics.g_np(self._state) @ action
        )

        # 含滑移的"真实"状态转移
        self._state = self._true_dynamics_step(self._state, action)
        self._step_count += 1

        # 更新路面滑移（variable模式下随机阶跃）
        if self.slip_config == "variable" and self._step_count % 200 == 0:
            self._slip_L, self._slip_R = self._sample_slip()

        # 更新参考路径索引
        self._path_idx = self.ref_path.advance(self._state, self._path_idx)

        obs = self._compute_obs()
        reward = self._compute_reward(obs, action)

        # 约束违反信息（用于评估，不进奖励）
        constraint_info = self._check_constraints(action)

        # 终止条件
        e_y = obs[0]
        terminated = abs(e_y) > 4.0  # 偏离过远
        truncated = self._step_count >= 300

        self._prev_obs = obs.copy()
        self._prev_u = action.copy()

        info = {
            "constraint_violations": constraint_info,
            "slip_L": self._slip_L,
            "slip_R": self._slip_R,
            "path_idx": self._path_idx,
            "nominal_xdot": self._last_nominal_xdot,
            "true_state": self._state.copy(),
        }
        return obs.astype(np.float32), float(reward), terminated, truncated, info

    def get_state(self) -> np.ndarray:
        """返回完整物理状态 [X, Y, psi, v]（用于GP和RCBF）"""
        return self._state.copy()

    def get_nominal_xdot(self, u: np.ndarray) -> np.ndarray:
        """标称模型预测的状态导数，用于GP残差收集"""
        return self.kinematics.f_np(self._state) + self.kinematics.g_np(self._state) @ u

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _true_dynamics_step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        含滑移的真实动力学（仿真"真实世界"）
        滑移使实际轮速缩减，从而产生扰动 d(x)
        """
        vL_cmd = self.vcfg.wheel_radius * u[0]
        vR_cmd = self.vcfg.wheel_radius * u[1]

        # 滑移降低实际速度
        vL_actual = vL_cmd * (1.0 - self._slip_L)
        vR_actual = vR_cmd * (1.0 - self._slip_R)

        X, Y, psi, v = x
        v_new = (vL_actual + vR_actual) / 2.0
        yaw_rate = (vR_actual - vL_actual) / self.vcfg.track_width

        xdot_true = np.array([
            v_new * np.cos(psi),
            v_new * np.sin(psi),
            yaw_rate,
            (v_new - v) / self.vcfg.dt,  # 近似加速度
        ])
        return x + self.vcfg.dt * xdot_true

    def _compute_obs(self) -> np.ndarray:
        """计算SAC观测量 [e_y, e_psi, e_v, kappa_ref, de_y, de_psi]"""
        X, Y, psi, v = self._state
        ref_X, ref_Y, ref_psi, ref_v, kappa = self.ref_path.query(self._path_idx)

        # 横向误差（沿参考点法向）
        dx = X - ref_X
        dy = Y - ref_Y
        e_y = -np.sin(ref_psi) * dx + np.cos(ref_psi) * dy

        # 航向误差
        e_psi = psi - ref_psi
        e_psi = np.arctan2(np.sin(e_psi), np.cos(e_psi))  # 归一化到 [-pi, pi]

        # 速度误差
        e_v = v - ref_v

        obs = np.array([e_y, e_psi, e_v, kappa, 0.0, 0.0], dtype=np.float32)

        # 误差变化率（用前一步差分近似）
        if self._prev_obs is not None:
            obs[4] = (obs[0] - self._prev_obs[0]) / self.vcfg.dt  # de_y/dt
            obs[5] = (obs[1] - self._prev_obs[1]) / self.vcfg.dt  # de_psi/dt

        return obs

    def _compute_reward(self, obs: np.ndarray, u: np.ndarray) -> float:
        e_y, e_psi, e_v = obs[0], obs[1], obs[2]
        delta_u = u - self._prev_u
        r = self.rcfg
        return -(
            r.w_lateral * e_y**2
            + r.w_heading * e_psi**2
            + r.w_velocity * e_v**2
            + r.w_smooth * np.sum(delta_u**2)
        )

    def _check_constraints(self, u: np.ndarray) -> Dict[str, bool]:
        """检查各约束是否违反（仅用于统计，不影响奖励）"""
        yaw_rate = self.kinematics.compute_yaw_rate(u)
        return {
            "yaw_rate": abs(yaw_rate) > self.vcfg.max_yaw_rate,
            "actuator_sat": np.any(np.abs(u) > self.vcfg.max_wheel_speed),
            "rate_limit": np.any(
                np.abs(u - self._prev_u) > self.vcfg.max_delta_speed
            ),
        }

    def _sample_slip(self) -> Tuple[float, float]:
        """根据路面工况采样当前滑移率"""
        if self.slip_config == "none":
            return 0.0, 0.0
        if self.slip_config == "variable":
            key = self.np_random.choice(["light", "medium", "heavy"])
        else:
            key = self.slip_config
        profile = SLIP_PROFILES[key]
        max_s = profile["max_slip"]
        asym = profile["asymmetry"]
        slip_L = self.np_random.uniform(0, max_s)
        # 左右不对称滑移（差速工况）
        slip_R = self.np_random.uniform(0, max_s * (1 + asym))
        slip_R = min(slip_R, 0.95)  # 物理上限
        return float(slip_L), float(slip_R)


# ------------------------------------------------------------------
# 参考路径生成器
# ------------------------------------------------------------------

class ReferencePath:
    """生成简单参考路径并提供最近点查询"""

    def __init__(self, path_type: str = "sine", num_points: int = 600):
        self.path_type = path_type
        self.num_points = num_points
        self._path: np.ndarray = None  # shape (N, 5): [X, Y, psi, v, kappa]

    def reset(self, rng: np.random.Generator):
        self._path = self._generate(rng)

    def _generate(self, rng: np.random.Generator) -> np.ndarray:
        s = np.linspace(0, 100, self.num_points)
        v_ref = 1.5

        if self.path_type == "straight":
            X = s
            Y = np.zeros_like(s)
        elif self.path_type == "circle":
            R = 10.0
            theta = s / R
            X = R * np.sin(theta)
            Y = R * (1 - np.cos(theta))
        elif self.path_type == "sine":
            A = 2.0
            freq = 0.05
            X = s
            Y = A * np.sin(2 * np.pi * freq * s)
        else:
            raise ValueError(f"Unknown path type: {self.path_type}")

        # 计算切线方向和曲率
        dX = np.gradient(X, s)
        dY = np.gradient(Y, s)
        psi = np.arctan2(dY, dX)

        ddX = np.gradient(dX, s)
        ddY = np.gradient(dY, s)
        kappa = (dX * ddY - dY * ddX) / (dX**2 + dY**2 + 1e-8) ** 1.5

        v = np.full_like(s, v_ref)

        return np.stack([X, Y, psi, v, kappa], axis=1)

    def get_init_pose(self) -> np.ndarray:
        """返回路径起点的初始状态 [X, Y, psi, v]"""
        return self._path[0, :4].copy()

    def query(self, idx: int) -> Tuple[float, float, float, float, float]:
        """查询路径点信息"""
        idx = min(idx, len(self._path) - 1)
        row = self._path[idx]
        return row[0], row[1], row[2], row[3], row[4]

    def advance(self, state: np.ndarray, current_idx: int) -> int:
        """推进路径索引到距当前位置最近的前向点"""
        X, Y = state[0], state[1]
        search_end = min(current_idx + 50, len(self._path))
        pts = self._path[current_idx:search_end, :2]
        dists = np.linalg.norm(pts - np.array([X, Y]), axis=1)
        return current_idx + int(np.argmin(dists))
