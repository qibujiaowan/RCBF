"""
基于高斯过程的扰动在线估计模块

功能：
  - 收集残差观测: y = x_dot_obs - f(x) - g(x)*u
  - 对状态空间各维度独立训练 GP
  - 输出扰动均值 mu_d(x) 和标准差 sigma_d(x)
  - 构造扰动集合 D(x) = mu_d ± k_c * sigma_d

实现策略（应对计算瓶颈）：
  - 维护固定大小的滑动窗口数据集（max_points 个最新观测）
  - 每 update_interval 步批量更新 GP 超参数
  - 推理时用当前 GP 参数直接计算后验，O(N) per query（已有数据固定时）
"""

import numpy as np
from typing import Tuple, Optional, List
from config import GPConfig


class IndependentGP:
    """
    单维度高斯过程，RBF核
    k(x, x') = output_scale * exp(-||x-x'||^2 / (2 * length_scale^2))
    """

    def __init__(self, cfg: GPConfig, dim_idx: int):
        self.cfg = cfg
        self.dim_idx = dim_idx
        self.length_scale = cfg.length_scale
        self.output_scale = cfg.output_scale
        self.noise_var = cfg.noise_var

        # 数据缓冲区
        self._X: np.ndarray = np.zeros((0, cfg.state_dim))  # 输入点
        self._y: np.ndarray = np.zeros(0)                   # 观测值（残差第dim维）

        # 缓存（避免重复计算）
        self._K_inv: Optional[np.ndarray] = None
        self._alpha: Optional[np.ndarray] = None  # K_inv @ y
        self._dirty: bool = True

    def add_observation(self, x: np.ndarray, y_val: float):
        """添加一条观测，必要时丢弃最旧的点（滑动窗口）"""
        if len(self._X) >= self.cfg.max_points:
            self._X = self._X[1:]
            self._y = self._y[1:]
        self._X = np.vstack([self._X, x.reshape(1, -1)])
        self._y = np.append(self._y, y_val)
        self._dirty = True

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """RBF核矩阵，shape (n1, n2)"""
        diff = X1[:, None, :] - X2[None, :, :]   # (n1, n2, d)
        sq_dist = np.sum(diff**2, axis=-1)        # (n1, n2)
        return self.output_scale * np.exp(-sq_dist / (2.0 * self.length_scale**2))

    def _update_cache(self):
        """预计算 K_inv 和 alpha（数据变化后调用）"""
        if not self._dirty or len(self._X) == 0:
            return
        K = self._rbf_kernel(self._X, self._X)
        K += (self.noise_var + 1e-6) * np.eye(len(self._X))
        try:
            L = np.linalg.cholesky(K)
            self._K_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(len(K))))
            self._alpha = self._K_inv @ self._y
        except np.linalg.LinAlgError:
            # Cholesky失败时加更大的正则化
            K += 1e-3 * np.eye(len(K))
            self._K_inv = np.linalg.pinv(K)
            self._alpha = self._K_inv @ self._y
        self._dirty = False

    def predict(self, x_query: np.ndarray) -> Tuple[float, float]:
        """
        返回 (mu, sigma) at x_query
        若无数据则返回零均值和单位方差
        """
        if len(self._X) == 0:
            return 0.0, 1.0

        self._update_cache()
        x_q = x_query.reshape(1, -1)
        k_star = self._rbf_kernel(x_q, self._X).flatten()  # (N,)
        k_ss = self.output_scale  # k(x*, x*)

        mu = float(k_star @ self._alpha)
        var = k_ss - float(k_star @ (self._K_inv @ k_star))
        var = max(var, 1e-8)  # 防止数值负值
        return mu, float(np.sqrt(var))

    def num_points(self) -> int:
        return len(self._X)


class DisturbanceGP:
    """
    多维扰动 GP 估计器
    d(x) ∈ R^{state_dim}，每维独立一个 GP

    使用方式：
      gp = DisturbanceGP(cfg)
      gp.add_residual(x, y)     # y = x_dot_obs - nominal_xdot
      mu, sigma = gp.predict(x) # 返回 shape (state_dim,) 的均值和标准差
    """

    def __init__(self, cfg: GPConfig):
        self.cfg = cfg
        self.state_dim = cfg.state_dim
        self.gps: List[IndependentGP] = [
            IndependentGP(cfg, i) for i in range(cfg.state_dim)
        ]
        self._step_count = 0
        self._pending_X: List[np.ndarray] = []
        self._pending_y: List[np.ndarray] = []

    def add_residual(self, x: np.ndarray, residual: np.ndarray):
        """
        添加一条残差观测
          x        : 状态向量, shape (state_dim,)
          residual : y = x_dot_obs - f(x) - g(x)*u, shape (state_dim,)
        """
        assert len(residual) == self.state_dim
        self._pending_X.append(x.copy())
        self._pending_y.append(residual.copy())
        self._step_count += 1

        if self._step_count % self.cfg.update_interval == 0:
            self._flush()

    def _flush(self):
        """将 pending 数据批量写入各 GP"""
        for x, y in zip(self._pending_X, self._pending_y):
            for i, gp in enumerate(self.gps):
                gp.add_observation(x, y[i])
        self._pending_X.clear()
        self._pending_y.clear()

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        返回扰动均值和标准差
          mu_d   : shape (state_dim,)
          sigma_d: shape (state_dim,)
        """
        mu_d = np.zeros(self.state_dim)
        sigma_d = np.ones(self.state_dim)  # 无数据时用单位标准差（保守）
        for i, gp in enumerate(self.gps):
            mu_d[i], sigma_d[i] = gp.predict(x)
        return mu_d, sigma_d

    def predict_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量预测
          X: shape (batch, state_dim)
          返回 mu (batch, state_dim), sigma (batch, state_dim)
        """
        batch_size = X.shape[0]
        mu_d = np.zeros((batch_size, self.state_dim))
        sigma_d = np.ones((batch_size, self.state_dim))
        for j in range(batch_size):
            mu_d[j], sigma_d[j] = self.predict(X[j])
        return mu_d, sigma_d

    def total_points(self) -> int:
        return self.gps[0].num_points() if self.gps else 0

    def reset(self):
        """重置所有 GP（新 episode 开始时可选调用）"""
        for gp in self.gps:
            gp._X = np.zeros((0, self.cfg.state_dim))
            gp._y = np.zeros(0)
            gp._dirty = True
            gp._K_inv = None
            gp._alpha = None
        self._pending_X.clear()
        self._pending_y.clear()
        self._step_count = 0
