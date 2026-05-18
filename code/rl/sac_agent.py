"""
SAC-RCBF 智能体

核心训练逻辑：
  1. SAC 策略网络输出 u_RL
  2. RCBF-QP 安全层修正得到 u* = u_RL + u_S
  3. 环境执行 u*，获得奖励和下一状态
  4. 策略梯度对 u* 反传（通过可微分QP层），而非对 u_RL 反传

关键设计（参考 Emam 2025）：
  - Critic 的 Q 值以 (obs, u*) 为输入，而非 (obs, u_RL)
  - 策略损失对 u* 求导，梯度经 cvxpylayers 的 KKT 反传到 u_RL
  - 这使 Actor 感知到安全层的行为，避免"安全层污染Q函数"
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from typing import Dict, Tuple, Optional

from rl.networks import GaussianActor, TwinCritic
from rl.replay_buffer import ReplayBuffer
from safety.rcbf_qp_layer import RCBFQPLayer
from gp.disturbance_gp import DisturbanceGP
from config import SACConfig, VehicleConfig, RCBFConfig, GPConfig


class SACAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        state_dim: int,
        action_scale: float,      # 动作范围 max_u
        sac_cfg: SACConfig,
        vcfg: VehicleConfig,
        rcbf_cfg: RCBFConfig,
        gp_cfg: GPConfig,
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_scale = action_scale
        self.cfg = sac_cfg
        self.device = device

        # --- 网络 ---
        self.actor = GaussianActor(obs_dim, action_dim, sac_cfg.hidden_dim).to(device)
        self.critic = TwinCritic(obs_dim, action_dim, sac_cfg.hidden_dim).to(device)
        self.critic_target = deepcopy(self.critic).to(device)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        # --- 优化器 ---
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=sac_cfg.learning_rate)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=sac_cfg.learning_rate)

        # --- 自动熵调整 ---
        self.target_entropy = -action_dim  # 目标熵（SAC默认）
        if sac_cfg.auto_entropy:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=sac_cfg.learning_rate)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.log_alpha = None
            self.alpha = sac_cfg.alpha

        # --- RCBF 安全层 ---
        self.safety_layer = RCBFQPLayer(
            vcfg=vcfg,
            rcbf_cfg=rcbf_cfg,
            differentiable=sac_cfg.use_diff_safety_layer,
        ).to(device)

        # --- GP 扰动估计 ---
        self.gp = DisturbanceGP(gp_cfg)

        # --- 经验回放 ---
        self.replay_buffer = ReplayBuffer(
            obs_dim=obs_dim,
            action_dim=action_dim,
            state_dim=state_dim,
            capacity=sac_cfg.replay_buffer_size,
            device=device,
        )

        # 内部状态
        self._prev_u: np.ndarray = np.zeros(action_dim)
        self.total_steps = 0

    # ------------------------------------------------------------------
    # 动作选择（在线推理）
    # ------------------------------------------------------------------

    def select_action(
        self,
        obs: np.ndarray,
        x_state: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        返回 (u_safe, u_rl)
          u_safe: 经 RCBF-QP 修正后的实际执行动作（已 scale）
          u_rl  : SAC 原始动作（已 scale，未经安全层）
        """
        if self.total_steps < self.cfg.warmup_steps:
            u_rl_raw = np.random.uniform(-1, 1, self.action_dim)
        else:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                u_rl_raw, _ = self.actor(obs_t, deterministic=deterministic)
            u_rl_raw = u_rl_raw.squeeze(0).cpu().numpy()

        u_rl = u_rl_raw * self.action_scale

        # GP 扰动估计
        mu_d, sigma_d = self.gp.predict(x_state)

        # RCBF-QP 安全修正（numpy 接口，无梯度）
        u_prev_t = torch.FloatTensor(self._prev_u).unsqueeze(0).to(self.device)
        u_safe = self.safety_layer.solve_np(
            u_rl=u_rl,
            x=x_state,
            u_prev=self._prev_u,
            mu_d=mu_d,
            sigma_d=sigma_d,
        )
        u_safe = np.clip(u_safe, -self.action_scale, self.action_scale)
        return u_safe, u_rl

    def update_prev_action(self, u: np.ndarray):
        self._prev_u = u.copy()

    # ------------------------------------------------------------------
    # 训练更新
    # ------------------------------------------------------------------

    def update(self) -> Dict[str, float]:
        if len(self.replay_buffer) < self.cfg.batch_size:
            return {}

        batch = self.replay_buffer.sample(self.cfg.batch_size)
        obs = batch["obs"]
        obs_next = batch["obs_next"]
        u_rl = batch["action_rl"]
        u_safe = batch["action_safe"]
        reward = batch["reward"]
        done = batch["done"]
        state = batch["state"]
        state_next = batch["state_next"]

        metrics = {}

        # === 1. 更新 Critic ===
        critic_loss, metrics_c = self._critic_update(
            obs, u_safe, reward, done, obs_next, state_next
        )
        metrics.update(metrics_c)

        # === 2. 更新 Actor（含可微分安全层梯度反传）===
        actor_loss, metrics_a = self._actor_update(obs, state)
        metrics.update(metrics_a)

        # === 3. 软更新目标网络 ===
        self._soft_update()

        # === 4. 自动熵调整 ===
        if self.cfg.auto_entropy:
            alpha_loss = self._alpha_update(obs)
            metrics["alpha_loss"] = alpha_loss
            self.alpha = self.log_alpha.exp().item()
            metrics["alpha"] = self.alpha

        return metrics

    def _critic_update(
        self,
        obs: torch.Tensor,
        u_safe: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        obs_next: torch.Tensor,
        state_next: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        with torch.no_grad():
            u_rl_next, log_pi_next = self.actor(obs_next)
            u_rl_next_scaled = u_rl_next * self.action_scale

            # 从 batch 中随机抽取 32 个子样本调用 solve_np，其余用 u_rl_next_scaled 近似
            # 子样本保证安全修正偏差被纳入训练信号，同时将 QP 调用次数从 256 降至 32（8x 提速）
            sub_n = min(32, u_rl_next_scaled.shape[0])
            idx = torch.randperm(u_rl_next_scaled.shape[0], device=self.device)[:sub_n]
            u_safe_next = u_rl_next_scaled.clone()
            u_safe_next_sub_np = np.stack([
                self.safety_layer.solve_np(
                    u_rl=u_rl_next_scaled[idx[i]].cpu().numpy(),
                    x=state_next[idx[i]].cpu().numpy(),
                    u_prev=np.zeros(self.action_dim),
                    mu_d=np.zeros(4),
                    sigma_d=np.ones(4),
                )
                for i in range(sub_n)
            ], axis=0)
            u_safe_next[idx] = torch.FloatTensor(u_safe_next_sub_np).to(self.device)

            q_target_next = self.critic_target.min_q(obs_next, u_safe_next)
            q_target = reward + (1.0 - done) * self.cfg.gamma * (
                q_target_next - self.alpha * log_pi_next
            )
            # Q值上界：单步最大奖励约0，下界按 gamma 折扣累积估算
            q_min = -self.action_scale ** 2 / (1.0 - self.cfg.gamma)
            q_target = q_target.clamp(q_min, 0.0)

        q1, q2 = self.critic(obs, u_safe)
        critic_loss = F.mse_loss(q1, q_target.squeeze(-1)) + F.mse_loss(q2, q_target.squeeze(-1))

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)
        self.critic_opt.step()

        return critic_loss, {
            "critic_loss": critic_loss.item(),
            "q1_mean": q1.mean().item(),
            "q2_mean": q2.mean().item(),
        }

    def _actor_update(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Actor 损失：最大化 Q(s, u*)
        若使用可微分安全层，梯度通过 QP 的 KKT 条件反传到 u_RL
        """
        u_rl, log_pi = self.actor(obs)
        u_rl_scaled = u_rl * self.action_scale

        if self.cfg.use_diff_safety_layer:
            # 可微分安全层路径：梯度保留
            # 用小子批次（16样本）做QP，降低cvxpylayers的逐样本调用开销
            sub_n = min(16, obs.shape[0])
            idx = torch.randperm(obs.shape[0], device=self.device)[:sub_n]
            u_rl_sub = u_rl_scaled[idx]
            state_sub = state[idx]
            u_prev_batch = torch.zeros_like(u_rl_sub)
            mu_d_batch = torch.zeros(sub_n, 4, device=self.device)
            sigma_d_batch = torch.ones(sub_n, 4, device=self.device)

            u_star_sub, u_s_sub, _ = self.safety_layer(
                u_rl=u_rl_sub,
                x=state_sub,
                u_prev=u_prev_batch,
                mu_d=mu_d_batch,
                sigma_d=sigma_d_batch,
            )
            # 子批次的Q值作为actor梯度信号，全批次obs用于alpha更新
            q_pi = self.critic.min_q(obs[idx], u_star_sub.float())
            actor_loss = (self.alpha * log_pi[idx] - q_pi).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_opt.step()
            return actor_loss, {
                "actor_loss": actor_loss.item(),
                "log_pi_mean": log_pi.mean().item(),
            }
        else:
            u_for_q = u_rl_scaled

        q_pi = self.critic.min_q(obs, u_for_q)
        actor_loss = (self.alpha * log_pi - q_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        return actor_loss, {
            "actor_loss": actor_loss.item(),
            "log_pi_mean": log_pi.mean().item(),
        }

    def _alpha_update(self, obs: torch.Tensor) -> float:
        with torch.no_grad():
            _, log_pi = self.actor(obs)
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        return alpha_loss.item()

    def _soft_update(self):
        tau = self.cfg.tau
        for p, p_t in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_t.data.copy_(tau * p.data + (1 - tau) * p_t.data)

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def store_transition(
        self,
        obs: np.ndarray,
        u_rl: np.ndarray,
        u_safe: np.ndarray,
        obs_next: np.ndarray,
        reward: float,
        done: bool,
        x_state: np.ndarray,
        x_next: np.ndarray,
        nominal_xdot: np.ndarray,
        true_xdot: np.ndarray,
    ):
        """存储转换并更新GP"""
        self.replay_buffer.add(obs, u_rl, u_safe, obs_next, reward, done, x_state, x_next)

        # 更新GP：残差 = 真实状态导数 - 标称模型预测
        residual = true_xdot - nominal_xdot
        self.gp.add_residual(x_state, residual)

        self.total_steps += 1

    def save(self, path: str):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
