"""
SAC 神经网络定义

Actor  : 高斯策略网络，输出动作均值和对数标准差（重参数化采样）
Critic : 双 Q 网络（twin critics，对抗过估计）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianActor(nn.Module):
    """
    SAC 高斯策略网络
    输入: 观测 s ∈ R^{obs_dim}
    输出: 经 tanh 压缩后的动作 a ∈ [-1, 1]^{action_dim}，以及对数概率

    动作在 [-1,1] 范围内，由外部 scale 变换到 [-max_u, max_u]
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回 (action, log_prob)
        action: tanh 压缩后，shape (..., action_dim)
        log_prob: shape (...,)
        """
        h = self.trunk(obs)
        mean = self.mean_layer(h)
        log_std = self.log_std_layer(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()

        dist = Normal(mean, std)

        if deterministic:
            raw_action = mean
        else:
            raw_action = dist.rsample()  # 重参数化采样

        # tanh 压缩 + 对应的 log prob 修正
        action = torch.tanh(raw_action)
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        log_prob -= torch.log(1 - action**2 + 1e-6).sum(dim=-1)

        return action, log_prob

    def get_mean(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.trunk(obs)
        return torch.tanh(self.mean_layer(h))


class TwinCritic(nn.Module):
    """
    双 Q 网络
    输入: (obs, action)
    输出: (Q1, Q2) 两个独立估计
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        input_dim = obs_dim + action_dim
        self.Q1 = MLP(input_dim, 1, hidden_dim)
        self.Q2 = MLP(input_dim, 1, hidden_dim)

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, action], dim=-1)
        return self.Q1(x).squeeze(-1), self.Q2(x).squeeze(-1)

    def min_q(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(obs, action)
        return torch.min(q1, q2)
