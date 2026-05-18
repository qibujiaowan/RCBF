"""
经验回放缓冲区
存储 (obs, action_rl, action_safe, obs_next, reward, done, x_state)
其中 action_rl 是 SAC 原始动作，action_safe 是经安全层修正后的实际执行动作
两者均需存储，用于可微分安全层的梯度计算
"""

import numpy as np
import torch
from typing import Dict, Tuple


class ReplayBuffer:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        state_dim: int,
        capacity: int = 100_000,
        device: str = "cpu",
    ):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.obs_next = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action_rl = np.zeros((capacity, action_dim), dtype=np.float32)
        self.action_safe = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)      # 当前物理状态 [X,Y,psi,v]
        self.state_next = np.zeros((capacity, state_dim), dtype=np.float32) # 下一步物理状态，用于修正 u_safe_next

    def add(
        self,
        obs: np.ndarray,
        action_rl: np.ndarray,
        action_safe: np.ndarray,
        obs_next: np.ndarray,
        reward: float,
        done: bool,
        state: np.ndarray,
        state_next: np.ndarray,
    ):
        self.obs[self.ptr] = obs
        self.obs_next[self.ptr] = obs_next
        self.action_rl[self.ptr] = action_rl
        self.action_safe[self.ptr] = action_safe
        self.reward[self.ptr] = reward
        self.done[self.ptr] = float(done)
        self.state[self.ptr] = state
        self.state_next[self.ptr] = state_next

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        to_t = lambda x: torch.FloatTensor(x[idx]).to(self.device)
        return {
            "obs":         to_t(self.obs),
            "obs_next":    to_t(self.obs_next),
            "action_rl":   to_t(self.action_rl),
            "action_safe": to_t(self.action_safe),
            "reward":      to_t(self.reward),
            "done":        to_t(self.done),
            "state":       to_t(self.state),
            "state_next":  to_t(self.state_next),
        }

    def __len__(self):
        return self.size
