"""
行为克隆预训练

用 MPC 教师在环境中采集 (obs, u_mpc) 示范数据，
对 actor 的 trunk + mean_layer 做监督回归，
使 SAC 从一个合理的初始策略出发，跳过随机探索阶段。
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple

from config import VehicleConfig, TrainConfig
from envs.tracked_vehicle_env import TrackedVehicleEnv
from rl.sac_agent import SACAgent
from pretrain.mpc_teacher import MPCTeacher


def collect_bc_data(
    env: TrackedVehicleEnv,
    mpc: MPCTeacher,
    action_scale: float,
    n_steps: int = 5000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    用 MPC 教师 rollout 环境，收集 (obs, u_normalized) 对。

    u_normalized = u_mpc / action_scale，对应 actor 的 tanh 输出空间 [-1, 1]。
    对已经超出 [-1,1] 的动作用 arctanh 做逆映射，但为简单起见直接 clip。
    """
    obs_list: List[np.ndarray] = []
    u_list: List[np.ndarray] = []

    obs, _ = env.reset()
    x_state = env.get_state()
    u_prev = np.zeros(2)
    collected = 0

    print(f"收集 BC 数据中（目标 {n_steps} 步）...")
    while collected < n_steps:
        u_mpc = mpc.get_action(x_state, env.ref_path, env._path_idx, u_prev)

        obs_list.append(obs.copy())
        # 归一化到 [-1, 1]：actor 输出 tanh(z)，故目标为 u/action_scale
        u_norm = np.clip(u_mpc / action_scale, -1.0, 1.0)
        u_list.append(u_norm)

        obs_next, _, terminated, truncated, _ = env.step(u_mpc)
        u_prev = u_mpc.copy()
        collected += 1

        if terminated or truncated:
            obs, _ = env.reset()
            x_state = env.get_state()
            u_prev = np.zeros(2)
        else:
            obs = obs_next
            x_state = env.get_state()

    print(f"  已收集 {collected} 步数据")
    return np.array(obs_list, dtype=np.float32), np.array(u_list, dtype=np.float32)


def pretrain_bc(
    agent: SACAgent,
    env: TrackedVehicleEnv,
    train_cfg: TrainConfig,
) -> None:
    """
    行为克隆预训练主函数。

    只优化 actor 的 trunk 和 mean_layer，log_std_layer 保持随机初始化，
    确保 SAC 后续自动熵调整仍能正常工作。
    """
    vcfg = agent.safety_layer.vcfg if hasattr(agent.safety_layer, "vcfg") else None
    # 通过 env 获取 vcfg
    vcfg = env.vcfg
    action_scale = agent.action_scale
    device = agent.device
    n_steps = train_cfg.bc_pretrain_steps
    n_epochs = train_cfg.bc_pretrain_epochs

    mpc = MPCTeacher(vcfg)

    # 1. 收集数据
    obs_np, u_np = collect_bc_data(env, mpc, action_scale, n_steps)

    obs_t = torch.FloatTensor(obs_np).to(device)
    u_t = torch.FloatTensor(u_np).to(device)
    dataset = TensorDataset(obs_t, u_t)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    # 2. 只训练 trunk + mean_layer
    actor = agent.actor
    bc_params = list(actor.trunk.parameters()) + list(actor.mean_layer.parameters())
    opt = torch.optim.Adam(bc_params, lr=1e-3)
    loss_fn = nn.MSELoss()

    print(f"BC 预训练中（{n_epochs} epochs）...")
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for obs_b, u_b in loader:
            u_pred = actor.get_mean(obs_b)   # tanh 输出，[-1, 1]
            loss = loss_fn(u_pred, u_b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(loader)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{n_epochs} | BC loss = {avg_loss:.4f}")

    print(f"BC 预训练完成，最终 loss = {avg_loss:.4f}")
