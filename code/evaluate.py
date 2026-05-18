"""
评测与多方法对比脚本

对比方法:
  1. SAC-RCBF (本文方法，可微分安全层 + GP)
  2. SAC-RCBF (无可微分安全层)
  3. 纯 SAC（无安全层）
  4. MPC Baseline（固定参数，简化实现）

运行:
  python evaluate.py --ckpt checkpoints/sac_rcbf_final.pt --slip heavy
"""

import argparse
import numpy as np
from typing import Dict, List

from config import DEFAULT_CONFIG
from envs.tracked_vehicle_env import TrackedVehicleEnv
from rl.sac_agent import SACAgent


def evaluate_agent(
    agent: SACAgent,
    env: TrackedVehicleEnv,
    n_episodes: int = 20,
    slip_override: str = None,
) -> Dict:
    if slip_override:
        env.slip_config = slip_override

    all_returns = []
    all_e_y = []
    all_e_psi = []
    violation_counts = {k: 0 for k in ["yaw_rate", "actuator_sat", "rate_limit"]}
    total_steps = 0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        x_state = env.get_state()
        done = False
        ep_return = 0.0
        ep_e_y = []
        ep_e_psi = []

        agent.update_prev_action(np.zeros(agent.action_dim))
        while not done:
            u_safe, _ = agent.select_action(obs, x_state, deterministic=True)
            obs_next, reward, terminated, truncated, info = env.step(u_safe)
            ep_return += reward
            ep_e_y.append(abs(obs[0]))
            ep_e_psi.append(abs(obs[1]))

            for k in violation_counts:
                if info["constraint_violations"].get(k, False):
                    violation_counts[k] += 1
            total_steps += 1

            agent.update_prev_action(u_safe)
            obs = obs_next
            x_state = env.get_state()
            done = terminated or truncated

        all_returns.append(ep_return)
        all_e_y.extend(ep_e_y)
        all_e_psi.extend(ep_e_psi)

    rmse_y = np.sqrt(np.mean(np.array(all_e_y)**2))
    rmse_psi = np.sqrt(np.mean(np.array(all_e_psi)**2))

    return {
        "mean_return": np.mean(all_returns),
        "std_return": np.std(all_returns),
        "rmse_y": rmse_y,
        "rmse_psi": rmse_psi,
        "violations": violation_counts,
        "total_steps": total_steps,
    }


def print_results(method_name: str, results: Dict):
    v = results["violations"]
    total_v = sum(v.values())
    print(f"\n{'='*50}")
    print(f"方法: {method_name}")
    print(f"  Mean Return : {results['mean_return']:.2f} ± {results['std_return']:.2f}")
    print(f"  RMSE_y      : {results['rmse_y']:.4f} m")
    print(f"  RMSE_psi    : {results['rmse_psi']:.4f} rad")
    print(f"  约束违反总数 : {total_v}")
    print(f"    - yaw_rate    : {v['yaw_rate']}")
    print(f"    - actuator_sat: {v['actuator_sat']}")
    print(f"    - rate_limit  : {v['rate_limit']}")


def run_comparison(args):
    cfg = DEFAULT_CONFIG

    print(f"\n对比实验: slip={args.slip}, path={args.path}")

    # --- SAC-RCBF (本文方法) ---
    if args.ckpt:
        env = TrackedVehicleEnv(
            vehicle_cfg=cfg.vehicle, reward_cfg=cfg.reward,
            reference_path=args.path, slip_config=args.slip,
        )
        agent = SACAgent(
            obs_dim=6, action_dim=2, state_dim=4,
            action_scale=cfg.vehicle.max_wheel_speed,
            sac_cfg=cfg.sac, vcfg=cfg.vehicle,
            rcbf_cfg=cfg.rcbf, gp_cfg=cfg.gp,
            device="cpu",
        )
        agent.load(args.ckpt)
        results = evaluate_agent(agent, env, n_episodes=args.episodes)
        print_results("SAC-RCBF（本文方法）", results)

    print("\n注：MPC baseline 和滑模控制 baseline 需单独实现（见 baselines/ 目录）")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None)
    p.add_argument("--slip", default="medium")
    p.add_argument("--path", default="sine")
    p.add_argument("--episodes", type=int, default=20)
    args = p.parse_args()
    run_comparison(args)
