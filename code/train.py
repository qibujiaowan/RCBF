"""
主训练入口

运行方式:
  python train.py
  python train.py --slip medium --path sine --steps 300000
"""

import argparse
import os
import numpy as np
import torch
from collections import deque
from torch.utils.tensorboard import SummaryWriter

from config import Config, DEFAULT_CONFIG
from envs.tracked_vehicle_env import TrackedVehicleEnv
from rl.sac_agent import SACAgent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--slip", default="medium", choices=["none","light","medium","heavy","variable"])
    p.add_argument("--path", default="sine", choices=["straight","circle","sine"])
    p.add_argument("--steps", type=int, default=300_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--no_diff_layer", action="store_true", help="关闭可微分安全层（baseline对比）")
    p.add_argument("--no_safety", action="store_true", help="关闭安全层（纯SAC对比）")
    p.add_argument("--tag", default="", help="实验标记")
    p.add_argument("--no_pretrain", action="store_true", help="关闭BC预训练（消融对比）")
    return p.parse_args()


def make_env(cfg: Config, slip: str, path: str):
    return TrackedVehicleEnv(
        vehicle_cfg=cfg.vehicle,
        reward_cfg=cfg.reward,
        reference_path=path,
        slip_config=slip,
    )


def evaluate(agent: SACAgent, env: TrackedVehicleEnv, n_episodes: int = 5):
    """评测：返回平均回报和约束违反次数"""
    returns = []
    violations = {k: 0 for k in ["yaw_rate", "actuator_sat", "rate_limit"]}

    for _ in range(n_episodes):
        obs, _ = env.reset()
        x_state = env.get_state()
        agent.update_prev_action(np.zeros(agent.action_dim))
        episode_return = 0.0
        done = False
        while not done:
            u_safe, u_rl = agent.select_action(obs, x_state, deterministic=True)
            obs_next, reward, terminated, truncated, info = env.step(u_safe)
            episode_return += reward
            for k in violations:
                if info["constraint_violations"].get(k, False):
                    violations[k] += 1
            agent.update_prev_action(u_safe)
            obs = obs_next
            x_state = env.get_state()
            done = terminated or truncated
        returns.append(episode_return)

    return np.mean(returns), violations


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = DEFAULT_CONFIG
    cfg.train.total_steps = args.steps
    cfg.vehicle.slip_config = args.slip
    if args.no_diff_layer:
        cfg.sac.use_diff_safety_layer = False

    os.makedirs(cfg.train.log_dir, exist_ok=True)
    os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)

    tag = f"_{args.tag}" if args.tag else ""
    run_name = f"{args.slip}_{args.path}{tag}"
    writer = SummaryWriter(log_dir=os.path.join(cfg.train.log_dir, run_name))

    env = make_env(cfg, args.slip, args.path)
    eval_env = make_env(cfg, args.slip, args.path)

    obs_dim = env.observation_space.shape[0]   # 6
    action_dim = env.action_space.shape[0]     # 2
    state_dim = 4                               # [X, Y, psi, v]

    agent = SACAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        state_dim=state_dim,
        action_scale=cfg.vehicle.max_wheel_speed,
        sac_cfg=cfg.sac,
        vcfg=cfg.vehicle,
        rcbf_cfg=cfg.rcbf,
        gp_cfg=cfg.gp,
        device=args.device,
    )

    # BC 预训练：在主循环前用 MPC 示范数据热启动 actor
    if not args.no_pretrain:
        from pretrain.bc_pretrain import pretrain_bc
        pretrain_bc(agent, env, cfg.train)

    obs, _ = env.reset(seed=args.seed)
    x_state = env.get_state()
    x_prev = x_state.copy()
    episode_return = 0.0
    episode_steps = 0
    episode_num = 0
    recent_returns = deque(maxlen=20)
    best_eval_return = float("-inf")

    print(f"开始训练: slip={args.slip}, path={args.path}, steps={args.steps}")
    print(f"  可微分安全层: {cfg.sac.use_diff_safety_layer}")
    print(f"  TensorBoard: tensorboard --logdir {cfg.train.log_dir}")

    for step in range(args.steps):
        u_safe, u_rl = agent.select_action(obs, x_state)

        obs_next, reward, terminated, truncated, info = env.step(u_safe)
        x_next = env.get_state()
        done = terminated or truncated

        # 计算真实状态导数（用于GP更新）
        true_xdot = (x_next - x_state) / cfg.vehicle.dt
        nominal_xdot = info["nominal_xdot"]

        agent.store_transition(
            obs=obs,
            u_rl=u_rl,
            u_safe=u_safe,
            obs_next=obs_next,
            reward=reward,
            done=done,
            x_state=x_state,
            x_next=x_next,
            nominal_xdot=nominal_xdot,
            true_xdot=true_xdot,
        )
        agent.update_prev_action(u_safe)

        # 训练更新
        if step >= cfg.sac.warmup_steps:
            for _ in range(cfg.sac.update_interval):
                metrics = agent.update()
            if metrics and step % 100 == 0:
                for k, v in metrics.items():
                    writer.add_scalar(f"train/{k}", v, step)

        episode_return += reward
        episode_steps += 1
        obs = obs_next
        x_prev = x_state
        x_state = x_next

        if done:
            recent_returns.append(episode_return)
            episode_num += 1
            gp_pts = agent.gp.total_points()
            writer.add_scalar("episode/return", episode_return, episode_num)
            writer.add_scalar("episode/length", episode_steps, episode_num)
            writer.add_scalar("episode/gp_points", gp_pts, episode_num)
            if episode_num % 10 == 0:
                avg_return = np.mean(recent_returns)
                print(
                    f"Step {step:7d} | Episode {episode_num:4d} | "
                    f"AvgReturn {avg_return:7.1f} | GP pts {gp_pts:4d}"
                )
                writer.add_scalar("episode/avg_return_20ep", avg_return, episode_num)
            obs, _ = env.reset()
            x_state = env.get_state()
            episode_return = 0.0
            episode_steps = 0
            agent.update_prev_action(np.zeros(action_dim))

        # 定期评测
        if step > 0 and step % cfg.train.eval_interval == 0:
            eval_return, violations = evaluate(agent, eval_env, cfg.train.eval_episodes)
            print(
                f"[EVAL] Step {step:7d} | Return {eval_return:7.1f} | "
                f"Violations: {violations}"
            )
            writer.add_scalar("eval/return", eval_return, step)
            for k, v in violations.items():
                writer.add_scalar(f"eval/violation_{k}", v, step)
            if eval_return > best_eval_return:
                best_eval_return = eval_return
                best_ckpt = os.path.join(cfg.train.checkpoint_dir, f"sac_rcbf{tag}_best.pt")
                agent.save(best_ckpt)
                print(f"  最佳模型已保存: {best_ckpt} (return={eval_return:.1f})")

        # 定期保存
        if step > 0 and step % cfg.train.save_interval == 0:
            ckpt_path = os.path.join(
                cfg.train.checkpoint_dir, f"sac_rcbf{tag}_step{step}.pt"
            )
            agent.save(ckpt_path)
            print(f"  已保存: {ckpt_path}")

    # 最终保存
    agent.save(os.path.join(cfg.train.checkpoint_dir, f"sac_rcbf{tag}_final.pt"))
    writer.close()
    print("训练完成。")
    return agent


if __name__ == "__main__":
    args = parse_args()
    train(args)
