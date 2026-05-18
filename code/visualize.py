"""
路径跟踪效果可视化脚本

运行示例:
  python visualize.py --ckpt checkpoints/sac_rcbf_round1_best.pt --slip none --path straight
  python visualize.py --ckpt checkpoints/sac_rcbf_round1_best.pt --slip medium --path sine
  python visualize.py --ckpt checkpoints/sac_rcbf_round1_best.pt --slip none --path sine --episodes 3
"""

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

matplotlib.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

from config import DEFAULT_CONFIG
from envs.tracked_vehicle_env import TrackedVehicleEnv
from rl.sac_agent import SACAgent


def rollout_episode(agent, env, seed=0):
    obs, _ = env.reset(seed=seed)
    agent.update_prev_action(np.zeros(agent.action_dim))

    traj_x, traj_y = [], []
    ref_x, ref_y = [], []
    e_y_hist, e_psi_hist = [], []
    u_hist = []
    reward_hist = []
    done = False

    while not done:
        x_state = env.get_state()
        traj_x.append(x_state[0])
        traj_y.append(x_state[1])

        ref_row = env.ref_path.query(env._path_idx)
        ref_x.append(ref_row[0])
        ref_y.append(ref_row[1])

        e_y_hist.append(obs[0])
        e_psi_hist.append(obs[1])

        u_safe, _ = agent.select_action(obs, x_state, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(u_safe)

        u_hist.append(u_safe.copy())
        reward_hist.append(reward)
        agent.update_prev_action(u_safe)
        done = terminated or truncated

    return {
        "traj_x": np.array(traj_x),
        "traj_y": np.array(traj_y),
        "ref_x": np.array(ref_x),
        "ref_y": np.array(ref_y),
        "e_y": np.array(e_y_hist),
        "e_psi": np.array(e_psi_hist),
        "u": np.array(u_hist),
        "rewards": np.array(reward_hist),
        "total_return": sum(reward_hist),
        "rmse_y": float(np.sqrt(np.mean(np.array(e_y_hist)**2))),
        "rmse_psi": float(np.sqrt(np.mean(np.array(e_psi_hist)**2))),
    }


def plot_results(episodes_data, args, save_path=None):
    n_ep = len(episodes_data)
    colors = plt.cm.tab10(np.linspace(0, 0.9, n_ep))

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"SAC-RCBF 路径跟踪效果  |  path={args.path}  slip={args.slip}  "
        f"ckpt={Path(args.ckpt).stem}",
        fontsize=13, fontweight="bold"
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax_xy   = fig.add_subplot(gs[:, 0:2])   # 左侧大图：XY轨迹
    ax_ey   = fig.add_subplot(gs[0, 2])     # 右上：横向误差
    ax_epsi = fig.add_subplot(gs[1, 2])     # 右中：航向误差
    ax_u    = fig.add_subplot(gs[2, 2])     # 右下：控制输入

    # --- XY 轨迹图 ---
    ref_x = episodes_data[0]["ref_x"]
    ref_y = episodes_data[0]["ref_y"]
    ax_xy.plot(ref_x, ref_y, "k--", lw=2, label="参考路径", zorder=5)

    for i, ep in enumerate(episodes_data):
        label = f"Episode {i+1}  (return={ep['total_return']:.0f}, RMSE_y={ep['rmse_y']:.3f}m)"
        ax_xy.plot(ep["traj_x"], ep["traj_y"], color=colors[i], lw=1.5,
                   alpha=0.85, label=label)
        ax_xy.plot(ep["traj_x"][0], ep["traj_y"][0], "o",
                   color=colors[i], ms=6, zorder=6)

    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.set_title("XY 轨迹")
    ax_xy.legend(fontsize=7.5, loc="upper left")
    ax_xy.set_aspect("equal")
    ax_xy.grid(True, alpha=0.3)

    # --- 横向误差 e_y ---
    for i, ep in enumerate(episodes_data):
        t = np.arange(len(ep["e_y"])) * DEFAULT_CONFIG.vehicle.dt
        ax_ey.plot(t, ep["e_y"], color=colors[i], lw=1.2, alpha=0.8)
    ax_ey.axhline(0, color="k", lw=0.8, ls="--")
    ax_ey.axhline(0.1, color="gray", lw=0.8, ls=":", label="±0.1m 目标")
    ax_ey.axhline(-0.1, color="gray", lw=0.8, ls=":")
    ax_ey.set_xlabel("时间 (s)")
    ax_ey.set_ylabel("e_y (m)")
    ax_ey.set_title("横向误差")
    ax_ey.legend(fontsize=7.5)
    ax_ey.grid(True, alpha=0.3)

    # --- 航向误差 e_psi ---
    for i, ep in enumerate(episodes_data):
        t = np.arange(len(ep["e_psi"])) * DEFAULT_CONFIG.vehicle.dt
        ax_epsi.plot(t, np.degrees(ep["e_psi"]), color=colors[i], lw=1.2, alpha=0.8)
    ax_epsi.axhline(0, color="k", lw=0.8, ls="--")
    ax_epsi.set_xlabel("时间 (s)")
    ax_epsi.set_ylabel("e_ψ (deg)")
    ax_epsi.set_title("航向误差")
    ax_epsi.grid(True, alpha=0.3)

    # --- 控制输入 ---
    cfg = DEFAULT_CONFIG.vehicle
    for i, ep in enumerate(episodes_data):
        t = np.arange(len(ep["u"])) * cfg.dt
        ax_u.plot(t, ep["u"][:, 0], color=colors[i], lw=1.2, alpha=0.7,
                  ls="-")
        ax_u.plot(t, ep["u"][:, 1], color=colors[i], lw=1.2, alpha=0.7,
                  ls="--")
    ax_u.axhline(cfg.max_wheel_speed, color="r", lw=0.8, ls=":", label=f"±{cfg.max_wheel_speed} rad/s")
    ax_u.axhline(-cfg.max_wheel_speed, color="r", lw=0.8, ls=":")
    # 图例说明实线/虚线含义（只标一次）
    ax_u.plot([], [], "k-",  lw=1.2, label="ωL")
    ax_u.plot([], [], "k--", lw=1.2, label="ωR")
    ax_u.set_xlabel("时间 (s)")
    ax_u.set_ylabel("轮速 (rad/s)")
    ax_u.set_title("控制输入")
    ax_u.legend(fontsize=7.5)
    ax_u.grid(True, alpha=0.3)

    # --- 统计汇总（文字） ---
    returns = [ep["total_return"] for ep in episodes_data]
    rmse_ys = [ep["rmse_y"] for ep in episodes_data]
    rmse_psis = [ep["rmse_psi"] for ep in episodes_data]
    summary = (
        f"共 {n_ep} 个 episode\n"
        f"Return: {np.mean(returns):.1f} ± {np.std(returns):.1f}\n"
        f"RMSE_y: {np.mean(rmse_ys):.4f} m  (目标 <0.10m)\n"
        f"RMSE_ψ: {np.mean(rmse_psis):.4f} rad  (目标 <0.30rad)"
    )
    fig.text(0.01, 0.01, summary, fontsize=9,
             verticalalignment="bottom",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"图像已保存: {save_path}")
    else:
        plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",     required=True,         help="checkpoint路径")
    p.add_argument("--slip",     default="none",        help="滑移工况: none/light/medium/heavy")
    p.add_argument("--path",     default="straight",    help="路径类型: straight/circle/sine")
    p.add_argument("--episodes", type=int, default=3,   help="可视化的episode数量")
    p.add_argument("--save",     default=None,          help="保存图像路径（不指定则弹窗显示）")
    args = p.parse_args()

    cfg = DEFAULT_CONFIG
    env = TrackedVehicleEnv(
        vehicle_cfg=cfg.vehicle,
        reward_cfg=cfg.reward,
        reference_path=args.path,
        slip_config=args.slip,
    )
    agent = SACAgent(
        obs_dim=6, action_dim=2, state_dim=4,
        action_scale=cfg.vehicle.max_wheel_speed,
        sac_cfg=cfg.sac, vcfg=cfg.vehicle,
        rcbf_cfg=cfg.rcbf, gp_cfg=cfg.gp,
        device="cpu",
    )
    agent.load(args.ckpt)
    print(f"已加载: {args.ckpt}")

    episodes_data = []
    for i in range(args.episodes):
        data = rollout_episode(agent, env, seed=i)
        episodes_data.append(data)
        print(f"  Episode {i+1}: return={data['total_return']:.1f}, "
              f"RMSE_y={data['rmse_y']:.4f}m, RMSE_psi={data['rmse_psi']:.4f}rad, "
              f"steps={len(data['e_y'])}")

    save_path = args.save or f"viz_{Path(args.ckpt).stem}_{args.path}_{args.slip}.png"
    plot_results(episodes_data, args, save_path=save_path)


if __name__ == "__main__":
    main()
