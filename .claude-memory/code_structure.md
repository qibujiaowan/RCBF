---
name: 代码框架结构与关键文件
description: D:/2026/project/code 下各模块职责、关键参数位置、已知问题
type: project
originSessionId: e5330314-1597-41e3-9ae6-f9c0f14f6d6b
---
## 代码根目录：D:/2026/project/code/

### 主要文件
| 文件 | 职责 |
|------|------|
| `train.py` | 训练主循环，含 TensorBoard 写入，支持 --steps/--slip/--path/--tag 参数 |
| `evaluate.py` | 评估脚本，输出 RMSE_y、RMSE_psi、约束违反次数 |
| `config.py` | **所有超参数集中在此**，修改调参直接改这个文件 |
| `.claude/commands/auto-train.md` | `/auto-train` skill 定义文件 |
| `logs/experiment_log.json` | 多轮迭代实验记录（参数+结果） |

### 模块结构
- `envs/tracked_vehicle_env.py` — 履带车仿真环境（含滑移模型）
- `safety/rcbf_qp_layer.py` — RCBF-QP 安全层（cvxpylayers，DPP形式）
- `safety/rcbf_constraints.py` — RCBF 约束计算
- `gp/disturbance_gp.py` — 高斯过程扰动估计
- `rl/sac_agent.py` — SAC 智能体（含可微分安全层路径）
- `rl/networks.py` — Actor/Critic 网络
- `rl/replay_buffer.py` — 经验回放（reward/done 不 unsqueeze）

### config.py 关键超参数位置
- SAC：`SACConfig`（lr=3e-4, gamma=0.99, batch_size=256, warmup_steps=1000）
- 奖励：`RewardConfig`（w_lateral=1.0, w_heading=0.5, w_velocity=0.3, w_smooth=0.1）
- 安全层：`RCBFConfig`（slack_penalty=1e4, alpha_yaw/sat/rate=1.0）
- 训练：`TrainConfig`（eval_interval=5000, save_interval=20000）

### 运行命令
```bash
# 训练
python -u train.py --steps 300000 --slip none --path straight

# 评估
python -u evaluate.py --ckpt checkpoints/sac_rcbf_final.pt --slip none --path straight

# TensorBoard
tensorboard --logdir D:/2026/project/code/logs
```

### GPU 状态
当前无 CUDA，PyTorch 版本为 2.11.0+cpu，只能 CPU 训练。

**Why**：代码细节易忘，下次会话直接定位修改位置。
**How to apply**：修改超参时去 config.py，修改训练逻辑去 train.py，调安全层去 rcbf_qp_layer.py。
