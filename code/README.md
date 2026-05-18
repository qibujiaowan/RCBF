# SAC-RCBF 履带车路径跟踪控制

**课题**：基于鲁棒控制屏障函数的安全强化学习履带车路径跟踪控制方法研究

基于 Emam et al. (IEEE RA-L 2025) 的 SAC-RCBF 框架，针对履带车差速驱动与独立滑移特性扩展。

---

## 项目完成情况

### 已完成

| 模块 | 文件 | 状态 | 说明 |
|------|------|------|------|
| 全局配置 | `config.py` | ✅ 完整 | 车辆参数、GP、RCBF、SAC、训练、奖励的统一配置入口 |
| 标称运动学模型 | `envs/vehicle_kinematics.py` | ✅ 完整 | 不含滑移的理想差速运动学 f(x), g(x)；同时支持 numpy（仿真）和 torch（可微分层） |
| 含滑移仿真环境 | `envs/tracked_vehicle_env.py` | ✅ 完整 | Gymnasium 环境，含左右履带独立滑移动力学；支持 4 种路面工况；提供 GP 残差收集接口 |
| GP 扰动估计 | `gp/disturbance_gp.py` | ✅ 完整 | 各状态维度独立 GP；滑动窗口稀疏截断（max_points=500）；每隔 update_interval 步批量更新 |
| RCBF 约束函数 | `safety/rcbf_constraints.py` | ✅ 骨架完整 | 横摆角速度（相对度1）、执行器饱和、控制变化率三类约束；滑移率 HOCBF 留 TODO |
| RCBF-QP 安全层 | `safety/rcbf_qp_layer.py` | ✅ 完整 | cvxpylayers 可微分 QP（训练用）+ OSQP numpy fallback（推理用）；含松弛变量保可行性 |
| SAC 神经网络 | `rl/networks.py` | ✅ 完整 | GaussianActor（tanh 重参数化）+ TwinCritic（双 Q 网络） |
| 经验回放 | `rl/replay_buffer.py` | ✅ 完整 | 同时存储 u_rl 和 u_safe，支持可微分安全层的梯度反传 |
| SAC-RCBF 智能体 | `rl/sac_agent.py` | ✅ 完整 | 完整训练循环；策略梯度对 u* 反传（经 cvxpylayers KKT 条件）；自动熵调整 |
| 训练入口 | `train.py` | ✅ 完整 | 命令行参数控制路面工况/路径类型/消融选项；含定期评测和 checkpoint 保存 |
| 评测脚本 | `evaluate.py` | ✅ 完整 | 多方法对比框架；计算 RMSE_y、RMSE_psi、约束违反次数 |

### 待完成（后续阶段工作）

| 内容 | 文件 | 说明 |
|------|------|------|
| 滑移率 HOCBF | `safety/rcbf_constraints.py` | 需先完成 Lie 导数理论推导（相对度分析），再填充 `h_slip` 方法 |
| Critic 目标值中的安全动作 | `rl/sac_agent.py` | 当前用 u_rl_next 近似 u*_next；严格实现需批量调用 QP（可选优化） |
| MPC baseline | `baselines/mpc_baseline.py` | 固定参数 MPC，用于对比实验（尚未创建） |
| 滑模控制 baseline | `baselines/smc_baseline.py` | 滑移观测器 + 滑模控制（尚未创建） |
| 训练曲线可视化 | `utils/plot.py` | 回报曲线、约束违反统计图（尚未创建） |

---

## 环境依赖

```
Python 3.13
torch          2.11.0+cpu
gymnasium      1.2.3
numpy          2.3.5
scipy          1.17.1
cvxpy          1.8.2
cvxpylayers    1.1.0      # 可微分 QP 层，用于安全层梯度反传
osqp           1.1.1      # QP 求解器
matplotlib     3.10.7
```

安装：

```bash
pip install -r requirements.txt
```

---

## 代码结构

```
code/
├── config.py                     # 全局超参数（车辆/GP/RCBF/SAC/训练/奖励）
├── train.py                      # 训练入口
├── evaluate.py                   # 评测与多方法对比
├── requirements.txt
│
├── envs/
│   ├── vehicle_kinematics.py     # 标称运动学模型 f(x), g(x)
│   └── tracked_vehicle_env.py    # 含滑移的 Gymnasium 仿真环境
│
├── gp/
│   └── disturbance_gp.py         # GP 扰动在线估计（mu_d, sigma_d）
│
├── safety/
│   ├── rcbf_constraints.py       # RCBF 障碍函数体系（h_yaw/h_sat/h_rate/h_slip）
│   └── rcbf_qp_layer.py          # RCBF-QP 安全层（可微分 + numpy 双模式）
│
└── rl/
    ├── networks.py               # GaussianActor + TwinCritic
    ├── replay_buffer.py          # 经验回放（存储 u_rl 和 u_safe）
    └── sac_agent.py              # SAC-RCBF 智能体（含可微分安全层训练逻辑）
```

---

## 运行流程

### 第一步：验证环境（推荐先跑）

关闭安全层，确认 SAC 基础通路和仿真环境无误：

```bash
cd D:/2026/project/code
python train.py --no_safety --steps 5000 --slip none --path straight
```

期望输出：每 10 个 episode 打印一次平均回报，无报错。

### 第二步：完整训练（本文方法）

```bash
python train.py --slip medium --path sine --steps 300000
```

可选参数：

| 参数 | 默认值 | 可选值 | 说明 |
|------|--------|--------|------|
| `--slip` | `medium` | `none` / `light` / `medium` / `heavy` / `variable` | 路面滑移工况 |
| `--path` | `sine` | `straight` / `circle` / `sine` | 参考路径类型 |
| `--steps` | `300000` | 任意整数 | 训练总步数 |
| `--seed` | `42` | 任意整数 | 随机种子 |
| `--no_diff_layer` | 关闭 | — | 关闭可微分安全层（消融实验 baseline） |
| `--no_safety` | 关闭 | — | 关闭安全层，纯 SAC（对比 baseline） |
| `--tag` | 空 | 字符串 | checkpoint 文件名标记 |

训练输出：
- 每 10 个 episode：打印平均回报和 GP 数据量
- 每 5000 步：打印评测结果（RMSE、约束违反次数）
- 每 20000 步：保存 checkpoint 到 `checkpoints/`

### 第三步：消融与对比实验

```bash
# 消融1：关闭可微分安全层
python train.py --slip medium --path sine --no_diff_layer --tag no_diff

# 消融2：关闭安全层（纯SAC）
python train.py --slip medium --path sine --no_safety --tag pure_sac

# 变附着工况鲁棒性测试
python train.py --slip variable --path sine --tag variable_slip
```

### 第四步：评测

```bash
python evaluate.py --ckpt checkpoints/sac_rcbf_final.pt --slip heavy --episodes 20
```

---

## 核心设计说明

### 控制数据流

```
履带车状态 x
    │
    ▼
SAC 策略网络 ──► u_RL（可能不安全）
                    │
                    ▼
GP 扰动估计 ──► RCBF-QP 安全层 ──► u* = u_RL + u_S（执行）
 D(x)=μ±kσ      min‖u_S‖²               │
    ▲            s.t. RCBF 条件           ▼
    │                              履带车环境（含滑移）
    └── 残差 y = ẋ_obs − f(x) − g(x)u* ─┘
```

### 可微分安全层的作用

标准 SAC+CBF 中，安全层对 RL 输出的修改对策略网络不透明，导致 Q 函数在约束边界附近学到异常低值，训练不稳定。本框架通过 cvxpylayers 将 QP 的 KKT 梯度反传到 Actor，使策略学习感知安全层行为，主动避开约束边界。

`--no_diff_layer` 参数可关闭此机制用于消融对比。

### 奖励函数

```
r = -w1*e_y² - w2*e_ψ² - w3*e_v² - w4*‖Δu‖²
```

安全约束**不进奖励**，完全由 RCBF 层硬性保证。

### 路面工况与 GP 扰动

仿真环境用含滑移的真实动力学，控制器只用不含滑移的标称模型。每步残差 `y = ẋ_obs − f(x) − g(x)u*` 自动送入 GP 训练，GP 输出的扰动集合 `D(x) = μ_d ± k_c·σ_d` 实时用于 RCBF-QP 的鲁棒安全条件。

---

## 已知限制

1. **HOCBF 未实现**：滑移率约束（`h_slip`）相对度可能为 2，需高阶 RCBF，待理论推导后填充。当前训练只包含横摆角速度、执行器饱和、控制变化率三类约束。
2. **torch CPU 版本**：当前 PyTorch 为 CPU 版，30 万步训练预计需要数小时。如需加速可换 GPU 版本。
3. **GP 计算复杂度**：GP 为 O(N³)，当前用滑动窗口（max_points=500）控制规模，大规模训练后期推理可能变慢。
