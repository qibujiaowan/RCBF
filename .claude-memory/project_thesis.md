---
name: 开题课题核心信息与进度
description: 履带车SAC-RCBF开题报告的题目、框架、文件路径和当前完成状态（2026-05-15更新）
type: project
originSessionId: bd095ebe-902a-4f40-8cf1-cfbca75477c9
---
## 课题题目
基于鲁棒控制屏障函数的安全强化学习履带车路径跟踪控制方法研究

## 核心框架（已确定）
以 Emam 等人（IEEE RA-L 2025）的 SAC-RCBF 框架为基础，针对履带车场景进行扩展：
- **标称模型**：不含滑移的理想差速运动学方程（已知，可标定）
- **扰动建模**：左右履带独立滑移的综合效应整体建模为未知扰动 $d(x)$
- **GP**：从系统残差在线估计扰动集合 $D(x) = \mu_d \pm k_c\sigma_d$
- **RCBF**：将 $D(x)$ 嵌入安全条件，对最坏情况扰动鲁棒
- **SAC**：学习路径跟踪策略，奖励函数不含安全惩罚项
- **可微分安全层**：策略梯度对最终安全输出 $u^*$ 反传，保证训练稳定性

## 三大问题与三大创新点对应关系
| 问题 | 创新点 |
|------|-------|
| 显式滑移辨识循环依赖 + MPC模型依赖 | GP扰动估计整体取代显式辨识 |
| 高阶相对度约束的RCBF设计缺口（Emam 2025标注为"留待后续"） | 针对滑移率约束的HOCBF理论扩展 |
| Safe RL软约束保证 + 朴素SAC+CBF训练不稳定 | 多约束RCBF体系 + 可微分安全层 |

## 关键文件路径（D:/2026/project/）
- `report/开题报告V5_SAC-RCBF框架.md` — **当前主版本**，已修复所有已知逻辑硬伤
- `report/开题答辩备问.md` — 答辩预设问题与回答思路（含问题4/5/6及仿真环境追问）
- `report/Emam2025_SAC-RCBF论文解读.md` — Emam 2025论文的详细中文解读
- `papers/Emam 等 - 2025 - Safe Reinforcement Learning Using Robust Control Barrier Functions.pdf`
- `code/CHANGELOG.md` — 所有代码改进记录（时间倒序）
- `code/logs/experiment_log.json` — 5轮auto-train实验记录与结论

## 当前进度（2026-05-15）
V5开题报告已完成。5轮10k步auto-train实验已完成，代码框架已修复所有已知bug。

### 代码框架状态（2026-05-15，已完成5轮auto-train）

**已修复的关键Bug（按时间排序）**
1. `safety/rcbf_qp_layer.py`：cvxpylayers DPP形式约束修正（2026-05-13）
2. `rl/sac_agent.py`：log_pi_next维度错误、float64/float32不匹配（2026-05-13）
3. `rl/replay_buffer.py`：reward/done不应unsqueeze（2026-05-13）
4. actor子采样：256→16次 solve_np（8×提速）（2026-05-13）
5. best-checkpoint保存机制（2026-05-13）
6. checkpoint覆盖：final.pt加{tag}（2026-05-13）
7. u_safe_next近似错误：critic用u_safe替代u_rl（2026-05-14）
8. critic子采样：256→32次 solve_np（8×提速）（2026-05-15）
9. 评测u_prev未更新：evaluate.py和train.py evaluate()均已修复（2026-05-15）

### 当前最优超参数（config.py，适合长期训练）
- `warmup_steps=1000`
- `update_interval=5`
- `eval_interval=2000`
- `learning_rate=1e-4`
- `w_heading=2.0, w_lateral=2.0, w_smooth=0.5`
- `alpha_yaw=0.5, alpha_rate=0.5`
- `total_steps=300_000`（默认，auto-train用--steps 10000覆盖）

### 5轮实验核心结论
10000步对SAC路径跟踪任务完全不足，Q值在step 2000-5000必然发散。
需至少100k步才能得到有意义结果。所有bug修复将在长期训练中发挥作用。

## Why：课题选择逻辑
履带滑移是核心挑战 → 现有方法（VBEKF/MPC）均依赖精确模型形成循环依赖 → 需要同时解决"模型不确定性"和"安全约束硬保证"两个问题 → SAC-RCBF框架恰好提供了这两个问题的联合解法 → 针对履带车的RCBF设计（尤其高阶相对度）是真实的理论贡献空间
