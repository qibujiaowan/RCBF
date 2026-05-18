# Changelog

所有代码改进按时间倒序记录。每条记录包含：改进时间、改进原因、具体改动。

---

## 2026-05-17

### Config: 三项改进后新5轮 auto-train 实验结论与最优超参数确认

**实验背景**

在BC预训练 + episode缩短（300步）+ batch_size增大（512）三项改进全部上线后，开展新5轮10k步验证实验（Round 1-5），系统排查超参数影响。

**实验结论（Round 1-5，10000步/轮）**

1. **三项改进显著有效**：评估 return 从旧版 -9395 提升至 -1808（**81%改善**），best_training_return 从 -2253 提升至 -1056.7（step 2000，53%改善）。
2. **最优配置确认为 R1 默认参数**（w_lateral=2.0, w_heading=2.0, alpha_yaw=0.5, gamma=0.99, tau=0.005, alpha=0.2），其余调参均未能超越。
3. **明确有害参数**：
   - `alpha_yaw < 0.5`（R2）：RCBF约束过紧，Q值发散加剧，评估return=-2649
   - `gamma = 0.995`（R3）：rate_limit违反在step2000爆发277次，Q值快速发散至-7948
   - `w_velocity = 0.5`（R5）：导致step4000 yaw_rate违反1168次（激进转向触发RCBF约束）
4. **best checkpoint 时间后移**：BC预训练后 best 从旧版 step 2000 推移到 step 4000-6000（说明训练早期更稳定），但10k步内仍无法收敛（RMSE_y>0.2m）。

**当前 config.py 已恢复为最优参数（R1配置）**：
- `w_lateral=2.0, w_heading=2.0, w_velocity=0.3, w_smooth=0.5`
- `alpha_yaw=0.5, alpha_rate=0.5`
- `gamma=0.99, tau=0.005, alpha=0.2`
- `batch_size=512, episode_length=300, bc_pretrain=True`

**下一步**：进行 100k+ 步长期训练以验证最终收敛效果。

---

## 2026-05-15

### Config: 5轮 auto-train 实验结论与超参数最终设置

**实验结论（Round 1-5，10000步/轮）**

5轮实验揭示了两个核心问题：
1. **步数严重不足**：10000步内 SAC Q值在 step 2000-5000 必然发散（deadly triad），最佳checkpoint几乎在学习刚开始时产生，无法代表收敛策略。后续训练需至少 100k 步。
2. **评测 bug**：`evaluate.py` 和 `train.py:evaluate()` 未调用 `agent.update_prev_action()`，导致 RCBF 始终从 `u_prev=0` 求解，与环境实际 `_prev_u` 不一致，rate_limit 违反次数被高估约 14000 次/20 episodes。

**已验证的有效配置（Round 5 最优结果）**
- `warmup_steps=1000`（vs 5000）：实际训练步数 9000 vs 5000，效果更充分
- `eval_interval=2000`：早期捕获最佳 checkpoint（step 2000 时 Return=-2253，是10000步内最好结果）
- `alpha_yaw=0.5, alpha_rate=0.5`（vs 1.0 或 0.3）：平衡约束强度与策略灵活性

**改动**

`config.py`（当前生效值，供长期训练使用）
- `warmup_steps`: 保持 1000
- `update_interval`: 5（Round 4 实验 update_interval=2 导致 off-policy 过大，还原）
- `eval_interval`: 2000（早期捕获最佳点）
- `learning_rate`: 1e-4（Round 5 实验值，相比 3e-4 更保守）

---

### Feat: MPC行为克隆预训练热启动 SAC actor

**原因**

5轮实验表明 SAC 在 step 2000-5000 必然发散，最佳 checkpoint return 约 -2253（10k步）。根因之一是随机初始化的 actor 需要大量探索才能学会基本的路径跟踪控制，大量早期样本质量极低，拖慢 critic 收敛。

**改动**

`pretrain/mpc_teacher.py`（新建）
- 基于标称运动学 `f(x)+g(x)u` 的线性化 MPC，预测步长 N=10
- 在参考轨迹点处做 Jacobian 线性化（A_k, B_k），用 scipy SLSQP 求解
- 代价函数：位置/航向/速度误差的二次型 + 控制变化率惩罚
- 约束：轮速幅值、偏航率、轮速变化率
- 实测 3 episode 平均 return = -19.4（随机策略约 -3700，提升约 190×）

`pretrain/bc_pretrain.py`（新建）
- `collect_bc_data()`：MPC 在环境中 rollout 采集 (obs, u_mpc/scale) 对
- `pretrain_bc()`：只优化 actor.trunk + actor.mean_layer，保留 log_std_layer 以维持熵调整机制
- 损失函数：`MSE(actor.get_mean(obs), u_target)`，实测 500 步数据 + 5 epochs 后 loss < 0.1

`config.py`
- `TrainConfig` 新增 `bc_pretrain_steps=5000`、`bc_pretrain_epochs=50`

`train.py`
- 在 `SACAgent` 初始化后、主循环前插入 `pretrain_bc()` 调用
- 新增 `--no_pretrain` 参数支持消融对比

**验证结果**（3000步对比实验，slip=none, straight）
- 有 BC 预训练：step 2000 eval return = **-433.9**
- 无 BC 预训练：step 2000 eval return = -734.7
- 提升幅度：**41%**

---

### Perf: 缩短 episode 长度（1000 → 300步）

**原因**

1000步 episode（50秒）在 10k 总步数内只有约 10 个 episode，梯度信号极度稀疏。缩短至 300步后每万步约 33 个 episode，Q值更新方差降低，best checkpoint 出现更早。

**改动**

`envs/tracked_vehicle_env.py`
- `truncated = self._step_count >= 1000` → `>= 300`
- `ReferencePath.num_points`: 2000 → 600（路径覆盖 300步 × 0.05s × 1.5m/s ≈ 22.5m，600点足够）

---

### Perf: 增大 batch_size（256 → 512）及 replay buffer（100k → 200k）

**原因**

batch_size=256 时每次梯度更新的信噪比低，Q值估计方差大，导致发散提前。batch=512 使每次更新更稳定（信噪比提升 √2 倍），同时扩大 buffer 保证样本多样性不下降。

**改动**

`config.py`
- `SACConfig.batch_size`: 256 → 512
- `SACConfig.replay_buffer_size`: 100_000 → 200_000

---

### Fix: 修复评测时 u_prev 未更新导致 rate_limit 违反虚高

**原因**

`evaluate.py` 和 `train.py` 的 `evaluate()` 函数在 episode 循环中均未调用 `agent.update_prev_action(u_safe)`，导致 RCBF 安全层始终以 `_prev_u=0` 求解 QP，而环境的 `_check_constraints` 使用实际更新的 `_prev_u`。两者不一致，使 rate_limit 违反次数被严重高估（实测约 14000+ 次 / 20 episodes，实际接近 0）。

**改动**

`evaluate.py`
- 每个 episode 开始前调用 `agent.update_prev_action(np.zeros(agent.action_dim))` 重置状态
- 每步后调用 `agent.update_prev_action(u_safe)` 保持与 RCBF 的 `_prev_u` 一致

`train.py`
- `evaluate()` 函数同步添加上述两处调用

---

### Perf: Critic u_safe_next 计算子采样（8× 提速）

**原因**

Round 2 训练实测：修复 `u_safe_next`（批量调用 256 次 `solve_np`）后，每次 `_critic_update` 耗时约 3.6 秒（vs. actor 子采样后的 ~0.1 秒），导致 10000 步耗时约 2.5 小时（每步约 0.9 秒，比预期的 5 步/秒 慢 5.5×）。速度瓶颈与 actor 修复前相同。

**改动**

`rl/sac_agent.py`
- `_critic_update()` 中仿照 actor 的子批次策略：从 batch（256 样本）中随机抽取 32 个子样本调用 `solve_np` 计算真实 `u_safe_next`，其余样本保留 `u_rl_next_scaled` 近似
- 子样本 32 个（vs. actor 的 16 个）以保留更多安全修正信号
- QP 调用次数：256 → 32（8× 提速），预计单步耗时从 ~0.9s 降至 ~0.2s

---

## 2026-05-14

### Fix: 修复 Q 值单调发散导致训练崩溃

**原因**

通过对 Round 4、5 的 TensorBoard 数据逐步分析，发现训练在 ~5000 步后必然发散，根因是三个相互耦合的问题：

1. **`u_safe_next` 近似错误**（最根本）：`_critic_update` 在计算 Bellman target 时，用 `u_rl_next`（未经QP修正的原始动作）代替 `u_safe_next`（经安全层修正的实际动作）。两者在约束激活时差距显著，导致每次 TD 更新都积累系统性偏差，`q1_mean` 从 step 1000 的 -0.22 单调下降至 step 9900 的 -800。

2. **无 Q 值上界约束**：Q 值无上界，一旦偏差积累便无法自我修正，最终触发 `critic_loss` 峰值（Round 4 最大 47869，Round 5 最大 67793），形成 deadly triad（函数近似 + bootstrapping + off-policy）。

3. **更新频率过高、warmup 样本过少**：`warmup_steps=1000`、`update_interval=1`，意味着 buffer 中有效样本极少时就开始高频更新，off-policy 程度极高，加速 Q 值偏移。

**改动**

`rl/replay_buffer.py`
- `add()` 新增 `state_next` 参数，buffer 中额外存储下一步物理状态 `[X, Y, psi, v]`，供 critic 更新时调用 QP 修正 `u_safe_next`

`rl/sac_agent.py`
- `store_transition()` 新增 `x_next` 参数并透传给 `replay_buffer.add()`
- `_critic_update()` 签名中 `state` 改为 `state_next`；target 计算时对 batch 中每个样本调用 `safety_layer.solve_np()` 得到真实的 `u_safe_next`，替换原来的 `u_rl_next_scaled` 近似
- 在 target 写入前加 `q_target.clamp(q_min, 0.0)`，上界 0（奖励非正），下界 `-max_u² / (1-γ) ≈ -10000`，防止 Q 值向负无穷漂移

`config.py`
- `warmup_steps`: 1000 → 5000（保证 buffer 积累足够多样的样本再开始更新）
- `update_interval`: 1 → 5（每 5 步环境交互后更新一次，降低 off-policy 程度）

`train.py`
- `store_transition()` 调用处新增 `x_next=x_next` 参数

---

## 2026-05-13

### Fix: 修复每轮训练 checkpoint 覆盖问题

**原因**

`train.py` 最终保存时路径硬编码为 `sac_rcbf_final.pt`，未使用 `--tag` 参数，导致每一轮训练都覆盖同一个文件，`evaluate.py` 加载的始终是上一轮的模型，评估结果不可信。

**改动**

`train.py`
- 最终保存路径改为 `sac_rcbf{tag}_final.pt`，确保每轮独立存储

---

### Perf: 可微分 QP 层 8× 提速

**原因**

`_actor_update` 中对整个 batch（256 样本）逐一调用 `cvxpylayers`，实测每次 `agent.update()` 耗时约 1084ms，训练速度约 1 步/秒，30 万步需约 86 小时，完全不可行。

**改动**

`rl/sac_agent.py`
- `_actor_update` 中从 batch 随机抽取 16 个子样本送入可微分 QP 层，其余逻辑不变。梯度信号依然有效，单步耗时降至约 124ms（8× 提速）

---

### Fix: 修复 Critic 梯度裁剪过激问题

**原因**

原 `clip_grad_norm_(critic, 1.0)` 裁剪阈值过小，在 critic_loss 峰值时截断了有效修正梯度，无法及时纠正 Q 值偏移。

**改动**

`rl/sac_agent.py`
- Critic 梯度裁剪阈值从 1.0 调整为 5.0

---

### Feat: 新增 best-checkpoint 保存机制

**原因**

训练存在先收敛后发散的现象（Round 4/5 在 step 5000 附近达到最优，之后 Q 值爆炸）。最终保存的 checkpoint 是发散后的模型，评估结果远差于训练中最优状态。

**改动**

`train.py`
- 每次定期评测（`eval_interval`）后，若当前 `eval_return` 优于历史最佳，则额外保存 `sac_rcbf{tag}_best.pt`

`auto-train.md`（skill）
- 评估命令改为加载 `_best.pt` 而非 `_final.pt`

---

### Config: 超参数调优（基于 5 轮 auto-train 实验）

**原因**

5 轮自动调参实验（Round 1-5）表明默认奖励权重偏小，约束惩罚系数不足，学习率选择影响稳定性。

**改动**

`config.py`
- `w_lateral`: 1.0 → 2.0（横向跟踪误差惩罚加倍，RMSE_y 从 0.125 降至 0.101）
- `w_heading`: 0.5 → 0.8（适度增大航向误差惩罚；1.5 过大会使 RMSE_psi 反而恶化）
- `w_smooth`: 0.1 → 0.2（加大控制平滑惩罚，抑制速率约束违反）
- `slack_penalty`: 1e4 → 1e5（提高约束松弛代价）
- `learning_rate`: 3e-4（保持，5e-4 在 10k+ 步时导致 critic 发散）

`auto-train.md`（skill）
- 每轮训练步数从 300000 → 10000（实测约 19 分钟/轮，可在单次会话内完成 5 轮）
