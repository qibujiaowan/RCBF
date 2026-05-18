# Auto Train Skill

自动训练循环：训练 → 评估 → 分析结果 → 调参 → 下一轮，共 5 轮。

每轮流程：
1. 读取实验记录文件 `logs/experiment_log.json`，了解历史结果和当前参数
2. 根据上一轮结果决定本轮调参策略，修改 `config.py`
3. 运行训练：`python -u train.py --steps 10000 --slip none --path straight --tag round<N>`
4. 训练完成后运行评估：`python -u evaluate.py --ckpt checkpoints/sac_rcbf_round<N>_best.pt --slip none --path straight --episodes 20`
5. 将本轮参数 + 评估结果写入 `logs/experiment_log.json`
6. 判断是否已完成 5 轮，若未完成则继续下一轮

调参优先级（根据指标决定）：
- RMSE_y > 0.1m → 加大 w_lateral 奖励权重
- RMSE_psi > 0.3rad → 加大 w_heading 奖励权重
- 约束违反次数多 → 增大 slack_penalty 或降低 alpha
- actor_loss 不收敛 → 降低 learning_rate
- Return 持续负增长 → 调整 gamma 或 batch_size
- README.md 中记录的尚未实现的功能，也可以在本轮尝试加入

开始执行第一步：读取实验记录，确定本轮编号，然后开始训练。
