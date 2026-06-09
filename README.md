# agentic_RL

Agentic Reinforcement Learning 技术调研与训练方案。

当前仓库从技术方案开始建设，主文档见：

- [Agentic RL 技术调研与训练方案](docs/agentic_rl_technical_research_and_training_plan.md)
- [训练框架选型与首次训练验证](docs/training_framework_selection_and_smoke.md)
- [2026-06-09 基线评测与 assistant-only SFT 实验日志](docs/experiments/2026-06-09_baseline_eval_and_sft.md)

文档覆盖数据准备、轨迹采集、监督微调、在线强化学习、奖励与信用分配、
轻量模型持续预训练、评测体系、系统架构、算力规划和分阶段落地计划。
当前主线目标为约 2B 参数模型，通过 CPT/mid-training、蒸馏 SFT 和可验证 RL
在 BFCL、τ²-bench 等公开指标上进入同体量第一梯队。第 13 章集中列出了数据、
环境、rollout、训练、分布式系统、评测和安全环节的常见难点、踩坑、诊断流程、
停训红线与解决方案。
