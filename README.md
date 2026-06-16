# Agentic RL — Task Orchestration & Accurate Tool Calling

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)

基于 **SFT（冷启动）→ GRPO（强化学习）** 的大模型 Agent 工具调用训练框架，专注于**任务编排 + 准确工具调用**场景。

## 技术方案

```
┌─────────────────────────────────────────────────────┐
│              训练流水线                               │
├──────────┬──────────────────────────────────────────┤
│ 数据     │ 19种工具Schema → 场景模板 → 合成10K轨迹   │
│ 清洗     │ 去重 + JSON校验 + 难度均衡 + 质量过滤     │
│ 训练     │ SFT冷启动(QLoRA) → GRPO(4维可验证奖励)    │
│ 评测     │ BFCL风格 + 工具精度 + 参数匹配 + JSON合法 │
└──────────┴──────────────────────────────────────────┘

算法: GRPO (Group Relative Policy Optimization)
     组内相对优势 | 无需Critic网络 | 可验证奖励
基座: Qwen2.5-7B-Instruct (Apache 2.0)
框架: TRL + vLLM + PyTorch
```

## 为什么选择这个方案？

| 维度 | 选择 | 理由 |
|------|------|------|
| **基座模型** | Qwen2.5-7B-Instruct | Apache 2.0 商用友好，原生函数调用，中英双语 |
| **训练算法** | GRPO (非 PPO/DPO) | 无需 Critic 网络（省一半显存），可验证奖励天然适合工具调用 |
| **奖励设计** | 4维可验证奖励 | 全部规则驱动，无需人工标注或 Reward Model |
| **训练框架** | TRL (HuggingFace) | 文档完善，社区活跃，7B 规模最佳选择 |
| **微调方式** | QLoRA (4-bit) | 单卡A100即可训练7B模型，显存友好 |

## 快速开始

```bash
# 1. 安装
git clone https://github.com/zealot52099/Agentic_RL-.git && cd Agentic_RL-
pip install -r requirements.txt

# 2. 生成数据
python scripts/run_data_pipeline.py

# 3. 训练 (SFT → GRPO)
python scripts/run_train.py --stage all

# 4. 评测
python scripts/run_eval.py --model_path ./output/grpo_model --eval_data ./data/test_raw.jsonl
```

## 项目结构

```
├── configs/             # YAML 配置 (模型/数据/GRPO)
├── src/
│   ├── data/            # 工具Schema + 数据生成 + 清洗 + 奖励函数
│   ├── train/           # SFT训练 + GRPO训练
│   └── eval/            # 工具精度评测 + BFCL评测 + 基准运转
├── scripts/             # 运行脚本 (数据/训练/评测)
├── USAGE_MANUAL.md      # 完整使用手册
└── requirements.txt
```

## 预期效果

| 指标 | 基座模型 | SFT后 | GRPO后 |
|------|----------|-------|--------|
| JSON格式输出率 | ~30% | ~85% | **~97%** |
| 正确工具选择率 | ~20% | ~60% | **~78%** |
| 参数填充准确率 | ~15% | ~50% | **~72%** |
| 复杂编排成功率 | ~10% | ~35% | **~58%** |

## 硬件要求

| 阶段 | 最低配置 | 推荐配置 | 耗时 |
|------|----------|----------|------|
| 数据生成 | CPU | — | 5 min |
| SFT训练 | RTX 4090 24GB | A100 40GB | 2-4 h |
| GRPO训练 | A100 40GB | A100 80GB | 4-8 h |
| 评测 | RTX 4090 24GB | A100 40GB | 0.5-1 h |

## 文档

- **[USAGE_MANUAL.md](USAGE_MANUAL.md)** — 完整使用手册（5步上手、调参指南、FAQ）
- **[agentic_RL_技术方案.md](agentic_RL_技术方案.md)** — 理论背景与完整技术方案

## License

MIT
