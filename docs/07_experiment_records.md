# 7. 实验过程记录

本文档合并历史实验日志，并规定后续实验记录格式。所有新实验都应记录到本文档，或在本文档中链接到单独的长日志。

## 记录原则

每次实验必须包含：

- 日期、job、项目路径、启动人和目标。
- base model 路径与 hash 或 revision。
- 训练数据路径、manifest、样本数、配比。
- 脚本路径和完整命令。
- SwanLab run 链接。
- 日志、metrics、checkpoint、eval 输出路径。
- 关键指标、异常、恢复动作和结论。
- 是否可作为后续实验基线。

## 当前远端路径

```text
Job: bifrost-2026060214414601-yans2
Project: /workspace/yans2@xiaopeng.com/agentic_rl_pipeline
```

## 历史实验摘要

### 2026-06-09 基线评测与 SFT 准备

目标：

- 跑通 Qwen3/Qwen2.5 小模型基础评测和 SFT 数据链路。
- 准备 xLAM、GSM8K、MMLU-Pro、WikiSQL 等 probe。

关键产物：

```text
scripts/remote/launch_sft_v2.sh
scripts/remote/evaluate_xlam_tool_calls.py
scripts/remote/evaluate_general_regression.py
scripts/remote/evaluate_mmlu_logprob.py
scripts/remote/evaluate_wikisql.py
```

结论：

- 初步建立了 “训练 -> checkpoint -> probe 评测” 的闭环。
- 内部 probe 不能等同官方 benchmark，需要结果标记。

### 2026-06-10 夜间连续队列与 WikiSQL

目标：

- 保证夜间 GPU 不空闲。
- 增加 WikiSQL SQL generation/execution 评测。

关键脚本：

```text
scripts/remote/run_night_successor_queue.sh
scripts/remote/run_wikisql_eval_queue.sh
scripts/remote/monitor_wikisql_eval.sh
```

结论：

- WikiSQL 评测需区分 SQL extraction、execution rate、execution accuracy、normalized SQL exact。
- SQL 只是 agent 能力的一部分，不能替代 MCP/tool 评测。

### 2026-06-11 24 小时 Instruction + RLVR 队列

目标：

- 延长训练任务，避免 GPU 空跑。
- 尝试 instruction SFT、verifiable GRPO/RLVR 和模型对比。

关键脚本：

```text
scripts/remote/run_24h_optimization_queue.sh
scripts/remote/monitor_24h_optimization_queue.sh
scripts/remote/train_verifiable_grpo.py
```

结论：

- RLVR 必须在 SFT 已有基本成功率后启动。
- reward mean 上升不等于任务成功率上升，必须看 verifier 分量。

### 2026-06-12 16 PPU Short-SFT Ablation

目标：

- 使用 16 PPU 做短 SFT 消融。
- 对比学习率、seed、模型和数据配方。

关键脚本：

```text
scripts/remote/run_16ppu_sft_ablation_20260612.sh
scripts/remote/monitor_16ppu_sft_ablation_20260612.sh
```

结论：

- 多实验并行能提高资源利用，但必须隔离日志和输出目录。
- 小差异必须多 seed 验证。

### 2026-06-15 SOTA-inspired V4

目标：

- 借鉴公开 agent 模型的数据混合和训练方法。
- 构建更贴近市场指标的后训练配方。

关键脚本：

```text
scripts/remote/prepare_sft_v4_agent_mixture.py
scripts/remote/run_sota_v4_ppu_queue_20260615.sh
```

结论：

- 数据混合必须同时兼顾 agent 指标和通用能力回归。
- 需要 market-aligned 结果标记，不能将内部 probe 当作官方分数。

### 2026-06-15 MCP SFT 稳定性诊断

目标：

- 排查 loss 震荡、训练不稳定和效果无提升。

发现：

- 旧 logger 对 gradient accumulation 的 loss 缩放不正确。
- 数据混合中短 completion 与长 completion 方差很高。
- BF16 全参数 + `1e-7` 学习率导致大多数更新被量化吞掉。
- clarify 与 no-tool 都是 `[]`，训练目标冲突。
- 固定验证集显示训练基本无效。

修正：

- 改为 supervised-token mean loss。
- LoRA adapter 保持 FP32。
- clarify 输出显式 JSON action。
- no-tool/clarify/positive 设置不同 loss weight。
- 增加固定验证集和 clipped step rate。

相关脚本：

```text
scripts/remote/train_assistant_only_sft.py
scripts/remote/train_lora_sft.py
scripts/remote/prepare_mcp_sft_v2.py
scripts/remote/evaluate_sft_loss.py
```

### 2026-06-18 MCP LoRA SFT v3

目标：

- 按修正后的数据和训练方案启动新 LoRA SFT。
- 接入 SwanLab。
- 记录完整日志。

模型：

```text
/publicdata/huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
```

数据：

```text
/workspace/yans2@xiaopeng.com/agentic_rl_pipeline/datasets/processed/mcp_lora_sft_v3_20260618/sft_v2_all/train_sft.jsonl
```

输出：

```text
/workspace/yans2@xiaopeng.com/agentic_rl_pipeline/output/mcp_lora_sft_v3_8ppu_20260618
/workspace/yans2@xiaopeng.com/agentic_rl_pipeline/logs/mcp_lora_sft_v3_8ppu_20260618.log
```

SwanLab：

```text
https://swanlab.cn/@yans2/agentic-rl-tool-calling/runs/46ldjiiczrmfpf5qm13z8
```

已知情况：

- 16 PPU 首次启动遇到 SwanLab API key 解析问题和 CQ memory warning。
- 已改为 8 PPU 训练。
- 早期 metrics 显示 loss 约 4.0，grad_norm 约 15-18，clipping rate 为 1.0，需要持续观察固定验证 loss。

## 后续实验记录模板

~~~markdown
## YYYY-MM-DD <run_name>

目标：

- 

环境：

```text
Job:
Project:
CUDA/PPU:
Base model:
```

数据：

```text
Train:
Validation:
Manifest:
```

命令：

```bash

```

SwanLab：

```text

```

关键指标：

| Step | Loss | Val Loss | Grad Norm | Clip Rate | Throughput | Notes |
|---:|---:|---:|---:|---:|---:|---|

评测：

| Benchmark | Label | Metric | Result |
|---|---|---|---:|

异常与处理：

- 

结论：

- 
~~~
