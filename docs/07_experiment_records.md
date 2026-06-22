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

### 2026-06-19 DPO Phase C v1 (bug) → v2 (Fixed)

目标：
- 在 1.5B SFT v3 基础上做 DPO，改善工具路由判别
- 修复 v1 clarify 崩塌问题

关键发现：
- v1: 偏好数据缺少 clarify 对 → clarify 0% (崩塌), beta=0.1 过低
- v2: 补充 clarify 偏好对 4,750 条 + beta→0.5 → 零退化 (100% val)
- **DPO 数据必须覆盖所有输出类型，否则遗忘未覆盖类型**

产物：
- DPO v1 (bug): `output/dpo_phase_c_20260619_013121` (退化)
- DPO v2 Fixed: `output/dpo_fixed_20260619_043239` (100% val)

### 2026-06-19 Coder7B SFT + DPO (MCP-only)

目标：1.5B→7B Coder，验证更大基座

结果：
- SFT: step 60-70 出现"顿悟" (grad 1665)，step 100 loss→0.68 收敛
- DPO (MCP-only): loss 从 0.69→0.001，训练耗时 6.6h
- **DPO 收敛但有害：BFCL V4 从 36.5%→11.5% (-25pp)**
- 原因：MCP-only DPO 覆盖了 Coder 原生的标准 function calling 能力

产物：
- SFT: `output/coder7b_sft_20260619_230410` (323MB)
- DPO: `output/coder7b_dpo_20260620_002247` (161MB)

### 2026-06-20 BFCL V4 Live 评测 + 内部 OOD

目标：首次在公开 benchmark 上区分模型

结果：
- 内部 smoke 验证集 100% (无效区分)
- OOD held-out 5000 条 100% (同样无效)
- **BFCL V4 首次拉开差距：1.5B DPO 80.0%, 1.5B SFT 62.4%, Coder7B DPO 11.5%, Coder7B SFT 36.5%**
- 内部评测完全无法区分模型能力

### 2026-06-21 混合格式 SFT (方向 A)

目标：修复 Coder7B 格式 mismatch → BFCL 回归

数据：28,448 条 (68.5% MCP + 31.5% 标准 function calling 格式)

结果：
- BFCL V4 Live: 36.5%→**82.4%** (+46pp)
- 仅 31.5% 标准格式数据，Coder7B 从垫底跃升至第一
- **证明 Coder 的 SQL/工具知识是完整的，SFT 只教了"输出格式纪律"**

产物：
- `output/coder7b_mixed_sft_20260621_001502` (323MB)

### 2026-06-21 混合格式 DPO (方向 B, 失败)

尝试 v1 和 v2 均不收敛 (loss 卡在 0.69)
- 根因：DPO 要求单一稳定输出分布，混合格式 (MCP JSON + STD JSON) 在 token 级别冲突
- 结论：**DPO 不兼容多输出格式场景，应使用 GRPO**

### 2026-06-21 1.5B DPO → GRPO (方向 P0, 成功)

目标：在 1.5B DPO 基础上验证 GRPO

改进：
- 分布外 prompt (BFCL) + temperature=1.0 + 粒度化 reward (0/0.5/1.0)
- reward_std 从 0→0.4-0.5, entropy 从 1e-7→0.2-1.5

结果：
- BFCL V4 Live: 80.0%→**83.5%** (+3.5pp)
- **SFT→DPO→GRPO 三阶段 pipeline 在 1.5B 上完整验证**
- 累计提升: 62.4%→80.0%→83.5% = +21pp

产物：
- `output/grpo_15b_dpo_20260621_153153/adapter`

### 2026-06-22 BFCL V3 SQL + 新 Benchmark 评测

目标：打通 SQL 和 Agentic 评测维度

SQL 结果：
- Coder7B 基座: Func 3% (知识完整，格式乱)
- Coder7B Mixed SFT: Func **99%**, Param **87%**, Exact **59%** (+96pp)
- 1.5B GRPO: Func 51%, Param 39%, Exact 17%

Agentic/Multi-Turn：BFCL Multi-Turn 100% JSON 合法, Web Search 100%

新 Benchmark 首次评测：
- WikiSQL: 0% (模型输出 JSON 函数调用，需 prompt 适配)
- Spider: 能生成 SQL 关键词
- Glaive FC: 0% (数据格式不兼容)

### 2026-06-22 RL 方法选型分析

候选方法评估：
- **GRPO**: ✅ 采用 (Dense 模型, 短 JSON, 粒度化 reward, 已验证 +3.5pp)
- GSPO: ❌ 解决 MoE 路由问题，Qwen2.5 是 Dense 不触发
- DAPO: ❌ 解决长 CoT (>1000 tokens) 熵坍缩，短 JSON 不触发
- Dr.GRPO: ❌ 修复长度偏差，输出均匀 50-100 tokens 优化很小
- PPO: ❌ 太重，需要 Critic 模型
- RLVR: ❌ reward 粒度不够 (仅二元)

文档路径：`docs/experiments/2026-06-20_experiment_log.md` (详细分析)

### 2026-06-22 GRPO Coder7B 能力 1+2 (进行中)

目标：GRPO 优化 SQL Exact (59%→70%+) + BFCL Live (82%→85%+)

配置：240 prompts (80 SQL + 160 BFCL Live), group=4, temp=1.0, beta=0.04

状态：🟢 训练中 (480 steps, ~1.7h)

BFCL V4 最终排名：
```
1.5B GRPO              83.5%  🥇  SFT→DPO→GRPO
Coder7B Mixed SFT      82.4%  🥈  混合格式 SFT
1.5B DPO               80.0%
1.5B SFT               62.4%
Coder7B SFT (MCP)      36.5%
Coder7B DPO (MCP)      11.5%
```



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
