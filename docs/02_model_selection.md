# 2. 模型选型

本文档记录 Agentic RL 小模型路线的模型梯队、选择原则、对照模型和当前推荐实验顺序。

## 选型目标

本项目目标不是从零训练大模型，而是在 7B 或更小模型上通过后训练提升 MCP/tool 选择能力，同时尽量保持通用能力。模型需要满足：

- 支持稳定的 HF Transformers 加载和保存。
- tokenizer/chat template 可控，便于 tool-call parser 固定。
- 长上下文能力足够容纳 server catalog、tool schema、用户上下文和 tool observation。
- 对 JSON、代码、SQL、结构化输出有较强先验。
- 在当前 PPU/GPU 环境中可以完成 LoRA 或小规模全参训练。

## 模型梯队

| 阶段 | 模型 | 用途 |
|---|---|---|
| 调试 | Qwen2.5-1.5B-Instruct / Qwen3-1.7B | 快速验证数据、loss、SwanLab、评测和恢复 |
| 主力小模型 | Qwen2.5-3B、Qwen2.5-Coder-3B、Qwen3-4B | Agent/tool 后训练和消融 |
| 7B 上限 | Qwen2.5-7B-Instruct、Qwen2.5-Coder-7B、Qwen3-8B | 接近同体量 SOTA 的最终候选 |
| 专用对照 | xLAM-2-3B-fc-r、xLAM-2-8B-fc-r | 函数调用专用模型对比 |
| 教师模型 | Qwen 30B+、DeepSeek/Qwen API、xLAM 8B | 生成轨迹、修复答案、构造 preference |

## 当前推荐

首轮训练推荐使用：

```text
/publicdata/huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
```

原因：

- 已在远端可用。
- 小模型便于快速暴露数据、loss 和 trainer 问题。
- Instruct 底座比 Base 更容易学习 no-tool、clarify 和安全边界。
- LoRA 成本低，适合先跑稳定性和数据质量验证。

稳定后迁移到：

```text
/publicdata/huggingface.co/Qwen/Qwen3-4B-Instruct-2507
/publicdata/huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct
/publicdata/huggingface.co/Qwen/Qwen2.5-7B-Instruct
```

具体以远端 `/publicdata/huggingface.co/Qwen` 实际存在权重为准。

## Base 与 Instruct 的取舍

| 底座 | 优点 | 风险 | 建议 |
|---|---|---|---|
| Base | 可做 CPT/mid-training，格式污染少 | 指令跟随弱，SFT 成本高，短期评测差 | 用于预训练原理和 CPT 实验，不作为首轮 MCP SFT 主线 |
| Instruct | 指令跟随、JSON、拒绝和对话能力较好 | 可能已有模板偏置，RL 时需控制 KL | 当前主线 |
| Coder | SQL、代码、工具参数和 SWE 任务更强 | 通用对话可能偏弱 | Agent/code/SWE 对照路线 |
| Function-calling 专用模型 | 工具调用强 | 通用能力与许可需确认 | 作为对照，不默认作为生产底座 |

## 选型评测矩阵

统一评测未经训练模型：

| 能力 | 评测 |
|---|---|
| MCP 内部工具路由 | `scripts/remote/evaluate_mcp_internal.py` |
| xLAM 工具调用 probe | `scripts/remote/evaluate_xlam_tool_calls.py` |
| IFEval | `scripts/remote/generate_ifeval_responses.py` + `scripts/remote/summarize_ifeval.py` |
| GSM8K/MMLU-Pro | `scripts/remote/evaluate_general_regression.py`、`scripts/remote/evaluate_mmlu_logprob.py` |
| WikiSQL | `scripts/remote/evaluate_wikisql.py` |
| SWE-bench | 官方 harness，需 Docker-capable worker |

只有同一 runner、同一 prompt、同一 parser、同一 max tokens 下的结果才可严格比较。

## 当前训练优先级

1. Qwen2.5-1.5B-Instruct LoRA SFT：验证数据和训练稳定性。
2. Qwen3-4B/Qwen2.5-Coder-3B LoRA SFT：比较 tool routing 与代码类任务。
3. 最优 3B/4B checkpoint 做 DPO 或 SimPO。
4. 达到内部任务成功率 15%-60% 后尝试 GRPO/RLVR。
5. 选出最优 recipe 后迁移 7B。

## 风险与处理

| 风险 | 表现 | 处理 |
|---|---|---|
| 模型太小导致复杂多轮能力上不去 | 多轮任务成功率低，但格式正确 | 先把目标拆成路由/参数/no-tool，再扩 3B/4B |
| Base 模型训练成本高 | SFT 很久仍不会指令 | 首轮用 Instruct，Base 仅做 CPT/原理实验 |
| Coder 模型通用能力回退 | IFEval/GSM8K/MMLU 下降 | 混入 10%-15% 通用回放，做固定回归门槛 |
| xLAM 对照不可比 | 只在函数调用强，其他能力不同 | 明确标注为专用对照，不作为同口径通用模型 |
| chat template 漂移 | 工具调用格式突然变差 | 每个 run 记录 tokenizer、template、parser 版本 |
