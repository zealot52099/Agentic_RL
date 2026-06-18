# 1. 数据收集与预处理

本文档记录 Agentic RL 项目的数据来源、预处理目标、处理前后样例、实现脚本路径和当前远端数据状态。它服务于主计划 `agentic_rl_mastery_and_small_model_training_plan.md` 中的“数据体系”部分。

## 目标

数据层必须支持四类能力：

- 工具调用：server/tool 路由、JSON 参数生成、并行调用和无关工具拒绝。
- MCP 多轮任务：跨 server 调用、观察结果续写、错误恢复和终止判断。
- 安全边界：no-tool、clarification、权限不足、提示注入和高风险操作确认。
- 通用能力保持：指令、推理、代码和数学回放，防止后训练导致能力回退。

所有训练数据统一进入 Agent Trace 或 SFT/RL 导出格式，禁止直接把 benchmark test prompt、gold answer、SWE-bench patch 或官方评测轨迹混入训练集。

## 数据来源

| 类别 | 当前来源 | 用途 | 状态 |
|---|---|---|---|
| MCP smoke trace | `scripts/mcp_agent_pipeline.py build-smoke` | 格式、路由、no-tool、clarify、RLVR 验证 | 已生成 smoke 级数据 |
| xLAM function calling | `datasets/processed/xlam-function-calling-60k` 或远端同名目录 | 单轮函数调用 SFT 和 probe | 已准备 |
| SWE-Gym/OpenHands | `datasets/processed/swe-gym-openhands-sft` 或远端同名目录 | 代码 agent、终端轨迹 | 已准备小规模 |
| 通用 SFT 回放 | `datasets/sources/tulu-*`、`datasets/sources/smol-*`、`datasets/sources/no-robots` | IFEval/GSM8K/MMLU/代码能力保持 | 已下载本地部分 |
| 公开评测集 | `datasets/eval_suite` | 只评测，不训练 | 已准备多项 |

远端主工作目录：

```text
/workspace/yans2@xiaopeng.com/agentic_rl_pipeline
```

当前 MCP SFT v3 数据目录：

```text
/workspace/yans2@xiaopeng.com/agentic_rl_pipeline/datasets/processed/mcp_lora_sft_v3_20260618
```

## 核心实现脚本

| 脚本 | 作用 |
|---|---|
| `scripts/mcp_agent_pipeline.py` | MCP trace 生成、打包、校验、污染检查、内部评测 |
| `agentic_rl/schema.py` | `mcp_agent_trace_v1` schema 校验 |
| `agentic_rl/fingerprint.py` | schema 指纹、server family split、contamination report |
| `agentic_rl/pipeline.py` | smoke trace 合成、SFT/preference/RLVR 导出 |
| `agentic_rl/sandbox.py` | 确定性 MCP 状态沙箱 |
| `agentic_rl/reward.py` | 工具调用 reward 与 verifier |
| `scripts/remote/prepare_mcp_sft_v1.py` | 旧版 MCP SFT 打包 |
| `scripts/remote/prepare_mcp_sft_v2.py` | 修复 no-tool/clarify，固定验证集，过滤超长样本，增加 loss weight |
| `scripts/remote/prepare_sft_v3_mixture.py` | 早期 instruction/RLVR 混合数据 |
| `scripts/remote/prepare_sft_v4_agent_mixture.py` | SOTA-inspired agent 混合数据 |
| `scripts/prepare_remote_datasets.py` | 远端基础数据准备 |
| `scripts/prepare_eval_suite.py` | 下载并记录公开评测资产 |

## 统一 Trace Schema

正式数据必须包含以下字段或可追溯引用：

```text
trace_id
split_group_id
server_catalog
tool_schemas
messages
assistant_action
tool_calls
tool_results
environment_state_before
environment_state_after
permissions
terminal_state
verifier_results
reward_components
generation_metadata
```

关键约束：

- `split_group_id` 使用语义家族，不使用展示名称。
- `schema_fingerprint` 忽略工具名和自然语言描述，防止改名泄漏。
- `server_family` 用于 OOD split。
- `parser_version` 和 `chat_template_version` 必须记录，否则评测结果不可比。
- tool result 必须通过 `call_id` 与 tool call 对齐。

## 处理前样例：原始 MCP Trace

以下是简化样例，展示 trace 进入打包脚本前的形态。

```json
{
  "trace_id": "mcp_smoke_000001",
  "split_group_id": "calendar.create_event",
  "server_catalog": [
    {
      "server": "calendar",
      "tools": [
        {
          "name": "calendar.create_event",
          "description": "Create a calendar event.",
          "input_schema": {
            "type": "object",
            "required": ["title", "date"],
            "properties": {
              "title": {"type": "string"},
              "date": {"type": "string"}
            }
          }
        }
      ]
    }
  ],
  "messages": [
    {"role": "user", "content": "Schedule project review for Friday."}
  ],
  "tool_calls": [
    {
      "name": "calendar.create_event",
      "arguments": {"title": "project review", "date": "Friday"}
    }
  ],
  "verifier_results": {
    "decision": "tool_call",
    "expected_calls": ["calendar.create_event"]
  }
}
```

## 处理后样例：SFT Row

`prepare_mcp_sft_v2.py` 输出训练器直接消费的 prompt/completion 行：

```json
{
  "id": "mcp_smoke_000001",
  "mixture_source": "mcp_positive",
  "prompt": "SYSTEM: You are an MCP tool-using assistant...\nMCP_SERVER_CATALOG:\n[...]\n\nUSER:\nSchedule project review for Friday.\n\nASSISTANT:",
  "completion": "[{\"name\":\"calendar.create_event\",\"arguments\":{\"title\":\"project review\",\"date\":\"Friday\"}}]",
  "verifier": {
    "kind": "mcp_tool_call",
    "decision": "tool_call",
    "expected_calls": [
      {"name": "calendar.create_event", "arguments": {"title": "project review", "date": "Friday"}}
    ]
  },
  "loss_weight": 1.0,
  "prompt_tokens": 512,
  "completion_tokens": 28
}
```

## 处理后样例：Clarification Row

旧版 clarify 与 no-tool 都输出 `[]`，会导致模型学不会追问。v2 已改为显式 JSON action。

```json
{
  "id": "mcp_smoke_missing_arg_000002",
  "mixture_source": "mcp_clarify",
  "prompt": "SYSTEM: Return [] when no tool applies. When required information is missing, return one JSON object with action=\"clarify\"...\nUSER:\nUse calendar.create_event for me.",
  "completion": "{\"action\":\"clarify\",\"missing\":[\"title\",\"date\"],\"message\":\"Please provide the required information: title, date.\"}",
  "verifier": {
    "kind": "mcp_tool_call",
    "decision": "clarify",
    "expected_action": "clarify"
  },
  "loss_weight": 2.0
}
```

## 处理后样例：No-Tool Row

```json
{
  "id": "mcp_smoke_no_tool_000003",
  "mixture_source": "mcp_no_tool",
  "prompt": "SYSTEM: Return [] when no tool applies...\nUSER:\nTell me a short joke.",
  "completion": "[]",
  "verifier": {
    "kind": "mcp_tool_call",
    "decision": "no_tool",
    "expected_calls": []
  },
  "loss_weight": 16.0
}
```

## 处理后样例：Preference Row

```json
{
  "id": "pref_000001",
  "prompt": "SYSTEM: ...\nUSER: Schedule project review for Friday.\nASSISTANT:",
  "chosen": "[{\"name\":\"calendar.create_event\",\"arguments\":{\"title\":\"project review\",\"date\":\"Friday\"}}]",
  "rejected": "[{\"name\":\"email.send\",\"arguments\":{\"subject\":\"project review\"}}]",
  "preference_type": "correct_tool_vs_similar_wrong_tool",
  "verifier": {"kind": "mcp_tool_call"}
}
```

## 处理后样例：RLVR Row

```json
{
  "id": "rlvr_000001",
  "prompt": "SYSTEM: ...\nUSER: Schedule project review for Friday.\nASSISTANT:",
  "verifier": {
    "kind": "mcp_tool_call",
    "expected_calls": [
      {"name": "calendar.create_event", "arguments": {"title": "project review", "date": "Friday"}}
    ],
    "reward_components": {
      "route": 0.10,
      "arguments": 0.15,
      "execution": 0.20,
      "task": 0.35,
      "recovery": 0.10,
      "safety": 0.10
    }
  }
}
```

## 当前已生成数据

远端 smoke 数据已完成以下版本：

| 数据目录 | 用途 | 说明 |
|---|---|---|
| `raw/` | 保留 20% OOD split | 用于污染检查和内部 OOD |
| `sft_v2/` | 修复后的 OOD train split | 不含完整 no-tool 分布，不作为当前主训练数据 |
| `raw_all/` | 20k 全量 trace | 用于训练覆盖 |
| `sft_v2_all/` | 当前 LoRA SFT 输入 | 含 positive、clarify、no-tool |

`sft_v2_all/manifest.json` 关键统计：

```json
{
  "train_rows": 19494,
  "validation_rows": 506,
  "train_counts": {
    "mcp_positive": 14744,
    "mcp_clarify": 2375,
    "mcp_no_tool": 2375
  },
  "validation_counts": {
    "mcp_clarify": 125,
    "mcp_no_tool": 125,
    "mcp_positive": 256
  }
}
```

## 标准处理命令

构建 smoke trace：

```bash
python scripts/mcp_agent_pipeline.py build-smoke \
  --output-dir datasets/processed/mcp_lora_sft_v3_20260618/raw \
  --count 20000 \
  --seed 20260618 \
  --holdout-percent 20
```

打包正式 trace：

```bash
python scripts/mcp_agent_pipeline.py package-traces \
  --input datasets/raw/mcp_traces.jsonl \
  --output-dir datasets/processed/mcp_agent_v1 \
  --holdout-percent 20
```

修复 SFT 标签并切分验证集：

```bash
python scripts/remote/prepare_mcp_sft_v2.py \
  --input datasets/processed/mcp_lora_sft_v3_20260618/raw_all/train_sft.jsonl \
  --model /publicdata/huggingface.co/Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir datasets/processed/mcp_lora_sft_v3_20260618/sft_v2_all \
  --seq-len 2048 \
  --validation-per-source 256 \
  --seed 20260618
```

污染检查：

```bash
python scripts/mcp_agent_pipeline.py check-contamination \
  --train datasets/processed/mcp_agent_v1/train_traces.jsonl \
  --eval datasets/processed/mcp_agent_v1/ood_eval_traces.jsonl
```

## 质量门槛

- JSON/schema valid rate 不低于 99.5%。
- 同一 `schema_fingerprint` 或 `split_group_id` 不得跨训练与 OOD 测试泄漏。
- clarify 不能再用 `[]`，必须输出显式追问动作。
- no-tool 样本比例不少于 8%，否则模型容易过度调用工具。
- 样本必须记录来源、license、处理脚本、token 长度和 verifier。
- 公开 benchmark test 数据只可进入 `datasets/eval_suite`，不得进入训练输入。

## 后续补强

当前数据是 smoke 级闭环，不是生产规模数据。下一步需要：

- 扩展到至少 60 个 MCP Server、500 个工具、8 万条多轮轨迹。
- 使用教师模型生成多路径轨迹，并通过 schema、执行和最终状态三层校验。
- 加入真实 MCP Server 回放，减少模板化数据的模拟器偏差。
- 用当前学生模型挖掘失败样本，形成 hard negative、preference 和 RLVR 数据。
