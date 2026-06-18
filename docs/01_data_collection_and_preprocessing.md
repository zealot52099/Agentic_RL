# 1. 数据收集与预处理

本文档记录 Agentic RL 项目的数据来源、原始格式、预处理脚本、处理后格式、产物路径和质量门槛。它服务于主计划 `agentic_rl_mastery_and_small_model_training_plan.md` 中的“数据体系”部分。

这篇文档不是只讲 SFT 数据。它覆盖四类资产：

- 训练数据：SFT、preference、RLVR。
- 验证数据：固定 validation split、内部 OOD split。
- 评测数据：IFEval、BFCL、SWE-bench、GSM8K、MMLU-Pro、WikiSQL、LiveCodeBench 等。
- 元数据：manifest、license、revision、schema fingerprint、污染检查报告。

## 总体目标

数据层必须支持以下能力：

- 工具调用：server/tool 路由、JSON 参数生成、并行调用和无关工具拒绝。
- MCP 多轮任务：跨 server 调用、观察结果续写、错误恢复和终止判断。
- 安全边界：no-tool、clarification、权限不足、提示注入和高风险操作确认。
- 通用能力保持：指令、推理、代码和数学回放，防止后训练导致能力回退。
- 可复现评测：内部 probe、官方复跑和厂商公开数字必须分开标记。

硬性边界：

- benchmark test prompt、gold answer、SWE-bench patch、hidden tests、官方评测轨迹不得进入训练数据。
- 所有训练样本必须保留来源、处理脚本、处理时间、license/revision 或本地 hash。
- 任何用于 OOD 评测的 server family、schema fingerprint 或 task family 不得泄漏到训练 split。

## 目录约定

本地项目根目录：

```text
D:\project\agentic_RL
```

远端主工作目录：

```text
/workspace/yans2@xiaopeng.com/agentic_rl_pipeline
```

历史远端数据根目录：

```text
/workspace/yans2@xiaopeng.com/agentic_rl/datasets
```

当前 MCP SFT v3 数据目录：

```text
/workspace/yans2@xiaopeng.com/agentic_rl_pipeline/datasets/processed/mcp_lora_sft_v3_20260618
```

公开评测资产目录：

```text
datasets/eval_suite
datasets/eval_suite/manifests/eval_suite_manifest.json
```

## 数据来源总览

| 来源 | 原始位置 | 处理后位置 | 进入训练 | 主要脚本 |
|---|---|---|---|---|
| MCP smoke / MCP trace | 由 `scripts/mcp_agent_pipeline.py` 生成，或正式 `datasets/raw/mcp_traces.jsonl` | `datasets/processed/mcp_*` | 是，导出 SFT/preference/RLVR | `scripts/mcp_agent_pipeline.py`、`scripts/remote/prepare_mcp_sft_v2.py` |
| xLAM function calling 60K | `datasets/sources/huggingface/xlam-function-calling-60k-parsed/*.parquet` | `datasets/processed/xlam-function-calling-60k/train.jsonl`、split 后 `train_sft.jsonl/eval.jsonl` | 是，主要用于单轮工具调用 SFT；heldout 可评测 | `scripts/prepare_remote_datasets.py`、`scripts/prepare_xlam_splits.py` |
| SWE-Gym/OpenHands | `datasets/sources/huggingface/SWE-Gym-OpenHands-SFT-Trajectories/*.parquet` | `datasets/processed/swe-gym-openhands-sft/train.jsonl` | 是，小规模代码 agent/终端轨迹 SFT | `scripts/prepare_remote_datasets.py`、`scripts/remote/prepare_sft_v4_agent_mixture.py` |
| 通用能力回放 | `datasets/sources/tulu-*`、`datasets/sources/smol-*`、`datasets/sources/no-robots` | SFT mixture 中的 broad/general rows | 是，用于防止能力回退 | `scripts/remote/prepare_sft_v3_mixture.py`、`scripts/remote/prepare_sft_v4_agent_mixture.py` |
| 合成 hard constraints | 脚本实时生成 | `rl_verifiable_hard.jsonl` 或 SFT mixture | 是，用于格式/指令/RLVR | `scripts/remote/prepare_sft_v4_agent_mixture.py` |
| tau2-bench / AgentBench / AgentTuning | `datasets/sources/tau2-bench`、`datasets/sources/AgentTuning` | manifest 记录；部分任务可转内部 eval | 默认评测/环境资产，不直接训练 | `scripts/prepare_remote_datasets.py` |
| IFEval/BFCL/SWE-bench/LiveCodeBench/HumanEval/MBPP/GSM8K/MMLU-Pro | `datasets/eval_suite` | eval-only prompt/runner 输入 | 否，评测专用 | `scripts/prepare_eval_suite.py` |

## 核心实现脚本

| 脚本 | 作用 |
|---|---|
| `scripts/mcp_agent_pipeline.py` | MCP trace 生成、打包、校验、污染检查、内部评测 |
| `agentic_rl/schema.py` | `mcp_agent_trace_v1` schema 校验 |
| `agentic_rl/fingerprint.py` | schema 指纹、server family split、contamination report |
| `agentic_rl/pipeline.py` | smoke trace 合成、SFT/preference/RLVR 导出 |
| `agentic_rl/sandbox.py` | 确定性 MCP 状态沙箱 |
| `agentic_rl/reward.py` | 工具调用 reward 与 verifier |
| `scripts/prepare_remote_datasets.py` | 将远端 parquet/repo 资产转 JSONL 并写 manifest |
| `scripts/prepare_xlam_splits.py` | xLAM 按 tool family 切分 train/heldout/eval，并渲染 SFT prompt |
| `scripts/remote/prepare_mcp_sft_v2.py` | 修复 MCP SFT 标签，固定验证集，过滤超长样本，增加 `loss_weight` |
| `scripts/remote/prepare_sft_v3_mixture.py` | 早期 instruction/RLVR 混合数据 |
| `scripts/remote/prepare_sft_v4_agent_mixture.py` | SOTA-inspired SFT/RLVR 混合数据 |
| `scripts/prepare_eval_suite.py` | 下载并记录公开评测资产 |

## 统一中间格式

不同来源不会天然长得一样。项目中有两个中间层：

1. **Agent Trace**：适合 MCP、多轮工具、sandbox、RLVR 和 OOD split。
2. **训练 row**：适合训练器直接读取，通常是 `prompt` + `completion` 或 `prompt` + `chosen/rejected`。

正式 MCP Agent Trace 应包含：

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

关键字段：

- `split_group_id`：语义任务家族，用于 train/OOD split。
- `schema_fingerprint`：忽略工具名和描述后的 schema 指纹，防止改名泄漏。
- `server_family`：用于保留真正 OOD server family。
- `parser_version`：工具调用 parser 版本。
- `chat_template_version`：prompt/template 版本。
- `verifier_results`：执行、最终状态、安全和恢复的判定证据。

## 阶段级标准 Schema

原始数据允许不同，但进入同一个训练或评测阶段后，消费格式必须一致。这里的“一致”指 required fields 和 action 表达一致；不同来源可以保留 optional metadata，供分桶统计、调试、评测和溯源使用。

### SFT Schema

SFT 训练器消费 `prompt` + `completion`。所有来源进入 SFT 前都必须转成这个形态。

Required fields：

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | string/int | 样本唯一 ID，建议稳定可复现 |
| `prompt` | string | 完整输入，包含 system/tool schema/user/history，结尾通常是 `ASSISTANT:` |
| `completion` | string | 只监督 assistant 输出，不包含 prompt |
| `mixture_source` | string | 来源分桶，例如 `mcp_positive`、`mcp_no_tool`、`xlam_tool_call`、`swe_trajectory`、`broad_instruction` |

Optional fields：

| 字段 | 类型 | 说明 |
|---|---|---|
| `loss_weight` | number | 样本或来源权重，用于补偿 no-tool 等短 completion |
| `verifier` | object | 可选 verifier，便于训练后回放或抽样检查 |
| `prompt_tokens` | int | 预处理时统计的 prompt token 数 |
| `completion_tokens` | int | 预处理时统计的 completion token 数 |
| `source` | string | 上游数据集或生成器 |
| `source_revision` | string | HF revision、git commit 或数据 hash |
| `license` | string | 数据许可 |
| `schema_fingerprint` | string | 工具 schema 指纹，主要用于 MCP/tool 数据 |
| `split_group_id` | string | 任务家族，主要用于 OOD split |
| `expected_calls` | array | 工具调用 gold，用于分析，不一定被 SFT trainer 使用 |

标准 SFT row：

```json
{
  "id": "sft_000001",
  "prompt": "SYSTEM: ...\nUSER:\nSchedule project review for Friday.\n\nASSISTANT:\n",
  "completion": "[{\"name\":\"calendar.create_event\",\"arguments\":{\"title\":\"project review\",\"date\":\"Friday\"}}]",
  "mixture_source": "mcp_positive",
  "loss_weight": 1.0,
  "verifier": {
    "kind": "mcp_tool_call",
    "expected_calls": [
      {"name": "calendar.create_event", "arguments": {"title": "project review", "date": "Friday"}}
    ]
  }
}
```

约束：

- `completion` 必须是模型应该生成的 assistant 内容，不能包含 user/system 文本。
- 工具调用类 completion 必须使用统一 action 表达，例如 JSON array 或显式 `clarify` JSON。
- no-tool 统一用 `[]`，clarify 不能用 `[]`。
- 如果样本来自通用回放，`verifier` 可以为空，但必须有 `mixture_source`。
- 训练前必须过滤超过目标 `seq_len` 的样本，或明确采用 packing/truncation 规则。

### Preference / DPO Schema

Preference 数据用于 DPO、SimPO 或其他偏好优化。核心是一条 prompt 配一个 chosen 和 rejected。

Required fields：

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | string/int | 偏好样本唯一 ID |
| `prompt` | string | 与 SFT 同风格的输入 |
| `chosen` | string | 更优输出 |
| `rejected` | string | 较差输出 |
| `preference_type` | string | 偏好类型，例如 `correct_tool_vs_similar_wrong_tool` |

Optional fields：

| 字段 | 类型 | 说明 |
|---|---|---|
| `verifier` | object | 说明 chosen 为什么正确、rejected 为什么错误 |
| `reward_chosen` | number | chosen 的 verifier/reward 分数 |
| `reward_rejected` | number | rejected 的 verifier/reward 分数 |
| `error_type` | string | rejected 的主要错误类型 |
| `mixture_source` | string | 来源分桶 |
| `source_trace_id` | string | 关联的原始 trace |

标准 preference row：

```json
{
  "id": "pref_000001",
  "prompt": "SYSTEM: ...\nUSER:\nSchedule project review for Friday.\n\nASSISTANT:\n",
  "chosen": "[{\"name\":\"calendar.create_event\",\"arguments\":{\"title\":\"project review\",\"date\":\"Friday\"}}]",
  "rejected": "[{\"name\":\"email.send\",\"arguments\":{\"subject\":\"project review\"}}]",
  "preference_type": "correct_tool_vs_similar_wrong_tool",
  "verifier": {
    "kind": "mcp_tool_call",
    "expected_calls": [
      {"name": "calendar.create_event", "arguments": {"title": "project review", "date": "Friday"}}
    ]
  }
}
```

推荐偏好类型：

- `correct_tool_vs_similar_wrong_tool`
- `correct_arguments_vs_wrong_arguments`
- `clarify_vs_guess_missing_arguments`
- `no_tool_vs_unnecessary_tool_call`
- `safe_refusal_vs_unauthorized_call`
- `recovery_vs_repeated_failure`

约束：

- chosen/rejected 必须共享同一个 prompt。
- rejected 最好是“近邻错误”，不是明显无关的垃圾输出。
- 偏好对必须按类型分桶汇报，否则容易被大量简单样本掩盖安全或澄清退化。

### RLVR / GRPO Schema

RLVR 数据用于在线采样和 verifier 打分。核心是 prompt + verifier，而不是固定 completion。

Required fields：

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | string/int | RL 样本唯一 ID |
| `prompt` | string | rollout 起点 |
| `verifier` | object | 可执行或可规则验证的判分器 |

Optional fields：

| 字段 | 类型 | 说明 |
|---|---|---|
| `reward_components` | object | route、arguments、execution、task、safety 等分量权重 |
| `reference_completion` | string | 可选参考答案，仅用于调试，不作为 RL 监督 |
| `environment_state` | object/string | 初始环境状态或快照引用 |
| `permissions` | object | 当前授权状态 |
| `max_turns` | int | 最大工具调用轮数 |
| `source_trace_id` | string | 关联 trace |
| `mixture_source` | string | 来源分桶 |

标准 RLVR row：

```json
{
  "id": "rlvr_000001",
  "prompt": "SYSTEM: ...\nUSER:\nSchedule project review for Friday.\n\nASSISTANT:\n",
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

约束：

- verifier 必须能自动打分，不能依赖教师模型自评。
- 格式正确只能作为 gate，不能让模型靠格式刷高分。
- 最终任务状态或 hidden assertion 应是主奖励来源。
- RLVR 数据不应在 SFT 中直接使用 `reference_completion` 训练，除非明确导出为 SFT row。

### Eval Schema

Eval 数据用于统一评测和回归。它可以长得接近 RLVR，但必须额外标明结果标签和 benchmark 信息。

Required fields：

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | string/int | 评测样本 ID |
| `prompt` | string | 模型输入 |
| `benchmark` | string | 例如 `internal_mcp_ood`、`xlam_probe`、`ifeval`、`bfcl` |
| `label_type` | string | `INTERNAL-PROBE`、`OUR-RERUN`、`OFFICIAL-RERUN`、`VENDOR-REPORTED` |
| `expected` 或 `verifier` | object/string | gold answer、expected call 或 verifier |

Optional fields：

| 字段 | 类型 | 说明 |
|---|---|---|
| `category` | string | 评测子类，例如 parallel、irrelevance、no-tool |
| `source_revision` | string | 数据集 revision |
| `evaluator_revision` | string | runner/evaluator revision |
| `parser_version` | string | 输出解析器版本 |
| `chat_template_version` | string | prompt 模板版本 |
| `max_tokens` | int | 生成长度 |
| `temperature` | number | 解码温度 |
| `scaffold` | object/string | SWE-bench/tau2 等 agent scaffold 配置 |

标准内部 eval row：

```json
{
  "id": "eval_mcp_ood_000001",
  "benchmark": "internal_mcp_ood",
  "label_type": "INTERNAL-PROBE",
  "prompt": "SYSTEM: ...\nUSER:\nFind unread high-priority emails and create a follow-up task.\n\nASSISTANT:\n",
  "verifier": {
    "kind": "mcp_tool_call",
    "expected_calls": [
      {"name": "email.search", "arguments": {"unread": true, "priority": "high"}},
      {"name": "tasks.create", "arguments": {"source": "email.search"}}
    ]
  },
  "category": "multi_server"
}
```

约束：

- Eval 数据默认不能进入训练。
- 内部 eval 与官方 benchmark 必须分开路径、分开指标、分开标签。
- 生成式评测必须记录 decoding、parser、template 和 evaluator revision。

### 阶段格式与来源映射

| 来源 | SFT | Preference/DPO | RLVR/GRPO | Eval |
|---|---|---|---|---|
| MCP trace | `prompt/completion/verifier` | `prompt/chosen/rejected/verifier` | `prompt/verifier` | `prompt/verifier` |
| xLAM | `prompt/completion/expected_calls` | 可由错误样本构造 | 不推荐直接 RL，除非有 verifier | heldout `prompt/expected_calls` |
| SWE-Gym/OpenHands | assistant turn SFT | 可由成功/失败轨迹构造 | 需要执行环境后再用 | 代码 agent probe |
| 通用回放 | `prompt/completion` | 少量偏好对可用 | 通常不用 | IFEval/GSM8K/MMLU 回归 |
| hard constraints | `prompt/completion/verifier` | 可构造 chosen/rejected | `prompt/verifier` | 格式/指令 probe |
| 公开 benchmark | 禁止进入训练 | 禁止进入训练 | 禁止进入训练 | 官方或内部复跑 |

## 来源一：MCP Smoke / MCP Trace

### 用途

MCP trace 是本项目最核心的数据源，能导出：

- SFT：格式、路由、参数、no-tool、clarify。
- Preference：正确工具 vs 相似错误工具，澄清 vs 猜参数。
- RLVR：prompt + verifier/reward，用于 GRPO。
- OOD eval：按 server family/schema fingerprint/task family 切分。

### 原始样例：MCP Trace

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

### 处理后样例：MCP SFT Row

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

### 处理后样例：Clarification Row

旧版 clarify 与 no-tool 都输出 `[]`，会导致模型学不会追问。v2 预处理将 clarify 改为显式 JSON action。

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

### 处理后样例：No-Tool Row

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

### 处理后样例：Preference Row

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

### 处理后样例：RLVR Row

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

### 命令

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

### 当前产物

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

## 来源二：xLAM Function Calling 60K

### 用途

xLAM 用于学习单轮函数调用的基本能力：

- 工具名选择。
- 参数 JSON 生成。
- 多工具/并行调用。
- tool family heldout 评测。

它不是 MCP 多轮环境数据，不包含真实 MCP server 状态变化和 tool observation。

### 原始位置

远端历史目录：

```text
/workspace/yans2@xiaopeng.com/agentic_rl/datasets/sources/huggingface/xlam-function-calling-60k-parsed/xlam-function-calling-60k.parquet
```

处理后基础 JSONL：

```text
/workspace/yans2@xiaopeng.com/agentic_rl/datasets/processed/xlam-function-calling-60k/train.jsonl
```

### 原始样例：xLAM parquet/json row

```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get weather by city.",
        "parameters": {
          "type": "object",
          "required": ["city"],
          "properties": {
            "city": {"type": "string"}
          }
        }
      }
    }
  ],
  "messages": [
    {"role": "user", "content": "What is the weather in Paris?"},
    {
      "role": "assistant",
      "tool_calls": [
        {
          "function": {
            "name": "get_weather",
            "arguments": "{\"city\":\"Paris\"}"
          }
        }
      ]
    }
  ],
  "extra": {"id": "xlam_000001"}
}
```

### 处理后样例：xLAM SFT Row

`scripts/prepare_xlam_splits.py` 会将 tools 和 user message 渲染成统一 prompt，并将 assistant tool calls 规范化为 JSON array。

```json
{
  "id": "xlam_000001",
  "family": "get_weather",
  "prompt_template_version": "xlam_tool_json_v1",
  "prompt": "You are a tool-calling assistant. Select only tools from the provided definitions. Return only a JSON array...\n\nTOOLS:\n[{...}]\n\nUSER:\nWhat is the weather in Paris?\n\nASSISTANT:\n",
  "completion": "[{\"arguments\":{\"city\":\"Paris\"},\"name\":\"get_weather\"}]",
  "expected_calls": [
    {"name": "get_weather", "arguments": {"city": "Paris"}}
  ],
  "tools": [...]
}
```

### 命令

将 parquet 转 JSONL 并写 manifest：

```bash
python scripts/prepare_remote_datasets.py
```

按 tool family 切分 train/heldout/eval：

```bash
python scripts/prepare_xlam_splits.py \
  --input datasets/processed/xlam-function-calling-60k/train.jsonl \
  --output-dir datasets/processed/xlam-function-calling-60k/splits \
  --holdout-modulus 10 \
  --holdout-bucket 0 \
  --eval-limit 512
```

### 质量要求

- `tools` 必须是 JSON list。
- assistant `tool_calls[].function.arguments` 如果是 string，必须能 parse 为 JSON object。
- heldout 单位是排序后的 expected tool-name set，即 `family`。
- heldout family 不进入 train。
- xLAM eval 是内部工具调用 probe，不等同 BFCL 官方分数。

## 来源三：SWE-Gym / OpenHands SFT Trajectories

### 用途

SWE-Gym/OpenHands 用于学习代码 agent 和终端轨迹：

- 读取问题。
- 操作文件。
- 执行命令。
- 根据测试反馈修复。
- 形成多轮 assistant 行为。

它可以用于 SFT，但不能把 SWE-bench Verified/Lite 的 gold patch 或 test patch 混入训练。

### 原始位置

```text
/workspace/yans2@xiaopeng.com/agentic_rl/datasets/sources/huggingface/SWE-Gym-OpenHands-SFT-Trajectories/swe-gym-openhands-sft.parquet
```

处理后基础 JSONL：

```text
/workspace/yans2@xiaopeng.com/agentic_rl/datasets/processed/swe-gym-openhands-sft/train.jsonl
```

### 原始样例：trajectory row

```json
{
  "messages": [
    {"role": "user", "content": "Fix the failing test in repository X."},
    {"role": "assistant", "content": "We need inspect the failure first."},
    {"role": "tool", "content": "pytest output ..."},
    {"role": "assistant", "content": "The bug is in parser.py. I will patch ..."}
  ],
  "repo": "example/repo",
  "instance_id": "swe_gym_000001"
}
```

### 处理后样例：assistant-only SFT Row

`prepare_sft_v4_agent_mixture.py` 中的 `swe_rows()` 会把每个 assistant turn 拆成独立训练样本，prompt 是之前的上下文，completion 是当前 assistant 内容。

```json
{
  "id": "swe-trajectory-0-turn-3",
  "source": "swe_gym_openhands",
  "mixture_source": "swe_trajectory",
  "prompt": "USER:\nFix the failing test in repository X.\n\nASSISTANT:\nWe need inspect the failure first.\n\nTOOL:\npytest output ...\n\nASSISTANT:\n",
  "completion": "The bug is in parser.py. I will patch ... "
}
```

### 命令

```bash
python scripts/prepare_remote_datasets.py
```

作为 v4 mixture 的输入：

```bash
python scripts/remote/prepare_sft_v4_agent_mixture.py \
  --v3-data datasets/processed/sft_v3/train_sft.jsonl \
  --tool-data datasets/processed/xlam-function-calling-60k/splits/train_sft.jsonl \
  --swe-data datasets/processed/swe-gym-openhands-sft/train.jsonl \
  --output-dir datasets/processed/sft_v4_agent_mixture \
  --total-rows 96000 \
  --seed 20260640
```

### 质量要求

- 只取 assistant turn 作为 completion。
- completion 过长的样本过滤或截断。
- prompt 最长上下文需要截断到训练 seq_len 可承受范围。
- 训练数据不能包含 SWE-bench test patch、hidden tests 或官方答案。

## 来源四：通用能力回放数据

### 用途

通用能力回放用于防止后训练过拟合工具调用，导致：

- IFEval 下降。
- GSM8K/MMLU-Pro 下降。
- 普通对话变差。
- 代码/数学能力回退。

建议在 SFT/DPO/RL 阶段保留 10%-15% 通用回放。

### 当前来源

```text
datasets/sources/tulu-3-sft-mixture
datasets/sources/tulu-personas-if
datasets/sources/smol-smoltalk
datasets/sources/no-robots
```

### 原始样例：messages parquet row

```json
{
  "messages": [
    {"role": "user", "content": "Explain gradient clipping in simple terms."},
    {"role": "assistant", "content": "Gradient clipping limits the size of an update..."}
  ],
  "source": "tulu"
}
```

### 处理后样例：general replay SFT Row

```json
{
  "id": "general_replay_000001",
  "mixture_source": "broad_instruction",
  "prompt": "USER:\nExplain gradient clipping in simple terms.\n\nASSISTANT:\n",
  "completion": "Gradient clipping limits the size of an update...",
  "loss_weight": 0.5
}
```

### 处理脚本

```text
scripts/remote/prepare_sft_v3_mixture.py
scripts/remote/prepare_sft_v4_agent_mixture.py
```

### 质量要求

- 不使用 IFEval/GSM8K/MMLU-Pro test prompt 本身作为训练回放。
- 保留数据来源字段，例如 `tulu_if`、`no_robots`、`broad_instruction`。
- 回放比例不能过高，否则会稀释工具调用能力。
- 回放比例不能为 0，否则容易出现通用能力回退。

## 来源五：合成 No-Tool、Hard Constraints 与 RLVR 数据

### 用途

合成数据用于补足公开数据缺少的行为边界：

- no-tool：工具无关时输出空调用。
- hard constraints：严格格式、关键词、行数、JSON key 等。
- RLVR：可验证奖励，减少纯文本偏好判断。

### 原始样例：由脚本构造的 no-tool prompt

```json
{
  "prompt": "TOOLS:\n[{...calendar tools...}]\n\nUSER:\nExplain why backups should be tested regularly.\n\nNo provided tool is relevant. Return only an empty JSON array.\n\nASSISTANT:\n",
  "completion": "[]"
}
```

### 原始样例：hard constraint prompt

```json
{
  "prompt": "Return only a JSON object about incident response. It must have exactly the keys plan, risks, and checks. Each value must be an array of exactly two strings. Include the exact phrase 'verify first' in exactly one string and never use the word 'easy'.\n\nASSISTANT:\n",
  "completion": "{\"plan\":[\"Define scope for incident response\",\"verify first\"],\"risks\":[\"Unexpected behavior\",\"Incomplete rollback\"],\"checks\":[\"Run deterministic tests\",\"Review recorded evidence\"]}",
  "verifier": {
    "kind": "nested_json_constraints",
    "keys": ["plan", "risks", "checks"],
    "array_length": 2,
    "required_phrase": "verify first",
    "forbidden": "easy"
  }
}
```

### 处理后产物

`prepare_sft_v4_agent_mixture.py` 会输出：

```text
train_sft.jsonl
rl_verifiable_hard.jsonl
manifest.json
```

其中：

- `train_sft.jsonl` 用于 SFT。
- `rl_verifiable_hard.jsonl` 用于 RLVR/GRPO。
- `manifest.json` 记录配比、seed 和设计说明。

## 来源六：tau2-bench、AgentBench、AgentTuning

### 用途

这些资产主要用于环境任务、agent prompt 和评测，不默认直接进入训练。

位置示例：

```text
datasets/sources/tau2-bench
datasets/sources/AgentTuning
```

manifest 会记录：

```json
{
  "name": "tau2-bench",
  "kind": "environment_tasks_and_evaluation",
  "source": "https://github.com/sierra-research/tau2-bench",
  "source_revision": "<git_commit>",
  "task_files": [
    {"path": ".../tasks.json", "records": 100}
  ]
}
```

如果未来转训练数据，必须先：

- 去除官方测试集。
- 去除 gold trajectories。
- 按 domain/environment family 做 OOD split。
- 记录 user simulator、tool environment 和 evaluator revision。

## 来源七：公开评测集

### 用途

公开评测集只用于评测，不进入训练。

当前资产由 `scripts/prepare_eval_suite.py` 下载并记录 manifest。

| Benchmark | 路径 | 用途 | 训练 |
|---|---|---|---|
| IFEval | `datasets/eval_suite/huggingface/google__IFEval` | 指令遵循 | 否 |
| BFCL | `datasets/eval_suite/huggingface/gorilla-llm__Berkeley-Function-Calling-Leaderboard` | 函数调用 | 否 |
| SWE-bench Lite/Verified | `datasets/eval_suite/huggingface/SWE-bench__*` | 软件工程 | 否 |
| LiveCodeBench Lite | `datasets/eval_suite/huggingface/livecodebench__code_generation_lite` | 代码生成 | 否 |
| HumanEval | `datasets/eval_suite/huggingface/openai__openai_humaneval` | 代码生成 | 否 |
| MBPP | `datasets/eval_suite/huggingface/google-research-datasets__mbpp` | 代码生成 | 否 |
| GSM8K | `datasets/eval_suite/huggingface/openai__gsm8k` | 数学推理 | 否 |
| MMLU-Pro | `datasets/eval_suite/huggingface/TIGER-Lab__MMLU-Pro` | 知识推理 | 否 |

### IFEval 样例

```json
{
  "key": 1000,
  "prompt": "Write a 300+ word summary ... Include the keyword exactly twice.",
  "instruction_id_list": ["keywords:existence", "length_constraints:number_words"],
  "kwargs": [...]
}
```

### BFCL 样例

```json
{
  "id": "BFCL_v3_simple_0001",
  "question": [[{"role": "user", "content": "Book a flight from SFO to JFK."}]],
  "function": [
    {
      "name": "book_flight",
      "parameters": {
        "type": "object",
        "required": ["from", "to"],
        "properties": {
          "from": {"type": "string"},
          "to": {"type": "string"}
        }
      }
    }
  ]
}
```

### SWE-bench 样例

```json
{
  "instance_id": "django__django-xxxxx",
  "repo": "django/django",
  "base_commit": "...",
  "problem_statement": "...",
  "test_patch": "...",
  "FAIL_TO_PASS": [...],
  "PASS_TO_PASS": [...]
}
```

注意：`test_patch` 是评测判题资产，不是训练答案。

### 评测资产准备命令

```powershell
python scripts/prepare_eval_suite.py --download
```

### 质量要求

- eval manifest 必须保存 exact revision、commit、hash。
- 训练脚本不得读取 `datasets/eval_suite` 下的 test 文件。
- 内部 probe、官方复跑和厂商报告数字必须分别标记。
- SWE-bench 结果必须记录 agent scaffold、工具、token budget、retry、Docker image revision。

## 当前数据状态

### MCP LoRA SFT v3

```text
Root:
/workspace/yans2@xiaopeng.com/agentic_rl_pipeline/datasets/processed/mcp_lora_sft_v3_20260618

Train:
sft_v2_all/train_sft.jsonl

Validation:
sft_v2_all/validation_sft.jsonl

Preference:
raw_all/train_preferences.jsonl

RLVR:
raw_all/train_rlvr.jsonl
```

统计：

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

### Evaluation suite

```text
datasets/eval_suite/manifests/eval_suite_manifest.json
```

已包含：

- IFEval。
- BFCL 数据和 evaluator staging。
- SWE-bench Lite/Verified metadata。
- LiveCodeBench Lite。
- HumanEval。
- MBPP。
- GSM8K。
- MMLU-Pro。

## Manifest 要求

每个 processed 数据目录都必须有 `manifest.json` 或纳入全局 manifest，至少包含：

```json
{
  "format_version": 1,
  "source": "source path or URL",
  "source_revision": "commit or dataset revision",
  "license": "license name",
  "processing_script": "scripts/...",
  "processing_command": "...",
  "seed": 20260618,
  "counts": {
    "train": 10000,
    "validation": 500,
    "eval": 500
  },
  "splits": {
    "unit": "server_family | schema_fingerprint | tool_family | task_family",
    "contamination_clean": true
  },
  "hashes": {
    "train_sft.jsonl": "sha256..."
  }
}
```

## 质量门槛

- JSON/schema valid rate 不低于 99.5%。
- 同一 `schema_fingerprint`、`split_group_id`、tool family 或 task family 不得跨训练与 OOD 测试泄漏。
- clarify 不能输出 `[]`，必须输出显式追问动作。
- no-tool 样本比例不少于 8%，否则模型容易过度调用工具。
- 样本必须记录来源、license、处理脚本、token 长度和 verifier。
- 公开 benchmark test 数据只可进入 `datasets/eval_suite`，不得进入训练输入。
- 合成数据不能只看格式成功，必须增加执行或 hidden assertion 验证。

## 后续补强

当前 MCP 数据是 smoke 级闭环，不是生产规模数据。下一步需要：

- 扩展到至少 60 个 MCP Server、500 个工具、8 万条多轮轨迹。
- 使用教师模型生成多路径轨迹，并通过 schema、执行和最终状态三层校验。
- 加入真实 MCP Server 回放，减少模板化数据的模拟器偏差。
- 用当前学生模型挖掘失败样本，形成 hard negative、preference 和 RLVR 数据。
- 为每个来源补齐真实样本 hash、license 审核记录和 contamination report。
