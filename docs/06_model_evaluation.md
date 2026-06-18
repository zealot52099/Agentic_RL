# 6. 模型评测

本文档统一模型评测集、指标、测试集样例、脚本路径和结果标记方式。

## 结果标记

所有结果必须标记来源：

| 标签 | 含义 |
|---|---|
| `OUR-RERUN` | 本项目训练模型，用固定 runner 复测 |
| `OFFICIAL-RERUN` | 官方开源权重，用同一 runner 复测 |
| `VENDOR-REPORTED` | 官方 model card 或 leaderboard 公布数字 |
| `INTERNAL-PROBE` | 项目内部 probe，只能做开发回归，不能等同官方榜单 |

严格比较只允许 `OUR-RERUN` vs `OFFICIAL-RERUN`。`VENDOR-REPORTED` 只能作为市场目标。

## 评测矩阵

| 能力 | 主评测 | 开发期评测 | 当前资产 |
|---|---|---|---|
| 工具调用 | BFCL V4 | xLAM/internal MCP probe | BFCL v3 asset + official evaluator staging |
| MCP 多服务器路由 | LiveMCPBench、MCP-Atlas | internal MCP OOD | 需继续补齐 |
| 多轮 Agent | tau2-bench、AppWorld | internal multi-turn traces | tau2 repo 已准备 |
| 指令遵循 | IFEval | fixed format checks | 已准备 |
| 数学 | GSM8K | fixed 256 subset | 已准备 |
| 知识推理 | MMLU-Pro | fixed direct-logprob subset | 已准备 |
| SQL | WikiSQL、Spider/BIRD | fixed WikiSQL probe | WikiSQL 已有历史评测 |
| 代码 | HumanEval+、MBPP+、LiveCodeBench | syntax/unit-test probe | 已准备 |
| 软件工程 | SWE-bench Verified/Lite | Lite smoke subset | 数据已准备，运行依赖 Docker |
| 安全 | MCPSecBench + 内部攻击集 | prompt-injection probe | 待扩展 |

本地评测资产目录：

```text
datasets/eval_suite
datasets/eval_suite/manifests/eval_suite_manifest.json
```

## 测试集样例

### IFEval 样例

路径：

```text
datasets/eval_suite/huggingface/google__IFEval/ifeval_input_data.jsonl
```

样例形态：

```json
{
  "key": 1000,
  "prompt": "Write a 300+ word summary ... Include the keyword exactly twice.",
  "instruction_id_list": ["keywords:existence", "length_constraints:number_words"],
  "kwargs": [...]
}
```

核心指标：

- prompt-level strict accuracy。
- instruction-level strict accuracy。

### BFCL 样例

路径：

```text
datasets/eval_suite/huggingface/gorilla-llm__Berkeley-Function-Calling-Leaderboard
```

样例形态：

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

核心指标：

- AST/execution accuracy。
- parallel/multiple/irrelevance/category scores。
- JSON parse/schema valid。

### SWE-bench 样例

路径：

```text
datasets/eval_suite/huggingface/SWE-bench__SWE-bench_Lite/data/test-00000-of-00001.parquet
datasets/eval_suite/huggingface/SWE-bench__SWE-bench_Verified/data/test-00000-of-00001.parquet
```

样例字段：

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

核心指标：

- resolved rate。
- infra failure rate。
- 每个实例 patch、日志、镜像 revision。

注意：SWE-bench 测的是模型加 agent scaffold，不同 token budget、工具、重试次数不可直接比较。

### WikiSQL 样例

样例形态：

```json
{
  "question": "What is the population of France?",
  "table_id": "1-10015132-16",
  "table": {
    "header": ["Country", "Population"],
    "rows": [["France", "67000000"]]
  },
  "sql": {
    "sel": 1,
    "conds": [[0, "=", "France"]]
  }
}
```

核心指标：

- SQL extraction rate。
- SQLite execution rate。
- execution result exact match。
- normalized SQL exact match。

### Internal MCP OOD 样例

```json
{
  "id": "mcp_ood_0001",
  "prompt": "SYSTEM: ...\nMCP_SERVER_CATALOG:\n[...]\nUSER:\nFind unread high-priority emails and create a follow-up task.\nASSISTANT:",
  "verifier": {
    "kind": "mcp_tool_call",
    "expected_calls": [
      {"name": "email.search", "arguments": {"unread": true, "priority": "high"}},
      {"name": "tasks.create", "arguments": {"source": "email.search"}}
    ]
  },
  "split_group_id": "email_to_task"
}
```

核心指标：

- server top-1/top-3 accuracy。
- tool accuracy / macro-F1 / no-tool F1。
- JSON/schema valid rate。
- argument exact/semantic accuracy。
- execution success。
- end-to-end task success。
- recovery rate。
- hallucinated tool rate。
- redundant call rate。

## 评测脚本

| 脚本 | 作用 |
|---|---|
| `scripts/prepare_eval_suite.py` | 下载并记录评测资产 |
| `scripts/remote/evaluate_mcp_internal.py` | 内部 MCP OOD 评测 |
| `scripts/remote/evaluate_xlam_tool_calls.py` | xLAM/tool-call probe |
| `scripts/remote/generate_ifeval_responses.py` | 生成 IFEval 响应 |
| `scripts/remote/summarize_ifeval.py` | 汇总 IFEval 指标 |
| `scripts/remote/evaluate_general_regression.py` | GSM8K 等通用回归 |
| `scripts/remote/evaluate_mmlu_logprob.py` | MMLU-Pro direct-logprob |
| `scripts/remote/evaluate_wikisql.py` | WikiSQL 生成与执行评测 |
| `scripts/compare_xlam_evals.py` | 对比 xLAM 评测输出 |
| `scripts/compare_general_evals.py` | 对比通用评测输出 |
| `scripts/analyze_xlam_errors.py` | 工具调用错误分型 |

## 评测门槛

训练 run 晋级到更贵的官方评测前，必须满足：

- 目标内部 probe 至少提升 1 个百分点。
- IFEval、GSM8K、MMLU-Pro 回退不超过 1-2 个百分点。
- JSON/tool syntax valid rate 高于 99%。
- 小于 2 个百分点的提升至少两个 seed 复现。
- 明确区分内部 probe、官方复跑和厂商报告数字。

## 常见坑

| 问题 | 处理 |
|---|---|
| 内部 xLAM exact 与 BFCL 官方分数混比 | 必须分开标记 |
| BFCL v3/v4 数据和 evaluator 混用 | 记录 evaluator commit 和 dataset revision |
| 生成式评测随机性 | 至少 3 次，报告均值、标准差和置信区间 |
| SWE-bench 基础设施失败算模型失败 | 单独统计 infra failure |
| LiveCodeBench 持续更新 | 固定 revision 和日期切片 |
| benchmark 泄漏到训练 | 训练数据构建时做 contamination scan |
