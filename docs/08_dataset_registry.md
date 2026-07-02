# 8. 数据集台账与指标影响

本文档维护项目中所有训练集、验证集和评测集的统一台账。目标是让每一次训练都能回答五个问题：

1. 这批数据从哪里来，为什么加入。
2. 样本数量是多少，创建日期是什么。
3. 进入训练或评测前后的格式是什么。
4. 用了哪个脚本生成，完整样例长什么样。
5. 加入之后，对 SQL、tool-call、多轮和通用指标有什么影响。

原则：公开 benchmark 的 test/eval 数据只作为评测资产，不得进入训练。内部 probe、官方 benchmark、公开论文数字必须分开标记。

## 快速索引

| ID | 阶段 | 数据集/资产 | 类型 | 创建日期 | 数量 | 远端路径 | 处理脚本 | 指标影响 |
|---|---|---|---|---|---:|---|---|---|
| `mcp_lora_sft_v3` | MCP SFT v3 | MCP smoke / trace SFT v2 all | 训练+验证 | 2026-06-18 | train 19,494; val 506 | `datasets/processed/mcp_lora_sft_v3_20260618/sft_v2_all` | `scripts/remote/prepare_mcp_sft_v2.py` | 修复 clarify/no-tool 同为 `[]` 的标签冲突；用于稳定 MCP/tool-call LoRA SFT。需用内部 MCP/xLAM 指标继续量化。 |
| `xlam_fc_60k` | Tool-call 基础 | xLAM Function Calling 60K | 训练+heldout eval | 2026-06-09 起 | 约 60K 原始样本，split 后以 manifest 为准 | `datasets/processed/xlam-function-calling-60k` | `scripts/prepare_remote_datasets.py`; `scripts/prepare_xlam_splits.py` | 提升单轮函数调用格式、tool name 和参数 JSON；不是 MCP 多轮环境数据，不能替代 BFCL 官方分数。 |
| `swe_gym_openhands` | Agent/代码轨迹 | SWE-Gym OpenHands SFT Trajectories | 训练 | 2026-06-09 起 | 以 processed manifest 为准 | `datasets/processed/swe-gym-openhands-sft/train.jsonl` | `scripts/prepare_remote_datasets.py`; `scripts/remote/prepare_sft_v4_agent_mixture.py` | 用于终端/代码 agent 轨迹回放；当前未作为 SQL/tool-call 主指标来源。 |
| `general_replay` | 能力保持 | Tulu/SmolTalk/No-Robots 等通用回放 | 训练 | 2026-06-18 起 | 按 mixture 抽样 | `datasets/sources/tulu-*`; `datasets/sources/smol-*`; `datasets/sources/no-robots` | `scripts/remote/prepare_sft_v3_mixture.py`; `scripts/remote/prepare_sft_v4_agent_mixture.py` | 用于控制 IFEval/GSM8K/MMLU-Pro 回退；影响需按每个训练阶段的 general eval 报告更新。 |
| `phase5_unified` | Data Agent SFT | Unified Data Agent action schema | 训练+验证 | 2026-06-26 | all 15,500; train 14,962; val 538 | `datasets/processed/phase5_unified_20260626` | `scripts/build_phase5_unified.py` | 建立统一 action schema；早期 1.5B smoke 主要验证训练稳定性，非最终效果结论。 |
| `phase15_multiturn_v1` | 多轮 Data Agent | Executable multi-turn traces v1 | 训练+评测 | 2026-06-30 | train traces 2,400; eval 360; validation turns 1,062 | `datasets/processed/phase15_data_agent_multiturn_20260630` | `scripts/remote/prepare_data_agent_multiturn.py` | 引入多轮 tool observation、状态推进和任务成功率评测；内部 executable probe，非官方 benchmark。 |
| `phase15_eval_v2` | 多轮 Data Agent | Multi-turn eval v2 | 评测 | 2026-06-30 | eval 2,000 | `datasets/processed/phase15_data_agent_multiturn_eval_v2_20260630` | `scripts/remote/prepare_data_agent_multiturn.py` | 扩大多轮内部评测覆盖；用于检查 multi-turn success、约束满足和执行稳定性。 |
| `phase15_clean_v4` | 多轮 Data Agent | Cleaned multi-turn SFT/RL/eval v4 | 训练+RL+评测 | 2026-06-30 | mixture train 38,223; multiturn-only 26,223; RL 10,000; eval 2,000 | `datasets/processed/phase15_multiturn_clean_v4_20260630` | `scripts/remote/prepare_phase15_multiturn_sft_mixture.py`; `scripts/remote/clean_phase15_multiturn_data.py` | 修复 synthetic trace 重复问题；推荐用于后续多轮 SFT/retention。保留 Phase6 SQL/tool replay，降低单轮 SQL/tool 指标回退风险。 |
| `phase16_sql_repair` | SQL SFT/GRPO | SQL repair + Spider + SQL-context + replay | 训练+评测 | 2026-07-01 | repair SFT 4,137; repair eval 64; GRPO 4,088; mixture SFT 12,133; taxonomy 113 | `datasets/processed/phase16_sql_repair_20260701` | `scripts/remote/prepare_phase16_sql_repair_data.py` | Phase16a executable repair accuracy 达 81.25%；旧 normalized exact 0% 被确认为评测设计问题。WikiSQL execution 目标仍需继续优化。 |
| `phase16_followup_assets` | SQL follow-up | Phase16b/16c train and eval assets | 训练+评测 | 2026-07-01 | DPO train 4,967; DPO holdout 256; GRPO train 4,088; smoke 256; WikiSQL eval 256; multi-turn eval 500; tool probe 307 | `datasets/processed/phase16_followup_assets_20260701` | `scripts/remote/prepare_phase16_followup_assets.py` | 固定 Phase16 后续训练/评测口径；Phase16c 使用 executable WikiSQL GRPO reward。 |
| `phase17_sql_error_sft` | SQL error SFT | Corrected WikiSQL v2 + Phase16c 错例 SFT | 训练+评测 | 2026-07-02 | Phase17 rows 380; repeated 3,040; replay 5,000; total 8,040; eval 256 | `datasets/processed/phase17_sql_error_sft_20260702_123712_phase17b` | `scripts/remote/prepare_phase17_sql_eval_and_sft.py`; `scripts/remote/evaluate_wikisql_v2.py`; `scripts/remote/run_phase17_sql_error_sft_ppu16.sh` | 修正 WikiSQL 大小写和 prompt 字段误导问题；corrected Phase16c baseline 为 execution accuracy 50.78%、execution rate 95.31%。Phase17 训练后影响待补。 |
| `eval_suite_public` | 通用/官方评测 | IFEval/BFCL/SWE-bench/LiveCodeBench/HumanEval/MBPP/GSM8K/MMLU-Pro | 评测 | 2026-06-10 起 | 按各 benchmark 官方 split | `datasets/eval_suite` | `scripts/prepare_eval_suite.py` | 用于公开指标对齐和能力回归。不得训练；官方 scorer 未接入的指标只能标记为内部/待接入。 |
| `wikisql_internal_probe` | SQL 评测 | WikiSQL internal execution probe | 评测 | 2026-06-10 起；v2 2026-07-02 | 256 | `datasets/processed/phase16_followup_assets_20260701/wikisql_eval_256.*`; Phase17 normalized variant | `scripts/remote/evaluate_wikisql.py`; `scripts/remote/evaluate_wikisql_v2.py` | Phase8 SQL-only GRPO 曾达 62.11% execution accuracy；Phase9/10 mixed 约 55.86%；Phase16c corrected v2 baseline 50.78%。内部 probe，非官方 WikiSQL benchmark。 |
| `sql_repair_execution_eval` | SQL repair 评测 | Executable SQL repair probe | 评测 | 2026-07-01 | 128 | `datasets/processed/phase16_followup_assets_20260701/sql_repair_execution_eval/sql_repair_execution_eval_128.jsonl` | `scripts/remote/prepare_sql_repair_execution_eval.py`; `scripts/remote/evaluate_sql_repair_execution.py` | Phase16a execution repair accuracy 81.25%，normalized SQL exact 28.12%；替代旧 0% exact-only 误导指标。 |
| `data_agent_action_probe` | Tool/Data Agent 评测 | Data Agent JSON action/tool-call probe | 评测 | 2026-07-01 | 307 | `datasets/processed/phase16_followup_assets_20260701/data_agent_tool_action_probe.jsonl` | `scripts/remote/prepare_phase16_followup_assets.py` | 用于 JSON/action/tool name/args 回归，避免 SQL-only 训练损伤 tool-call。影响按 post-eval 更新。 |
| `mcp_xlam_array_smoke` | Tool smoke | MCP array-format smoke probe | 评测 | 2026-07-01 | 5 | `datasets/processed/phase16_followup_assets_20260701/mcp_xlam_array_tool_probe.jsonl` | `scripts/remote/prepare_phase16_followup_assets.py` | 仅 smoke test，样本太少，不能作为 headline metric。 |

## 指标影响摘要

| 数据/阶段 | 加入前参考 | 加入后观察 | 结论 |
|---|---|---|---|
| Phase5 unified action schema | 各数据源输出格式不统一 | 统一为 `tool_call/no_tool/clarify` action；训练链路先跑通 | 格式统一是必要基础，但单靠 Phase5 不足以保证 SQL execution 提升。 |
| Phase8 SQL-only GRPO | Phase5 之后 SQL execution 仍偏低 | WikiSQL internal 256 probe: extraction 100%, execution rate 88.67%, execution accuracy 62.11% | SQL-only executable reward 对 WikiSQL 有收益，但容易牺牲 tool/general retention，需要后续混合保持。 |
| Phase9 mixed SQL+tool GRPO | Phase8 SQL-only 62.11% | WikiSQL execution accuracy 55.86%, execution rate 80.08% | 混合训练保留 tool-call，但 SQL reward 被稀释。 |
| Phase10 staged SQL->mixed GRPO | Phase9 55.86% | WikiSQL execution accuracy 55.86%, execution rate 79.69% | staged retention 没有恢复 SQL-only 峰值，说明需要更强 schema/value grounding 和错误修复数据。 |
| Phase16a SQL repair SFT | legacy repair normalized exact 0% | executable repair accuracy 81.25%; normalized exact 28.12% | 旧 repair exact 评测不合理；带 schema/table/error feedback 的 execution-based repair 能真实反映修复能力。 |
| Phase17 corrected WikiSQL v2 | old probe 存在大小写/字段名误导 | Phase16c corrected baseline: execution accuracy 50.78%, execution rate 95.31% | 评测口径修正后，当前主要错误集中在 where/value/column 与 missing aggregation；Phase17b 训练影响待训练后补。 |

## 完整数据样例

以下样例用于说明每个数据资产进入训练/评测时的消费格式。实际文件可能包含更多 metadata；新增数据必须在本节补一个完整单条样例。

### `mcp_lora_sft_v3` SFT 样例

```json
{
  "id": "mcp_smoke_calendar_000001",
  "mixture_source": "mcp_positive",
  "prompt": "SYSTEM: Return [] when no tool applies. When required information is missing, return one JSON object with action=\"clarify\".\nMCP_SERVER_CATALOG:\n[{\"server\":\"calendar\",\"tools\":[{\"name\":\"calendar.create_event\",\"description\":\"Create a calendar event\",\"parameters\":{\"type\":\"object\",\"required\":[\"title\",\"date\"],\"properties\":{\"title\":{\"type\":\"string\"},\"date\":{\"type\":\"string\"}}}}]}]\nUSER:\nSchedule project review for Friday.\nASSISTANT:",
  "completion": "[{\"action\":\"tool_call\",\"server\":\"calendar\",\"tool\":\"calendar.create_event\",\"arguments\":{\"title\":\"project review\",\"date\":\"Friday\"}}]",
  "loss_weight": 1.0,
  "metadata": {
    "source": "mcp_smoke",
    "schema_fingerprint": "calendar.create_event:v1",
    "split": "train"
  }
}
```

### `xlam_fc_60k` SFT 样例

```json
{
  "id": "xlam_train_000001",
  "prompt_template_version": "xlam_tool_json_v1",
  "prompt": "You are a tool-calling assistant. Select only tools from the provided definitions. Return only a JSON array.\n\nTOOLS:\n[{\"name\":\"get_weather\",\"description\":\"Get weather by city\",\"parameters\":{\"type\":\"object\",\"required\":[\"city\"],\"properties\":{\"city\":{\"type\":\"string\"}}}}]\n\nUSER:\nWhat is the weather in Paris?\n\nASSISTANT:\n",
  "completion": "[{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}]",
  "expected_calls": [
    {
      "name": "get_weather",
      "arguments": {
        "city": "Paris"
      }
    }
  ],
  "metadata": {
    "source": "xlam-function-calling-60k",
    "split_group_id": "get_weather",
    "split": "train"
  }
}
```

### `phase5_unified` Data Agent SFT 样例

```json
{
  "id": "phase5_spider_000001",
  "source": "spider",
  "schema": "data_agent_action_v1",
  "messages": [
    {
      "role": "system",
      "content": "You are a Data Agent. Return exactly one JSON action."
    },
    {
      "role": "user",
      "content": "For database spider:concert_singer, find the names of singers older than 30."
    }
  ],
  "completion": "{\"action\":\"tool_call\",\"calls\":[{\"name\":\"execute_sql\",\"arguments\":{\"database\":\"spider:concert_singer\",\"sql\":\"SELECT name FROM singer WHERE age > 30\"}}]}",
  "metadata": {
    "action": "tool_call",
    "task_family": "sql_generation",
    "split": "train"
  }
}
```

### `phase15_clean_v4` Multi-turn SFT 样例

```json
{
  "id": "phase15_mt_clean_000001_turn03",
  "task_id": "phase15_mt_clean_000001",
  "prompt": "SYSTEM: You are a Data Agent. Use JSON actions only.\nTOOLS:\n- execute_sql(database, sql)\n- inspect_schema(database)\n- final_answer(answer)\n\nUSER:\nFind active enterprise customers with unpaid invoices and summarize the total amount.\nASSISTANT:{\"action\":\"tool_call\",\"calls\":[{\"name\":\"inspect_schema\",\"arguments\":{\"database\":\"sales_demo\"}}]}\nTOOL:\n{\"tables\":{\"customers\":[\"id\",\"name\",\"segment\",\"status\"],\"invoices\":[\"customer_id\",\"amount\",\"paid\"]}}\nASSISTANT:",
  "completion": "{\"action\":\"tool_call\",\"calls\":[{\"name\":\"execute_sql\",\"arguments\":{\"database\":\"sales_demo\",\"sql\":\"SELECT c.name, SUM(i.amount) AS unpaid_amount FROM customers c JOIN invoices i ON c.id = i.customer_id WHERE c.segment = 'enterprise' AND c.status = 'active' AND i.paid = 0 GROUP BY c.name\"}}]}",
  "metadata": {
    "source": "phase15_multiturn_clean_v4",
    "turn_index": 3,
    "split": "train",
    "verifier": "sqlite_state"
  }
}
```

### `phase16_sql_repair` SFT 样例

```json
{
  "id": "phase16_repair_real_failure_000001",
  "source": "phase10_wikisql_failed_prediction",
  "prompt": "You are a SQL repair assistant.\nQuestion: What is the number of players from Canada?\nTable schema: table(col0 TEXT, col1 TEXT, col2 TEXT)\nHeaders: col0=Player, col1=Country, col2=Score\nPrevious SQL: SELECT col0 FROM table WHERE col1 = 'Canada'\nExecution feedback: result does not answer the question; aggregation is missing.\nReturn only the corrected SQL.\n",
  "completion": "SELECT COUNT(col0) FROM table WHERE col1 = 'Canada'",
  "loss_weight": 1.5,
  "metadata": {
    "failure_type": "wrong_missing_aggregation",
    "split": "train"
  }
}
```

### `phase16c_grpo_train` RLVR 样例

```json
{
  "id": "phase16c_wikisql_grpo_000001",
  "query": "Generate SQLite SQL for the question. Use only physical columns col0, col1, ...\nQuestion: What is the total attendance where team is Boston?\nTable schema: table(col0 TEXT, col1 TEXT, col2 REAL)\nHeaders: col0=Team, col1=City, col2=Attendance\nSample rows: [[\"Boston\", \"Boston\", 12000], [\"Chicago\", \"Chicago\", 9000]]\nReturn only SQL.",
  "solution": "SELECT SUM(col2) FROM table WHERE col0 = 'Boston'",
  "task_type": "wikisql_exec",
  "verifier": {
    "kind": "sqlite_execution",
    "database": "datasets/processed/phase16_followup_assets_20260701/wikisql_eval_256.sqlite",
    "gold_result": [[12000]]
  },
  "metadata": {
    "source": "wikisql",
    "split": "train"
  }
}
```

### `phase17_sql_error_sft` 样例

```json
{
  "id": "phase17_wrong_where_or_value_000001",
  "source": "phase16c_corrected_wikisql_v2_error",
  "prompt": "Generate SQLite SQL. Use only physical SQL columns named col0, col1, ... Header names are descriptions only, not SQL identifiers.\nQuestion: What is the height of Hato Mayor?\nTable schema: table(col0 TEXT, col1 TEXT, col2 TEXT, col3 TEXT, col4 TEXT)\nHeaders: col0=Municipality, col1=Province, col2=Population, col3=Area, col4=Height\nSample rows: [[\"hato mayor\", \"hato mayor\", \"70000\", \"1200\", \"20\"]]\nPrevious model SQL: SELECT col4 FROM table WHERE col4 = 'hato mayor'\nError category: wrong_where_or_value_or_column\nReturn only the corrected SQL.\n",
  "completion": "SELECT col4 FROM table WHERE col0 = 'hato mayor'",
  "loss_weight": 1.5,
  "metadata": {
    "corrected_eval": "wikisql_v2_casefolded",
    "split": "train"
  }
}
```

### `wikisql_internal_probe` 评测样例

```json
{
  "id": "wikisql_eval_000001",
  "question": "What is the number of teams from Boston?",
  "table_id": "wikisql_000001",
  "sqlite_table": "table_000001",
  "header": ["Team", "City", "Wins"],
  "types": ["text", "text", "real"],
  "physical_columns": ["col0", "col1", "col2"],
  "sample_rows": [["boston", "boston", 10], ["chicago", "chicago", 7]],
  "gold_sql": "SELECT COUNT(col0) FROM table_000001 WHERE col1 = 'boston'",
  "gold_result": [[1]],
  "metadata": {
    "benchmark": "WikiSQL-derived internal probe",
    "scoring": "read-only SQLite execution; result rows compared order-insensitively",
    "split": "eval"
  }
}
```

### `sql_repair_execution_eval` 样例

```json
{
  "id": "sql_repair_exec_000001",
  "question": "What is the average score for Canada?",
  "schema": "table(col0 TEXT, col1 TEXT, col2 REAL)",
  "headers": {
    "col0": "Player",
    "col1": "Country",
    "col2": "Score"
  },
  "previous_sql": "SELECT COUNT(col2) FROM table WHERE col1 = 'Canada'",
  "execution_feedback": "SQL executed but result does not match expected result.",
  "expected_result": [[8.5]],
  "gold_sql": "SELECT AVG(col2) FROM table WHERE col1 = 'Canada'",
  "verifier": {
    "kind": "sqlite_execution",
    "database": "datasets/processed/phase16_followup_assets_20260701/wikisql_eval_256.sqlite"
  }
}
```

### `data_agent_action_probe` 评测样例

```json
{
  "id": "data_agent_action_probe_000001",
  "prompt": "SYSTEM: You are a Data Agent. Return exactly one JSON action.\nAVAILABLE_TOOLS:\n[{\"name\":\"inspect_schema\"},{\"name\":\"execute_sql\"},{\"name\":\"final_answer\"}]\nUSER:\nList unpaid invoice totals by customer. The schema is already known.\nASSISTANT:",
  "expected_action": {
    "action": "tool_call",
    "calls": [
      {
        "name": "execute_sql",
        "arguments": {
          "database": "data_agent_eval",
          "sql": "SELECT customer_id, SUM(amount) FROM invoices WHERE paid = 0 GROUP BY customer_id"
        }
      }
    ]
  },
  "metrics": [
    "json_valid",
    "action_exact",
    "tool_name_exact",
    "arguments_semantic"
  ]
}
```

### `eval_suite_public` IFEval 样例

```json
{
  "key": 1001,
  "prompt": "Write a 100-word paragraph about renewable energy. Include the word solar at least twice.",
  "instruction_id_list": [
    "length_constraints:number_words",
    "keywords:frequency"
  ],
  "kwargs": [
    {
      "num_words": 100
    },
    {
      "keyword": "solar",
      "frequency": 2
    }
  ]
}
```

### `eval_suite_public` BFCL 样例

```json
{
  "id": "BFCL_v3_simple_0001",
  "question": [
    [
      {
        "role": "user",
        "content": "Book a flight from SFO to JFK."
      }
    ]
  ],
  "function": [
    {
      "name": "book_flight",
      "parameters": {
        "type": "object",
        "required": ["from", "to"],
        "properties": {
          "from": {
            "type": "string"
          },
          "to": {
            "type": "string"
          }
        }
      }
    }
  ]
}
```

### `eval_suite_public` GSM8K 样例

```json
{
  "question": "Janet has 3 apples and buys 5 more. How many apples does she have?",
  "answer": "Janet has 3 + 5 = 8 apples. #### 8",
  "metadata": {
    "benchmark": "GSM8K",
    "split": "test",
    "train_use": false
  }
}
```

### `eval_suite_public` MMLU-Pro 样例

```json
{
  "question": "Which of the following best explains ...?",
  "options": ["A ...", "B ...", "C ...", "D ..."],
  "answer": "C",
  "category": "computer science",
  "metadata": {
    "benchmark": "MMLU-Pro",
    "scoring": "direct-logprob preferred; generation is auxiliary",
    "train_use": false
  }
}
```

## 维护规则

新增或修改任何数据集时，必须同步更新：

1. 快速索引表：ID、类型、创建日期、数量、路径、脚本、指标影响。
2. 完整数据样例：至少一条可读 JSON，字段不能只写省略号。
3. 指标影响摘要：如果尚未评测，写 `待补` 和预期评测路径；不要填猜测数字。
4. 污染边界：训练集和评测集必须标明是否可训练。
5. 关联实验记录：在 `docs/07_experiment_records.md` 中记录本批数据被哪个 Phase 使用。

建议每个 processed 数据目录都保留 `manifest.json`，至少包含：

```json
{
  "dataset_id": "phase17_sql_error_sft",
  "created_at": "2026-07-02",
  "source": ["wikisql", "phase16c_predictions", "base_replay"],
  "processing_script": "scripts/remote/prepare_phase17_sql_eval_and_sft.py",
  "counts": {
    "train": 8040,
    "eval": 256
  },
  "train_use": true,
  "eval_use": true,
  "contamination_policy": "public benchmark test answers excluded from training; internal probe split fixed",
  "known_effect": "baseline corrected WikiSQL v2 execution accuracy 50.78%; post-train result pending"
}
```
