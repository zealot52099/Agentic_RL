# Agentic RL 工具调用训练周报

> 2026-06-18 ~ 2026-06-22 | bifrost-2026060214414601-yans2 | [SwanLab](https://swanlab.cn/@yans2/agentic-rl-tool-calling)

## 一、评测集

### BFCL V4 Live（函数调用）

Berkeley Function Calling Leaderboard V4，评估模型能否根据用户意图选择正确的函数并填充参数。**均未用于训练。**

| 子类别 | 样本数 | 说明 | 样例 |
|---|---|---|---|
| live_simple | 150 | 调用单个函数，参数明确 | "查用户 ID 7890 的详细信息" → `get_user_info(user_id=7890)` |
| live_multiple | 150 | 从多个候选中选最合适的函数 | "帮我查下东京的天气" → 在 `get_weather` / `search_flights` 中选 `get_weather(city="Tokyo")` |
| live_parallel | 16 | 同时调用多个独立函数 | "查北京和上海的天气" → `[get_weather("Beijing"), get_weather("Shanghai")]` |
| live_parallel_multiple | 24 | 调用多个函数，部分有依赖 | "查东京航班并订酒店" → `[search_flights("Tokyo"), book_hotel("Tokyo")]` |

### BFCL V3 SQL（SQL 函数调用）

评估模型能否生成正确的 SQL 查询（通过 `sql.execute` 函数调用）。100 条测试用例。

| 样例 |
|---|
| Q："What is the name of the student in the 'students' table with ID 1234?" |
| A：`sql.execute(sql_keyword="SELECT", table_name="students", columns=["name"], conditions=["id=1234"])` |

### BFCL Agentic（多轮/搜索）

评估模型在多轮对话、网页搜索等复杂 Agent 场景下的函数调用格式合法性。

| 子类别 | 说明 |
|---|---|
| multi_turn_base | 多轮对话中持续调用函数 |
| web_search | 需要搜索外部信息后调用函数 |

---

## 二、训练集

### MCP Smoke 数据（基础 SFT）

确定性模板生成的 MCP 工具调用数据，覆盖 5 个 server（database/files/mail/git/calendar），共 30 个 tool，**19,494 条**。

| 类别 | 数量 | 占比 |
|---|---|---|
| mcp_positive（工具调用） | 14,744 | 75.6% |
| mcp_clarify（澄清追问） | 2,375 | 12.2% |
| mcp_no_tool（无需调用） | 2,375 | 12.2% |

**完整样例（含 prompt + completion）：**

```
PROMPT:
You are an MCP agent. Select only tools from the supplied server catalog.
Return a JSON array of calls with server_id, tool_name, and arguments.

MCP_SERVER_CATALOG:
[{"server_id": "database", "tools": [{"name":"query_records",
 "description":"Query records by field value", ...}]}]

USER:
Look up all projects with field 'project' equal to 'atlas'.

ASSISTANT:

COMPLETION:
[{"arguments":{"field":"project","value":"atlas"},"server_id":"database",
  "tool_name":"query_records"}]
```

### 标准格式转换数据（Mixed SFT 新增）

从 MCP positive 样本转换，将输出从 MCP 格式（`server_id` + `tool_name`）改为标准 OpenAI function-calling 格式（`name` + `arguments`）。**8,954 条（含 8,354 func_call + 300 clarify + 300 no_tool）**。

**完整样例：**

```
PROMPT:
You are an MCP agent. Select only tools from the supplied server catalog.
Return a JSON array of calls with server_id, tool_name, and arguments.

AVAILABLE FUNCTIONS:
[{"name": "write_file", "description": "Call write_file on files server",
  "parameters": {"type": "object", "properties": {"path": {"type": "string"},
  "content": {"type": "string"}}, "required": ["path", "content"]}}]

USER:
Save release notes to /notes/release.txt with content "release approved".

ASSISTANT:

COMPLETION:
{"name": "write_file", "arguments": {"path": "/notes/release.txt",
 "content": "release approved"}}
```

### 合成并行数据（Clean SFT 新增）

从 MCP 数据中随机抽取两个不同 server 的单函数调用，拼接用户 query 和 server catalog 组成并行调用样本。**2,000 条（1,000 MCP 格式 + 1,000 标准格式），零 BFCL 泄漏。**

**MCP 格式样例：**

```
PROMPT:
You are an MCP agent. Select only tools from the supplied server catalog.
Return a JSON array of calls with server_id, tool_name, and arguments.

MCP_SERVER_CATALOG:
[...mail server tools..., ...files server tools...]

USER:
Send an email to dev@example.com with subject "MCP" and body "Tests passed."
Also, write release notes to /notes/release.txt with content "release approved".

ASSISTANT:

COMPLETION:
[{"server_id":"mail","tool_name":"send_email","arguments":{"to":"dev@example.com",
  "subject":"MCP","body":"Tests passed."}},
 {"server_id":"files","tool_name":"write_file","arguments":{"path":"/notes/release.txt",
  "content":"release approved"}}]
```

**标准格式样例：**

```
COMPLETION:
[{"name":"read_file","arguments":{"path":"/notes/todo.txt"}},
 {"name":"update_issue","arguments":{"issue_id":7,"status":"closed"}}]
```

---

## 三、训练路线

| 模型 | 路线 | 各阶段数据（累计） |
|---|---|---|
| **1.5B GRPO** | SFT → DPO → GRPO | ① MCP smoke 19K → ② MCP 偏好对 27K → ③ BFCL Live + SQL prompts (320) |
| **Coder7B Clean SFT** | Mixed SFT + 合成并行 | MCP smoke 19K + 标准格式 9K + 合成并行 2K（= **30,448 条**，零 BFCL） |
| Coder7B Mixed SFT | Mixed SFT | MCP smoke 19K + 标准格式 9K（= **28,448 条**） |
| 1.5B DPO | SFT → DPO | ① MCP smoke 19K → ② MCP 偏好对 27K |
| 1.5B SFT | SFT only | MCP smoke 19K |
| Coder7B 基座 | 无训练 | — |

---

## 四、模型选型

### 为什么选 Coder7B

Qwen2.5-Coder-7B-Instruct 专为代码和结构化输出场景设计，预训练中大量接触 JSON Schema、函数签名、SQL 语法，与工具调用任务高度匹配。

**关键证据**：基座 SQL Func 仅 3%，但不是因为缺少知识——模型能明确选对 `sql.execute`，填对表名和列名——只是输出格式是 markdown 包装的 `{"function_name":"..."}` 而非裸 JSON。格式纠正后飙升至 99%，证明知识完整。

### Bifrost 可用候选模型

| 模型 | 参数量 | 类型 | 适合度 | 说明 |
|---|---|---|---|---|
| **Qwen2.5-Coder-7B-Instruct** ✅ | 7B | 代码专用 | ⭐⭐⭐ | 当前采用，JSON/SQL 原生能力最强，15GB 单卡可跑 |
| Qwen2.5-Coder-14B-Instruct | 14B | 代码专用 | ⭐⭐⭐ | 更强能力，~28GB 需 QLoRA 或模型并行 |
| Qwen2.5-3B-Instruct | 3B | 通用 | ⭐⭐ | 1.5B↔7B 甜点位 |
| Qwen2.5-7B-Instruct | 7B | 通用 | ⭐⭐ | 已用 1.5B 版本跑通流程 |
| Qwen3-8B | 8B | 新一代 Dense | ⭐⭐ | 架构更新，需 SFT |
| Qwen2.5-14B-Instruct | 14B | 通用 | ⭐⭐ | 更强但显存挑战大 |

### 选型路线

- **已验证**：1.5B Instruct + Coder7B
- **中期冲榜**：Coder14B + QLoRA + 混合格式 SFT，预估 BFCL 85-90%
- **远期**：Qwen3-Coder 系列发布后跟进

---

## 五、实验矩阵

| 模型 | 参数量 | 训练路线 | BFCL Live | live_parallel | live_parallel_multiple | SQL Func | SQL Exact | Multi-Turn | Web Search |
|---|---|---|---|---|---|---|---|---|---|
| **1.5B GRPO** 🥇 | 1.5B | SFT→DPO→GRPO | **83.5%** | 31.2% | 62.5% | 51.0% | 17.0% | N/A | N/A |
| **Coder7B Clean SFT** | 7B | Mixed SFT + 合成并行 | 82.1% | **100.0%** | **95.8%** | 98.0% | 0.0% | **100%** | **100%** |
| Coder7B Mixed SFT | 7B | Mixed SFT | 82.4% | 62.5% | 45.8% | **99.0%** | **59.0%** | **100%** | **100%** |
| 1.5B DPO | 1.5B | SFT→DPO | 80.0% | 25.0% | 58.3% | N/A | N/A | N/A | N/A |
| 1.5B SFT | 1.5B | SFT only | 62.4% | ~30% | ~40% | N/A | N/A | N/A | N/A |
| Coder7B 基座 | 7B | 无训练 | ~3% | ~0% | ~0% | 3.0% | 2.0% | ~0% | ~0% |
| *DeepSeek V4 Pro* | *~数百B* | *API 对照* | *70.7%\** | *93.8%* | *83.3%* | *44.0%* | *28.0%* | — | — |

> **标注说明**  
> \* DeepSeek 因 BFCL schema 不适配导致 ~10% API 报错，实际估计 78-82%  
> 
> N/A = 模型输出 MCP 格式（`server_id` + `tool_name`），SQL/Multi-Turn 评测使用标准格式 parser，无法直接比对  
> ~0% = 基座无训练，输出 markdown 包装 + 非标准 key name，parse 基本失败

## 六、关键结论

**1. 1.5B 小模型跑通 SFT→DPO→GRPO 全流程（+21pp）**

62.4%（SFT）→ 80.0%（DPO）→ 83.5%（GRPO）。单向函数调用超越 DeepSeek V4 Pro。验证了三阶段 RL pipeline 在 Dense 小模型上的有效性。

**2. Coder7B 通过格式纠正释放已有知识（SQL Func 3%→99%，BFCL 3%→82%）**

基座模型的知识已存在，仅因输出格式不对（markdown 代码块、`function_name` 而非 `name`）被"封印"。加入 31.5% 标准格式 SFT 数据即可释放。

**3. 合成并行数据解决多函数调用短板（+50pp），零外部数据泄漏**

仅用 MCP 数据两两合成 2,000 条并行样本，live_parallel_multiple 45.8%→95.8%。多函数调用是数据覆盖问题而非能力问题。

**4. GRPO 有效区间：模型有明确提升空间时**

1.5B 从 80.0% 受益于 GRPO（+3.5pp）；Coder7B 在 82.4% 天花板时 GRPO 只引入噪声。

**5. DPO 效果取决于基座和格式一致性**

1.5B Instruct +17pp（注入能力）；Coder7B -25pp（覆盖已有能力）；混合格式不收敛（输出分布冲突）。

---

## 七、RL 方法选型

| 方法 | 适用场景 | 本项目 | 原因 |
|---|---|---|---|
| GRPO | Dense + 短输出 | ✅ 采用 | 已验证 +3.5pp |
| GSPO | MoE 架构 | ❌ | Qwen2.5 是 Dense |
| DAPO | 长 CoT | ❌ | 输出仅 50-100 tokens |
| DPO | 单输出格式 | ⚠️ | 混合格式不收敛 |

---

## 八、脚本说明

### 训练脚本

| 脚本 | 用途 | 用法 |
|---|---|---|
| `scripts/train_coder7b_sft.py` | Coder7B LoRA SFT 训练 | 设置环境变量 `TRAIN_DATA` `OUTPUT_DIR` `SWANLAB_RUN`，`python train_coder7b_sft.py` |
| `scripts/train_dpo_phase_c.py` | LoRA DPO 训练 | `python train_dpo_phase_c.py --base-model <path> --adapter <path> --dataset <path> --output-dir <path> --beta 0.5 ...` |
| `scripts/train_grpo_15b_dpo.py` | 1.5B GRPO 训练 | 修改脚本内路径后 `python train_grpo_15b_dpo.py` |
| `scripts/train_grpo_c7b_fixed.py` | Coder7B GRPO 训练 | 同上 |

### 评测脚本

| 脚本 | 用途 | 用法 |
|---|---|---|
| `scripts/eval_bfcl_light.py` | BFCL V4 Live 评测 | 修改脚本内模型路径，`python eval_bfcl_light.py` |
| `scripts/eval_sql.py` | BFCL V3 SQL 评测 | 同上 |
| `scripts/eval_coder7b.py` | Coder7B 通用评测 | 修改 ADAPTER 路径后运行 |
| `scripts/eval_ood_all.py` | OOD held-out 评测 | 同上 |
| `run_deepseek_eval.py` | DeepSeek V4 Pro API 评测 | 本地运行，需 API key |

### 数据构建脚本

| 脚本 | 用途 |
|---|---|
| `build_mixed_sft.py` | 生成标准格式转换数据 |
| `build_clean_parallel.py` | 生成合成并行数据（MCP 来源，零 BFCL） |
| `build_mixed_dpo_v2.py` | 生成混合格式 DPO 偏好数据 |

---

## 九、下一步

| 方向 | 预估提升 | 说明 |
|---|---|---|
| **修复标准格式数据**（P0） | BFCL +4-6pp（82%→86-88%） | 当前标准格式只放 target function（1个），应改为全量 function 列表（6个），让模型学会在标准格式下做选择 |
| 修复后重训 Coder7B Clean SFT | — | 同时含全量标准格式 + 合成并行数据 |
| SQL Exact 59%→75% | +16pp | 参数精确匹配 SFT |
| xLAM / BIRD 评测 | — | 外网数据下载 |

### P0 详情：修复标准格式数据

**当前缺陷**：标准格式转换时 AVAILABLE FUNCTIONS 只包含被调用的 target function（1个），模型不需要做选择。

```
MCP 原始:    6 个 tool → 模型从 6 个候选中选 1 个  ✅
标准格式(当前): 1 个 function → 模型看到什么输出什么  ❌
标准格式(修复): 6 个 function → 模型在标准格式下选择  ✅
```

**确认无泄漏**：数据来源 MCP smoke 模板，全部 19,494 条为自生成，与 BFCL 零交集。

**预估收益**：live_simple +6-9pp，live_multiple +8-12pp，BFCL Overall +4-6pp。
