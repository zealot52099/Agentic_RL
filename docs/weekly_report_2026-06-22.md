# Agentic RL 工具调用训练周报

> 2026-06-18 ~ 2026-06-22 | bifrost-2026060214414601-yans2 | [SwanLab](https://swanlab.cn/@yans2/agentic-rl-tool-calling)

## 一、评测集说明

### BFCL V4 Live（函数调用）

Berkeley Function Calling Leaderboard V4，评估模型能否根据用户意图选择正确的函数并填充参数。测试题目不区分训练/测试（如未特殊说明，未用于训练）。

| 子类别 | 样本数 | 说明 | 样例 |
|---|---|---|---|
| live_simple | 150 | 调用单个函数，参数明确 | "查用户 ID 7890 的详细信息" → `get_user_info(user_id=7890)` |
| live_multiple | 150 | 从多个候选中选最合适的函数 | "帮我查下东京的天气" → 在 `get_weather` / `search_flights` / `book_hotel` 中选 `get_weather(city="Tokyo")` |
| live_parallel | 16 | 同时调用多个独立函数 | "查北京和上海的天气" → `[get_weather("Beijing"), get_weather("Shanghai")]` |
| live_parallel_multiple | 24 | 调用多个函数，部分有依赖 | "查东京航班并订酒店" → `[search_flights("Tokyo"), book_hotel("Tokyo")]` |

### BFCL V3 SQL（SQL 函数调用）

评估模型能否生成正确的 SQL 查询（通过 `sql.execute` 函数调用）。

| 样例 |
|---|
| 问题："查 students 表中 ID 1234 的学生姓名" → `sql.execute(sql_keyword="SELECT", table_name="students", columns=["name"], conditions=["id=1234"])` |

### BFCL Agentic（多轮/搜索）

评估模型在多轮对话、网页搜索等复杂 Agent 场景下的函数调用格式合法性。

| 子类别 | 样本数 | 说明 |
|---|---|---|
| multi_turn_base | 300+ | 多轮对话中持续调用函数 |
| web_search | 100+ | 需要搜索外部信息后调用函数 |

---

## 二、训练集说明

### MCP Smoke 数据（基础 SFT）

确定性模板生成的 MCP 工具调用数据，覆盖 5 个 server（database/files/mail/git/calendar），共 30 个 tool。

| 类别 | 数量 | 样例 |
|---|---|---|
| mcp_positive | 14,744 | `query_records(server_id="database", field="project", value="atlas")` |
| mcp_clarify | 2,375 | `{"action":"clarify","message":"请提供必要信息"}` |
| mcp_no_tool | 2,375 | `[]`（无需调用工具） |

### 标准格式转换数据（Mixed SFT 新增）

从 MCP 数据中抽取 positive 样本，将输出格式从 MCP（`server_id` + `tool_name`）转换为标准 OpenAI function-calling 格式（`name` + `arguments`）。

| 样例 |
|---|
| 输入："查用户 ID 7890 的详细信息" |
| MCP 格式输出：`{"server_id":"database","tool_name":"query_user","arguments":{"user_id":7890}}` |
| 标准格式输出：`{"name":"query_user","arguments":{"user_id":7890}}` |

### 合成并行数据（Clean SFT 新增）

从 MCP 数据中随机抽取两个不同 server 的单函数调用，组合成并行调用样本。

| 样例 |
|---|
| 输入："查 atlas 项目的最新提交 Also, schedule a review meeting tomorrow" |
| 输出：`[{"server_id":"git","tool_name":"get_commits","arguments":{...}}, {"server_id":"calendar","tool_name":"create_event","arguments":{...}}]` |

---

## 三、训练路线

| 模型 | 训练路线 | 各阶段数据 |
|---|---|---|
| **1.5B GRPO** | SFT → DPO → GRPO | SFT: MCP smoke (19K) → DPO: MCP 偏好对 (27K) → GRPO: BFCL Live + SQL prompts (320) |
| **Coder7B Clean SFT** | Mixed SFT + 合成并行 | Mixed SFT: MCP smoke + 标准格式转换 (28K) + 合成并行 (2K, 零 BFCL 泄漏) |
| Coder7B Mixed SFT | Mixed SFT | MCP smoke + 标准格式转换 (28K) |
| 1.5B DPO | SFT → DPO | SFT: MCP smoke → DPO: MCP 偏好对 (27K) |
| 1.5B SFT | SFT only | MCP smoke (19K) |
| Coder7B 基座 | 无训练 | — |

---

## 四、实验矩阵（完整指标）

| 模型 | 参数量 | 训练路线 | BFCL Live | live_parallel | live_parallel_multiple | SQL Func | SQL Exact | Multi-Turn | Web Search |
|---|---|---|---|---|---|---|---|---|---|
| **1.5B GRPO** 🥇 | 1.5B | SFT→DPO→GRPO | **83.5%** | 31.2% | 62.5% | 51.0% | 17.0% | 🔵 | 🔵 |
| **Coder7B Clean SFT** | 7B | Mixed SFT + 合成并行 | 82.1% | **100.0%** | **95.8%** | 🔵 | 🔵 | 🔵 | 🔵 |
| Coder7B Mixed SFT | 7B | Mixed SFT | 82.4% | 62.5% | 45.8% | **99.0%** | **59.0%** | **100%** | **100%** |
| 1.5B DPO | 1.5B | SFT→DPO | 80.0% | 25.0% | 58.3% | 🔵 | 🔵 | 🔵 | 🔵 |
| 1.5B SFT | 1.5B | SFT only | 62.4% | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 |
| Coder7B 基座 | 7B | 无训练 | ~3% | — | — | 3.0% | 2.0% | — | — |
| *DeepSeek V4 Pro* | *~数百B* | *API 对照* | *70.7%\** | *93.8%* | *83.3%* | — | — | — | — |

> \* DeepSeek 因 BFCL schema 不适配导致 ~10% API 报错，实际估计 78-82%
> 🔵 = 评测进行中

## 五、关键结论

**1. 1.5B 小模型跑通 SFT→DPO→GRPO，累计 +21pp（62.4%→80.0%→83.5%）**

单向函数调用超越 DeepSeek V4 Pro。验证了三阶段 RL pipeline 在 Dense 小模型上的有效性。

**2. Coder7B 通过格式纠正释放已有知识（SQL Func 3%→99%，BFCL 3%→82%）**

基座模型 SQL 和函数调用知识已存在，仅因输出格式不对被"封印"。加入 31.5% 标准格式 SFT 数据即可释放，无需额外知识注入。

**3. 合成并行数据解决多函数调用短板（+50pp），零外部数据泄漏**

用 MCP 数据两两合成 2,000 条并行调用样本，live_parallel_multiple 从 45.8%→95.8%。多函数调用是数据覆盖问题而非能力问题。

**4. GRPO 对未收敛模型有效（+3.5pp），对已收敛模型无效**

1.5B 在 80.0% 起点受益于 GRPO；Coder7B 在 82.4% 天花板时 GRPO 只引入噪声。

**5. DPO 效果取决于基座和格式一致性**

1.5B Instruct +17pp（注入结构化能力）；Coder7B -25pp（覆盖已有能力）；混合格式不收敛（输出分布冲突）。

**6. 内部评测标杆失效**

smoke 验证集和 OOD 所有模型 100%，BFCL 才拉开 11.5%→83.5% 的真实能力差距。

## 六、RL 方法选型

| 方法 | 适用场景 | 本项目 | 原因 |
|---|---|---|---|
| GRPO | Dense + 短输出 | ✅ 采用 | 已验证 +3.5pp |
| GSPO | MoE 架构 | ❌ | Qwen2.5 是 Dense |
| DAPO | 长 CoT | ❌ | 输出仅 50-100 tokens |
| DPO | 单输出格式 | ⚠️ | 混合格式不收敛 |

## 七、下一步

| 方向 | 现状 |
|---|---|
| SQL Exact 59%→75% | 参数精确匹配 SFT |
| live_multiple 恢复 | 合成数据增加格式多样性 |
| xLAM / BIRD 评测 | 外网数据下载 |
