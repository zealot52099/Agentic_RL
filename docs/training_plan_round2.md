# 第二轮训练计划

> 2026-06-23 | 基于第一轮实验结论

## 一、第一轮回顾与遗留问题

| 问题 | 现状 | 根因 |
|---|---|---|
| SQL Exact 偏低 | 59% | 训练数据无 SQL 生成 |
| live_multiple 退化 | 94%→81% | 标准格式只含 1 个 function，没教选择 |
| 多函数并行 | 已解决 | ✅ 合成并行数据有效 |
| Agentic | 已满分 | ✅ 格式纪律完美迁移 |
| Coder7B GRPO 无效 | 已确认 | 天花板处 RL 只加噪声 |

## 二、新增训练数据

### 工具调用

| 数据集 | 数量 | 特点 |
|---|---|---|
| Glaive FC v2 | 1,000 | 多样化工具描述，OpenAI format |
| Hermes FC v1 | 1,000 | 多轮 + 单轮混合 |
| ToolACE | 1,000 | 真实 API schema，参数复杂度高 |

### SQL 生成

| 数据集 | 数量 | 特点 |
|---|---|---|
| Spider Train | 7,000 | 多表 JOIN、嵌套查询、完整 schema |
| SQL Create Context | 500 | 含字段描述的 CREATE TABLE 场景 |

### 已有数据（保留）

| 数据集 | 数量 | 用途 |
|---|---|---|
| MCP Smoke | 14,744 | 基础工具选择能力 |
| 合成并行 | 2,000 | 多函数并行调用 |
| MCP clarify/no-tool | 4,750 | 澄清/拒调能力 |

## 三、数据整合方案

### 步骤 1：修复标准格式（P0）

当前标准格式转换时 AVAILABLE FUNCTIONS 只含 target function（1个）。修复为含全量 6 个 server 的所有 tool。

### 步骤 2：SQL 数据转换为函数调用格式

```python
# Spider 原始
{"question": "How many students?", "query": "SELECT COUNT(*) FROM students"}

# 转换后
{
  "prompt": "AVAILABLE FUNCTIONS:\n[sql.execute(sql_keyword, table_name, columns, conditions)]\n\nUSER:\nHow many students?",
  "completion": {"name":"sql.execute","arguments":{"sql_keyword":"SELECT","table_name":"students","columns":["COUNT(*)"],"conditions":[]}}
}
```

### 步骤 3：开源工具调用数据标准化

Glaive/Hermes/ToolACE 已接近标准 format，只需统一 output key 为 `{"name":"...","arguments":{...}}`。

## 四、最终训练配方

| 数据源 | 数量 | 占比 | 教会什么 |
|---|---|---|---|
| MCP Smoke（原始） | 14,744 | 40% | 基础工具选择 |
| 标准格式全量（修复后） | 8,354 | 23% | 标准 format 下选择 |
| Spider SQL | 5,000 | 14% | SQL 精确生成 |
| 开源工具调用 | 3,000 | 8% | 多样化工具 |
| 合成并行 | 2,000 | 5% | 多函数调用 |
| clarify/no-tool（MCP+标准） | 3,750 | 10% | 拒调/澄清 |
| **总计** | **~36,848** | 100% | |

## 五、训练配置

| 参数 | 值 | 说明 |
|---|---|---|
| 基座 | Qwen2.5-Coder-7B-Instruct | 同上轮 |
| 方法 | LoRA r=32, alpha=64 | 同上轮 |
| lr | 5e-6, warmup=50 | 同上轮 |
| steps | 600 | 36K 样本约 0.5 epoch |
| batch | 16 (2×8) | 同上轮 |

## 六、预期指标

| 指标 | 当前 | 目标 | 提升来源 |
|---|---|---|---|
| BFCL Live Overall | 82.1% | **86-88%** | 修复标准格式 + 开源工具数据 |
| live_multiple | 80.7% | **88-92%** | 标准格式全量 function 列表 |
| SQL Exact | 59% | **72-78%** | Spider SQL 训练数据 |
| live_parallel | 100% | 保持 | 已满分 |
| live_parallel_multiple | 95.8% | 保持 | 已接近满分 |
| BFCL Multi-Turn | 100% | 保持 | 已满分 |

## 七、风险与缓解

| 风险 | 缓解措施 |
|---|---|
| 数据稀释（MCP 占比下降） | 保留 40% MCP 数据作为锚点 |
| SQL 数据与工具调用冲突 | SQL 函数是独立的 `sql.execute`，不与其他工具重叠 |
| 训练时间延长 | 600 steps 仍可控（~40min） |
