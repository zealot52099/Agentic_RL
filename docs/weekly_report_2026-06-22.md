# Agentic RL 工具调用训练周报

> 2026-06-18 ~ 2026-06-22 | bifrost-2026060214414601-yans2 | [SwanLab](https://swanlab.cn/@yans2/agentic-rl-tool-calling)

## 一、实验矩阵（完整指标）

| 模型 | 参数量 | 训练路线 | BFCL Live | live_parallel | live_parallel_multiple | SQL Func | SQL Exact | BFCL Multi-Turn | BFCL Web Search |
|---|---|---|---|---|---|---|---|---|---|
| **1.5B GRPO** 🥇 | 1.5B | SFT→DPO→GRPO | **83.5%** | 31.2% | 62.5% | 51.0% | 17.0% | — | — |
| **Coder7B Clean SFT** | 7B | Mixed SFT + 合成并行 | 82.1% | **100.0%** | **95.8%** | — | — | — | — |
| Coder7B Mixed SFT | 7B | Mixed SFT | 82.4% | 62.5% | 45.8% | **99.0%** | **59.0%** | **100%** | **100%** |
| 1.5B DPO | 1.5B | SFT→DPO | 80.0% | 25.0% | 58.3% | — | — | — | — |
| 1.5B SFT | 1.5B | SFT only | 62.4% | — | — | — | — | — | — |
| Coder7B SFT (MCP) | 7B | MCP SFT | 36.5% | — | — | — | — | — | — |
| Coder7B 基座 | 7B | 无训练 | ~3% | — | — | 3.0% | — | — | — |
| *DeepSeek V4 Pro* | *~数百B* | *闭源 SOTA* | *70.7%\** | *93.8%* | *83.3%* | — | — | — | — |

> \* DeepSeek 因 BFCL schema 不适配导致 ~10% API 报错，实际估计 78-82%

## 二、关键发现

**1. 1.5B 小模型跑通 SFT→DPO→GRPO 全流程，累计 +21pp**

1.5B Instruct 从 62.4%（SFT）→ 80.0%（DPO, +17pp）→ 83.5%（GRPO, +3.5pp），单向函数调用超越 DeepSeek V4 Pro。

**2. Coder7B 格式训练释放已有知识（+96pp SQL）**

基座 Coder7B SQL Func 仅 3%，不是因为缺少 SQL 知识，而是输出格式不对（markdown 代码块 + `function_name` 而非 `name`）。混合格式 SFT 仅纠正格式，即让 Func 飙升至 99%。

**3. 合成并行数据解决多函数调用短板（+50pp）**

live_parallel_multiple 从 45.8%→95.8%——仅用 MCP 数据两两合成 2,000 条并行样本，零 BFCL 泄漏。多函数调用是数据覆盖问题，不是能力问题。

**4. DPO 对专用模型有害（-25pp）**

Coder7B MCP DPO 使 BFCL 从 36.5%→11.5%。DPO 将 Coder 拉向 MCP 单一格式，覆盖了其原生标准 function calling 能力。1.5B Instruct 却受益 (+17pp)——DPO 注入小模型缺少的结构化能力。

**5. GRPO 对已收敛模型无效**

Coder7B 在 BFCL 82.4% 时，两次 GRPO 尝试都导致下降。1.5B 成功是因为起点低 (80%→83.5%)。GRPO 的有效区间是模型有明确提升空间时。

**6. 内部评测完全失效**

smoke 验证集和 OOD held-out 所有模型均 100%。BFCL 才拉开 11.5%→83.5% 的巨大差距。

## 三、RL 方法选型结论

| 方法 | 本场景适用？ | 原因 |
|---|---|---|
| **GRPO** | ✅ | Dense 模型 + 短 JSON + 粒度奖励 |
| GSPO | ❌ | 解决 MoE 路由问题，Qwen2.5 是 Dense |
| DAPO | ❌ | 解决长 CoT (>1000 tokens) 熵坍缩 |
| DPO | ⚠️ | 仅单格式有效，混合格式不收敛 |
| PPO | ❌ | 需额外 Critic 模型，太重 |

## 四、遗留问题与下一步

| 问题 | 方向 |
|---|---|
| SQL Exact 59% → 75% | 参数精确匹配 SFT / 归一化 fix |
| live_multiple 退化 (-13pp) | 合成数据增加格式多样性 |
| WikiSQL 0% | prompt 适配裸 SQL 格式 |
| xLAM / BIRD 评测 | 外网下载数据 |
