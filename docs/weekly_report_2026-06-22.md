# Agentic RL 工具调用训练周报

> 2026-06-18 ~ 2026-06-22 | bifrost-2026060214414601-yans2 | [SwanLab](https://swanlab.cn/@yans2/agentic-rl-tool-calling)

## 实验矩阵

| 模型 | 参数量 | 训练路线 | BFCL Live | live_parallel | live_parallel_multiple | SQL Func | SQL Exact | BFCL Multi-Turn | BFCL Web Search |
|---|---|---|---|---|---|---|---|---|---|
| **1.5B GRPO** | 1.5B | SFT→DPO→GRPO | **83.5%** | 31.2% | 62.5% | 51.0% | 17.0% | — | — |
| **Coder7B Clean SFT** | 7B | Mixed SFT + 合成并行 | 82.1% | **100.0%** | **95.8%** | — | — | — | — |
| Coder7B Mixed SFT | 7B | Mixed SFT | 82.4% | 62.5% | 45.8% | **99.0%** | **59.0%** | **100%** | **100%** |
| 1.5B DPO | 1.5B | SFT→DPO | 80.0% | 25.0% | 58.3% | — | — | — | — |
| 1.5B SFT | 1.5B | SFT only | 62.4% | — | — | — | — | — | — |
| Coder7B 基座 | 7B | 无训练 | ~3% | — | — | 3.0% | — | — | — |
| *DeepSeek V4 Pro* | *~数百B* | *API 对照* | *70.7%\** | *93.8%* | *83.3%* | — | — | — | — |

> \* DeepSeek 因 BFCL schema 不适配导致 ~10% API 报错，实际估计 78-82%

## 关键结论

**1. 1.5B 小模型跑通 SFT→DPO→GRPO，累计 +21pp（62.4%→80.0%→83.5%）**

单向函数调用超越 DeepSeek V4 Pro。验证了三阶段 RL pipeline 在 Dense 小模型上的有效性。

**2. Coder7B 通过格式纠正释放已有知识（SQL Func 3%→99%，BFCL 3%→82%）**

基座模型 SQL 和函数调用知识已存在，仅因输出格式不对被"封印"。加入 31.5% 标准格式 SFT 数据即可释放，无需额外知识注入。

**3. 合成并行数据解决多函数调用短板（+50pp），零外部数据泄漏**

用 MCP 数据两两合成 2,000 条并行调用样本，live_parallel_multiple 从 45.8%→95.8%。多函数调用是数据覆盖问题而非能力问题。

**4. GRPO 对未收敛模型有效（+3.5pp），对已收敛模型无效**

1.5B 在 80.0% 起点受益于 GRPO；Coder7B 在 82.4% 天花板时 GRPO 只引入噪声。RL 有效区间是模型有明确提升空间时。

**5. DPO 效果取决于基座和格式一致性**

1.5B Instruct +17pp（注入结构化能力）；Coder7B -25pp（覆盖已有能力）；混合格式不收敛（输出分布冲突）。DPO 仅适用于单一稳定输出格式。

**6. 内部评测标杆失效**

smoke 验证集和 OOD 所有模型 100%，BFCL 才拉开 11.5%→83.5% 的真实能力差距。

## RL 方法选型

| 方法 | 适用场景 | 本项目 | 原因 |
|---|---|---|---|
| GRPO | Dense + 短输出 | ✅ 采用 | 已验证 +3.5pp |
| GSPO | MoE 架构 | ❌ | Qwen2.5 是 Dense |
| DAPO | 长 CoT | ❌ | 输出仅 50-100 tokens |
| DPO | 单输出格式 | ⚠️ | 混合格式不收敛 |
| PPO | 通用 | ❌ | 需额外 Critic 模型 |

## 下一步

| 方向 | 现状 |
|---|---|
| SQL Exact 59%→75% | 参数精确匹配 SFT |
| live_multiple 恢复 | 合成数据增加格式多样性 |
| xLAM / BIRD 评测 | 外网数据下载 |
