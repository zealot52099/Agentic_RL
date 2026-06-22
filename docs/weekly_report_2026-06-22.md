# Agentic RL 工具调用训练周报

> 周期：2026-06-18 ~ 2026-06-22
> Job：bifrost-2026060214414601-yans2
> SwanLab：https://swanlab.cn/@yans2/agentic-rl-tool-calling

## 一、实验目标

在 MCP 工具调用场景下，验证 SFT→DPO→GRPO 三阶段强化学习 pipeline 的有效性，并通过 BFCL V4 / SQL / Agentic 多维度 benchmark 量化能力。

## 二、核心成果

### 最终模型排名 (BFCL V4 Live)

| 模型 | 参数量 | score | 路线 |
|---|---|---|---|
| **Coder7B Clean SFT** | 7B | **82.1%** | Mixed SFT + 合成并行 |
| 1.5B GRPO | 1.5B | **83.5%** | SFT→DPO→GRPO 三阶段 |
| DeepSeek V4 Pro | ~数百B | ~70-80%* | 闭源 SOTA 对照 |

### 关键亮点

- **1.5B 小模型跑通 SFT→DPO→GRPO 全流程**，累计提升 +21pp (62.4%→80.0%→83.5%)，单向函数调用超越 DeepSeek V4 Pro
- **Coder7B 格式训练 +96pp**：基座 SQL 知识已存在，仅通过 31.5% 标准格式数据释放
- **合成并行数据 +50pp**：live_parallel_multiple 45.8%→95.8%，无需 BFCL 泄漏数据
- **Agentic 全满分**：BFCL Multi-Turn 100%，Web Search 100%

## 三、方法验证

### SFT 训练

| 发现 | 量化 |
|---|---|
| 基座 Coder7B SQL 知识完整，仅格式不对 | Func 3%→99% |
| 混合格式 SFT 有效 | BFCL +46pp |
| 并行能力是数据覆盖问题，非能力问题 | live_parallel +37pp |

### DPO 偏好优化

| 发现 | 量化 |
|---|---|
| 1.5B DPO 正向 | +17pp |
| Coder7B DPO 有害 | -25pp（覆盖已有能力） |
| 混合格式 DPO 不收敛 | loss 卡在 0.69 |
| DPO 需偏好数据覆盖全部输出类型 | clarify 缺失→崩塌 |

### GRPO 强化学习

| 发现 | 量化 |
|---|---|
| 1.5B GRPO 有效 | +3.5pp |
| Coder7B GRPO 无效 | 天花板已到，只加噪声 |
| GRPO 成功条件：分布外 prompt + 粒度 reward + 高温 | 三项缺一不可 |

## 四、RL 方法对比结论

| 方法 | 适用场景 | 本项目适用？ |
|---|---|---|
| GRPO | Dense 模型 + 短 JSON | ✅ 采用 |
| GSPO | MoE 架构 | ❌ Qwen2.5 是 Dense |
| DAPO | 长 CoT (>1000 tokens) | ❌ 不触发 |
| DPO | 单输出格式 | ⚠️ 混合格式不收敛 |

## 五、Benchmark 覆盖

| 能力 | Benchmark | 最佳分数 | 模型 |
|---|---|---|---|
| 函数调用 | BFCL V4 Live | **83.5%** | 1.5B GRPO |
| 并行多函数 | BFCL live_parallel | **100%** | Coder7B Clean SFT |
| SQL 函数调用 | BFCL V3 SQL Func | **99.0%** | Coder7B Mixed SFT |
| SQL 函数调用 | BFCL V3 SQL Exact | **59.0%** | Coder7B Mixed SFT |
| Agentic | BFCL Multi-Turn | **100%** | Coder7B Mixed SFT |
| Agentic | BFCL Web Search | **100%** | Coder7B Mixed SFT |

## 六、遗留问题

| 问题 | 优先级 | 方向 |
|---|---|---|
| SQL Exact 59% 偏低 | P0 | 参数精确匹配 SFT 数据 |
| live_multiple 退化 | P1 | 合成数据格式多样性 |
| WikiSQL 0%（格式 mismatch） | P2 | prompt 适配 |
| xLAM eval 未打通 | P2 | 外网下载数据 |

## 七、关键教训

1. **内部评测必须用外部 benchmark 标定**：smoke 数据全满分，BFCL 才拉开 11.5%→83.5% 差距
2. **DPO 是把双刃剑**：小模型注入能力 (+17pp)，大模型覆盖能力 (-25pp)
3. **训练教格式，不教知识**：Coder7B SQL +96pp 纯靠格式纪律
4. **测试数据不能进训练集**：BFCL 泄漏导致高估 +6-12pp
5. **RL 不是万能药**：模型天花板时 GRPO 加噪声，降天花板时 GRPO 加增益

## 八、产物清单

| 模型 | BFCL Live | 路径 |
|---|---|---|
| 🥇 1.5B GRPO | 83.5% | `output/grpo_15b_dpo_20260621_153153/adapter` |
| 🥈 Coder7B Clean SFT | 82.1% | `output/coder7b_clean_parallel_sft_20260622_183244/adapter` |
| Coder7B Mixed SFT | 82.4% | `output/coder7b_mixed_sft_20260621_001502/adapter` |
| 📄 详细日志 | — | `docs/experiments/2026-06-20_experiment_log.md` |
| 📄 实验记录 | — | `docs/07_experiment_records.md` |
