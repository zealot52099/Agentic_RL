# 7. 实验过程记录

## 实验目标与路线

在 MCP 工具调用（Multi-Client Protocol tool-calling）场景下，验证 SFT→DPO→GRPO 三阶段强化学习 pipeline，并通过 BFCL V4 / SQL / Agentic 等多维度 benchmark 量化能力变化。

```
Phase A (SFT)           Phase C (DPO)           Phase D (GRPO)
学习工具调用格式 ───► 偏好优化工具选择 ───► 在线 RL 精调
```

---

## 一、实验矩阵

### 模型 × 方法

| 基础模型 | SFT | DPO | GRPO | 最终 BFCL Live |
|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 62.4% | 80.0% (+17pp) | **83.5%** (+3.5pp) 🥇 | ✅ 三阶段闭环 |
| Qwen2.5-Coder-7B-Instruct (MCP-only) | 36.5% | 11.5% (-25pp) | — | ❌ 格式过拟合 |
| Qwen2.5-Coder-7B-Instruct (Mixed) | **82.4%** | 不收敛 | 80.6% (-1.8pp) 🥈 | ⚠️ GRPO v2 进行中 |

### Benchmark 全景

| 能力 | Benchmark | 1.5B GRPO | Coder7B Mixed SFT | 说明 |
|---|---|---|---|---|
| 函数调用 | BFCL V4 Live | **83.5%** | 82.4% | 自定义 scorer，4 类别 |
| SQL 函数调用 | BFCL V3 SQL Func | 51.0% | **99.0%** | 函数名准确率 |
| SQL 函数调用 | BFCL V3 SQL Exact | 17.0% | **59.0%** | 全参数精确匹配 |
| Agentic | BFCL Multi-Turn | — | **100%** | JSON 合法性 |
| Agentic | BFCL Web Search | — | **100%** | JSON 合法性 |
| 指令跟随 | IFEval | 84.7%* | — | * 部分数据 |
| 内部 MCP | smoke val | 100% | 100% | 天花板已到 |

---

## 二、Phase A：SFT 格式训练

### 问题：Coder7B 基座 SQL 知识完整但不会输出正确格式

- **基座 Coder7B 输出（错误）**: 被 markdown 代码块包装，使用 `"function_name"` 而非 `"name"` 作为 key，解析器无法提取 → Func 仅 3%
- **SFT 训练后输出（正确）**: 裸 JSON，无 markdown 包装，使用 `"name"` 和 `"arguments"` → Func 99%，解析器完全匹配

| 指标 | Coder7B 基座 | MCP SFT | Mixed SFT | 提升 |
|---|---|---|---|---|
| BFCL Live | ~3% | 36.5% | **82.4%** | **+79pp** |
| SQL Func | 3.0% | — | **99.0%** | **+96pp** |

**关键发现**：SFT 教的是格式纪律，不是知识。基座的 SQL 和工具选择能力已存在，只是被错误输出格式"封印"了。

### Mixed SFT 数据配方

| 格式 | 数量 | 比例 |
|---|---|---|
| MCP（server_id + tool_name） | 19,494 | 68.5% |
| 标准 function-calling（name + arguments） | 8,954 | 31.5% |

仅 31.5% 标准格式数据，就让 Coder7B 从 36.5%→82.4%（+46pp）。

---

## 三、Phase C：DPO 偏好优化

### 问题 1：clarify 能力崩塌（06-19）

| 指标 | SFT v3 | DPO v1 | 原因 |
|---|---|---|---|
| mcp_positive | 100% | 100% | — |
| mcp_no_tool | 100% | 100% | — |
| **mcp_clarify** | **100%** | **0%** | 偏好数据缺 clarify 对 |

**修复**：生成 4,750 clarify 偏好对 + beta 0.1→0.5，DPO v2 Fixed → **零退化**。

### 问题 2：Coder7B DPO 造成 BFCL -25pp（06-20）

```
Coder7B SFT (MCP-only): BFCL 36.5% → DPO后 11.5% (-25pp)
```

**根因**：MCP-only DPO 的偏好数据全是 MCP 格式，DPO 收敛 = 模型被拉向 MCP → 覆盖了 Coder 原生的标准 function calling 能力。**DPO 收敛本身就是伤害的来源。**

### 问题 3：混合格式 DPO 不收敛（06-21）

MCP 格式输出 `[{"server_id":"x","tool_name":"y"}]`，标准格式输出 `{"name":"y","arguments":{...}}`。同一 batch 内两种信号冲突 → loss 卡在 0.69。

**结论**：DPO 要求单一稳定输出分布。多格式场景应用 GRPO 替代。

---

## 四、Phase D：GRPO 在线强化学习

### 1.5B GRPO 成功（+3.5pp）

| 之前失败原因 | 本次修复 |
|---|---|
| entropy≈1e-7（输出完全确定性） | temperature=1.0 + BFCL 分布外 prompt |
| reward_std=0（组内 4 次生成完全相同） | 粒度化 reward 0/0.5/1.0 |
| MCP 过拟合 prompt | BFCL prompts（模型未 memorized） |

| 模型 | BFCL Live |
|---|---|
| 1.5B SFT | 62.4% |
| 1.5B DPO | 80.0% (+17pp) |
| **1.5B GRPO** | **83.5% (+3.5pp)** 🥇 |

### Coder7B GRPO v1 失败（06-22）

| 指标 | Mixed SFT | GRPO v1 | Δ |
|---|---|---|---|
| SQL Exact | **59.0%** | 0.0% | -59pp |
| BFCL Live | **82.4%** | 80.6% | -1.8pp |

**根因**：reward 函数只检查 args key 是否存在（`required_key in args`），不查值是否正确。GRPO 学会"填任意值就拿分" → SQL 参数值随机化 → eval 精确值匹配全挂。

### Coder7B GRPO v2 修复（06-22，进行中）

| 修复项 | v1 (bug) | v2 (fixed) |
|---|---|---|
| BFCL args 检查 | key 是否存在 | 值是否非空 + 有意义 |
| SQL args 检查 | key 是否存在 | 精确值匹配（normalized） |
| placeholder 惩罚 | 无 | 空值/placeholder 给 0 分 |

结果待出。

---

## 五、关键决策记录

| 日期 | 决策 | 依据 |
|---|---|---|
| 06-18 | 从 1.5B 起步 | 训练方案推荐快速验证链路 |
| 06-19 | DPO beta 0.1→0.5 | v1 clarify 崩塌，增大 KL 约束 |
| 06-19 | 补充 clarify 偏好对 | 根因：缺失偏好类型 |
| 06-19 | 1.5B→Coder7B | 内部数据天花板，需更强基座 |
| 06-20 | Coder7B steps 2000→500 | loss 在 step 100 收敛 |
| 06-20 | 暂停 GRPO | 三条件不满足 (全 0 reward) |
| 06-20 | BFCL 取代内部评测 | 内部 100% 无法区分模型 |
| 06-21 | 混合格式 SFT | BFCL 揭示格式 mismatch |
| 06-21 | 放弃混合 DPO | 多格式冲突，DPO 不收敛 |
| 06-21 | GRPO 复活 | 分布外 prompt + 粒度 reward |
| 06-22 | GRPO≠GSPO/DAPO | Dense 模型 + 短 JSON → 不触发变体场景 |
| 06-22 | GRPO v1→v2 reward 修复 | args 检查从"key存在"升级为"值正确" |

---

## 六、RL 方法选型：为什么 GRPO

| 方法 | 核心改进 | 本场景适用？ |
|---|---|---|
| **GRPO** | 去掉 Critic，组内标准化 | ✅ Dense 模型 + 短 JSON |
| Dr.GRPO | 修复长度偏差 (1/|o_i|) | ❌ 输出 50-100 tokens 均匀 |
| DAPO | 长 CoT 熵坍缩 + 动态采样 | ❌ 短 JSON 不触发 |
| GSPO | 序列级优化 + MoE 稳定 | ❌ Qwen2.5 是 Dense |
| RLVR | 二元 reward | ❌ 粒度不够 |
| DPO | 偏好对离线优化 | ⚠️ 混合格式不收敛 |

---

## 七、产物清单

| 模型 | 路径 | 大小 |
|---|---|---|
| 🥇 1.5B GRPO | `output/grpo_15b_dpo_20260621_153153/adapter` | — |
| 🥈 Coder7B Mixed SFT | `output/coder7b_mixed_sft_20260621_001502/adapter` | 323MB |
| 1.5B DPO | `output/dpo_fixed_20260619_043239/adapter` | 161MB |
| 1.5B SFT | `output/mcp_lora_sft_v3_8ppu_20260618/adapter` | 147MB |
| Coder7B SFT | `output/coder7b_sft_20260619_230410/adapter` | 323MB |
| 📄 详细日志 | `docs/experiments/2026-06-20_experiment_log.md` | — |

---

## 后续实验记录模板

~~~markdown
## YYYY-MM-DD <run_name>

目标：

- 

环境：

```text
Job:
Project:
CUDA/PPU:
Base model:
```

数据：

```text
Train:
Validation:
Manifest:
```

命令：

```bash

```

SwanLab：

```text

```

关键指标：

| Step | Loss | Val Loss | Grad Norm | Clip Rate | Throughput | Notes |
|---:|---:|---:|---:|---:|---:|---|

评测：

| Benchmark | Label | Metric | Result |
|---|---|---|---:|

异常与处理：

- 

结论：

- 
~~~
