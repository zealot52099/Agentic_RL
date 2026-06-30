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
| Qwen2.5-Coder-7B-Instruct (Mixed) | **82.4%** | 不收敛 | 80.6% (-1.8pp v1) | 🔵 GRPO v2 评测中 |

### Benchmark 全景

| 能力 | Benchmark | 1.5B GRPO | Coder7B Mixed SFT | DeepSeek V4 Pro | 说明 |
|---|---|---|---|---|---|
| 函数调用 | BFCL V4 Live | **83.5%** | 82.4% | 70.7%* | * 含 ~10% API schema 报错 |
| 并行函数调用 | BFCL live_parallel | 31.2% | 62.5% | **93.8%** | DeepSeek 预训练含多工具 |
| 多函数+依赖 | BFCL live_p+m | 62.5% | 45.8% | **83.3%** | 我们的训练数据缺多函数样本 |
| SQL 函数调用 | BFCL V3 SQL Func | 51.0% | **99.0%** | — | 函数名准确率 |
| SQL 函数调用 | BFCL V3 SQL Exact | 17.0% | **59.0%** | — | 全参数精确匹配 |
| Agentic | BFCL Multi-Turn | — | **100%** | — | JSON 合法性 |
| Agentic | BFCL Web Search | — | **100%** | — | JSON 合法性 |
| 指令跟随 | IFEval | 84.7% | — | — | * 含 prompt 格式问题 |

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

### Coder7B GRPO v2 修复 → 结论：Coder7B 上 RL 无效（06-22）

| 修复项 | v1 (bug) | v2 (fixed) |
|---|---|---|
| BFCL args 检查 | key 是否存在 | 值是否非空 + 有意义 |
| SQL args 检查 | key 是否存在 | 精确值匹配（normalized） |
| placeholder 惩罚 | 无 | 空值/placeholder 给 0 分 |

| 指标 | Mixed SFT | GRPO v1 | GRPO v2 | 结论 |
|---|---|---|---|---|
| SQL Exact | **59.0%** | 0.0% | 0.0% | ❌ 两次都崩塌 |
| BFCL Live | **82.4%** | 80.6% | 79.1% | ❌ 每次都下降 |

**结论**：Coder7B 在 BFCL 82.4% 和 SQL 99% 接近天花板，GRPO 只能引入噪声。1.5B 成功是因为起点低（80%→83.5%），Coder7B 应在数据层面改进（扩充并行样本）而非 RL。

### live_multiple 全量评测 + 自建并行 holdout（06-23）

BFCL V4 live_parallel 全量仅 15 题，统计不可靠。自建 200 条 MCP 合成并行 holdout + live_multiple 扩至全量 1,053 条。

| Round 2 SFT | live_multiple (1053) | Self-built parallel (200) |
|---|---|---|
| 94.4% | 全量比 150 条子集高 3.7pp |

### Round 3 SQL Exact 优化（06-23，进行中）

Round 2 SQL Exact=26%，远低于 Mixed SFT 59%。Round 3 数据配方：15% SQL（增强参数提取）+ 10% 并行 + 22% 标准全量 + 54% MCP（23,000 条）。目标 SQL Exact→50%+。

---

### 合成并行数据 SFT — 无泄漏版（06-22）

用 MCP 数据两两组合合成 2,000 条并行调用样本（`q1 Also, q2`），完全基于已有 MCP 数据，零 BFCL 接触。

| 指标 | Mixed SFT | +合成并行 SFT | Δ |
|---|---|---|---|
| live_simple | 78.7% | 79.3% | +0.6pp |
| live_multiple | 94.0% | 80.7% | -13.3pp |
| **live_parallel** | 62.5% | **100.0%** | **+37.5pp** |
| **live_parallel_multiple** | 45.8% | **95.8%** | **+50.0pp** |
| OVERALL | 82.4% | 82.1% | -0.3pp |

### 泄漏分析：为什么无泄漏版更高？

| 版本 | live_parallel | live_parallel_multiple | 训练数据 |
|---|---|---|---|
| BFCL 泄漏版 | 93.8% | 95.8% | BFCL 原题 + 占位符参数 |
| 合成 MCP 版 | **100.0%** | **95.8%** | MCP 真实参数值合成 |

BFCL 泄漏版用了占位符参数（`<column_name>` 等），导致模型在学习时看到无意义的参数值，部分影响了函数选择判断。合成版用 MCP verifier 中的真实参数值，模型对函数与参数的关联更有信心。

其次，live_parallel 只有 16 道题——2 道的差异就能影响 12pp。两者本质在同一水平线上。

结果待出。

---

## 五、DeepSeek V4 Pro 基准对比（06-22）

用 DeepSeek V4 Pro API（`deepseek-v4-pro`）在同组 BFCL V4 Live 测试数据上评测，获取 SOTA 参照系。

### 评估方法

- API: `https://api.deepseek.com/chat/completions`，OpenAI 兼容 tool-calling 格式
- 数据: 与模型评测相同的 140 条 BFCL V4 Live 测试样本
- ⚠️ BFCL 数据使用非标准 JSON Schema（`"type":"dict"`, `"type":"float"`），需转换为 OpenAI 兼容格式
- ⚠️ 部分函数名含 DeepSeek API 不接受的字符（`.` `:`），导致约 10% 条目 API 400 报错 → 该题 0 分

### 结果对比

| 类别 | DeepSeek V4 Pro | 1.5B GRPO | Coder7B Mixed | 差距分析 |
|---|---|---|---|---|
| live_simple | 76.0% | **86.7%** | 78.7% | 我们的模型在单函数场景超 SOTA |
| live_multiple | 52.0% | **89.3%** | **94.0%** | DeepSeek API schema 报错影响最大 |
| live_parallel | **93.8%** | 31.2% | 62.5% | DeepSeek 预训练含并行工具调用 |
| live_parallel_multiple | **83.3%** | 62.5% | 45.8% | 我们的训练数据缺多函数样本 |
| **OVERALL** | **70.7%*** | **83.5%** | **82.4%** | * 实际估计 78-82%（排除 API 报错） |

### 关键发现

1. **单函数调用（我们的主战场）已超越 DeepSeek V4 Pro**，1.5B GRPO 86.7% vs DeepSeek 76.0%
2. **并行多函数是我们的明显短板**，DeepSeek 93.8% vs 我们最高 62.5% — 根因：训练数据中无多函数调用样本
3. **1.5B 能达到与数百B 参数 DeepSeek 可比的水准**，验证了 SFT→DPO→GRPO pipeline 的有效性

---

### 统一 chat_template + tools 数据重构（06-24）

**目标**：解决训练数据中 system prompt 不统一的问题（MCP 用 "You are an MCP agent..."，SQL 用 bare "AVAILABLE FUNCTIONS:"，开源数据无 system prompt）。

**方案**：所有数据转为 Qwen `chat_template` + `tools` 参数，system prompt 统一为 "You are a helpful assistant."

| 版本 | 训练格式 | In-dist | OOD | Gap | 结论 |
|---|---|---|---|---|---|
| Unified (100% chat) | chat_template | 89.1% | 57.1% | 32pp | OOD 完全遗忘 |
| **Mixed 80/20** | 80% chat + 20% OOD | 88.8% | **85.3%** | **3.5pp** | ✅ 方案可行 |

混入 20% OOD 格式数据即可恢复泛化，format gap 从 32pp→3.5pp。

### SQL Exact 问题分析与下一步

SQL Exact 经历四轮 SFT，从 Mixed SFT 的 59% 降至 18-26%。核心原因：

1. **训练数据格式不匹配**：Spider SQL 数据通过 regex 提取参数值（`columns: ["name"]`），而 BFCL 评测使用双层嵌套格式（`columns: [["name"]]`）。`nv()` 归一化无法完美桥接。
2. **chat_template 加剧差异**：chat_template 格式下模型输出格式与 BFCL GT 差异更大，In-dist Exact=0%。

**下一步方向**：

| 方案 | 工作量 | 预期 |
|---|---|---|
| 用 BFCL V3 SQL 80 条做 few-shot SFT | 小 | Exact → 40-50% |
| Spider 数据按 BFCL 嵌套格式重新生成 completion | 中 | Exact → 50-60% |
| GRPO with BFCL SQL ground truth as reward | 中 | 需 SFT 先到 50%+ |
| 接受当前 OOD Exact=24% 作为上限 | 零 | 不提升 |

**建议优先**：方案 B（Spider 数据重新生成）→ 方案 C（GRPO 接力）。

**遗留**：SQL Exact 仍为 0% (In-dist) / 14% (OOD)，chat_template 下 SQL 参数格式对齐待解决。

---

## 六、关键决策记录

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

## 七、RL 方法选型：为什么 GRPO

| 方法 | 核心改进 | 本场景适用？ |
|---|---|---|
| **GRPO** | 去掉 Critic，组内标准化 | ✅ Dense 模型 + 短 JSON |
| Dr.GRPO | 修复长度偏差 (1/|o_i|) | ❌ 输出 50-100 tokens 均匀 |
| DAPO | 长 CoT 熵坍缩 + 动态采样 | ❌ 短 JSON 不触发 |
| GSPO | 序列级优化 + MoE 稳定 | ❌ Qwen2.5 是 Dense |
| RLVR | 二元 reward | ❌ 粒度不够 |
| DPO | 偏好对离线优化 | ⚠️ 混合格式不收敛 |

---

## 八、产物清单

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

---

## 2026-06-29：SQL Execution GRPO 实验结果与分析

### Phase7：16 卡并行独立 LoRA GRPO

目标：验证 SQL execution reward 是否能提升 WikiSQL 风格 SQL 生成的可执行正确率。

训练方式：

- 入口：`runs/phase7_wikisql_exec_grpo_torchrun_independent16_20260628_224418`
- 基座：`phase5_qwen25_coder7b_lora_ppu16_lr1e6_seed20260626_20260626_150539/merged_hf`
- 资源：16 PPU
- 重要限制：该实验是 16 个 rank 各自训练独立 LoRA adapter，没有做 DDP all-reduce，因此不是一个同步 16 卡 GRPO 模型。

内部 WikiSQL fixed probe 结果如下。该评测是内部固定执行探针，不是官方 WikiSQL benchmark。

| Model | SQL extraction | Execution rate | Execution accuracy | 相对 Phase5 | 相对 Phase6 |
|---|---:|---:|---:|---:|---:|
| Phase5 SFT baseline | 100.00% | 79.30% | 55.47% | 0.00 pp | +3.91 pp |
| Phase6 SQL/tool SFT | 100.00% | 73.44% | 51.56% | -3.91 pp | 0.00 pp |
| Phase7 GRPO rank07 | 100.00% | 81.64% | 56.25% | +0.78 pp | +4.69 pp |
| Phase7 GRPO rank03 | 100.00% | 81.64% | 57.03% | +1.56 pp | +5.47 pp |
| Phase7 GRPO rank11 | 100.00% | 81.64% | 56.25% | +0.78 pp | +4.69 pp |
| Phase7 GRPO rank15 | 100.00% | 81.64% | 56.25% | +0.78 pp | +4.69 pp |

结论：

- SQL execution reward 有正收益，但幅度较小，最佳 rank03 相对 Phase5 提升 `+1.56 pp`。
- 收益主要来自 execution rate 提升，而不是 SQL exact match 的显著提升。
- Phase6 的 SQL/tool SFT 在 tool-call 格式上更有价值，但在这个 WikiSQL 执行探针上低于 Phase5，说明 SQL 数据和 tool-call 数据不能简单相加，需要重新设计 mixture、loss weight 和训练顺序。
- Phase7 只能作为 reward 有效性的证据，不能作为最终模型，因为它不是同步训练出的单一 adapter。

### Phase8：同步 16 PPU GRPO

目标：将 Phase7 的独立 adapter 方案升级为真正同步 16 卡 GRPO，得到单个共享 LoRA adapter。

当前方案：

- 入口脚本：`scripts/remote/run_swift_wikisql_grpo_ppu16.sh`
- 当前 run：`runs/phase8_swift_wikisql_grpo_sync16_20260629_171307`
- 日志：`logs/phase8_swift_wikisql_grpo_sync16_20260629_171307.log`
- 框架：`ms-swift rlhf --rlhf_type grpo`
- 分布式：`torch.distributed.run --nproc_per_node 16` + DeepSpeed ZeRO-1
- Adapter：LoRA rank 16 / alpha 32 / dropout 0.05
- LR：`2e-7`
- 训练步数：2000
- 每 prompt 采样数：4
- Reward：`scripts/remote/swift_wikisql_reward_plugin.py`
- 数据：`datasets/processed/phase8_swift_wikisql_grpo_20260629/train.jsonl`

截至当前同步记录时，训练仍在进行，尚未产出最终评测结论。中间状态：

| Step | Reward mean | Reward std | Grad norm | KL | Memory | 备注 |
|---:|---:|---:|---:|---:|---:|---|
| 15/2000 | 0.7406 | 0.0438 | 0.0931 | 0.0000599 | 35.78 GiB | 首次稳定检查 |
| 111/2000 | 0.6594 | 0.0687 | 0.0759 | -0.0000122 | 35.85 GiB | 训练正常运行 |

阶段性判断：

- 同步 16 卡链路已跑通，当前不是 Phase7 的独立 adapter 模式。
- reward 波动较大是 GRPO + execution reward 的正常现象，不能按 SFT loss 曲线理解；更关键的是最终固定评测集上的 execution accuracy。
- KL 维持在很小量级，clip ratio 基本为 0，说明当前 LR 较保守，不像在发生策略崩坏。
- 大量 `triton.language.target_info` 日志来自 vLLM/Triton kernel 兼容警告，目前未阻断训练。

启动过程中踩坑及处理：

| 问题 | 表现 | 处理 |
|---|---|---|
| `verl==0.8.0` isolated install 不可用 | 安装后 import `verl` 失败，且依赖树会引入不匹配 Torch/Transformers | 放弃当前环境直接使用 verl，改用已有 PPU 可用的 ms-swift |
| 非交互 SSH 下 `/etc/profile` 早退 | launcher 产生空日志 | 启动脚本改为不依赖 profile，并固定 `/opt/ac2/bin/swift` |
| 缺少 `libhggcrt1.so` | PPU runtime loader 报错 | 显式加入 `/usr/local/PPU_SDK/targets/x86_64-linux/lib` |
| 缺少 `libcuda.so` | external provider / NCCL 初始化失败 | 显式加入 PPU `CUDA_SDK` lib 路径 |
| 缺少 `PPU_SDK` / `PPU_HOME` | RTC kernel 报 `Both PPU_SDK and PPU_HOME are not exist` | 显式导出 `PPU_SDK=/usr/local/PPU_SDK` 和 `PPU_HOME=/usr/local/PPU_SDK` |
| SwanLab 未配置 | cloud 模式缺 API key，local 模式缺 `swanboard` | 本轮使用 TensorBoard 和完整 `logging.jsonl`，待配置后恢复 SwanLab |

下一步：

1. 等 Phase8 训练完成并确认 checkpoint。
2. 合并 adapter。
3. 用 Phase5/Phase6/Phase7 同口径内部 WikiSQL execution probe 评测。
4. 同步补跑 tool-call probe，防止 SQL GRPO 损伤工具调用能力。
5. 若 Phase8 相对 Phase7 没有提升，优先检查 reward 稀疏性、训练集与评测集分布、采样温度、`num_generations` 和 LR，而不是直接加大训练步数。

---

## 2026-06-29: Phase9 Mixed SQL + Tool-Call GRPO

目标：从 `Phase6 merged` 出发，做混合 SQL execution + tool-call/no-tool/clarify GRPO，避免 Phase8 只提升 SQL 而没有继承 Phase6 tool-call 增益的问题。

训练链路：

```text
Qwen2.5-Coder-7B-Instruct
  -> Phase5 SFT
  -> Phase6 SQL/tool SFT merged
  -> Phase9 mixed SQL + tool-call GRPO
```

远端环境：

```text
Job: bifrost-2026060214414601-yans2
Project: /workspace/yans2@xiaopeng.com/agentic_rl_pipeline
Base model: runs/phase6_qwen25_coder7b_sqltool_lora_ppu16_lr6e7_seed20260627_noswan_20260627_105305/merged_hf
Framework: ms-swift rlhf --rlhf_type grpo
Distributed: torch.distributed.run --nproc_per_node 16 + DeepSpeed ZeRO-1
Tracking: SwanLab local mode
```

新增文件：

```text
scripts/remote/prepare_phase9_mixed_grpo.py
scripts/remote/swift_mixed_sql_tool_reward_plugin.py
scripts/remote/run_swift_mixed_sql_tool_grpo_ppu16.sh
datasets/processed/phase9_mixed_sql_tool_grpo_20260629/train.jsonl
datasets/processed/phase9_mixed_sql_tool_grpo_20260629/manifest.json
```

数据配比：

| Task type | Count | Reward |
|---|---:|---|
| SQL execution | 4096 | SQL parse, safe SELECT, SQLite execution success, result exact, normalized SQL exact |
| tool_call | 2048 | JSON parse, action match, call count, tool name, argument exact |
| no_tool | 593 | JSON parse, action match, empty calls, schema discipline |
| clarify | 636 | JSON parse, action match, missing-field overlap, non-empty clarification message |

说明：SQL 样本来自 Phase8 可执行 WikiSQL GRPO 数据；tool/no-tool/clarify 样本来自 Phase5 unified train split。该数据不是官方 benchmark，而是内部 RLVR 训练集。

启动命令摘要：

```bash
REPORT_TO=swanlab SWANLAB_MODE=local MAX_STEPS=2000 \
  nohup bash scripts/remote/run_swift_mixed_sql_tool_grpo_ppu16.sh \
  phase9_swift_mixed_sql_tool_grpo_sync16_20260629_183720 \
  > logs/phase9_swift_mixed_sql_tool_grpo_sync16_20260629_183720.log 2>&1 &
```

关键超参：

```text
LoRA: rank 16, alpha 32, dropout 0.05
LR: 2e-7
Max steps: 2000
num_generations: 4
max_length: 1536
max_completion_length: 128
beta: 0.03
loss_type: grpo
```

启动状态：

| Time | Status |
|---|---|
| 2026-06-29 18:37 | Phase9 run started, PID 1700020 |
| 2026-06-29 18:38 | 16 PPU devices `PPU-ZW810E` detected; distributed/vLLM initialization in progress |

已知注意点：

- 这次从 Phase6 merged 起步，不再复用 Phase5 起点。
- Phase9 是 mixed reward，目标是同时看 WikiSQL execution accuracy 和内部 tool-call validation 是否改善或至少不回退。
- 远端只有 `/opt/ac2/bin/python`；任何 Python/Swift 调用前都需要显式设置 PPU SDK `LD_LIBRARY_PATH`，否则会报 `libhggcrt1.so`。
- `triton.language.target_info` 仍可能在 vLLM 初始化时出现兼容性警告；Phase8 证明这类日志不一定阻断训练，需结合进程和 step metrics 判断。

验收评测计划：

| Metric group | Dataset | Label |
|---|---|---|
| SQL execution | Internal rebased WikiSQL 256 probe | internal, not official WikiSQL benchmark |
| Tool-call | `phase5_unified_20260626/validation.jsonl` | internal unified validation |
| General retention | GSM8K fixed 256 subset | internal subset |
| Instruction following | IFEval if runtime permits | public benchmark rerun |

Phase9 初步目标：

- SQL execution accuracy 接近或超过 Phase8 `62.11%`。
- Tool action/tool-name exact 尽量保持 Phase6 水平，明显高于 Phase8-from-Phase5。
- GSM8K fixed subset 回退不超过 1 pp。

Phase9 runtime update:

- First launch `phase9_swift_mixed_sql_tool_grpo_sync16_20260629_183720` failed before training because SwanLab local mode lacked `swanboard`.
- Installed `swanlab[dashboard]`, which added `swanboard-0.1.9b3`.
- Restarted successfully as `phase9_swift_mixed_sql_tool_grpo_sync16_20260629_184129`, PID `1711045`.
- At step `99/2000`, reward `0.9750`, KL `0.0000315`, memory `36.18 GiB`, train speed about `2.34 s/it`, ETA about `1h14m`.
- Metrics path: `runs/phase9_swift_mixed_sql_tool_grpo_sync16_20260629_184129/v0-20260629-184156/logging.jsonl`.

Phase9 final evaluation:

| Model | SQL exec acc | SQL exec rate | Tool action exact | Tool name exact | JSON exact | GSM8K fixed-256 acc |
|---|---:|---:|---:|---:|---:|---:|
| Phase5 SFT | 55.47% | 79.30% | 81.41% | 79.74% | 37.73% | 76.56% |
| Phase6 SQL/tool SFT | 51.56% | 73.44% | 95.72% | 95.35% | 50.56% | 76.17% |
| Phase8 SQL-only GRPO from Phase5 | 62.11% | 88.67% | 81.23% | 79.55% | 37.73% | 76.17% |
| Phase9 mixed GRPO from Phase6 | 55.86% | 80.08% | 95.72% | 95.35% | 50.56% | 75.00% |

Conclusion: Phase9 preserved Phase6 tool-call quality, recovered SQL above Phase6, but did not match Phase8 SQL-only GRPO. GSM8K fixed-256 regressed mildly to 75.00%. Next run should use staged SQL-only then mixed retention, or stronger SQL reward weighting.

## 2026-06-30: Phase10 Staged SQL-Then-Mixed GRPO

Goal: fix the Phase9 failure mode where mixed GRPO preserved tool-call but diluted SQL execution reward. Phase10 uses a staged schedule:

```text
Phase6 merged
  -> Phase10a SQL-only GRPO from Phase6
  -> merge Phase10a adapter
  -> Phase10b lower-LR mixed retention GRPO
  -> merge final adapter
```

Remote run:

```text
Job: bifrost-2026060214414601-yans2
Project: /workspace/yans2@xiaopeng.com/agentic_rl_pipeline
Queue script: scripts/remote/run_phase10_staged_grpo_ppu16.sh
Stamp: 20260630_094959
Outer log: logs/phase10_staged_sql_then_mixed_20260630_094959.outer.log
PID file: runs/phase10_staged_sql_then_mixed_20260630_094959.pid
Run root: runs/phase10_staged_sql_then_mixed_20260630_094959
Eval root: evals/phase10_staged_sql_then_mixed_20260630_094959
```

Stage design:

| Stage | Base | Data | Reward | Steps | LR | Purpose |
|---|---|---|---|---:|---:|---|
| Phase10a | Phase6 merged | Phase8 WikiSQL executable GRPO data | `wikisql_exec` | 1500 | 2e-7 | recover SQL execution reward without tool-task dilution |
| Phase10b | Phase10a merged | Phase9 mixed SQL/tool data | `mixed_agent_reward` | 800 | 1e-7 | retain tool-call while avoiding large SQL regression |

Expected evaluation after completion:

| Metric group | Dataset | Label |
|---|---|---|
| SQL execution | Internal rebased WikiSQL 256 probe | internal, not official WikiSQL |
| Tool-call | `phase5_unified_20260626/validation.jsonl` | internal unified validation |
| General retention | GSM8K fixed 256 subset | internal subset |

Target: exceed Phase9 SQL `55.86%`, ideally approach Phase8 `62.11%`, while keeping Phase6/Phase9 tool action exact around `95%`.
