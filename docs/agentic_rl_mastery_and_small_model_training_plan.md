# Agentic RL 全流程掌握与小模型训练修订方案

> 配套数据台账见 `docs/08_dataset_registry.md`，其中维护训练集、评测集、完整样例、创建日期、数量和加入后的指标影响。

## 0. 目标与判断

本方案是对“19 个工具 + 10K 合成轨迹 + SFT -> GRPO”入门方案的升级版。原方案适合理解工具调用训练的最小闭环，但不足以支撑两个目标：

1. 系统掌握 Agentic RL 的全流程能力，包括数据准备、预训练原理、后训练、RL、评测、工程稳定性和常见故障处理。
2. 基于 7B 或更小模型做除预训练之外的训练，在常用通用指标和 Agent 指标上尽量接近同体量 SOTA。

这里的核心原则是：先建立可验证闭环，再扩大数据和模型；先让小实验真实学习，再谈长训和 RL；所有结论都必须由固定验证集、公开 benchmark 或可复现内部 probe 支撑。

## 1. 能力地图

### 1.1 需要掌握的技术层级

| 层级 | 要掌握的内容 | 验收方式 |
|---|---|---|
| 数据 | trace schema、工具 schema、MCP server catalog、数据去重、OOD split、污染检测 | 能解释每条数据为何可训练、为何不泄漏测试集 |
| 预训练原理 | tokenization、next-token prediction、数据配比、长上下文、scaling law、退火、checkpoint | 不做大规模预训练，但能读懂训练曲线和数据配方 |
| SFT | assistant-only loss、template、LoRA/QLoRA/全参、FSDP、batch/token weighting | 固定验证 loss 下降，JSON/schema 指标提升 |
| 偏好优化 | DPO、IPO、SimPO、ORPO、拒绝采样、hard negative | 能构造“正确工具 vs 相似错误工具”等偏好对 |
| RLVR | GRPO、Dr.GRPO/DAPO、可验证奖励、KL、group success rate、reward hacking | reward 上升且真实执行成功率提升 |
| Agent 环境 | MCP sandbox、真实工具执行、状态机、权限、超时、错误恢复 | 能复现任务轨迹并验证最终状态 |
| 评测 | BFCL、xLAM、IFEval、GSM8K、MMLU-Pro、WikiSQL、tau2/AppWorld/SWE-bench | 内部、官方复测、公开数字分开标记 |
| 工程 | provenance、SwanLab、日志归档、checkpoint eval、失败恢复 | 每次实验可定位代码、数据、模型和环境 |

### 1.2 学习顺序

1. 读懂一个最小工具调用训练项目。
2. 自己实现数据 schema、reward、eval，不急着训练。
3. 用 1.5B 或 3B 跑 LoRA smoke，观察真实学习。
4. 扩展到 4B/7B，加入通用能力回放。
5. 加入偏好优化和 GRPO。
6. 接入真实 MCP sandbox 和公开 benchmark。
7. 建立模型卡和失败案例库。

## 2. 模型路线

### 2.1 推荐模型梯队

| 阶段 | 模型 | 目的 |
|---|---|---|
| 调试 | Qwen2.5-1.5B-Instruct / Qwen3-1.7B | 快速验证数据、loss、reward、评测链路 |
| 主力小模型 | Qwen2.5-3B、Qwen3-4B、Qwen2.5-Coder-3B | Agent/tool 能力训练和消融 |
| 7B 上限 | Qwen2.5-7B-Instruct、Qwen2.5-Coder-7B、Qwen3-8B 降级参考 | 同体量强基线和最终对齐 |
| 专用对照 | xLAM-2-3B-fc-r / xLAM-2-8B-fc-r | 函数调用专用模型对照 |
| 教师 | Qwen 30B+、DeepSeek/Qwen API、xLAM 8B | 生成轨迹、修复答案、构造偏好对 |

### 2.2 选型原则

优先选 instruct 模型而非 base 模型做后训练。Base 模型需要补齐指令跟随，成本更高。Coder 模型适合工具、SQL、代码执行和 SWE 类任务，但通用对话可能弱一些。

7B 以下模型的核心不是“什么都学”，而是通过数据和 reward 精准提升：

- 工具选择。
- 参数生成。
- no-tool 判断。
- 澄清问题。
- 错误恢复。
- 权限和安全边界。

## 3. 数据体系

### 3.1 统一 Trace Schema

所有 Agent 数据都统一转换为如下结构：

```text
trace_id
split_group_id
server_catalog
tool_schemas
messages
assistant_action
tool_calls
tool_results
environment_state_before
environment_state_after
permissions
terminal_state
verifier_results
reward_components
generation_metadata
```

必须保留以下字段：

- `schema_fingerprint`：用于防止同一工具改名后泄漏到测试集。
- `server_family`：用于 OOD split。
- `parser_version`：不同 parser 会导致 BFCL/MCP 指标不可比。
- `chat_template_version`：template 变化会明显影响工具调用输出。

### 3.2 数据来源

| 数据类型 | 作用 | 训练占比建议 |
|---|---|---:|
| 单轮函数调用 | 学工具名、参数 JSON、并行调用 | 20%-25% |
| MCP 多 server 路由 | 学 server/tool 选择、候选工具干扰 | 20%-30% |
| 多轮执行轨迹 | 学 observation、状态变化、终止条件 | 15%-25% |
| no-tool | 学不该调用工具时直接回答 | 8%-12% |
| clarification | 学缺参时追问，不猜参数 | 8%-12% |
| error recovery | 学超时、空结果、权限不足、重试 | 8%-12% |
| safety/security | 学拒绝越权、提示注入、危险操作确认 | 5%-8% |
| 通用回放 | 防止 IFEval/GSM8K/MMLU/代码能力回退 | 10%-15% |

### 3.3 数据生成

每个任务先生成 blueprint，再生成自然语言和轨迹。

Blueprint 必须包含：

- 用户目标。
- 可用 server 和工具。
- 必须调用的工具。
- 可选路径。
- 禁止动作。
- 缺失参数。
- 环境初始状态。
- 期望最终状态。
- 验证器。

生成流程：

1. 收集工具 schema，归一化字段名、类型、required、默认值。
2. 计算 schema fingerprint 和 server family。
3. 构建任务 blueprint。
4. 加入干扰工具：同名、近义名、同参数不同语义、过时工具。
5. 教师模型生成用户请求和候选轨迹。
6. 执行或模拟轨迹，验证 schema、权限、状态变化。
7. 成功样本进入 SFT。
8. 失败样本转成偏好对或错误恢复数据。
9. 用当前学生模型挖失败样本，定向补数据。

### 3.4 必须避免的数据坑

| 问题 | 症状 | 解决 |
|---|---|---|
| no-tool 过短 | loss 权重很低，模型仍乱调工具 | 对 no-tool 加样本权重或使用 balanced batch |
| clarification 标成 `[]` | 模型不会追问，只会空调用 | 明确输出 clarify action 或自然语言追问 |
| 工具 schema 泄漏 | 内部测试虚高 | 按 schema fingerprint/server family 切分 |
| 教师幻觉 | 学会不存在工具或错误参数 | schema 校验 + 执行校验 + 最终状态校验 |
| replay 太长 | 截断破坏答案 | 过滤、摘要或提高 seq_len |
| 数据全是正例 | 工具幻觉率高 | 至少 15%-25% no-tool/clarify/safety |
| 只训练单轮 | 多轮状态漂移 | 多轮 observation 和 state summary 训练 |
| 输出格式混乱 | parser 指标低 | 固定 chat template 和 tool call grammar |

## 4. 预训练原理学习

本项目不建议从零预训练 7B 模型，但必须掌握预训练原理，否则无法判断后训练失败原因。

### 4.1 需要掌握的点

- 自回归 next-token loss。
- tokenizer 对 JSON、代码、中文、工具名的影响。
- 数据配比对能力的影响。
- 长上下文训练和 RoPE scaling 的风险。
- learning rate schedule、warmup、cosine decay。
- gradient clipping、loss spike、NaN。
- BF16/FP16/FP32 master weight 的区别。
- checkpoint averaging 和退火数据。

### 4.2 推荐实践

用 100M-500M 或 1B 以下模型做 toy pretraining：

- 1GB-10GB 混合文本。
- 1K-4K context。
- 训练 1K-10K step。
- 观察 loss、perplexity、样本生成。

目的不是得到可用模型，而是理解训练动态。

## 5. 后训练路线

### 5.1 阶段 A：格式与路由 SFT

目标：

- JSON/schema valid > 99%。
- 工具名幻觉 < 2%。
- no-tool F1 > 85%。
- clarify 识别率 > 80%。

训练建议：

- 先用 1.5B/3B LoRA 跑 100-300 step smoke。
- 确认固定验证 loss 下降。
- 确认 adapter 或权重确实更新。
- 再扩大到 1K-5K step。

注意：

- 小模型优先 LoRA，排除数据问题。
- 全参训练必须有 FP32 master 或 FSDP/DeepSpeed。
- 不要只看 train loss，要看固定验证集分 source loss。

### 5.2 阶段 B：完整 SFT

目标：

- 学会多轮工具调用和状态更新。
- 保持通用能力不大幅下降。
- 内部 MCP OOD success 明显提升。

建议配置：

| 模型 | 方法 | 起步配置 |
|---|---|---|
| 1.5B/3B | LoRA/全参 | lr 1e-5 到 5e-5 LoRA，或全参 5e-7 到 2e-6 |
| 4B | LoRA/FSDP | LoRA 1e-5 到 4e-5；全参需 FP32 master |
| 7B | QLoRA/LoRA/FSDP | QLoRA 1e-4 级，LoRA 1e-5 到 3e-5 |

### 5.3 阶段 C：拒绝采样与偏好优化

数据构造：

- 对每个任务采样 4-8 个回复。
- 执行或验证每个回复。
- 成功轨迹作为 chosen。
- 相似错误、缺参猜测、越权调用作为 rejected。

偏好对类型：

- 正确工具 vs 相似错误工具。
- 正确 server vs 错误 server。
- 澄清 vs 猜参数。
- 拒绝危险操作 vs 越权调用。
- 一次成功 vs 重复调用。
- 正确恢复 vs 放弃任务。

算法选择：

- DPO：最稳，适合第一版。
- SimPO/IPO：可作为消融。
- ORPO：可尝试，但要严格看回归指标。

### 5.4 阶段 D：RLVR / GRPO

启动条件：

- SFT 已有 15%-60% 可验证成功率。
- JSON/schema valid 已高。
- reward 不全是 0 或 1。
- 有稳定 sandbox 或静态 verifier。

奖励设计：

```text
format_gate: 必须通过，否则后续奖励为 0
route_reward: 0.10
argument_reward: 0.15
execution_reward: 0.20
task_success_reward: 0.35
recovery_reward: 0.10
safety_reward: 0.10
penalty: 多余调用、重复调用、超长、危险操作、无效 JSON
```

监控项：

- reward mean/std。
- group success rate。
- skipped group rate。
- KL。
- completion length。
- call count。
- JSON valid。
- task success。
- 各 reward component。

常见坑：

| 问题 | 表现 | 处理 |
|---|---|---|
| reward hacking | 输出关键词骗 task completion | 用执行状态做主奖励 |
| 全 0 reward | GRPO 无学习信号 | 降低任务难度或先补 SFT |
| 全 1 reward | 无相对优势 | 增加 hard negative |
| KL 爆炸 | 格式崩坏、能力回退 | 增大 beta、降低 lr、混回放 |
| 长度膨胀 | completion 越来越长 | 长度惩罚和最大调用数 |
| 只学格式 | JSON valid 高但成功率低 | 降低格式奖励权重，把格式作为 gate |

## 6. 小模型逼近 SOTA 的策略

### 6.1 什么叫“接近 SOTA”

不能只看内部指标。需要三类数字：

- 官方公开数字：leaderboard 或论文数字。
- 官方代码复测：同 prompt、同 parser、同版本。
- 内部 probe：用于开发，不对外声称 SOTA。

### 6.2 7B 以下模型的优化重点

小模型容量有限，应避免泛泛地学所有任务。优先优化：

1. 工具选择边界。
2. 参数 schema。
3. no-tool/clarify。
4. 安全拒绝。
5. 多轮状态一致性。
6. 通用能力保持。

### 6.3 推荐实验矩阵

第一轮：

| 实验 | 模型 | 方法 | 目标 |
|---|---|---|---|
| E1 | Qwen3-1.7B | LoRA SFT | 验证数据和评测 |
| E2 | Qwen2.5-3B | LoRA SFT | 工具调用小模型基线 |
| E3 | Qwen3-4B | LoRA SFT | 主力路线 |
| E4 | Qwen2.5-Coder-3B | LoRA SFT | 代码/SQL/tool 对照 |

第二轮：

| 实验 | 方法 | 目标 |
|---|---|---|
| E5 | hard negative DPO | 改善相似工具混淆 |
| E6 | clarify/no-tool 加权 SFT | 降低工具幻觉 |
| E7 | GRPO static verifier | 提升 JSON 和参数 |
| E8 | RLVR sandbox | 提升真实任务成功率 |

第三轮：

- 迁移最佳配方到 7B。
- 与 xLAM-2-3B/8B、Qwen 同体量模型、公开 leaderboard 对齐。
- 写模型卡。

### 6.4 通用能力保持

每轮训练都跑：

- IFEval。
- GSM8K。
- MMLU-Pro。
- HumanEval+/MBPP+ 或轻量代码集。
- WikiSQL/SQL 执行。

回退阈值：

- 任一通用指标回退 > 2 个百分点，阻断进入下一阶段。
- 如果 Agent 指标提升很大但通用能力下降，可尝试：
  - 增加 10%-20% replay。
  - 降低 lr。
  - 降低 LoRA rank 或训练步数。
  - DPO/RL 加 KL。

## 7. 评测体系

### 7.1 开发期内部评测

| 指标 | 目的 |
|---|---|
| JSON parse rate | 格式基础 |
| schema valid rate | 参数结构 |
| server accuracy | MCP 路由 |
| tool accuracy | 工具选择 |
| argument exact/semantic | 参数正确性 |
| no-tool F1 | 不乱调用 |
| clarify accuracy | 缺参追问 |
| execution success | 工具可执行 |
| task success | 最终状态正确 |
| recovery rate | 错误恢复 |
| safety violation | 权限和注入 |

### 7.2 公开评测

| Benchmark | 作用 | 备注 |
|---|---|---|
| BFCL V4 | 函数调用官方对比 | 必须用官方 parser/harness |
| xLAM eval | 工具调用内部/开源对照 | 明确是否官方 |
| IFEval | 指令保持 | 防止格式训练损伤 |
| GSM8K | 数学推理回归 | 小模型常用 |
| MMLU-Pro | 通用知识和推理 | 用 direct-logprob 更稳 |
| WikiSQL | SQL 生成和执行 | 报 extraction/execution/exact |
| HumanEval+/MBPP+ | 代码能力 | 7B 以下重要 |
| tau2-bench | 多轮任务 | 需要适配环境 |
| AppWorld | 应用级 Agent | 需要环境 |
| SWE-bench Lite/Verified | 软件工程 | 必须容器隔离 |

### 7.3 报告格式

所有结果必须标记：

- `official_public`：引用公开榜单。
- `official_rerun`：本地官方代码复测。
- `internal_probe`：内部开发集。
- `synthetic_train_like`：与训练分布相近，只能看趋势。

## 8. 工程规范

每个实验目录必须包含：

```text
training_config.json
train.log
train_metrics.jsonl
provenance.json
exit_code
logs.sha256
swanlab/
adapter/ or hf/
checkpoints/
evals/
```

每次训练前必须检查：

- 模型路径。
- 数据 manifest。
- 数据 hash。
- tokenizer 和 chat template。
- train/eval split 是否泄漏。
- 权重是否会真实更新。
- SwanLab 是否可写。
- GPU/PPU 是否空闲。

每次训练后必须检查：

- loss 是否下降。
- 固定验证是否改善。
- 各 source loss 是否异常。
- 参数是否真实更新。
- 是否有 NaN/Traceback。
- benchmark 是否回退。

## 9. 已知问题与解决方案

### 9.1 BF16 小学习率静默无效

症状：loss 震荡，固定验证不变，权重 hash 变但参数几乎不动。

原因：BF16 参数无 FP32 master，更新小于 BF16 量化步长。

解决：

- LoRA adapter 保持 FP32。
- 全参训练用 FSDP/DeepSpeed FP32 master。
- 做参数变化 sentinel。

### 9.2 PPU 多卡 FP32 LoRA all-reduce 失败

症状：Hggc invalid argument / NCCL custom allreduce error。

解决：

- 单 PPU FP32 LoRA 并行做超参搜索。
- 多卡时用 BF16 梯度或平台支持的 FSDP。
- 不要强行 16 卡 DDP FP32 adapter。

### 9.3 no-tool 学不会

原因：`[]` 只有 2 token，token-level loss 权重太低。

解决：

- no-tool 样本加权。
- balanced batch。
- no-tool 输出可扩展为明确 decision JSON。
- 增加 hard negative。

### 9.4 clarify 和 no-tool 混淆

原因：两者都标成 `[]`。

解决：

- clarify 输出独立 action。
- verifier 独立统计 clarify accuracy。
- 缺参任务中禁止猜默认值。

### 9.5 内部指标虚高

原因：训练集和验证集共享工具 schema、server family 或模板。

解决：

- 按 fingerprint split。
- held-out server family。
- paraphrase 后仍保持 split。
- benchmark contamination scan。

## 10. 推荐 8 周学习与实验计划

### 第 1 周：基础闭环

- 实现 trace schema、reward、内部 eval。
- 用 1.5B 跑 LoRA smoke。
- 学会看 loss、grad、token/s、显存。

### 第 2 周：数据质量

- 构造 no-tool、clarify、hard negative。
- 建立 OOD split。
- 做数据 manifest 和污染检测。

### 第 3 周：SFT 主实验

- 3B/4B LoRA SFT。
- SwanLab 可视化。
- 固定验证 + 内部 MCP eval。

### 第 4 周：通用能力回归

- IFEval、GSM8K、MMLU-Pro、WikiSQL。
- 调整 replay 比例。

### 第 5 周：偏好优化

- 拒绝采样。
- DPO/SimPO 消融。
- 分析相似工具混淆。

### 第 6 周：RLVR

- 静态 verifier GRPO。
- sandbox reward。
- 监控 reward hacking。

### 第 7 周：公开 benchmark

- BFCL、xLAM、IFEval、WikiSQL。
- 尝试 tau2/AppWorld。
- SWE-bench 只在容器就绪后跑。

### 第 8 周：7B 迁移与模型卡

- 把最佳数据和配方迁移到 7B。
- 和同体量开源模型对齐。
- 输出模型卡、失败分析、下一轮数据计划。

## 11. 对原 README 方案的保留与修改

保留：

- SFT -> GRPO 主线。
- 可验证奖励。
- QLoRA/LoRA 降低成本。
- 工具调用评测。

修改：

- 19 tools/10K synthetic 仅作为入门，不作为正式训练规模。
- BFCL 风格评测不能替代官方 BFCL。
- task completion 奖励不能用关键词，必须尽量用执行状态。
- no-tool、clarify、safety 必须单独建模。
- 训练必须有固定验证、参数更新检查和通用能力回归。
- GRPO 只能在 SFT 成功率合适时启动。
- 所有实验必须接入 SwanLab 和 provenance。

## 12. 当前项目下一步建议

基于现有实验，下一步优先做：

1. 用 `mcp_sft_v2_seed60` 的最佳 LoRA 配方继续扩大数据。
2. 将 `lr=4e-5, rank=32` 作为当前 LoRA SFT 默认候选。
3. 对最佳 adapter 运行内部 MCP OOD、xLAM、IFEval、GSM8K、MMLU-Pro、WikiSQL。
4. 构造 hard negative 偏好对，启动 DPO。
5. 修复真实 MCP sandbox 环境，准备 RLVR。
6. 等容器就绪后再跑 SWE-bench，不提前声明 resolved rate。

## 13. 需要用户决策的问题

如果要继续推进，我建议你后续在以下问题上做选择：

1. 主力模型优先 4B 还是 7B。
2. 目标更偏 MCP/tool routing，还是代码/SQL/SWE agent。
3. 是否接受使用外部闭源教师生成数据。
4. 训练最终形态是 LoRA adapter、合并权重，还是全参模型。
5. 是否需要对接真实业务 MCP server。
