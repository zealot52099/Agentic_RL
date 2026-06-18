# MCP Agent 后训练实施与运行手册

## 1. 已实现范围

本次实现将 MCP 后训练从方案变成了可执行的最小闭环：

- `agentic_rl/schema.py`：版本化 Agent Trace 校验，保留 server catalog、
  tool schema、call ID、原始结果、环境状态、权限、终止状态和奖励证据。
- `agentic_rl/fingerprint.py`：忽略名称和描述的 schema 指纹、工具家族 OOD
  切分及污染检查。
- `agentic_rl/sandbox.py`：确定性状态沙箱，覆盖文件、日历、邮件、数据库和
  Git 操作，支持权限不足、超时和状态回滚。
- `agentic_rl/reward.py`：格式门控的路由、参数、执行、任务、恢复和安全奖励。
- `agentic_rl/pipeline.py`：trace 合成、SFT/DPO/RLVR 导出和内部指标汇总。
- `scripts/remote/evaluate_mcp_internal.py`：本地 HF 模型推理与内部 OOD 评测。
- `scripts/remote/train_mcp_dpo.py`：MCP 偏好优化入口。
- 现有 `train_verifiable_grpo.py` 已支持 `mcp_tool_call` verifier。

当前沙箱和模板是训练基础设施 smoke test，不是计划中 20 万 turns 的正式语料。
正式数据必须经同一 schema 和 verifier 打包，不能简单复制模板扩量。

## 2. 数据生产

### 2.1 Trace 合同

每条正式轨迹必须满足 `mcp_agent_trace_v1`，并包含：

- `split_group_ids`：目标操作的稳定语义家族，不使用展示名称。
- `server_catalog`：该回合真实可见的服务器和工具。
- `messages`：完整多轮上下文，tool result 必须通过 call ID 对齐。
- `environment_state`：可重放快照或快照引用及哈希。
- `permissions`：已授权权限、确认策略和不可逆操作标记。
- `verifier_results`：执行、最终状态、恢复和安全判定证据。
- `generation_metadata`：教师 revision、温度、seed、parser 和环境版本。

教师输出必须经过三层过滤：

1. JSON Schema 与 call ID 校验。
2. 沙箱或真实 MCP Server 执行。
3. 隐藏最终状态断言；不能用教师自评代替。

正式目标为至少 60 个服务器、500 个工具、8 万条多轮轨迹。推荐批次：

| 批次 | turns | 目的 |
|---|---:|---|
| Gold smoke | 2 万 | 格式、路由、no-tool、澄清 |
| Multi-server | 7 万 | 干扰工具、重名、跨服务器组合 |
| Recovery | 2 万 | 超时、空结果、失效工具、部分成功 |
| Safety | 1 万 | 权限、确认、注入、不可逆操作 |
| General replay | 2 万 | 指令、推理、代码能力保持 |
| Student failures | 6 万 | 基于当前 checkpoint 的定向补弱 |

将生产 trace 打包：

```bash
python scripts/mcp_agent_pipeline.py package-traces \
  --input datasets/raw/mcp_traces.jsonl \
  --output-dir datasets/processed/mcp_agent_v1 \
  --holdout-percent 20
```

打包命令发现工具家族或 task ID 泄漏会失败。BFCL、LiveMCPBench、MCP-Atlas、
tau2-bench、AppWorld 和 SWE-bench 测试数据不得进入该输入。

## 3. 训练顺序

### 3.1 统一基线

确认三个模型目录后运行：

```bash
bash scripts/remote/run_mcp_baseline_eval.sh
```

比较 Qwen3.5-4B、Qwen3-4B-Instruct-2507 和 xLAM-2-3B-fc-r。内部结果必须标记
为 `internal_probe_not_official_benchmark`，不能冒充 BFCL 或 MCP-Atlas 分数。

### 3.2 SFT

先用 2 万条 gold 数据运行 100–200 step smoke，再扩展到完整数据。当前训练器是
BF16 全参数 DDP；4B 模型长序列正式训练建议迁移到 FSDP/DeepSpeed。

```bash
MODEL=/publicdata/huggingface.co/Qwen/Qwen3-4B-Instruct-2507 \
GPU_COUNT=4 SFT_STEPS=200 SEQ_LEN=8192 \
bash scripts/remote/run_mcp_agent_smoke.sh
```

当前 `bifrost-2026051921173700-yans2` 已有 Qwen3-4B-Instruct-2507，但截至
2026-06-15 未发现 Qwen3.5-4B。因此脚本默认使用兼容底座；Qwen3.5 权重就绪后通过
`MODEL` 和 `RUN_NAME` 环境变量切换，不能把两个底座的结果混在同一实验目录。

只有内部 OOD 的 JSON valid ≥99.5%、工具幻觉率 <2%，且任务成功率达到 15%，
才进入偏好优化或 RL。低于门槛时先增加失败定向 SFT。

### 3.3 DPO

```bash
torchrun --standalone --nproc_per_node=4 scripts/remote/train_mcp_dpo.py \
  --model runs/qwen35_mcp_sft/hf \
  --dataset datasets/processed/mcp_agent_v1/train_preferences.jsonl \
  --output-dir runs/qwen35_mcp_dpo \
  --learning-rate 5e-7 --beta 0.1
```

偏好对分为正确工具/近邻干扰、澄清/猜参数、拒绝/越权和恢复/重复失败四类。
每类单独报告胜率，避免大量简单路由对掩盖安全回归。

### 3.4 RLVR

小规模验证可以复用仓库 GRPO 脚本：

```bash
torchrun --standalone --nproc_per_node=4 \
  scripts/remote/train_verifiable_grpo.py \
  --model runs/qwen35_mcp_dpo \
  --dataset datasets/processed/mcp_agent_v1/train_rlvr.jsonl \
  --output-dir runs/qwen35_mcp_grpo \
  --steps 300 --group-size 8 --learning-rate 1e-7
```

该脚本适合静态 next-action RLVR。正式多轮在线训练使用
`verl + SGLang + MCP sandbox`，每次 action 后由环境返回 observation 并更新状态。
环境错误样本必须从 policy loss 排除，不能当作模型 0 reward。

课程池保持 20%–80% 成功率。持续出现全对/全错组时调整难度或采样温度，
不要靠增大 group size 硬撑。

## 4. 评测与发布门槛

内部 OOD：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/remote/evaluate_mcp_internal.py \
  --model runs/qwen35_mcp_sft/hf \
  --traces datasets/processed/mcp_agent_v1/traces/ood_eval.jsonl \
  --output-dir evals/qwen35_mcp_sft/mcp_internal \
  --model-label qwen35_mcp_sft
```

正式发布必须同时运行：

- BFCL V4：AST、执行、多调用、irrelevance。
- LiveMCPBench 或 MCP-Atlas：海量工具路由和跨服务器完成率。
- tau2-bench、AppWorld：多轮状态任务。
- MCPSecBench 与内部攻击集：越权、注入和危险调用。
- IFEval、GSM8K、MMLU-Pro、HumanEval+/MBPP+/LiveCodeBench：能力回归。
- SWE-bench Verified/Lite：扩展软件工程能力；需可复现容器环境。

晋级条件：

- 内部 JSON/schema valid ≥99.5%，工具幻觉率 <2%，no-tool F1 ≥90%。
- 权限违规率 <1%，不可逆错误必须为零或经过明确用户确认。
- BFCL、MCP 路由和 tau2 相对基线分别达到计划提升。
- 通用指标任一回退超过 2 个百分点即阻断发布。

所有生成式评测至少使用 3 个 seed，保存逐题预测，报告均值、标准差和置信区间。
官方公开数字、官方代码复测和内部 probe 在表格中使用不同标签。

## 5. 尚需外部资源

- Qwen3.5-4B 与 xLAM-2 模型权重需在 job 上存在或上传。
- LiveMCPBench、MCP-Atlas、AppWorld 和 MCPSecBench 尚未纳入当前本地资产。
- SWE-bench Verified 需要 Docker 或等价隔离执行环境；没有容器时不能声称 resolved。
- 正式 `verl + SGLang` 在线环境需要独立部署，当前仓库实现的是其数据、验证和
  reward 合同，以及静态 GRPO smoke 路径。

## 6. 日志与环境依赖

每个正式训练目录必须保存：

- `train.log`：所有 rank 的完整 stdout/stderr。
- `train_metrics.jsonl`：step、loss、梯度、学习率、吞吐、显存等结构化指标。
- `training_config.json`：模型、数据、超参数、world size 和运行时版本。
- `provenance.json`：脱敏后的环境、依赖、设备信息及数据/模型/脚本 SHA-256。
- `exit_code`、`logs.sha256` 和 `logs_bundle.tar.gz`：退出状态及最终日志归档。

校验对容器的依赖分为三类：

| 校验 | 容器依赖 |
|---|---|
| Trace schema、JSON、参数 exact、污染检查、静态 reward | 不依赖容器，纯 Python 可运行 |
| 模型生成、logprob、BFCL 静态类别、IFEval、MMLU-Pro | 不强制容器，但必须固定模型运行时、parser 和依赖版本 |
| MCP 真实执行、tau2-bench、AppWorld、代码执行 | 依赖可重置环境；推荐或要求容器隔离 |
| SWE-bench Verified/Lite resolved rate | 依赖对应仓库镜像和 Docker/等价容器运行时 |
| 权限、注入和危险操作安全评测 | 必须在沙箱或最小权限测试服务中运行，不得连接生产账户 |

内部状态沙箱可以脱离容器运行，但它只能验证确定性状态逻辑，不能替代真实 MCP
Server 的网络、认证、超时和副作用测试。
