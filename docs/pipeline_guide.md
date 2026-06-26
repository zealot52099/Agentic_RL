# Agentic RL Pipeline Guide

本文档记录当前 `agentic_rl_pipeline` 的可执行路线。目标是在尚无业务数据前，先用开源数据统一成面向 Data Agent 的生产输出格式，提升工具调用、SQL 生成、澄清/拒答和安全边界能力。

## 2026-06-26 Phase5: Unified Data Agent SFT

### 目标

Phase5 不再分别优化 BFCL/xLAM/Spider/WikiSQL 的原始输出格式，而是统一到 Data Agent 运行时更容易消费的 action schema：

```json
{"action":"tool_call","calls":[{"name":"execute_sql","arguments":{"database":"spider:<db_id>","sql":"SELECT ..."}}]}
{"action":"tool_call","calls":[{"name":"mcp_server.tool_name","arguments":{"key":"value"}}]}
{"action":"no_tool","calls":[]}
{"action":"clarify","missing":["table_schema"],"message":"需要补充数据库表结构后才能生成可靠 SQL。"}
```

这样训练目标和后续生产解析器一致；官方 benchmark 仍单独保留原始格式评测，不能和内部统一格式指标混算。

### 数据位置

生成脚本：

```text
scripts/build_phase5_unified.py
```

远端输出：

```text
/workspace/yans2@xiaopeng.com/agentic_rl_pipeline/datasets/processed/phase5_unified_20260626/train.jsonl
/workspace/yans2@xiaopeng.com/agentic_rl_pipeline/datasets/processed/phase5_unified_20260626/validation.jsonl
/workspace/yans2@xiaopeng.com/agentic_rl_pipeline/datasets/processed/phase5_unified_20260626/audit_examples.jsonl
/workspace/yans2@xiaopeng.com/agentic_rl_pipeline/datasets/processed/phase5_unified_20260626/manifest.json
```

当前 manifest 摘要：

```json
{
  "schema": "data_agent_action_v1",
  "counts": {"all": 15500, "train": 14962, "validation": 538},
  "action_counts": {"tool_call": 13234, "no_tool": 1105, "clarify": 1161},
  "source_plan": {"phase3": 6500, "mcp": 3500, "parallel": 2000, "spider": 3500}
}
```

主要 mixture 包括：MCP 正样本、Spider SQL、WikiSQL/BFCL 风格 SQL、标准函数调用、并行工具调用、no-tool、clarify。Spider 当前本地文件缺少 `tables.json`/schema，因此只能训练 `execute_sql` 输出格式和 SQL 表达式，不能充分训练 schema linking。

### 当前训练方案

优先目标是先跑通稳定训练，再恢复 7B LoRA/QLoRA：

```text
model: /publicdata/huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct
script: /workspace/yans2@xiaopeng.com/agentic_rl_pipeline/scripts/train_phase5_full_sft.py
run: phase5_unified_coder15b_full_fp32_lr2e7_seed20260626_20260626_125859
output: /workspace/yans2@xiaopeng.com/agentic_rl_pipeline/output/phase5_unified_coder15b_full_fp32_lr2e7_seed20260626_20260626_125859
log: /workspace/yans2@xiaopeng.com/agentic_rl_pipeline/logs/phase5_unified_coder15b_full_fp32_lr2e7_seed20260626_20260626_125859.log
steps: 300
seq_len: 2048
micro_batch_size: 1
grad_accum_steps: 8
learning_rate: 2e-7
warmup_steps: 50
dtype: fp32
attention: sdpa
optimizer: AdamW non-fused
swanlab: local mode
```

SwanLab cloud 当前缺少 API key。local dashboard 可用：

```bash
swanlab watch /workspace/yans2@xiaopeng.com/agentic_rl_pipeline/output/phase5_unified_coder15b_full_fp32_lr2e7_seed20260626_20260626_125859/swanlab
```

### 启动前资源处理

本次启动前已终止占用 GPU 的旧 VLLM 进程：

```text
25922 25923 25924 25925
```

之后又终止了多次卡住的 Phase5 smoke run，避免 D 状态残留占用设备。

### 已遇到的问题

1. 7B LoRA 和 3B LoRA 在权重加载后卡住，没有写出 metrics。
   - 现象：Python 子进程进入 `D` 状态，GPU 仅占用少量显存。
   - 根因倾向：远端系统 Python 的 torch 栈被 pip 覆盖到 `torch 2.11.0+cu130`，而历史成功训练记录使用的是 `torch 2.8.0+ali.9.ppu2.0.0.cu129`。
   - 处理：停止继续使用 PEFT 路径硬试，先启用不依赖 PEFT 的 1.5B 全参 fallback。

2. CPU 加载后 `.to(cuda)` 路径卡住。
   - 现象：1.5B 全参 BF16 也卡在模型加载后。
   - 处理：改为 `from_pretrained(device_map={"": 0})` 直接加载到 GPU，并将 attention 切到 `sdpa`。

3. SwanLab cloud 未配置 API key。
   - 现象：`swanlab.init(mode="cloud")` 报 `api key not configured (no-tty)`。
   - 处理：安装 `swanlab[dashboard]`，先用 `mode="local"` 保存可视化日志。要接入云端，需要在 job 中执行 `swanlab.login(api_key=...)` 或配置相应环境变量。

4. BF16 全参数训练 step 2 NaN。
   - 现象：step 1 loss 有限但 grad_norm 为 NaN，step 2 loss 变 NaN。
   - 处理：改为 FP32 参数、关闭 autocast、关闭 fused optimizer、学习率降到 `2e-7`。

### 当前健康信号

稳定版已写出 metrics：

```json
{"step":1,"loss":0.783116240054369,"grad_norm":37.63218688964844,"lr":8e-09}
{"step":10,"loss":0.6331170462071896,"grad_norm":35.20143127441406,"lr":4.4e-08}
{"step":20,"loss":0.860849391669035,"grad_norm":26.396486282348633,"lr":8.4e-08}
```

这只是训练稳定性指标，不是最终效果指标。完成后需要跑内部统一格式 eval 和官方格式 eval，并明确标注来源。

### 后续计划

1. 等待 1.5B FP32 fallback 完成 300 step，检查 loss 曲线、JSON/action 格式、SQL/tool call 小样本。
2. 若稳定，将相同 direct-load/no-fused 经验迁移到 3B 或恢复 LoRA 环境。
3. 恢复 7B 推荐先重置 job 镜像或安装与平台匹配的 torch wheel，再重新跑 LoRA SFT。
4. 配置 SwanLab cloud API key 后，将 `SWANLAB_MODE` 切回 `cloud`。
5. 评测必须拆分：
   - 内部统一格式：JSON valid、action accuracy、tool name/args、SQL exact/execution、no-tool/clarify F1。
   - 官方格式：BFCL、IFEval、GSM8K、MMLU-Pro、WikiSQL/Spider 等，单独报告。
