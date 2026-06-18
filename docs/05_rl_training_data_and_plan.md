# 5. RL 训练数据及方案

本文档描述 preference optimization 与 RLVR/GRPO 的数据、奖励、训练条件和实现路径。

## 阶段定位

SFT 解决“会不会写出正确动作”，RL 解决“在多个可行动作中是否更偏好成功轨迹”。RL 不应该过早启动。只有当 SFT 在内部 MCP OOD 上具备基本成功率时，RL 才有足够探索信号。

启动条件：

- JSON/schema valid rate 不低于 99.5%。
- 内部 MCP 任务成功率约 15%-60%。
- no-tool、clarify、positive 三类都有可测改善。
- 固定 validation loss 下降。
- 评测 runner 和 reward 稳定，无明显 reward hacking。

## 数据类型

| 数据 | 格式 | 用途 |
|---|---|---|
| Preference pairs | prompt/chosen/rejected/verifier | DPO/SimPO，修正选择边界 |
| RLVR prompts | prompt/verifier/reward_components | GRPO，在线采样并用 verifier 打分 |
| Failure traces | prompt/model_output/error_type | 定向补弱，生成 hard negative |
| Replay data | SFT rows | 防止 RL 后格式和通用能力坍塌 |

## Preference 数据样例

```json
{
  "id": "pref_route_0001",
  "prompt": "SYSTEM: ...\nMCP_SERVER_CATALOG:\n[calendar, email]\nUSER:\nSchedule project review for Friday.\nASSISTANT:",
  "chosen": "[{\"name\":\"calendar.create_event\",\"arguments\":{\"title\":\"project review\",\"date\":\"Friday\"}}]",
  "rejected": "[{\"name\":\"email.send\",\"arguments\":{\"subject\":\"project review\",\"body\":\"Friday\"}}]",
  "preference_type": "correct_tool_vs_similar_wrong_tool",
  "verifier": {
    "kind": "mcp_tool_call",
    "expected_calls": [
      {"name": "calendar.create_event", "arguments": {"title": "project review", "date": "Friday"}}
    ]
  }
}
```

## RLVR 数据样例

```json
{
  "id": "rlvr_mcp_0001",
  "prompt": "SYSTEM: ...\nUSER:\nSchedule project review for Friday.\nASSISTANT:",
  "verifier": {
    "kind": "mcp_tool_call",
    "expected_calls": [
      {"name": "calendar.create_event", "arguments": {"title": "project review", "date": "Friday"}}
    ],
    "permissions": {"calendar.write": true},
    "terminal_state": {"calendar_contains": "project review"}
  }
}
```

## Reward 设计

MCP reward 采用门控式：

| 分量 | 权重 | 说明 |
|---|---:|---|
| 格式正确 | gate | JSON parse/schema valid 是后续奖励前提 |
| 路由正确 | 0.10 | server/tool name 正确 |
| 参数正确 | 0.15 | required fields、类型、边界和语义正确 |
| 执行成功 | 0.20 | sandbox/真实 MCP 执行成功 |
| 最终任务完成 | 0.35 | 环境最终状态满足 hidden assertion |
| 错误恢复 | 0.10 | 超时、空结果、权限不足后的正确恢复 |
| 权限与安全 | 0.10 | 不越权，不执行危险操作 |

扣分项：

- 多余调用、重复调用。
- 猜测缺失参数。
- 调用不存在工具。
- 超长轨迹。
- 不可逆错误。
- 忽略提示注入或权限边界。

实现位置：

```text
agentic_rl/reward.py
scripts/remote/train_verifiable_grpo.py
```

## DPO/SimPO 方案

当前 DPO 入口：

```bash
torchrun --standalone --nproc_per_node=4 scripts/remote/train_mcp_dpo.py \
  --model output/mcp_lora_sft_v3_8ppu_20260618 \
  --dataset datasets/processed/mcp_lora_sft_v3_20260618/raw_all/train_preferences.jsonl \
  --output-dir output/mcp_dpo_v1 \
  --learning-rate 5e-7 \
  --beta 0.1
```

偏好对类型必须分桶汇报：

- 正确工具 vs 相似错误工具。
- 澄清 vs 猜参数。
- no-tool vs 过度调用工具。
- 安全拒绝 vs 越权调用。
- 恢复路径 vs 重复失败。

## GRPO/RLVR 方案

当前最小 GRPO 入口：

```bash
torchrun --standalone --nproc_per_node=4 scripts/remote/train_verifiable_grpo.py \
  --model output/mcp_lora_sft_v3_8ppu_20260618 \
  --dataset datasets/processed/mcp_lora_sft_v3_20260618/raw_all/train_rlvr.jsonl \
  --output-dir output/mcp_grpo_v1 \
  --steps 300 \
  --group-size 8 \
  --learning-rate 1e-7
```

中长期目标：

```text
verl + SGLang/vLLM + MCP sandbox
```

原因：

- 需要异步 rollout。
- 需要真实或模拟 MCP 环境。
- 需要按任务维护状态、权限、超时和 tool observation。
- 需要更稳定的 KL、group reward、reject sampling 和 checkpoint eval。

## 监控指标

RL 训练必须记录：

- reward mean/std。
- success rate。
- skipped/invalid group rate。
- JSON/schema valid rate。
- route/argument/execution/task/safety 各分量。
- 平均调用数、重复调用率、轨迹长度。
- KL 或 policy drift 指标。
- token/second、rollout latency、GPU utilization。
- IFEval/GSM8K/MMLU-Pro 回归。

## 风险与处理

| 风险 | 表现 | 处理 |
|---|---|---|
| 过早 RL | reward 无信号，输出乱 | 先 SFT 到 15%+ 内部成功率 |
| reward hacking | 格式很好但任务失败 | 最终环境状态作为主奖励，格式仅 gate |
| 只学会调用工具 | no-tool F1 下降 | 加 no-tool preference 和负奖励 |
| 猜参数 | 缺字段仍调用 | 澄清样本和澄清 reward 单独统计 |
| RL 导致通用退化 | IFEval/GSM8K/MMLU 下降 | 混入 replay，设置 KL 和回归门槛 |
| 模拟器偏差 | sandbox 高分，真实 MCP 失败 | 引入真实 MCP 回放和失败响应 |
| 长轨迹刷分 | 调用越来越多 | 调用数、长度和重复调用扣分 |
