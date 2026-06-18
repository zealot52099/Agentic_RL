# 4. SFT 数据及方案

本文档定义 MCP/tool SFT 的数据格式、训练顺序、实现脚本、当前运行方案和验收门槛。

## 训练目标

SFT 先解决“会不会正确表达动作”的问题：

- 在多个 MCP Server 和大量候选工具中选择正确 server/tool。
- 判断何时不需要工具，输出 `[]`。
- 缺少必要参数时澄清，不猜参数。
- 生成符合 JSON Schema 的参数。
- 看见 tool result 后继续、修正、重试或终止。
- 在安全和权限边界下拒绝或请求确认。

## 数据配比

| 数据类型 | 建议占比 | 当前 smoke 状态 |
|---|---:|---|
| MCP positive tool-call | 50%-70% | 已有 |
| no-tool | 8%-12% | 已有 |
| clarification | 8%-12% | 已有 |
| error recovery | 5%-10% | 待扩展 |
| safety/security | 5%-8% | 待扩展 |
| general replay | 10%-15% | 数据源已准备，当前 MCP v3 首轮未大量混入 |

当前训练数据：

```text
/workspace/yans2@xiaopeng.com/agentic_rl_pipeline/datasets/processed/mcp_lora_sft_v3_20260618/sft_v2_all/train_sft.jsonl
```

验证数据：

```text
/workspace/yans2@xiaopeng.com/agentic_rl_pipeline/datasets/processed/mcp_lora_sft_v3_20260618/sft_v2_all/validation_sft.jsonl
```

## 处理前样例

```json
{
  "id": "mcp_0001",
  "prompt": "SYSTEM: Return [] when no tool applies or required information is missing.\nMCP_SERVER_CATALOG:\n[...]\nUSER:\nUse calendar.create_event for me.\nASSISTANT:",
  "completion": "[]",
  "verifier": {
    "decision": "clarify",
    "expected_calls": []
  }
}
```

问题：clarify 与 no-tool 都是 `[]`，模型无法学会追问。

## 处理后样例

```json
{
  "id": "mcp_0001",
  "mixture_source": "mcp_clarify",
  "prompt": "SYSTEM: Return [] when no tool applies. When required information is missing, return one JSON object with action=\"clarify\", a missing field-name array, and a concise message. Do not guess required arguments.\nMCP_SERVER_CATALOG:\n[...]\nUSER:\nUse calendar.create_event for me.\nASSISTANT:",
  "completion": "{\"action\":\"clarify\",\"missing\":[\"title\",\"date\"],\"message\":\"Please provide the required information: title, date.\"}",
  "loss_weight": 2.0,
  "prompt_tokens": 621,
  "completion_tokens": 24
}
```

## 实现脚本

| 脚本 | 作用 |
|---|---|
| `scripts/remote/prepare_mcp_sft_v2.py` | 修复 clarify/no-tool、固定验证集、token 长度过滤、source loss weight |
| `scripts/remote/train_lora_sft.py` | 当前主力 LoRA SFT 训练器，支持 SwanLab |
| `scripts/remote/train_assistant_only_sft.py` | 全参/普通 SFT 训练器，含全局 supervised-token loss 修正 |
| `scripts/remote/evaluate_sft_loss.py` | 固定验证 loss 评估 |
| `scripts/remote/capture_run_provenance.py` | 记录代码、环境、数据和命令 provenance |
| `scripts/remote/run_mcp_lora_v2_queue_20260616.sh` | LoRA v2 并行消融历史脚本 |
| `scripts/remote/run_mcp_ppu16_queue_20260615.sh` | 16 PPU MCP SFT 历史脚本 |

## 当前推荐训练命令

```bash
torchrun --standalone --nproc_per_node=8 scripts/remote/train_lora_sft.py \
  --model /publicdata/huggingface.co/Qwen/Qwen2.5-1.5B-Instruct \
  --train-data datasets/processed/mcp_lora_sft_v3_20260618/sft_v2_all/train_sft.jsonl \
  --output-dir output/mcp_lora_sft_v3_8ppu_20260618 \
  --steps 1500 \
  --seq-len 2048 \
  --micro-batch-size 1 \
  --grad-accum-steps 4 \
  --learning-rate 3e-5 \
  --warmup-steps 30 \
  --lora-r 32 \
  --lora-alpha 64 \
  --lora-dropout 0.05 \
  --swanlab-project agentic-rl-tool-calling \
  --swanlab-run-name mcp_lora_sft_v3_8ppu_20260618
```

## 日志与可视化

必须保存：

```text
logs/<run_name>.log
output/<run_name>/train_metrics.jsonl
output/<run_name>/metadata.json
output/<run_name>/checkpoint-*/
docs/experiments/<date>-<run_name>.md
```

SwanLab 必须记录：

- `loss`
- `loss_ema`
- `step_loss`
- `grad_norm`
- `mean_grad_norm`
- `clipped_step_rate`
- `lr`
- `supervised_tokens_per_second`
- `max_memory_gib`

## 验收门槛

进入 DPO/RL 前至少满足：

- 固定 validation loss 下降，而不只是训练 loss 下降。
- JSON parse rate 不低于 99.5%。
- 工具幻觉率低于 2%。
- no-tool F1 和 clarification accuracy 明显优于 base。
- IFEval/GSM8K/MMLU-Pro 回退不超过 2 个百分点。
- 参数变化 sentinel 显示 LoRA 或全参确实发生有效更新。

## 主要坑

| 问题 | 表现 | 解决方案 |
|---|---|---|
| clarify 与 no-tool 混淆 | 模型只输出 `[]` | clarify 使用显式 JSON action |
| 短 completion 权重过低 | no-tool 学不会 | 对 no-tool 使用较高 `loss_weight` |
| BF16 全参小 LR 无效 | loss 看似震荡，参数几乎没变 | LoRA FP32 adapter 或 FSDP FP32 master params |
| per-example loss 误导 | 两 token 样本和长答案等权 | 使用 supervised-token mean 与 source weight |
| 只看训练 loss | 过拟合或数据混合导致错判 | 固定 validation split，每个 checkpoint 评估 |
| 16 卡通信/显存问题 | PCCL/NCCL timeout 或 OOM | 先 8 PPU 稳定跑，必要时缩 seq_len/batch |
