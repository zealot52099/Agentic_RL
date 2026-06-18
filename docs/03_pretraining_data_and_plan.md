# 3. 预训练数据及方案

本文档解释本项目如何学习和验证预训练原理，以及在资源有限条件下如何设计 CPT/mid-training。当前主线是后训练，预训练只做小规模理解和能力补强，不做从零训练。

## 定位

预训练阶段的目标：

- 掌握 next-token prediction、token mix、长上下文、checkpoint、恢复和 loss 曲线解释。
- 验证 TorchTitan/FSDP 等基础训练栈。
- 为 Base 模型做小规模 CPT/mid-training 消融。
- 不把单一工具调用数据直接当作大比例 CPT 数据，避免通用能力遗忘。

## 数据类别

| 数据 | 用途 | 建议比例 |
|---|---|---:|
| 通用文本/指令混合 | 保持语言和指令能力 | 50%-70% |
| 代码/SQL/终端文本 | 提升工具参数、代码和 SWE 能力 | 15%-25% |
| 工具文档/API schema | 学习工具语义、字段描述和 JSON schema | 5%-15% |
| Agent trace 序列化文本 | 学习 observation/action 模式 | 5%-10% |
| 数学/推理 | 保持 GSM8K/MMLU-Pro | 5%-10% |

## 当前本地可用数据

```text
datasets/sources/tulu-3-sft-mixture
datasets/sources/tulu-personas-if
datasets/sources/smol-smoltalk
datasets/sources/no-robots
datasets/processed/xlam-function-calling-60k
datasets/processed/swe-gym-openhands-sft
```

远端数据以 `/workspace/yans2@xiaopeng.com/agentic_rl_pipeline/datasets` 为准。

## 处理前样例

通用指令数据通常是 message 格式：

```json
{
  "messages": [
    {"role": "user", "content": "Explain why gradient clipping is useful."},
    {"role": "assistant", "content": "Gradient clipping limits the norm..."}
  ],
  "source": "tulu"
}
```

工具/API 文档可序列化为普通文本：

```json
{
  "server": "calendar",
  "tool": "calendar.create_event",
  "description": "Create a calendar event.",
  "schema": {
    "required": ["title", "date"],
    "properties": {
      "title": {"type": "string"},
      "date": {"type": "string"}
    }
  }
}
```

## 处理后样例

CPT/mid-training 数据可以统一成纯文本或 packed token stream：

```json
{
  "id": "cpt_api_doc_calendar_0001",
  "text": "MCP Server: calendar\nTool: calendar.create_event\nDescription: Create a calendar event.\nRequired arguments: title, date.\nExample: ...",
  "source": "api_schema_doc",
  "token_count": 128
}
```

## 实现脚本与配置

| 路径 | 作用 |
|---|---|
| `configs/torchtitan/qwen3_0.6b_smoke.toml` | TorchTitan 0.6B 随机初始化 smoke |
| `configs/torchtitan/qwen3_1.7b_base_single_gpu_smoke.toml` | Qwen3-1.7B 单卡真实权重 smoke |
| `configs/torchtitan/qwen3_1.7b_base_4gpu_smoke.toml` | Qwen3-1.7B 多卡 smoke |
| `configs/torchtitan/qwen3_1.7b_cpt_4gpu.toml` | CPT/mid-training 配置 |
| `scripts/remote/run_torchtitan_smoke.sh` | 远端 TorchTitan smoke 启动 |
| `scripts/remote/stage_qwen3_1.7b_base.sh` | Qwen3-1.7B Base 权重 staging |
| `scripts/remote/finalize_and_validate_hf_export.py` | DCP/HF 导出与验证 |

## 训练框架

推荐分工：

- CPT/mid-training：TorchTitan，关注 FSDP2、checkpoint、吞吐和恢复。
- SFT/LoRA：Transformers + PEFT/DDP 或 TRL。
- Agentic RL：verl/SGLang 是终局方向；当前先用项目内 GRPO smoke。

## 最小实验

1. 0.6B random-init，5 steps：验证 tokenizer、forward/backward、checkpoint。
2. 1.7B pretrained，20 steps：验证 HF 权重导入、loss 合理性、恢复。
3. 1.7B pretrained，500 steps：验证多卡吞吐和稳定性。
4. 固定 1B token pilot：比较数据配方，不追求最终指标。

## 验收标准

- step 0 loss 与 Transformers 前向交叉检查。
- 训练 20 step 后保存、销毁进程、恢复继续。
- 定期导出 HF checkpoint 并完成固定 prompt 推理。
- IFEval/GSM8K/MMLU-Pro 不发生明显回退。
- 数据配方、token 数、checkpoint hash、代码 revision 全部记录。

## 风险与处理

| 风险 | 表现 | 处理 |
|---|---|---|
| 把 random-init smoke 当有效训练 | loss 下降但能力归零 | 明确标注 smoke，正式实验强制记录 base checkpoint |
| 工具数据比例过高 | 通用能力下降 | Agent 数据 CPT 占比先控制在 5%-10% |
| 训练框架和 SFT 模板不一致 | 下游 tool-call 格式不稳定 | CPT 只学习文本分布，SFT 固定 chat template |
| 远端无法下载权重 | job 内拉取失败 | 本地下载、SHA256 校验、SCP 上传 |
| checkpoint 只能恢复不能推理 | DCP/HF 格式混淆 | 每个阶段做 HF export 验收 |
