# Agentic RL 任务编排 + 工具调用 — 完整使用手册

> **场景**: 任务编排并准确调用工具  
> **方案**: SFT（冷启动）→ GRPO（强化学习优化）  
> **基座模型**: Qwen2.5-7B-Instruct（Apache 2.0，原生函数调用支持）  
> **框架**: TRL + vLLM + PyTorch

---

## 目录

1. [环境准备](#1-环境准备)
2. [项目结构](#2-项目结构)
3. [快速开始（5 步上手）](#3-快速开始)
4. [第一步：生成训练数据](#4-第一步生成训练数据)
5. [第二步：SFT 冷启动训练](#5-第二步sft-冷启动训练)
6. [第三步：GRPO 强化学习训练](#6-第三步grpo-强化学习训练)
7. [第四步：模型评测](#7-第四步模型评测)
8. [第五步：模型推理与部署](#8-第五步模型推理与部署)
9. [完整流程一键运行](#9-完整流程一键运行)
10. [调参与优化指南](#10-调参与优化指南)
11. [常见问题](#11-常见问题)

---

## 1. 环境准备

### 1.1 硬件要求

| 训练阶段 | 最低 GPU | 推荐 GPU | 预计时间 |
|----------|----------|----------|----------|
| 数据生成 | CPU | — | ~5 分钟 |
| SFT (LoRA) | 1× RTX 4090 24GB | 1× A100 40GB | 2–4 小时 |
| GRPO (LoRA) | 1× A100 40GB | 1× A100 80GB | 4–8 小时 |
| 评测 | 1× RTX 4090 24GB | 1× A100 40GB | 30–60 分钟 |

### 1.2 软件安装

```bash
# 克隆项目
git clone https://github.com/zealot52099/Agentic_RL-.git
cd Agentic_RL-

# 创建虚拟环境
conda create -n agentic_rl python=3.10 -y
conda activate agentic_rl

# 安装依赖
pip install -r requirements.txt

# (可选) 安装 Flash Attention 2 加速训练
pip install flash-attn --no-build-isolation
```

### 1.3 登录 WandB（可选）

```bash
wandb login
```

---

## 2. 项目结构

```
Agentic_RL-/
├── README.md                      # 项目说明
├── USAGE_MANUAL.md                # 本手册
├── requirements.txt               # Python 依赖
│
├── configs/                       # 配置文件
│   ├── model_config.yaml          # 模型选型配置
│   ├── data_config.yaml           # 数据流水线配置
│   └── grpo_config.yaml           # GRPO 训练超参数
│
├── src/                           # 核心源码
│   ├── data/
│   │   ├── tool_schema.py         # 工具 Schema 定义（19 种工具）
│   │   ├── synthetic_data.py      # 合成训练数据生成
│   │   ├── data_cleaner.py        # 数据清洗与质量过滤
│   │   └── reward_funcs.py        # 多维度可验证奖励函数
│   │
│   ├── train/
│   │   ├── sft_trainer.py         # SFT 冷启动训练（QLoRA）
│   │   └── grpo_trainer.py        # GRPO 强化学习训练
│   │
│   └── eval/
│       ├── tool_accuracy.py       # 工具调用精度评测
│       ├── bfcl_eval.py           # BFCL 风格评测
│       └── benchmark_runner.py    # 评测运转器
│
├── scripts/                       # 运行脚本
│   ├── run_data_pipeline.py       # 数据流水线
│   ├── run_train.py               # 训练流水线
│   └── run_eval.py                # 评测脚本
│
├── tests/                         # 单元测试
├── examples/                      # 示例 Notebook
└── data/                          # 生成的数据（自动创建）
```

---

## 3. 快速开始

5 步完成从数据到模型的完整流程：

```bash
# Step 1: 生成训练数据 (5 分钟)
python scripts/run_data_pipeline.py

# Step 2: SFT 冷启动训练 (2-4 小时)
python scripts/run_train.py --stage sft

# Step 3: GRPO 强化学习优化 (4-8 小时)
python scripts/run_train.py --stage grpo --grpo_model ./output/sft_model

# Step 4: 评测模型 (30 分钟)
python scripts/run_eval.py --model_path ./output/grpo_model --eval_data ./data/test_raw.jsonl

# Step 5: 交互式测试
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('./output/grpo_model', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained('./output/grpo_model', device_map='auto', torch_dtype='auto', trust_remote_code=True)

prompt = 'Search for the latest AI news, calculate the average sentiment score, and email the report to team@company.com'
messages = [{'role': 'user', 'content': prompt}]
inputs = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors='pt', add_generation_prompt=True).to(model.device)
outputs = model.generate(inputs, max_new_tokens=2048, temperature=0.1)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
"
```

---

## 4. 第一步：生成训练数据

### 4.1 数据流水线概述

```
工具 Schema 库 (19 tools)
        │
        ▼
场景模板匹配 (3 难度: easy/medium/hard)
        │
        ▼
合成器生成 10,000 条工具调用轨迹
        │
        ▼
数据清洗 (去重/JSON校验/难度均衡)
        │
        ▼
训练集 80% / 验证集 10% / 测试集 10%
        │
        ▼
双格式输出: SFT格式 + GRPO格式
```

### 4.2 自定义数据参数

编辑 `configs/data_config.yaml` 或直接修改 `scripts/run_data_pipeline.py`：

```python
# 修改生成数量
NUM_SAMPLES = 20000  # 默认 10000

# 修改难度分布
DIFFICULTY_DIST = {
    "easy": 0.1,    # 减少简单场景
    "medium": 0.4,
    "hard": 0.5,    # 增加复杂编排场景
}

# 修改工具类别（在 data_config.yaml 中）
tools:
  categories:
    - "search"
    - "calculator"
    - "database"
    - "api_call"
    - "file_ops"
    - "code_exec"
    - "email"
    - "calendar"
```

### 4.3 添加自定义工具

编辑 `src/data/tool_schema.py`，在 `TOOL_LIBRARY` 字典中添加新工具：

```python
"my_custom_tool": ToolSchema(
    name="my_custom_tool",
    description="Description of what this tool does",
    category="api_call",
    parameters=[
        ToolParameter("param1", "string", "First parameter"),
        ToolParameter("param2", "integer", "Second parameter", required=False),
    ],
),
```

### 4.4 生成输出

```
data/
├── train_raw.jsonl        # 训练集原始数据 (8,000 条)
├── val_raw.jsonl          # 验证集 (1,000 条)
├── test_raw.jsonl         # 测试集 (1,000 条)
├── train_sft.jsonl        # SFT 训练格式 (chat messages)
├── val_sft.jsonl          # SFT 验证格式
├── test_sft.jsonl         # SFT 测试格式
├── train_grpo.jsonl       # GRPO 训练格式 (prompts + expected tools)
├── val_grpo.jsonl         # GRPO 验证格式
├── test_grpo.jsonl        # GRPO 测试格式
└── tool_library.json      # 工具库参考文件
```

---

## 5. 第二步：SFT 冷启动训练

### 5.1 训练目标

在 GRPO 之前，先用 8,000 条工具调用示例进行监督微调，让模型：
- 学会工具调用的 JSON 输出格式
- 理解任务编排的基本模式
- 建立多步工具调用的能力基线

### 5.2 启动 SFT 训练

```bash
python scripts/run_train.py --stage sft \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --sft_data ./data/train_sft.jsonl \
    --sft_val_data ./data/val_sft.jsonl \
    --sft_output_dir ./output/sft_model \
    --sft_epochs 3 \
    --sft_lr 2e-4 \
    --batch_size 4 \
    --max_seq_length 4096
```

### 5.3 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_name_or_path` | Qwen/Qwen2.5-7B-Instruct | 基座模型（可选 Aura-7b, Qwen3-8B-LAM-v4） |
| `--sft_epochs` | 3 | 训练轮数（2-5 推荐） |
| `--sft_lr` | 2e-4 | 学习率（LoRA 通常 1e-4 到 5e-4） |
| `--batch_size` | 4 | 每卡批量（24GB 显存用 4，40GB 可用 8） |
| `--max_seq_length` | 4096 | 最大序列长度 |

### 5.4 监控训练

```bash
# 启动 TensorBoard
tensorboard --logdir ./output/sft_model
```

关注指标：
- **Loss 下降曲线**：应平滑下降，无剧烈震荡
- **Eval Loss**：与 Train Loss 差距不应持续扩大（否则过拟合）

### 5.5 预期结果

| 指标 | SFT 前 | SFT 后 |
|------|--------|--------|
| JSON 格式输出率 | ~30% | ~85% |
| 正确工具选择率 | ~20% | ~60% |
| 参数填充准确率 | ~15% | ~50% |

---

## 6. 第三步：GRPO 强化学习训练

### 6.1 为什么 GRPO 优于 PPO/DPO

- **PPO**：需要额外训练 Critic 网络（内存 ×2），不适合 7B 单卡场景
- **DPO**：无探索能力，无法发现更优的工具调用策略
- **GRPO**：组内相对优势，无需 Critic，直接利用可验证奖励（JSON 合法性、工具存在性），**最适合工具调用场景**

### 6.2 奖励函数设计

GRPO 使用四维可验证奖励，每个生成样本的 8 个回复互相比较：

```
R_total = 0.45 × R_tool_accuracy    # 工具名称 + 参数正确
        + 0.20 × R_json_validity    # 输出是合法 JSON
        + 0.25 × R_task_completion  # 任务目标达成
        + 0.10 × R_efficiency       # 工具调用数量合理
```

**关键优势**：所有奖励都是**可验证的**——不需要人工标注或 Reward Model，直接通过规则计算。

### 6.3 启动 GRPO 训练

```bash
python scripts/run_train.py --stage grpo \
    --grpo_model ./output/sft_model \
    --grpo_data ./data/train_grpo.jsonl \
    --grpo_output_dir ./output/grpo_model \
    --grpo_epochs 3 \
    --grpo_lr 2e-5 \
    --grpo_g 8 \
    --beta 0.01 \
    --batch_size 4 \
    --max_completion_length 4096
```

### 6.4 GRPO 参数详解

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `--grpo_g` | 8 | 每组生成 8 个回复（G=8 适合 40GB 显存，更大 = 更稳定但更慢） |
| `--beta` | 0.01 | KL 惩罚系数（0.001=激进探索，0.04=保守保持，0.01=平衡） |
| `--grpo_lr` | 2e-5 | GRPO 学习率（应低于 SFT，1e-6 到 5e-5） |
| `--grpo_epochs` | 3 | 训练轮数（2-5，观察奖励曲线决定） |

### 6.5 监控 GRPO 训练

```bash
tensorboard --logdir ./output/grpo_model
```

关键监控指标：

```
├── reward/total                  # 总奖励（应逐步上升）
├── reward/tool_call_accuracy     # 工具准确性（目标 >0.8）
├── reward/json_validity          # JSON 合法性（目标 >0.9）
├── reward/task_completion        # 任务完成率（目标 >0.7）
├── kl                            # KL 散度（应 <0.1，超过则增大 beta）
└── completion_length             # 生成长度分布（检测长度爆炸）
```

### 6.6 训练稳定性检查

| 症状 | 原因 | 解决 |
|------|------|------|
| `reward` 不上升 | 学习率太低 或 G 太小 | 增大 LR 到 5e-5，或 G 到 16 |
| `kl` 飙升 >0.2 | 策略漂移过快 | 增大 beta 到 0.04 |
| `completion_length` 爆炸 | 无长度惩罚 | 添加长度奖励惩罚项 |
| `json_validity` 突然下降 | 模型忘记格式 | 混入 10% SFT 数据 |

### 6.7 预期提升

| 指标 | SFT 后 | GRPO 后 | 提升 |
|------|--------|---------|------|
| JSON 格式输出率 | ~85% | ~97% | +12% |
| 正确工具选择率 | ~60% | ~78% | +18% |
| 参数填充准确率 | ~50% | ~72% | +22% |
| 复杂编排成功率 | ~35% | ~58% | +23% |

---

## 7. 第四步：模型评测

### 7.1 运行评测

```bash
python scripts/run_eval.py \
    --model_path ./output/grpo_model \
    --eval_data ./data/test_raw.jsonl \
    --output ./output/eval_report.json
```

### 7.2 评测指标体系

#### 工具调用精度

| 指标 | 含义 | 目标值 |
|------|------|--------|
| Function Name Accuracy | 正确选择了哪个工具 | > 0.75 |
| Parameter Accuracy | 参数填充正确 | > 0.70 |
| JSON Validity Rate | 输出是合法 JSON | > 0.95 |
| Exact Match Rate | 工具名+参数完全匹配 | > 0.60 |
| Task Success Rate | 多步编排整体成功 | > 0.50 |
| Efficiency Score | 工具调用数量合理 | > 0.80 |

#### BFCL 风格评测

| 类别 | 含义 | 目标值 |
|------|------|--------|
| Single-Turn | 单轮单工具调用 | > 0.85 |
| Multi-Turn | 多轮序列调用 | > 0.70 |
| Parallel | 并行多工具调用 | > 0.60 |
| Agentic | 复杂代理编排场景 | > 0.50 |

### 7.3 比较基线与已训练模型

```bash
# 评测基座模型（训练前）
python scripts/run_eval.py \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --eval_data ./data/test_raw.jsonl \
    --output ./output/baseline_report.json

# 评测 SFT 模型
python scripts/run_eval.py \
    --model_path ./output/sft_model \
    --eval_data ./data/test_raw.jsonl \
    --output ./output/sft_report.json

# 评测 GRPO 模型
python scripts/run_eval.py \
    --model_path ./output/grpo_model \
    --eval_data ./data/test_raw.jsonl \
    --output ./output/grpo_report.json

# 对比
python -c "
import json
for name in ['baseline', 'sft', 'grpo']:
    with open(f'./output/{name}_report.json') as f:
        r = json.load(f)
    ta = r['tool_accuracy']
    print(f'{name}: exact_match={ta[\"exact_match_rate\"]:.2%}, json_valid={ta[\"json_validity_rate\"]:.2%}')
"
```

---

## 8. 第五步：模型推理与部署

### 8.1 交互式推理

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "./output/grpo_model"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
)

def run_agent(prompt: str) -> str:
    messages = [
        {"role": "system", "content": "You are an AI assistant with tool access. Use JSON to call tools."},
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, return_tensors="pt", add_generation_prompt=True
    ).to(model.device)

    outputs = model.generate(
        inputs, max_new_tokens=2048, temperature=0.1,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)

# 测试
print(run_agent("Search for the weather in Beijing and send it to user@example.com"))
```

### 8.2 vLLM 高性能部署

```bash
# 合并 LoRA 权重
python -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct', torch_dtype='auto')
model = PeftModel.from_pretrained(model, './output/grpo_model')
model = model.merge_and_unload()
model.save_pretrained('./output/grpo_model_merged')
"

# 启动 vLLM 服务
python -m vllm.entrypoints.openai.api_server \
    --model ./output/grpo_model_merged \
    --port 8000 \
    --max-model-len 8192
```

### 8.3 导出 GGUF（本地部署）

```bash
# 安装 llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp && make -j

# 转换为 GGUF
python convert_hf_to_gguf.py ../output/grpo_model_merged --outtype q4_k_m

# 本地推理
./main -m grpo_model_merged.gguf -p "Search for AI news" -n 512
```

---

## 9. 完整流程一键运行

```bash
#!/bin/bash
# full_pipeline.sh — 一键运行完整流水线

set -e

echo "========================================"
echo "Agentic RL Full Pipeline"
echo "========================================"

# Step 1: Data
echo "[1/5] Generating training data..."
python scripts/run_data_pipeline.py

# Step 2: SFT
echo "[2/5] SFT Cold-Start Training..."
python scripts/run_train.py --stage sft --sft_epochs 3

# Step 3: GRPO
echo "[3/5] GRPO Reinforcement Learning..."
python scripts/run_train.py --stage grpo \
    --grpo_model ./output/sft_model --grpo_epochs 3

# Step 4: Evaluate
echo "[4/5] Evaluating model..."
python scripts/run_eval.py \
    --model_path ./output/grpo_model \
    --eval_data ./data/test_raw.jsonl \
    --output ./output/final_report.json

# Step 5: Summary
echo "[5/5] Pipeline Complete!"
python -c "
import json
with open('./output/final_report.json') as f:
    r = json.load(f)
ta = r['tool_accuracy']
print(f'Results:')
print(f'  Function Name Accuracy: {ta[\"function_name_accuracy\"]:.2%}')
print(f'  JSON Validity Rate:     {ta[\"json_validity_rate\"]:.2%}')
print(f'  Exact Match Rate:       {ta[\"exact_match_rate\"]:.2%}')
"
```

---

## 10. 调参与优化指南

### 10.1 场景适配

| 场景特点 | SFT 数据 | GRPO 参数 | 预期难点 |
|----------|----------|-----------|----------|
| **简单单步调用** | easy=50%, 5K 样本 | G=4, beta=0.04 | 极少，收敛快 |
| **多步序列编排** | medium=60%, 15K 样本 | G=8, beta=0.01 | 步骤依赖关系学习 |
| **复杂 Agent 场景** | hard=50%, 20K+ 样本 | G=16, beta=0.005 | 长时域信用分配 |

### 10.2 GPU 显存优化

```python
# 如果 OOM，依次尝试：
# 1. 减小 batch_size
batch_size = 2  # 默认 4

# 2. 增大梯度累积
gradient_accumulation_steps = 64  # 默认 32

# 3. 减小 Group Size
G = 4  # 默认 8，每组少生成一半

# 4. 使用更激进的量化
load_in_4bit = True  # 4-bit QLoRA
```

### 10.3 奖励函数权重调整

根据实际场景调整 `configs/grpo_config.yaml` 中的 reward 权重：

```yaml
# 严格精确场景（金融/医疗）
rewards:
  tool_call_accuracy: 0.60    # 提高准确率权重
  json_validity: 0.25
  task_completion: 0.10
  efficiency: 0.05

# 探索性场景（研究/创意）
rewards:
  tool_call_accuracy: 0.30
  json_validity: 0.15
  task_completion: 0.40       # 提高任务完成权重
  efficiency: 0.15
```

---

## 11. 常见问题

### Q1: SFT 训练 Loss 不下降

```bash
# 增大学习率
--sft_lr 5e-4

# 或检查数据质量
python -c "
import json
with open('./data/train_sft.jsonl') as f:
    for i, line in enumerate(f):
        if i < 3:
            print(json.loads(line)['messages'][1]['content'][:200])
"
```

### Q2: GRPO 奖励一直为 0

检查模型是否正确输出了 JSON 格式：
```bash
python scripts/run_eval.py --model_path ./output/sft_model --eval_data ./data/val_raw.jsonl
```

如果 JSON Validity Rate < 50%，说明 SFT 阶段不足，需要增加 SFT epoch 或数据量。

### Q3: 显存不足 (OOM)

```bash
# 使用更小的模型
--model_name_or_path Qwen/Qwen2.5-1.5B-Instruct

# 或减少 GRPO Group Size
--grpo_g 4
```

### Q4: 工具调用"幻觉"（调用不存在的工具）

在系统提示词中明确限制可用工具范围，并在 reward 函数中增加对未知工具名的惩罚。

### Q5: 如何用私有数据微调

```python
# 准备私有数据的格式（与 train_sft.jsonl 相同）：
# {"messages": [
#   {"role": "system", "content": "..."},
#   {"role": "user", "content": "用户任务描述"},
#   {"role": "assistant", "content": "{\"tool_calls\": [...]}"},
#   {"role": "tool", "content": "工具返回结果"},
#   {"role": "assistant", "content": "最终回复"}
# ]}

# 然后替换数据路径：
python scripts/run_train.py --stage sft --sft_data ./data/my_private_data.jsonl
```

### Q6: 模型评测分数不理想

1. 检查训练数据量是否足够（建议 8K+ SFT + GRPO）
2. 检查 SFT 阶段的 JSON Validity 是否 > 80%
3. 尝试增大 GRPO 的 Group Size（G=16）
4. 考虑换用更强的基座模型（如 Aura-7b）

---

## 参考资源

- [技术方案文档](agentic_RL_技术方案.md) — 完整理论背景
- [GRPO 论文 (DeepSeek-R1)](https://arxiv.org/abs/2501.12948)
- [Qwen 函数调用文档](https://github.com/QwenLM/Qwen-Agent)
- [BFCL Leaderboard](https://berkeley-function-calling-leaderboard.com/)
- [TRL GRPO 文档](https://huggingface.co/docs/trl/grpo_trainer)

---

> **技术支持**: 提交 Issue 至 https://github.com/zealot52099/Agentic_RL-/issues
