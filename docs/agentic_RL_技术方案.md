# Agentic RL 大模型训练技术方案

> **版本**: v1.0  
> **日期**: 2026-06-16  
> **摘要**: 本文档系统总结了基于 Agentic RL（强化学习）对大语言模型进行后训练（Post-Training）的完整技术方案，覆盖数据收集 → 数据清洗 → 模型选型 → 模型训练 → 效果评测五个阶段。内容基于 2024–2026 年最新论文与工业实践，涵盖 DeepSeek-R1、GPT-4、Claude、Qwen、Llama 等主流模型的相关技术。

---

## 目录

1. [第一阶段：数据收集](#1-第一阶段数据收集)
2. [第二阶段：数据清洗](#2-第二阶段数据清洗)
3. [第三阶段：模型选型](#3-第三阶段模型选型)
4. [第四阶段：模型训练](#4-第四阶段模型训练)
5. [第五阶段：效果评测](#5-第五阶段效果评测)
6. [参考文献](#6-参考文献)

---

## 1. 第一阶段：数据收集

### 1.1 数据分类体系

Agentic RL 训练需要三类核心数据：

| 数据类型 | 用途 | 典型来源 | 重要性 |
|----------|------|----------|--------|
| **SFT 数据** | 冷启动、格式对齐 | 人工标注 / 模型蒸馏 / 合成 | 最高 |
| **偏好数据 (Preference Pairs)** | DPO/RLHF 训练 | 人工对比标注 / LLM-as-Judge / 业务指标 | 最高 |
| **轨迹数据 (Trajectories)** | Agentic RL / GRPO 训练 | 真实工具调用 / 环境交互 / 规则引擎合成 | 高 |

### 1.2 SFT 数据构建

#### 1.2.1 数据来源策略（按优先级排序）

**第一优先级：蒸馏高质量模型（Distillation）**

- DeepSeek-R1 论文验证：对小型模型（≤32B），从强模型蒸馏远比从头 RL 训练更具成本效益
- 方法：使用 DeepSeek-R1 / GPT-4 / Claude 等强模型生成高质量 CoT 回复，作为 SFT 数据
- 成本参考：DeepSeek-R1 的 SFT 数据构建成本约 $10K（80 万条样本）
- 典型流程：强模型生成 → 质量过滤 → 格式标准化 → SFT 训练

**第二优先级：真实端到端轨迹（Real End-to-End Trajectories）**

- 2025 年论文 "Demystifying RL in Agentic Reasoning" 关键发现：**真实工具调用轨迹远优于拼接合成轨迹**
- 来源：用户在真实环境中完成任务的完整操作序列
- 优势：包含自然的错误恢复、探索行为，合成数据难以模拟

**第三优先级：自动化合成（Synthetic Generation）**

- **O-Researcher (2026)**：多智能体工作流自动生成研究级轨迹数据
  - Planner + Tool-User + Summarizer 三阶段协作
  - 每问题生成 8 条回复，仅保留"甜区"（非平凡且非不可解）问题
- **AgenticQwen (2026)**：基于行为树结构的任务生成
  - 使用 LangGraph 管线 + 大型 LLM 创建虚拟工具集、策略和任务
  - 三阶段：任务与工具生成 → 任务求解（模拟工具/用户）→ 评分过滤
- **AIF-GEN (ICML 2025)**：首个面向传统 RLHF 和终身 RLHF 的合成偏好数据平台，生成 18 个合成数据集

#### 1.2.2 数据量级参考

| 训练阶段 | 数据量 | 说明 |
|----------|--------|------|
| 冷启动 SFT | 5K–50K 条 | 高质量标注或蒸馏数据，覆盖核心能力域 |
| 大规模 SFT | 200K–800K 条 | DeepSeek-R1 使用 80 万条拒绝采样数据 |
| 偏好对（DPO） | 10K–100K 对 | ToolPref-Pairwise-30K 为工具使用场景的参考量级 |
| RL 训练提示词 | 1K–10K 条 | 需高多样性和适当难度分布 |

### 1.3 偏好数据构建

#### 1.3.1 主流收集方法

**方法一：人工对比标注**

- 标注员对同一 prompt 的两个回复进行二选一
- 关键改进（SAC 论文, 2024）：增加"两者都不好"选项，过滤低质量对
- 标注员间一致性（Inter-annotator Agreement）需 > 0.7

**方法二：LLM-as-a-Judge**

- 使用 GPT-4/Claude 等强模型作为评判者
- 多维评分：帮助性、安全性、准确性、完整性、格式合规性
- 注意：需校准 LLM Judge 的偏差（位置偏差、长度偏差、风格偏差）

**方法三：业务指标驱动**

- Self-Preference (2026)：从业务指标（如金融场景的问题解决率、周转率）自动构建偏好数据
- 聚类对话历史 → 计算条件概率比 → 自动标注偏好
- 避免主观人工评估和模型诱导偏差

**方法四：VickreyFeedback 经济机制 (2024)**

- 将偏好数据收集建模为货币化经济 + 拍卖机制
- 处理非传递性/循环偏好关系
- 提升成本效率的同时保持模型性能

### 1.4 交互环境数据（用于 Agentic RL）

静态 RL 数据集越来越不够用。2026 年调查总结了四类动态训练环境：

| 环境类别 | 代表项目 | 适用场景 |
|----------|----------|----------|
| **规则验证型** | Logic-RL, Reasoning Gym, SynLogic, Enigmata | 数学推理、逻辑推理 |
| **代码执行型** | AppWorld, MLGym, R2E-Gym, SWE-rebench | 代码生成、软件工程 |
| **游戏交互型** | TextArena, KORGym, PuzzleJAX | 策略规划、多步决策 |
| **模型驱动型** | Absolute Zero, Genie 3 | 开放式探索 |

**核心原则**：可扩展的 LLM RL 训练必须利用**提供可验证反馈的动态环境**（自动化合成 + 步骤级反馈）。

### 1.5 数据构建的质量控制

**难度增强策略**：

1. **复杂度增强**：多跳推理、多轮对话、多工具协同
2. **不确定性增强**：信息模糊、干扰项、不可回答问题
3. **专业性增强**：领域专家级问题（如 GPQA 风格）

**"甜区"筛选**（O-Researcher 方法）：

- 过滤"平凡"问题（所有回复均获高分）→ 无训练信号
- 过滤"难解"问题（所有回复均获低分）→ 模型无法学习
- 保留中间难度问题 → 最大化训练信号

---

## 2. 第二阶段：数据清洗

### 2.1 数据质量问题识别

根据 SAC 论文（arXiv 2410.01957, 2024），人类反馈数据存在**六类不可靠来源**：

| 问题类型 | 描述 | 检测方法 |
|----------|------|----------|
| **双方质量都差** | 两个备选回复都不好 | RM 双低分过滤 |
| **偏好模糊** | 两个回复质量接近 | 分数差距 < 阈值 |
| **标注员错误** | 注意力不集中或误操作 | 与 Gold RM 交叉验证 |
| **风格 vs 内容** | 偏好格式优美但内容错误 | 内容/格式解耦评分 |
| **安全-有用冲突** | 安全但无用 vs 有用但不安全 | 多维评分矩阵 |
| **格式伪影** | 因格式差异而非内容差异产生的偏好 | 格式标准化后重评 |

### 2.2 数据清洗流水线

#### 2.2.1 三层过滤架构

```
原始数据
  │
  ├── 第一层：规则过滤
  │   ├── 去重（精确匹配 + MinHash/LSH 近似去重）
  │   ├── 长度过滤（过短 < 10 tokens / 过长 > 阈值）
  │   ├── 语言检测与过滤
  │   ├── 特殊字符/乱码检测
  │   └── 模板化/重复内容检测
  │
  ├── 第二层：评分过滤
  │   ├── 困惑度 (Perplexity) 评分
  │   ├── Reward Model 评分
  │   ├── 影响力函数 (Influence Function) 评分
  │   └── 多维质量打分（内容、格式、安全性）
  │
  └── 第三层：LLM 质量评估
      ├── LLM-as-Judge 综合评判
      ├── 事实性验证
      ├── 安全与合规审查
      └── 领域专家审查（关键数据）
```

#### 2.2.2 关键清洗技术

**1. 去重（Deduplication）**

| 方法 | 适用场景 | 精度 |
|------|----------|------|
| 精确匹配 | 完全相同样本 | 100% |
| MinHash + LSH | 近似重复检测 | 可调（Jaccard 阈值） |
| 语义去重 (Embedding) | 语义相似但表述不同 | 高（需设定相似度阈值） |
| N-gram 重叠 | 部分重叠的文本块 | 中等 |

社区实践：Anthropic HH-RLHF 数据集经过去重后，移除了 18,170 条对齐伪影和 51,954 条精确重复，从约 17 万条降至 99,228 条干净指令。

**2. 模型依赖型数据估值 (TIF, 2025)**

传统方法假设数据质量是内禀属性，但 TIF（Truncated Influence Function）证明**数据质量是模型依赖的**：

- **中等影响力（IF）数据最有价值** → 提供最丰富的学习信号
- 极高 IF 数据 → 噪音/过拟合风险
- 极低 IF 数据 → 对训练无贡献，浪费计算
- LossDiff-IRM 评分函数（仅需前向传播，无需梯度）：结合损失差异和隐式奖励边际
- **效果**：仅用 50–64% 数据实现 **+13.58% WinRate 提升**

**3. Hölder-DPO 内置鲁棒清洗 (2025)**

首个具备**可证明降权属性（Redescending Property）**的 DPO 变体：

- 训练过程中自动识别并降低噪声标签权重
- 发现 Anthropic HH-RLHF 数据集约 **~25% 污染率**
- 推荐二阶段流程：先用 Hölder-DPO 过滤噪声 → 再用 Vanilla DPO 训练

**4. Source-Aware Cleaning (SAC, 2024)**

使用 RewardBench 顶级 Reward Model 委员会进行源感知清洗：

- 发布 HH-Clean 数据集
- 训练模型在 DPO/IPO/SLiC/KTO 上均达到 **~72% Win-Tie Rate** 优于原始数据
- 推荐：标注时强制加入"两者都不好"选项

**5. 自动化 Reward Model 质量调整 (EMNLP 2024)**

- Reward Model 分配的分数差是有效的数据质量指标
- 自动调整 RM 训练权重：低质量数据降权，减少噪声影响
- 稳定 RM 训练并显著提升 RLHF 对齐性能

### 2.3 数据配比策略

| 能力维度 | 推荐数据占比 | 说明 |
|----------|-------------|------|
| 推理能力（数学/代码/逻辑） | 30–40% | RL 训练的核心增益区 |
| 指令遵循 | 20–25% | 保障基础对话能力 |
| 安全与拒答 | 15–20% | 对齐安全性要求 |
| 通用知识 | 10–15% | 防止能力退化 |
| 工具使用 | 10–15% | Agent 场景专项能力 |

---

## 3. 第三阶段：模型选型

### 3.1 2025–2026 主流开源基座模型对比

#### 3.1.1 旗舰模型架构概览

| 维度 | Qwen 3 / Qwen 2.5 | Llama 4 / Llama 3.1 | DeepSeek V3 / R1 |
|------|-------------------|---------------------|-------------------|
| **架构** | Dense + MoE + GQA + RoPE | Dense / MoE + GQA | MoE (671B 总参, 37B 激活) |
| **参数规模** | 0.5B–72B (Qwen 2.5); 32B (Qwen 3) | 1B–405B (Llama 3); 400B MoE (Llama 4) | 7B–671B |
| **上下文长度** | 32K–128K | 128K (Llama 3.1); 10M (Llama 4) | 128K |
| **许可证** | Apache 2.0 | Llama Community License | MIT |
| **中文能力** | 最优 | 一般 | 优 |
| **英文能力** | 优 | 最优 | 优 |
| **数学推理** | 优 | 良 | 最优 |
| **代码能力** | 优 | 最优 | 最优 |
| **推理速度** | 快 (Dense 架构) | 中等 | 较慢 (671B MoE) |
| **训练成本** | 未公开 | 未公开 | V3: $5.6M（极低） |
| **推理成本** | 中等 | 中等 | GPT-4 的 1/20–1/50 |

#### 3.1.2 Benchmark 性能参考

| Benchmark | Qwen 3 32B | Llama 4 Maverick | DeepSeek-V3 | DeepSeek-R1 |
|-----------|-----------|-------------------|-------------|-------------|
| MMLU | 82.3 | 88.6 | 87.2 | — |
| HumanEval | 81.2 | 92.8 | 89.9 | — |
| MATH | — | — | 90.2 | 97.3 (AIME 2024) |
| Arena Elo | 1300+ | 1350+ | 1330+ | ~1350 |

### 3.2 模型选型决策框架

#### 3.2.1 决策树

```
1. 主要语言？
   ├── 中文为主 → Qwen 2.5/3 (最佳中文性能，Apache 2.0 商用友好)
   └── 英文/国际 → 进入 2

2. 核心任务？
   ├── 代码生成 → Llama 3.1 / DeepSeek-Coder V2
   ├── 数学推理 → DeepSeek-R1 蒸馏版 (7B/14B/32B)
   ├── 通用对话 → Llama 3.1 或 Qwen 2.5
   └── 长文档处理 → Llama 4 (10M 上下文)

3. 硬件预算？
   ├── 消费级 GPU (≤24GB) → Qwen 2.5-7B (4-bit) / Mistral 7B
   ├── 单卡 A100 (40/80GB) → Qwen 2.5-32B / Llama 3.1-70B (QLoRA)
   └── 多卡集群 → DeepSeek-V3 (671B MoE)

4. 合规要求？
   ├── 中国境内 → Qwen / ChatGLM / Baichuan
   └── 全球 → Llama / Mistral / Qwen
```

#### 3.2.2 按场景推荐

| 应用场景 | 推荐基座模型 | 核心理由 |
|----------|-------------|----------|
| **中文企业应用** | Qwen 2.5-14B/72B | 中文最佳，Apache 2.0 商用 |
| **全球 SaaS 产品** | Llama 3.1-8B/70B | 英文生态最强 |
| **数学/科学推理** | DeepSeek-R1-Distill-Qwen-32B | 对标 o1，MIT 许可 |
| **代码生成** | DeepSeek-Coder V2 / Qwen 2.5-Coder | HumanEval 领先 |
| **长文档处理** | Llama 4 (10M) / Baichuan 4 | 超长上下文 |
| **资源受限部署** | Qwen 2.5-7B (4-bit AWQ) | 仅需 ~4GB VRAM |
| **学术研究** | Llama 3.1 | 学术标准，可复现性最强 |
| **成本极致优化** | DeepSeek-V3 API | GPT-4 的 1/50 推理成本 |

### 3.3 模型规模选择指南

| 训练方式 | 推荐参数规模 | 最低 GPU 配置 |
|----------|-------------|--------------|
| Full Fine-Tune | 7B–14B | 4× A100 80GB |
| LoRA/QLoRA | 7B–72B | 1× RTX 4090 24GB (7B) 至 2× A100 (70B) |
| GRPO Full | 1.5B–7B | 1× A100 40GB (7B, G=8) |
| GRPO Full | 32B | H200 141GB |
| GRPO Full | 70B+ | 2× B200 |
| 仅推理 (4-bit AWQ) | 任意 | 按规模 4–160 GB |

**关键建议**：
- 对于 RL 训练，**建议从 1.5B–7B 蒸馏模型开始**，验证流水线后再扩展到更大规模
- AAAI 2026 论文证明：在 DeepSeek-R1-Distill-Qwen-1.5B 上用 4× A40 24 小时内即可达到 AIME24 46.7%（超越 o1-preview），成本仅 $42

---

## 4. 第四阶段：模型训练

### 4.1 RL 算法全景对比

#### 4.1.1 算法演进路线

```
PPO (2017, 2022 年适配 LLM)
  │  需要: Reward Model + Critic + Reference + Policy (4模型同驻)
  │  移除二阶优化 (vs TRPO)
  ▼
DPO (2023)
  │  移除 Reward Model 和 Rollout（直接偏好优化）
  │  无探索能力，但 2-4× 更便宜
  ▼
GRPO (2024, DeepSeek)
  │  移除 Critic Network（组内相对优势）
  │  更适合可验证奖励（RLVR）
  ▼
Dr.GRPO / DAPO / CISPO (2025)
  │  修复 GRPO 的长度偏差、梯度消失、过度裁剪
  ▼
MaxRL / DPPO / AR3PO (2025–2026)
    改进 pass@k、信任域、rollout 效率
```

#### 4.1.2 算法详细对比

| 算法 | 内存占用 | 探索能力 | 稳定性 | 最佳场景 |
|------|---------|---------|--------|----------|
| **PPO** | 最高 (~4× 模型) | 强 | 中（需精细调参） | 有高质量 RM + 大 GPU 预算 |
| **DPO** | 低 (~2× 模型) | 无 | 高 | 风格对齐、安全拒答 |
| **GRPO** | 中等 (~2.5× 模型) | 强 | 中 | 可验证奖励（数学/代码推理） |
| **REINFORCE++** | 低 | 中等 | 中 | 轻量替代 PPO |
| **DAPO** | 中等 | 强 | 高 | GRPO 的稳定替代（推荐） |
| **SimPO** | 低 (~1.5× 模型) | 无 | 高 | 无需参考模型的偏好优化 |
| **KTO** | 低 | 无 | 高 | 仅有二元反馈（好/坏）的场景 |

### 4.2 推荐训练策略：按目标选择

#### 4.2.1 策略一：风格对齐 / 安全拒答 → DPO 路线

```
SFT 基座模型
  │
  ├── 偏好数据构建（人工标注 + LLM Judge）
  │
  ├── DPO 训练
  │   参数：β = 0.1, LR = 5e-7, epochs = 1-3
  │
  └── 效果验证（AlpacaEval / Arena-Hard）
```

- **适用场景**：让模型更礼貌、遵循指令格式、添加安全拒答行为
- **优势**：2-4× 比 PPO 便宜，训练稳定，监督损失直接下降
- **局限**：无探索能力，受偏好数据质量上限约束

#### 4.2.2 策略二：推理能力增强 → GRPO + RLVR 路线（核心路线）

这是 2025–2026 年最重要且效果最好的训练路线，也是 DeepSeek-R1 的核心方法：

```
基座模型（或 SFT 初始化模型）
  │
  ├── 阶段1: 冷启动 SFT（可选但推荐）
  │   数据：蒸馏高质量 CoT 数据（5K–50K 条）
  │   目的：建立基本格式和推理能力
  │
  ├── 阶段2: GRPO + 可验证奖励 (RLVR)
  │   算法：GRPO
  │   奖励：基于可验证规则（数学答案匹配、代码测试通过等）
  │   组大小 G：8–32
  │   KL 系数 β：0.001–0.04
  │   学习率：1e-6（蒸馏模型）至 4e-5（大模型）
  │
  ├── 阶段3: 拒绝采样 + 大规模 SFT（可选）
  │   从 RL 模型采样，过滤正确答案
  │   DeepSeek-R1 使用 80 万条拒绝采样数据
  │
  └── 阶段4: 全场景 RL（可选）
       混合多种奖励信号（有帮助性、安全性、推理能力等）
```

**关键超参数配置**：

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| Group Size (G) | 16–32 | 更大的 G 降低方差但线性增加 rollout 成本；14 被验证为最平滑 |
| KL 系数 (β) | 0.001–0.04 | DeepSeek-R1 用 0.001；TRL 默认 0.04；过低导致策略漂移 |
| Clip Ratio (ε) | 0.2 (标准) / 10.0 (GRPO) | GRPO 使用异常大的 clip ratio |
| 学习率 | 1e-6 (小蒸馏模型) / 4e-5 (大模型) | 小模型需更低学习率以避免不稳定 |
| Max Completion Length | 2048–32768 | 根据任务复杂度；推理需长 CoT |
| 优化器 | AdamW 8bit / Paged AdamW | 8-bit 量化优化器大幅降低内存 |

#### 4.2.3 策略三：Agent 工具使用 → GRPO on Trajectories

```
SFT 初始化（真实工具调用轨迹，非拼接合成轨迹）
  │
  ├── 环境：模拟工具/用户交互环境
  │
  ├── GRPO on Agent Trajectories
  │   奖励：任务完成 + 工具使用效率 + 格式合规
  │   框架：verl + SGLang / Agent-Lightning (Microsoft) / Agent-R1 (USTC)
  │   特点：长轨迹需考虑过程奖励 (PRM)
  │
  └── 探索增强：审议策略（deliberative strategy）
      少量高质量工具调用 > 冗余冗长调用
```

### 4.3 核心算法深度解析

#### 4.3.1 GRPO (Group Relative Policy Optimization)

**核心机制**：对于每个 prompt x，采样 G 个回复 {y_1, ..., y_G}，计算组内标准化优势：

```
A_i = (r_i - mean(r)) / std(r)
```

**关键理论发现 (2025)**：
- GRPO 的策略梯度是一个 **U-统计量**，使其渐近等价于拥有理想价值函数的 Oracle
- 可验证奖励下的 GRPO 可写为 **KL 正则化对比损失**
- GRPO 成功概率收敛到**高于参考模型的固定点** — 解释了为何 GRPO 放大成功

**常见陷阱与对策**：

| 陷阱 | 影响 | 对策 |
|------|------|------|
| 全正确/全错误组 | 优势退化为 0，无学习信号 | 分母加 epsilon；动态采样过滤 |
| 长度偏差 (Dr.GRPO) | 错误回复被人为拉长获取更高奖励 | 恒定 token 归一化替代序列长度平均 |
| 梯度消失 (DAPO) | 训练停滞 | 非对称裁剪 (ε_high=0.28, ε_low=0.2) |
| 过度裁剪 (CISPO) | 稀有 token 梯度被截断 | Stop-Gradient 替代硬掩码 |
| 熵崩塌 | 策略过早收敛，输出多样性丧失 | 调高 KL 系数；增大采样温度 |

#### 4.3.2 DAPO (Decoupled Alignment Policy Optimization, 2025)

GRPO 的关键改进版，字节跳动/清华提出：

- **非对称裁剪**：ε_high=0.28（鼓励探索），ε_low=0.2（防止退化）
- **动态采样**：自动过滤全对/全错组，确保有效训练信号
- **Token 级损失**：替代序列级损失聚合，精细梯度分配
- **超长回复奖励塑造**：防止模型产生无限长输出
- **效果**：Qwen2.5-32B 在 AIME 2024 达到 50 分，**仅需 DeepSeek-R1-Zero 50% 训练步数**

#### 4.3.3 DPO (Direct Preference Optimization) 及衍生变体

**核心公式**：

```
L_DPO = -E[log σ(β × (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]
```

即直接最大化"好回复相对于差回复的对数概率比"。

**2025–2026 衍生变体**：

| 变体 | 改进点 | 效果 |
|------|--------|------|
| **SimPO** | 移除参考模型；用平均对数概率作隐式奖励 | +6.4 AlpacaEval 2 |
| **KTO** | 支持二元好/坏反馈替代成对比较 | 更灵活的数据需求 |
| **ORPO** | 合并 SFT + 偏好优化到单阶段 | 训练更高效 |
| **Hölder-DPO** | 内置鲁棒降权，自动检测噪声标签 | +抗噪能力 |

### 4.4 奖励建模

#### 4.4.1 奖励类型体系

```
奖励信号
  │
  ├── 结果奖励 (ORM: Outcome Reward Model)
  │   ├── 可验证奖励 (RLVR)：数学答案匹配、代码测试通过
  │   ├── 标量 RM：训练好的神经网络奖励模型
  │   └── LLM-as-a-Judge：使用强 LLM 评分
  │
  ├── 过程奖励 (PRM: Process Reward Model)
  │   ├── 步骤级标注：人工标注每步是否正确
  │   └── 自动 PRM：从结果标签自动推断步骤质量
  │
  └── 混合奖励
      ├── 多维度评分：有用性 + 安全性 + 准确性 + 格式
      └── 约束奖励：格式、长度、工具使用规范
```

#### 4.4.2 ORM vs PRM 选型决策

| 维度 | ORM (结果奖励) | PRM (过程奖励) |
|------|---------------|---------------|
| 标注成本 | 低（仅需结果标签） | 高（需步骤级标注） |
| 信号密度 | 稀疏 | 密集，样本效率更高 |
| 适用任务 | 数学、代码（可验证结果） | 长文生成、工具使用、安全性 |
| 2025-2026 趋势 | GRPO + 可验证奖励为主流 | 前向模型上，简单多数投票 > 训练后的 PRM |

**关键发现（2025）**：

- DeepSeek-R1 论文证明：纯结果奖励 + GRPO 即可产生逐步推理，**无需步骤级标注**
- "Is PRM Necessary?" 论文：对 DeepSeek-R1/QwQ-32B 等前向模型，**训练的 PRM 表现不如简单多数投票**
- **PRM 仍胜出的 4 种场景**：
  1. 最终答案模糊的长时域 Agent 任务
  2. 工具使用安全性（惩罚特定中间危险行为）
  3. 多智能体训练 (MARL) — 组内信用分配失效
  4. 推理时搜索（Beam Search, MCTS）

#### 4.4.3 奖励工程最佳实践

**可验证奖励 (RLVR) 设计原则**：

```
1. 精确匹配 → 二元 0/1（数学答案）
2. 测试通过率 → 连续 [0, 1]（代码执行：通过测试数/总测试数）
3. 格式合规 → 二元 0/1（结构验证：是否有 <think>/<answer> 标签）
4. 约束满足 → 逐项扣分（输出规范：长度、格式等）

奖励组合权重示例：
  R_total = 1.0 × R_correctness + 0.1 × R_format + 0.05 × R_length_penalty
```

**SPARK 框架 (2025)**：策略与奖励协同进化

- 回收 rollout 和正确性数据训练模型自身作为 RM
- 消除独立 RM 和昂贵人工偏好数据需求
- 在 7 个推理基准上平均增益 9.7%

### 4.5 训练基础设施

#### 4.5.1 开源框架

| 框架 | 适用场景 | 特点 |
|------|----------|------|
| **verl** (ByteDance) | 分布式 RL (GRPO/PPO/RLVR) | 事实标准，支持异步调度 |
| **TRL** (HuggingFace) | DPO/GRPO/PPO | 中小规模，文档完善 |
| **OpenRLHF** | RLHF 全流程 | 广泛的算法支持 |
| **Agent-Lightning** (Microsoft) | Agent RL | 框架无关，通过可观测性钩子 |
| **Agent-R1** (USTC) | Agent 轨迹 RL | 步骤级 MDP 抽象，原生 PRM 支持 |

#### 4.5.2 训练优化技术

| 技术 | 收益 | 实现方式 |
|------|------|----------|
| **vLLM 异步 Rollout** | 2–3× 加速 | rollout 服务器与训练器分离 |
| **动态内存卸载** | 30–40% VRAM 节省 | 非活跃模块卸载到 CPU |
| **多 Token 预测 (MTP)** | 推测解码加速 | DeepSeek-R1 基础设施 |
| **8-bit 优化器** | 50% 优化器内存节省 | AdamW 8bit / Paged AdamW |
| **QLoRA** | 4× 内存减少 | 4-bit 量化 + LoRA 适配器 |
| **梯度累积** | 模拟大批量 | 累积步数 128–512 |

#### 4.5.3 GPU 配置参考

| 模型规模 | 训练方式 | GPU 需求 | 预计训练时间 |
|----------|---------|---------|-------------|
| 1.5B | GRPO G=8, T=2048 | 1× A40 48GB | ~24 小时 |
| 7B | GRPO G=8, T=512 | 1× A100 40GB | 2–5 天 |
| 7B | GRPO G=16, T=4096 | 4× A100 40GB | 1–3 天 |
| 32B | GRPO G=8, T=1024 | H200 141GB | 3–7 天 |
| 70B | GRPO G=8, T=1024 | 2× B200 | 7–14 天 |
| 671B (DeepSeek-V3) | GRPO (R1-Zero) | 648× H800 | ~198 小时 (~$202K) |

### 4.6 训练稳定性保障

#### 4.6.1 常见问题诊断与解决

| 问题 | 症状 | 排查优先级 | 解决方案 |
|------|------|-----------|----------|
| **奖励黑客** | 奖励上升但实际质量下降 | 1. 检查奖励函数漏洞 | 添加惩罚项；混合多奖励信号 |
| **策略崩溃** | 输出退化为重复或无意义文本 | 1. KL 系数太低 | 增大 KL 系数；降低 LR |
| **训练不收敛** | Loss 震荡不下降 | 1. LR 太高 2. G 太小 | 降低 LR；增大 Group Size |
| **长度爆炸** | 输出越来越长 | 1. 缺少长度惩罚 | 添加长度惩罚；用 Dr.GRPO |
| **熵崩塌** | 输出多样性骤降 | 1. 策略过早收敛 | 增大温度；调高 clip ε_high |

#### 4.6.2 训练期间监控指标

```
核心监控：
├── 奖励曲线（按任务类型分组的平均奖励）
├── KL 散度（当前策略 vs 参考策略，监测策略漂移）
├── 回复长度分布（均值、方差、P99，检测长度爆炸）
├── 输出熵（token 级别多样性，检测熵崩塌）
├── Rollout 质量（正确率、格式合规率）
├── GPU 利用率
└── 学习率调度曲线
```

---

## 5. 第五阶段：效果评测

### 5.1 评测基准体系

#### 5.1.1 基准分类全景

```
评测基准
  │
  ├── 通用对话评测
  │   ├── AlpacaEval 2.0: 805 条指令，有用性，LC Win Rate
  │   ├── MT-Bench: 80 条多轮对话，8 领域，GPT-4 评分 1–10
  │   ├── Arena-Hard: 250 个主题簇，500 查询，Win Rate vs GPT-4
  │   └── Chatbot Arena: 众包人类评测，Elo 评分
  │
  ├── 知识评测
  │   ├── MMLU / MMLU-Pro: 多学科知识理解
  │   ├── GPQA Diamond: 研究生级科学问答
  │   └── TruthfulQA: 事实性/真实性
  │
  ├── 推理评测
  │   ├── AIME 2024: 竞赛级数学（30 题）
  │   ├── MATH-500: 竞赛数学
  │   ├── GSM8K: 小学数学应用题
  │   ├── BigBench-Hard: 复杂推理
  │   └── DROP: 阅读理解 + 数值推理
  │
  ├── 代码评测
  │   ├── HumanEval / HumanEval+: 函数生成
  │   ├── LiveCodeBench: 实时竞赛题
  │   ├── SWE-Bench Verified: 真实软件工程任务 (2294 题)
  │   └── MBPP: 入门级 Python
  │
  ├── 工具使用评测
  │   ├── BFCL V4: 函数调用评测
  │   └── τ-Bench: 工具使用代理评测
  │
  └── 安全评测
      ├── HarmBench: 有害内容生成评测
      ├── JailbreakBench: 越狱鲁棒性
      └── TruthfulQA: 真实性评测
```

#### 5.1.2 各基准详细参数

| 基准 | 类型 | 样本数 | 评判者 | 主要指标 | 局限性 |
|------|------|--------|--------|----------|--------|
| **AlpacaEval 2.0** | 单轮有用性 | 805 | GPT-4 | LC Win Rate | 可被 length hack |
| **MT-Bench** | 多轮对话 | 80 对话×2 | GPT-4 | 1-10 平均分 | 样本量小；GPT 偏向 |
| **Arena-Hard** | 高难度指令 | 500 | GPT-4 | Win Rate | — |
| **Chatbot Arena** | 开放式 | 众包 | 人类 | Elo 评分 | 慢且昂贵 |
| **MMLU-Pro** | 知识 | 12K+ | 准确率 | Accuracy | 静态数据集 |
| **GPQA Diamond** | 科学推理 | 198 | 准确率 | Accuracy | 样本量小 |
| **AIME 2024** | 竞赛数学 | 30 | 答案匹配 | Pass@1 | 仅数学 |
| **SWE-Bench** | 代码工程 | 2294 | 测试通过 | Resolved Rate | 难度极高 |

### 5.2 评测策略

#### 5.2.1 三阶段评测金字塔

```
            ┌─────────────┐
            │  人工评测    │  ← 顶层：50-100 条，质量终裁
            │   (Gold)    │     A/B 盲评 + 多维打分
           ┌─┴───────────┴─┐
           │  自动化深度评测 │  ← 中层：多基准 + LLM Judge
           │  (Benchmarks) │     全量基准 + 与基线对比
          ┌─┴─────────────┴─┐
          │  快速回归评测    │  ← 底层：每次训练后运行
          │  (Regression)  │     小规模子集 < 30min
          └─────────────────┘
```

**第一层：快速回归评测（每次训练迭代后）**

- 小型基准子集（如 GSM8K-100, HumanEval, 格式合规率）
- 运行时间：< 30 分钟
- 目的：快速检测训练退化或灾难性遗忘

**第二层：自动化深度评测（里程碑/Checkpoint）**

- 全量基准运行
- LLM-as-a-Judge 多维度评分
- 与基线模型对比
- 运行时间：数小时
- 目的：全面能力评估

**第三层：人工评测（最终验收）**

- 50–100 条实际业务场景样本
- 与竞品盲评对比 (A/B Test)
- 多维度评分：准确性、有用性、安全性、流畅性
- 目的：最终质量裁决

#### 5.2.2 LLM-as-a-Judge 最佳实践

**基准 LLM Judge 的选择与校准**：

```
推荐 Judge 模型：GPT-4-Turbo / Claude 3.5 Sonnet / DeepSeek-R1

校准策略：
1. 位置偏差消除：每条样本判断两次，交换回复顺序，取平均
2. 长度偏差控制：使用 Length-Controlled Win Rate
3. 多维度评分矩阵（推荐，而非单一分数）：

   ┌──────────────────────────────────────┐
   │ 维度        │ 权重  │ 评分 (1-5)     │
   ├─────────────┼───────┼────────────────┤
   │ 指令遵循    │ 25%   │ 是否准确完成指令│
   │ 事实准确性  │ 25%   │ 内容是否事实正确│
   │ 回复有用性  │ 20%   │ 是否真正帮到用户│
   │ 连贯性/流畅 │ 15%   │ 语言是否自然流畅│
   │ 安全性      │ 15%   │ 是否符合安全规范│
   └──────────────────────────────────────┘

4. 周期性人工校验：随机抽样 10% 确认 Judge 评分与人工判断一致性
5. 多 Judge 投票：使用 3 个不同 LLM Judge 投票，降低单一 Judge 偏差
```

### 5.3 主流模型评测参考值

| 模型 | Arena Elo | AlpacaEval 2 LC | MT-Bench | MMLU | AIME 2024 |
|------|-----------|-----------------|----------|------|-----------|
| GPT-4-Turbo | ~1250 | 30.2 (0613) | 9.18 | 86.4 | ~20 |
| Claude 3.5 Sonnet | ~1280 | 40.5 | 9.00 | 88.7 | ~15 |
| DeepSeek-R1 | ~1350 | — | — | — | 79.8 (Pass@1) |
| Qwen 2.5-72B | ~1280 | — | ~8.9 | 86.1 | ~30 |
| Llama 3.1-405B | ~1320 | — | ~9.0 | 88.6 | ~25 |

> 注：数据来源为 2025–2026 年公开基准，实际值因评测配置不同可能有浮动。

### 5.4 评测注意事项与陷阱

#### 5.4.1 基准污染与博弈

- **ICLR 2025 关键发现**：即使是**空模型**（输出固定字符串）也能在 AlpacaEval 2.0、Arena-Hard-Auto、MT-Bench 上获得高排名
- 对策：**不要只看单一基准分数，必须三角验证**
- 推荐使用**私有评测集**（500–1500 条业务流量采样）作为补充

#### 5.4.2 学术评测 vs 生产评测（2026 最佳实践）

| 维度 | 学术评测 | 生产评测 |
|------|----------|----------|
| 核心问题 | "哪个模型最聪明？" | "我的系统在我的流量上工作吗？" |
| 数据集 | 固定精选数据集 | 流量采样 + 漂移追踪 |
| 指标 | 单一标量分数 | 评分矩阵 (Rubric Baskets) |
| 频率 | 论文发表时一次性 | 持续 CI 门禁 |
| 更新方式 | 静态不变 | 对抗性样本持续更新 |

**推荐 2026 模式**：

1. 用 3–4 个公开基准（MMLU-Pro + τ-bench + GPQA Diamond + Chatbot Arena）圈定候选模型
2. 运行私有评测（500–1500 条流量采样，按路由评分）
3. 包含人工标注的生产轨迹、合成变体和对抗探测
4. 持续监控线上指标（用户满意度、任务完成率、拒答率）

---

## 6. 总结：端到端技术方案路线图

### 6.1 推荐技术栈总览

```
┌──────────────────────────────────────────────────────────────┐
│                    Agentic RL 训练全流程                       │
├──────────────┬───────────────────────────────────────────────┤
│ 阶段1: 数据  │ 蒸馏(强模型生成) → 环境交互轨迹 → 合成增强      │
│              │ 偏好对构建: 人工标注 + LLM Judge + 业务指标     │
├──────────────┼───────────────────────────────────────────────┤
│ 阶段2: 清洗  │ 三层过滤: 规则去重 → RM评分 → LLM质量评估      │
│              │ 模型依赖型数据估值 (TIF/LossDiff-IRM)          │
│              │ Hölder-DPO 内置鲁棒清洗                        │
├──────────────┼───────────────────────────────────────────────┤
│ 阶段3: 选型  │ 中文→Qwen 2.5 │ 英文→Llama 3.1                 │
│              │ 推理→DeepSeek-R1-Distill │ 代码→DeepSeek-Coder│
│              │ 推荐起步规模: 7B-14B (QLoRA) / 1.5B-7B (GRPO) │
├──────────────┼───────────────────────────────────────────────┤
│ 阶段4: 训练  │ 主路线: SFT → GRPO + 可验证奖励 (RLVR)         │
│              │ 框架: verl (分布式) / TRL (中小规模)           │
│              │ 风格对齐补充: DPO/SimPO                        │
│              │ Agent训练: GRPO on Trajectories + PRM          │
├──────────────┼───────────────────────────────────────────────┤
│ 阶段5: 评测  │ 三层金字塔: 快速回归 → 深度基准 → 人工终裁      │
│              │ 核心基准: Arena + AlpacaEval + AIME + SWE-Bench│
│              │ 生产评测: 私有数据集 + 持续CI门禁               │
└──────────────┴───────────────────────────────────────────────┘
```

### 6.2 按目标速查

| 目标 | 数据 | 算法 | 基座模型 | 硬件（最小） | 评测重点 |
|------|------|------|----------|-------------|----------|
| **推理增强** | CoT蒸馏 5K-50K | GRPO+RLVR | DeepSeek-R1-Distill-7B | 1×A100 40GB | AIME, MATH-500 |
| **风格对齐** | 偏好对 10K-100K | DPO/SimPO | Qwen 2.5-7B | 1×A100 40GB | AlpacaEval, Arena |
| **Agent工具使用** | 真实轨迹 5K-50K | GRPO on Traj | Qwen 2.5-7B / Llama 3.1-8B | 1×A100 40GB | BFCL, τ-Bench |
| **全面对齐** | SFT 50K + 偏好 20K | SFT→DPO→GRPO | Qwen 2.5-14B/72B | 4×A100 80GB | 全基准 + 私有评测 |

### 6.3 核心经验总结

1. **从小模型开始验证流水线**：在 1.5B–7B 模型上验证数据、算法、评测的完整链路，再扩展到更大规模。AAAI 2026 论文证明 $42 即可在 4×A40 上跑通完整 GRPO 训练。

2. **蒸馏优于从头 RL（对中小模型）**：DeepSeek-R1 团队结论——对 ≤32B 模型，从强模型蒸馏推理能力远比从头 RL 训练经济高效。

3. **可验证奖励是成功关键**：数学、代码等有确定性答案的领域最适合 RLVR 训练；模糊领域需混合多种奖励信号并持续验证。

4. **数据质量 > 数据数量**：TIF 论文证明 50% 的高价值数据胜于 100% 的原始数据；Hölder-DPO 发现数据集中约 25% 的噪声标签广泛存在。

5. **不要迷信单一基准**：2025 年已有论文证明空模型可在主流基准上获得高排名；必须三角验证 + 私有评测 + 人工抽检。

6. **训练与推理协同设计**：vLLM 异步 rollout、多 Token 预测 (MTP)、动态内存卸载等基础设施优化能带来 2–3× 的训练加速。

7. **RLHF 不会消亡，但角色在演变**：RLHF 正从通用对齐手段转变为薄层专用工具——用于风格、语调、品牌声音；核心推理能力提升已全面转向 RLVR 路线。

---

## 参考文献

1. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv:2501.12948, 2025/2026.
2. DAPO: Decoupled Alignment Policy Optimization. ByteDance/Tsinghua, 2025.
3. Dr. GRPO: Understanding R1-Zero-Like Training: A Critical Perspective. arXiv:2503.20783, 2025.
4. GRPO with Verifiable Rewards: Effective Loss, Dynamics, and Success Amplification. arXiv:2503.06639, 2025.
5. Reinforcement Learning for Reasoning in Small LLMs: What Works and What Doesn't. AAAI 2026.
6. DisCO: Discriminative Constrained Optimization for Large Reasoning Models. NeurIPS 2025.
7. O-Researcher: Open Ended Deep Research Model via Multi-Agent Distillation and Agentic RL. arXiv:2601.03743, 2026.
8. Demystifying RL in Agentic Reasoning: Design Choices and Impacts. arXiv, 2025.
9. AIF-GEN: Open-Source Platform and Synthetic Dataset Suite for RLHF. ICML 2025.
10. VickreyFeedback: Cost-efficient Data Construction for RLHF. arXiv:2409.18417, 2024.
11. Self-Preference: Automated Method for Preference-Aligned Data from Business Metrics. Springer, 2026.
12. Hölder-DPO: Robust Alignment with Built-in Data Valuation. arXiv:2505.17859, 2025.
13. SAC: Source-Aware Cleaning for Human Feedback. arXiv:2410.01957, 2024.
14. TIF: Towards Understanding Valuable Preference Data for LLM Alignment. arXiv:2510.13212, 2025.
15. Reward Modeling Requires Automatic Adjustment Based on Data Quality. EMNLP 2024 Findings.
16. RLHF Algorithms Ranked: An Extensive Evaluation Across Diverse Tasks. EMNLP 2025 Industry Track.
17. A Technical Survey of Reinforcement Learning Techniques for LLMs. arXiv:2507.04136, 2025.
18. SPARK: Synergistic Policy And Reward Co-Evolving Framework. arXiv:2509.22624, 2025.
19. Cheating Automatic LLM Benchmarks: Null Models Achieve High Win Rates. ICLR 2025 Oral.
20. Long-Context LLMs Through the Data Lens: Training, Reasoning, and Evaluation. HKUST-GZ, 2026.
21. Post-Training in 2026: GRPO, DAPO, RLVR & Beyond. llm-stats.com, 2026.
22. Pairwise-RL: A Unified Pairwise Framework for RLHF. ByteDance, arXiv:2504.04950, 2025.
23. A Comprehensive Survey on Learning from Rewards for LLMs. EMNLP 2025 Findings.
24. Reward Modeling for Human Preferences. UC Berkeley Tech Report (EECS-2025-82), 2025.
25. ToolRM: Reward Models for Tool-Use Scenarios. 2025.
26. LAIMARK: Self-Generated Training Curriculum for GRPO. 2026.
27. AgenticQwen: Open-Source Data Flywheels for Small Agentic LMs. 2026.
28. Back to Basics: Revisiting REINFORCE for RLHF. Ahmadian et al., 2024.
29. SimPO: Simple Preference Optimization with a Reference-Free Reward. 2024.
30. ORPO: Monolithic Preference Optimization without Reference Model. 2024.

---

> **文档说明**：本文档基于 2024–2026 年公开论文、技术报告与工业实践编写，部分数字为近似值或基于特定评测配置的结果，实际应用时需要根据具体场景进行适配验证。
