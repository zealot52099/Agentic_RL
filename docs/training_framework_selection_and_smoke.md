# 训练框架选型与首次训练验证

更新时间：2026-06-09

## 1. 当前约束

- 目标模型：优先验证 Qwen3-1.7B-Base，候选 Qwen3.5-2B-Base。
- 训练阶段：CPT/mid-training、SFT、RFT、Agentic RL。
- 当前算力：目标 job 有 4×H800 80GB；检查时 GPU 1 空闲，其余卡有已有进程。
- 当前软件：PyTorch 2.11、TorchTitan 0.2、Transformers 5.9、vLLM 0.21。
- 当前网络：GitHub 可访问，Hugging Face 与 ModelScope 无法从 job 直连。
- 当前数据：xLAM function-calling 60K、SWE-Gym OpenHands SFT、tau2-bench、
  AgentTuning 已落到持久化 workspace。

## 2. 多角度对比

评分为 1-5，越高越适合当前任务。吞吐评分强调 2B 模型和 4×H800，而不是超大规模集群。

| 框架 | CPT/预训练 | SFT | Agentic RL | 2B/4卡效率 | Qwen/HF 兼容 | 可控性 | 当前落地成本 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TorchTitan | 5 | 3 | 2 | 5 | 4 | 5 | 5 |
| Megatron-LM/Core | 5 | 3 | 2 | 4 | 3 | 5 | 2 |
| Nanotron | 4 | 3 | 1 | 4 | 3 | 5 | 2 |
| Transformers + Accelerate/DeepSpeed | 3 | 5 | 3 | 3 | 5 | 3 | 3 |
| LLaMA-Factory / ms-swift | 2 | 5 | 2 | 3 | 5 | 2 | 3 |
| TRL | 1 | 5 | 3 | 2 | 5 | 3 | 2 |
| verl | 1 | 3 | 5 | 4 | 4 | 4 | 2 |
| OpenRLHF | 1 | 3 | 4 | 4 | 4 | 4 | 2 |

### TorchTitan

优势：

- PyTorch 原生 FSDP2、TP、PP、CP、DCP checkpoint 和 `torch.compile`。
- 当前环境已安装且原生注册 Qwen3 0.6B/1.7B/4B/8B。
- 包含 Qwen3 Hugging Face state-dict adapter，可从 HF safetensors 初始化。
- 对 2B、4 卡这类中小规模实验，部署复杂度显著低于 Megatron。
- 训练代码透明，便于做数据配比、loss mask、优化器和稳定性实验。

限制：

- 默认文本数据注册较简化，需要项目侧提供数据适配器。
- SFT 的 assistant-only loss、packing 和多模态模板能力不如专用 SFT 框架。
- 不是完整 Agentic RL 平台，在线 rollout、环境调度和奖励计算应交给 verl。

### Megatron-LM/Core

优势是大规模 3D/4D 并行、吞吐和成熟度。当前 2B/4×H800 场景无法充分摊薄其
转换、安装、配置和调试成本。若后续扩展到 30B+ 或多节点，再做同数据吞吐赛马。

### Nanotron

代码清晰，适合预训练研究和可复现实验，但当前未安装、Qwen 资产和团队生态弱于
TorchTitan。作为独立复核后端有价值，不作为首轮主线。

### Transformers / DeepSpeed / SFT 封装框架

模型与数据生态最好，适合 SFT/LoRA 和快速兼容新架构。全参数 CPT 时，训练循环、
分布式 checkpoint、编译和并行策略的组合更分散。LLaMA-Factory/ms-swift 很适合
数据配方验证，但不应成为核心预训练基础设施。

### verl / OpenRLHF / TRL

这些框架解决的是后训练和 RL，不应被强行用作 CPT 后端。verl 在多轮 Agent loop、
异步 rollout、FSDP/Megatron worker 和 vLLM/SGLang 推理方面最符合本项目终局。
TRL 适合奖励函数和小规模 GRPO PoC。

## 3. 结论：分阶段双后端

不存在一个框架同时在 CPT、SFT 和异步 Agentic RL 上最优。当前推荐：

1. CPT/mid-training：TorchTitan。
2. SFT：首轮沿用 TorchTitan 验证全参数链路；需要 assistant-only loss、packing
   和 LoRA 快速实验时，增加 Transformers/TRL 或 ms-swift 适配层。
3. 在线 Agentic RL：verl + vLLM；环境接口保持框架无关。
4. 超过单节点或 30B+：用固定 token budget 对 TorchTitan 与 Megatron-Core 做
   吞吐、显存、收敛和 checkpoint 恢复赛马。

这不是“两套重复系统”：TorchTitan 负责稳定地更新模型，verl 负责环境 rollout
和策略优化。二者通过 HF checkpoint、tokenizer、chat template 和统一 trace schema
衔接。

## 4. 首次训练策略

### 4.1 冒烟测试

- 模型：TorchTitan 原生 Qwen3-0.6B。
- 初始化：第一轮允许随机初始化，只验证 tokenizer、数据、前后向、优化器、
  checkpoint 和日志链路；不能将其结果当作有效 CPT。
- 数据：xLAM 60K 本地 JSONL，经项目适配器序列化消息与工具调用。
- 设备：只使用空闲 GPU 1，不干扰 GPU 0/2/3 上已有任务。
- 参数：BF16，sequence length 512，micro batch 1，5 steps。

### 4.2 有效 CPT

冒烟通过后必须换为 Qwen3-1.7B-Base 或 Qwen3.5-2B-Base 的真实 Base 权重。
远端模型站点不可达，采用本地下载、校验 SHA256、SCP/对象存储上传。有效训练前
增加三项验收：

1. 权重加载后 step 0 loss 与 Transformers 前向交叉校验。
2. 训练 20 step 后保存、销毁进程、恢复并继续 5 step。
3. 导出 HF checkpoint，用 vLLM/Transformers 完成固定 prompts 推理。

## 5. 首轮配置后的扩展顺序

1. `0.6B random-init / 1 GPU / 5 steps`：基础链路。
2. `0.6B pretrained / 1 GPU / 20 steps`：HF 权重导入和恢复。
3. `1.7B pretrained / 1 GPU / 100 steps`：真实 CPT 小样本。
4. `1.7B pretrained / 4 GPU FSDP2 / 500 steps`：吞吐和稳定性。
5. 固定 10B token pilot：比较 Qwen3-1.7B 与 Qwen3.5-2B 的增益/成本。

## 6. 关键风险

| 风险 | 表现 | 处理 |
|---|---|---|
| 把随机初始化冒烟当 CPT | loss 能下降但通用能力归零 | 明确标记 smoke；正式实验强制记录 base checkpoint hash |
| xLAM 直接做 CPT | 工具格式提升但通用能力遗忘 | 与通用、代码、数学数据混合；限制 agent 数据比例 |
| 对全对话计算 loss | 学到 system/user 文本复述 | SFT 阶段实现 assistant/tool-action loss mask |
| tokenizer/chat template 漂移 | BFCL 格式错误、RL logprob 不一致 | 全阶段固定 tokenizer hash 和模板版本 |
| 远端下载失败 | job 内无法拉权重 | 本地下载、分片校验、断点上传、持久化缓存 |
| 误占用 GPU | 干扰同 job 任务或 OOM | 启动前查 PID；本次显式 `CUDA_VISIBLE_DEVICES=1` |
| checkpoint 只能恢复不能推理 | DCP 与 HF 格式混淆 | 定期做 DCP resume 与 HF export 两类测试 |
| 只比较 tokens/s | 高吞吐但 loss/能力异常 | 同时报 MFU、峰值显存、loss/token、恢复时间和评测增益 |

## 7. 2026-06-09 实测结果

已在 `bifrost-2026051921173700-yans2` 的空闲 GPU 1 完成：

- TorchTitan 0.2 + PyTorch 2.11 启动成功。
- Qwen3-0.6B 原生模型构建成功，TorchTitan 报告模型参数约 596M。
- xLAM 60K JSONL 加载成功，数据适配器生效。
- BF16、sequence length 512、batch size 1 完成 step 1-5。
- 稳态吞吐约 5.6K-6.9K tokens/s；峰值显存约 6.13 GiB。
- 完整 DCP checkpoint `step-5` 保存成功。
- 第二次进程从 `step-5` 恢复，继续完成 step 6-7 并保存 `step-7`。
- 单卡 `step-5` checkpoint 恢复约 41 秒，末步保存约 14 秒。

日志与 checkpoint 位于：

```text
/workspace/yans2@xiaopeng.com/agentic_rl/runs/qwen3_0.6b_smoke/
```

本次是基础设施冒烟，不是有效模型训练：使用的是随机初始化权重，且 xLAM 数据当前
序列化用于 CPT 链路测试，尚未实现 SFT assistant-only loss mask。正式训练的阻塞项
是取得并校验 Qwen3-1.7B-Base/Qwen3.5-2B-Base 权重与构建混合 CPT 数据。

## 8. Qwen3-1.7B-Base 真实权重训练验收

2026-06-09 后续取得真实 Base 权重：

```text
/publicdata/huggingface.co/Qwen/Qwen3-1.7B-Base
```

`model.safetensors` 大小为 3,441,185,608 bytes，SHA256 为：

```text
6df85b39330e5a425ee36253d0f894e4387e4f0a15b9c53cb467d668e6b3a841
```

直接从 `/publicdata` NAS 加载时，读取超过 TorchTitan 默认 100 秒通信超时，训练
进程收到 SIGKILL。将权重暂存到节点本地
`/tmp/agentic_rl_models/Qwen3-1.7B-Base`，校验 SHA256 一致，并将通信超时提高到
600 秒后，HF state-dict 导入耗时降到约 1.5 秒。

### 单卡验收

- GPU：1×H800。
- 上下文：1024。
- 真实 HF Base 权重导入成功。
- 初始训练 loss：约 0.88。
- 完成 2 step 并保存完整 checkpoint。
- 峰值显存：约 15.82 GiB。

### 四卡 FSDP2 验收

- GPU：4×H800，FSDP shard degree 4。
- 上下文：2048，local batch 1，global batch 4。
- 完成 step 1-10，随后从 `step-10` 恢复到 `step-12`。
- 稳态每卡吞吐约 11K-12.5K tokens/s。
- 稳态 MFU 约 13%-15%。
- 首轮峰值显存约 12.17 GiB/卡；恢复后约 13.74 GiB/卡。
- HF 权重分布式导入约 1.5-1.7 秒。
- 四卡完整 checkpoint 恢复约 28 秒，保存约 22-25 秒。
- 一个完整 optimizer checkpoint 约 11GB。

远端结果：

```text
/workspace/yans2@xiaopeng.com/agentic_rl/runs/qwen3_1.7b_base_4gpu_smoke/
```

至此真实 Base 初始化、FSDP2、分布式数据、优化器、DCP 保存与恢复均已打通。
但当前训练数据仍只有 xLAM/SWE-Gym 等 Agent 后训练数据，不满足通用 CPT 的数据
配比要求。正式 2000-step 长训需先建立通用文本、代码、数学和 Agent 数据 mixture，
否则容易出现灾难性遗忘。xLAM 长训还应放到 assistant-only loss 的 SFT 阶段，而
不是直接作为唯一 CPT 语料。
