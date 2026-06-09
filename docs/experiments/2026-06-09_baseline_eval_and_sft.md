# 2026-06-09 基线评测与 SFT 准备实验日志

## 目标

1. 将 TorchTitan FSDP/DCP checkpoint 导出为 Hugging Face 格式。
2. 建立不可变的 xLAM held-out 工具调用评测集。
3. 在完全相同的 prompt、采样参数和 scorer 下比较：
   - 原始 Qwen3-1.7B-Base。
   - TorchTitan Agent 数据短训 checkpoint。
4. 准备 assistant-only SFT 数据和后续训练入口。

## 环境

- Job：`bifrost-2026051921173700-yans2`
- GPU：4×NVIDIA H800 80GB
- Python：3.12.3
- PyTorch：2.11.0+cu130
- TorchTitan：0.2.0+gite98ae995
- Transformers：5.9.0
- vLLM：0.21.0
- 原始权重：
  `/publicdata/huggingface.co/Qwen/Qwen3-1.7B-Base`
- 节点本地权重缓存：
  `/tmp/agentic_rl_models/Qwen3-1.7B-Base`
- 原始权重 SHA256：
  `6df85b39330e5a425ee36253d0f894e4387e4f0a15b9c53cb467d668e6b3a841`

## 已知前置结果

- 原始 HF Base 权重可被 TorchTitan 正确导入。
- 4×H800 FSDP2 已训练到 step 12。
- step 10 到 step 12 的 checkpoint 恢复通过。
- 短训使用 xLAM 数据的全 token causal LM loss，不是规范 assistant-only SFT。
- 当前短训总 token 很少，只用于系统验收，不预期产生显著能力提升。

## 实验记录

### E1：DCP 到 Hugging Face 导出

状态：完成。

计划：

- 从四卡 `step-12` 恢复。
- 额外训练 1 step 到 `step-13`。
- 设置 `last_save_model_only=true`、`last_save_in_hf=true`。
- 使用 BF16 导出，避免不必要的 FP32 文件膨胀。
- 补齐 `config.json`、tokenizer 和 generation config。
- 用 Transformers 完成加载与固定 prompt 推理。

注意：导出模型比 `step-12` 多一个训练 step，所有比较记录为 `step-13`。

结果：

- 四卡从 `step-12` 恢复耗时约 57 秒。
- 额外 step 13 loss 为 0.9117。
- 成功导出 BF16 Hugging Face safetensors。
- 合并模型文件约 3.3GB。
- Transformers 成功加载为 `Qwen3ForCausalLM`。
- 参数量 1,720,574,976，logits 全部有限。

### E2：xLAM held-out

状态：完成。

划分原则：

- 固定 seed 和脚本版本。
- 按工具名/工具集合分组后哈希划分，避免同一 API family 同时出现在训练和 held-out。
- 保存样本 ID、工具 schema、用户请求和标准 tool calls。
- 评测集永不回流本轮训练。

指标：

- JSON 可解析率。
- 工具名准确率。
- 参数 JSON 精确匹配率。
- 完整 tool-call 集合准确率。
- 多工具调用数量准确率。
- 无额外文本的严格格式率。
- 推理 token、延迟和失败原因分布。

划分结果：

- 训练：55,043 条。
- held-out：4,957 条。
- 固定 eval：256 条，覆盖 256 个不同 held-out tool family。
- 总工具 family：11,761。
- held-out family：1,130。

### E3：配对基线

状态：完成。

统一设置：

- 相同 held-out 样本。
- 相同 prompt 模板。
- greedy decoding，temperature 0。
- 相同 max tokens 和停止条件。
- 保存原始 response，不做静默 JSON 修复。
- scorer 允许提取 Markdown code fence，但分别报告 strict 和 normalized 结果。

结果：

| 模型 | JSON 解析 | 完整调用 | 工具名 | 调用数量 |
|---|---:|---:|---:|---:|
| Qwen3-1.7B-Base | 96.88% | 41.41% | 78.12% | 82.03% |
| TorchTitan step-13 | 97.27% | 42.19% | 80.08% | 83.98% |
| Assistant-only SFT step-100 | 99.61% | 47.27% | 87.50% | 93.75% |

Base 对比 TorchTitan step-13：

- 完整调用提升 0.78 个百分点，bootstrap 95% CI `[-1.17, 3.12]`。
- 增益不显著，不能证明全 token 短训配方有效。

Base 对比 assistant-only SFT：

- 完整调用提升 5.86 个百分点，95% CI `[0.78, 10.94]`。
- 无序完整调用提升 6.64 个百分点，95% CI `[1.56, 11.72]`。
- 工具名提升 9.38 个百分点，95% CI `[4.30, 14.45]`。
- 调用数量提升 11.72 个百分点，95% CI `[7.03, 16.41]`。
- 该 pilot 的主要增益来自正确工具选择、调用数量和 JSON 格式。

### E4：Assistant-only SFT pilot

状态：完成。

配置：

- Base：Qwen3-1.7B-Base。
- 4×H800，DDP 全参数。
- 100 optimizer steps。
- global batch 32。
- sequence length 2048。
- prompt token label 全部为 `-100`。
- 只监督 canonical tool-call JSON 和 EOS。
- 峰值学习率 5e-6，10 step warmup，cosine decay。

训练结果：

- completion 监督 token：126,565。
- loss：3.75 降至约 0.71。
- 峰值显存：约 16.31GiB/卡。
- 训练耗时：约 155 秒，不含模型保存。
- HF 输出：
  `/workspace/yans2@xiaopeng.com/agentic_rl/runs/qwen3_1.7b_assistant_sft_v1_step100/hf`

### E5：通用能力回归 probe

状态：完成。

数据：

- GSM8K 官方 test parquet，固定 revision，按问题 SHA256 选择 256 条。
- MMLU-Pro 官方 test parquet，按 category 轮询选择 256 条。

| 模型 | GSM8K | MMLU-Pro |
|---|---:|---:|
| Qwen3-1.7B-Base | 66.02% | 14.45% |
| Assistant-only SFT step-100 | 69.92% | 16.80% |

限制：

- 这是固定子集回归 probe，不是官方全量成绩。
- MMLU-Pro 使用生成式 `Answer: X` 解析，Base/SFT 解析率分别只有 44.53%/50%。
- 下一版应使用 A-J 选项条件 logprob 或官方 harness，避免答案未在 token budget 内输出。
- 当前结果至少没有显示明显灾难性遗忘，但不能据此声称通用能力得到可靠提升。

### E6：MMLU-Pro 直接选项概率评测

状态：完成。

为避免生成式 `Answer: X` 的格式和 token budget 干扰，固定使用与 E5 相同的
256 条 category-stratified 样本，直接比较 A-J 十个单 token 在答案位置的条件
logit。

| 模型 | MMLU-Pro direct logprob |
|---|---:|
| Qwen3-1.7B-Base | 30.08% |
| Assistant-only SFT step-100 | 30.47% |

成对 bootstrap：

- 差值：+0.39 个百分点。
- 95% CI：`[-2.34, 2.73]` 个百分点。
- 6 条由错变对，5 条由对变错，245 条不变。
- 结论：没有观察到显著通用知识退化，也没有可靠提升。

### E7：工具调用错误分型

状态：完成。

Assistant-only SFT step-100 的 256 条 held-out 样本：

| 类型 | 样本数 | 占比 |
|---|---:|---:|
| 完全正确且严格格式 | 105 | 41.02% |
| 调用正确但带额外文本 | 15 | 5.86% |
| 参数错误 | 101 | 39.45% |
| 工具名错误 | 16 | 6.25% |
| 调用数量错误 | 15 | 5.86% |
| JSON 解析失败 | 1 | 0.39% |
| 仅调用顺序错误 | 3 | 1.17% |

按期望调用数量：

| 期望调用数 | Base ordered exact | SFT ordered exact |
|---|---:|---:|
| 1 | 50.00% | 61.11% |
| 2 | 41.34% | 51.40% |
| 3+ | 38.98% | 30.51% |

关键判断：

- 格式冷启动已经有效：strict format 从 0% 提升到 41.02%，解析失败降到 0.39%。
- 调用数量也明显改善，但参数错误成为首要瓶颈。
- 三工具及以上复杂任务出现退化，说明当前 100-step pilot 更偏向学习短而规范的
  输出，复杂组合与长参数监督不足。
- 下一轮不能只增加同分布训练步数，应提高困难参数样本、多调用样本和一般指令
  replay 的比例。

## 数据来源与校验

- GSM8K：
  `https://huggingface.co/datasets/openai/gsm8k`
- 本地文件 SHA256：
  `EE7B8DA9E381DF27B9E3F7758A159AB2BDAA4DBAA910546CBBC47E0CB44E4F59`
- MMLU-Pro：
  `https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro`
- 本地文件 SHA256：
  `0E24A191921C2F453518A537A8B2117BD137E7714D4EF1565E9BA06C1ECB9AD8`

## 复现命令与产物

主要脚本：

- `scripts/prepare_xlam_splits.py`
- `scripts/remote/train_assistant_only_sft.py`
- `scripts/remote/evaluate_xlam_tool_calls.py`
- `scripts/remote/evaluate_general_regression.py`
- `scripts/remote/evaluate_mmlu_logprob.py`
- `scripts/compare_xlam_evals.py`
- `scripts/compare_general_evals.py`
- `scripts/analyze_xlam_errors.py`

远端主要产物：

- HF 导出：
  `/workspace/yans2@xiaopeng.com/agentic_rl/runs/qwen3_1.7b_base_4gpu_smoke/checkpoint/step-13`
- SFT 权重：
  `/workspace/yans2@xiaopeng.com/agentic_rl/runs/qwen3_1.7b_assistant_sft_v1_step100/hf`
- xLAM 评测：
  `/workspace/yans2@xiaopeng.com/agentic_rl/evals/xlam_tool_family_v1`
- 通用生成式 probe：
  `/workspace/yans2@xiaopeng.com/agentic_rl/evals/general_regression_v1`
- MMLU-Pro direct logprob：
  `/workspace/yans2@xiaopeng.com/agentic_rl/evals/mmlu_logprob_v1`

## 下一轮实施方案

1. 构建 SFT v2 混合数据：40% 参数困难样本、25% 三工具及以上样本、20%
   普通单/双工具样本、15% 通用指令与数学 replay。
2. 为参数困难样本增加近邻负例：相似工具名、缺失必填字段、额外字段、类型错误、
   枚举值错误和用户实体复制错误。
3. 从 Base 重新训练 500 steps，每 100 steps 保存并运行固定 xLAM、GSM8K 和
   MMLU-Pro direct-logprob 回归。
4. 进入 RL 前设置门槛：held-out unordered exact 至少 55%，参数错误率低于
   30%，3+ 调用 ordered exact 不低于 Base，MMLU-Pro 回归不超过 2 个百分点。
5. 达到门槛后再使用 verl 启动可验证奖励 RL；奖励分解为 JSON 合法性、工具名、
   参数 schema、参数值、调用数量和完整任务成功，避免单一总分奖励被投机。

## 决策门槛

- 若短训模型只降低训练 loss、held-out 无提升：停止此全 token 配方。
- 若工具名提升但参数下降：加强 schema/参数监督和困难负例。
- 若 JSON 合法率低：先做格式冷启动 SFT，不进入 RL。
- 若通用能力回归超过 3%：提高通用数据比例并降低 Agent 学习率/训练 token。
- 未完成 Base 基线前，不启动大规模 SFT/RL。
