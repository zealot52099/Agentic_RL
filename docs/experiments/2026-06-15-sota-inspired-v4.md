# SOTA-Inspired Agent Post-Training V4

## Public evidence

- Qwen3 uses staged post-training: long-CoT cold start, reasoning RL,
  thinking-mode fusion, then general RL for instruction following and agents.
- Qwen3.5 attributes much of its agent gain to scaling the diversity,
  difficulty, and generality of RL environments rather than optimizing one
  benchmark.
- DeepSeek-R1 combines cold-start SFT, GRPO with verifiable rewards, rejection
  sampling, a second SFT stage, and further RL over both reasoning and general
  preference tasks.
- xLAM/APIGen generates function-calling data and filters it through format,
  execution, and semantic verification. xLAM also balances heterogeneous agent
  trajectory sources instead of training on one tool distribution.
- Qwen3-Coder scales code and synthetic data, uses an older strong model to
  clean noisy samples, and trains on repository-scale interaction trajectories.

Exact proprietary data mixtures are not public. The recipe below is an
engineering adaptation of the disclosed principles, not a reproduction claim.

## V4 mixture

| Component | Share | Purpose |
|---|---:|---|
| Verified xLAM tool calls | 25% | Tool selection and arguments |
| Irrelevant-tool negatives | 10% | Learn when not to call tools |
| Broad/complex instructions | 35% | Preserve IFEval and general behavior |
| SWE-Gym/OpenHands action turns | 10% | Multi-step software agent behavior |
| GSM8K replay | 10% | Limit reasoning regression |
| Hard compositional constraints | 10% | SFT and verifiable RL curriculum |

Benchmark test data from IFEval, BFCL, SWE-bench, and WikiSQL is excluded.

## Training

1. Continue from the strongest current Qwen3-4B v3 checkpoint for 2,000 low-LR
   SFT steps.
2. Evaluate xLAM, IFEval, GSM8K, MMLU-Pro, and WikiSQL.
3. If regression gates pass, run 300-step RLVR on hard constraints.
4. Promote only when IFEval improves and no core metric regresses by more than
   two points.

The RL implementation now clones generated tensors outside inference mode and
uses harder constraints intended to produce non-zero within-group reward
variance.

## Sources

- Qwen3 technical report and official release blog
- Qwen3.5 official release blog
- DeepSeek-R1 paper
- xLAM paper and APIGen dataset card
- Qwen3-Coder official release blog
