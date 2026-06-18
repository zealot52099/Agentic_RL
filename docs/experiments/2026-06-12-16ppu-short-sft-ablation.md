# 2026-06-12 16-PPU Short-SFT Ablation

## Objective

The 12,000-step Qwen2.5-Coder-3B v3 continuation reduced the internal xLAM
unordered exact score from 56.25% to 54.69%, GSM8K from 67.97% to 62.50%, and
official IFEval strict prompt accuracy from 33.09% to 31.79%. MMLU-Pro
direct-logprob and WikiSQL execution accuracy were flat. This experiment tests
whether the regression is primarily caused by excessive continuation length or
an overly high learning rate.

## Controlled Matrix

All four runs start from
`qwen2.5_coder_3b_sft_v2_hard90_lr1p5e6_seed18/hf`, use the same deterministic
SFT v3 mixture, global batch size 32, sequence length 2,048, completion-only
loss, bf16, FlashAttention 2, gradient checkpointing, and cosine decay.

| Run order | Steps | Peak LR | Seed |
|---|---:|---:|---:|
| 1 | 2,000 | 3e-7 | 20260630 |
| 2 | 4,000 | 3e-7 | 20260631 |
| 3 | 4,000 | 5e-7 | 20260632 |
| 4 | 6,000 | 3e-7 | 20260633 |

The matrix prioritizes final exports over intermediate checkpoints to limit
storage amplification from four concurrent full-parameter runs.

The PPU runtime permits one C4D/NCCL communication domain per node. Training
therefore runs sequentially with one 16-rank DDP job. Gradient accumulation is
2, preserving the original global batch size of 32. After all training jobs,
the four evaluation bundles run concurrently on disjoint four-PPU groups.

## Evaluation And Selection

Each run is evaluated independently on its own four-PPU partition:

- Internal xLAM held-out tool-call probe
- GSM8K regression probe
- MMLU-Pro direct-logprob probe
- Internal WikiSQL execution probe
- Official IFEval scorer

A candidate is promotable only if it improves IFEval over the v2 starting point
without reducing xLAM, GSM8K, or MMLU-Pro by more than two percentage points.
WikiSQL execution accuracy is the primary SQL gate; normalized SQL string exact
match remains diagnostic only.

## Failure Handling

- Each run records training and evaluation exit codes separately.
- A failed training run does not stop the remaining experiments.
- Existing valid Hugging Face exports are reused after restart.
- Training uses one 16-rank communication domain; evaluation GPU IDs are
  explicit to prevent cross-worker collisions.
- The queue writes one completion marker per worker and a global `COMPLETED`
  marker after all workers exit.

## Paths

- Queue: `scripts/remote/run_16ppu_sft_ablation_20260612.sh`
- Monitor: `scripts/remote/monitor_16ppu_sft_ablation_20260612.sh`
- Logs: `runs/ppu16_sft_ablation_20260612`
- Metrics: `evals/ppu16_sft_ablation_20260612`
