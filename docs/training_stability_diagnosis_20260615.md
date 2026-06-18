# MCP SFT Stability Diagnosis (2026-06-15)

## Scope

- Job: `bifrost-2026060214414601-yans2`
- Run: `runs/qwen3_4b_mcp_v1_6k_lr1e7_seed50`
- Model input: `runs/qwen3_4b_sft_v4_sota_2k_lr2e7_seed40/hf`
- Configuration: 16-way DDP, BF16 parameters, global batch 32, LR `1e-7`,
  gradient accumulation 2, completion-only loss.

## Findings

### 1. The displayed loss was scaled incorrectly

The old logger summed two accumulation losses and divided only by world size.
It did not divide by `grad_accum_steps=2`. A displayed loss of 8-12 therefore
represented an example-averaged loss of roughly 4-6.

It also logged one optimization step every 20 steps instead of the mean over
the 20-step interval. This made a heterogeneous mixture look more unstable.

### 2. The data mixture creates high loss variance

Supervised completion lengths differ substantially:

| Source | Rows | Mean supervised tokens | Notable issue |
|---|---:|---:|---|
| MCP positive | 52,800 | 53.6 | Generally easy for the input model |
| MCP no-tool | 14,400 | 2.0 | Target is only `[]` |
| MCP clarify | 9,600 | 2.0 | Target is also `[]`, so it does not teach clarification text |
| General replay | 19,200 | 187.3 | 14.2% exceed sequence length 2048 |

The old trainer averaged per-example losses. A two-token rejection example and
a long replay answer therefore had equal batch weight. Random changes in source
composition produced large step-to-step changes.

### 3. Full-parameter updates were rounded away

The model was loaded directly with BF16 parameters and optimized with AdamW
without FP32 master parameters. The learning rate was only `1e-7`.

Weight comparison between input and final models showed:

- Files have different hashes, so saving did occur.
- Most sampled parameter elements are exactly unchanged.
- Changed values move in increments of `6.1035e-05`, consistent with BF16
  quantization.
- Mean absolute parameter changes were around `1e-10`.

The intended updates were usually smaller than one BF16 representable step and
were discarded when written back to model weights.

### 4. Fixed-subset validation confirms ineffective learning

The same 32 examples per source were evaluated at input, step 2000, step 4000,
and step 6000.

| Checkpoint | Overall token loss | MCP positive | No-tool | Clarify | Replay |
|---|---:|---:|---:|---:|---:|
| Input | 1.5463 | 0.8337 | 15.7363 | 14.2109 | 1.4224 |
| Step 2000 | 1.5464 | 0.8327 | 15.7598 | 14.2383 | 1.4222 |
| Step 4000 | 1.5455 | 0.8347 | 15.7383 | 14.2324 | 1.4208 |
| Step 6000 | 1.5473 | 0.8332 | 15.7637 | 14.2090 | 1.4236 |

The changes are noise-level. The run did not improve the intended no-tool or
clarification behavior.

### 5. Gradient clipping was active almost continuously

Old pre-clipping norms were typically 50-200 and reached 358. With corrected
global token normalization, a two-step smoke test reported norms around 12.
Both still exceed `max_grad_norm=1.0`.

This is not the primary failure, but future runs must report clipping rate and
compare thresholds rather than treating the raw norm as an optimization loss.

## Corrections Applied

`scripts/remote/train_assistant_only_sft.py` now:

- Computes exact global supervised-token mean loss.
- Handles gradient accumulation without per-example weighting distortion.
- Logs interval loss, step loss, EMA, mean gradient norm, and clipping rate.
- Uses `DDP.no_sync()` for non-final accumulation micro-batches.
- Advances sampler epochs explicitly.

The corrected 16-PPU smoke test completed successfully and reported token loss
around 1.3 instead of the incorrectly scaled 8-12 range.

Fixed-loss evaluation outputs are in:

`evals/mcp_loss_diagnosis_20260615`

## Required Next Training Design

1. Run a LoRA validation experiment first. PEFT is installed and adapters can
   remain FP32 while the frozen base model stays BF16.
2. For full-parameter training, use FSDP with sharded FP32 master
   parameters/optimizer states and BF16 autocast. Native FSDP is importable in
   the PPU environment, but a communication and checkpoint smoke test is still
   required.
3. Replace clarification targets of `[]` with actual clarification responses.
   Keep no-tool and clarify as separate behaviors.
4. Use source-balanced batches or explicit source weights. Do not rely on
   example-mean loss across two-token and hundreds-token completions.
5. Filter or repack replay samples that exceed 2048 tokens.
6. Add a frozen, fixed validation split by source and evaluate it every
   checkpoint. Training loss alone is not an acceptance signal.
7. Reject a run automatically when fixed validation loss and a parameter-change
   sentinel fail to improve after the first 100-200 steps.
