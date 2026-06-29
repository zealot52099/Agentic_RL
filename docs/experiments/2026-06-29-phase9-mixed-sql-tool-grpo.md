# 2026-06-29 Phase9 Mixed SQL + Tool-Call GRPO

This experiment starts from the Phase6 merged model and runs synchronized 16-PPU GRPO with a mixed reward. The purpose is to combine the SQL execution gain observed in Phase8 with the tool-call routing and JSON-format gain from Phase6.

## Setup

```text
Job: bifrost-2026060214414601-yans2
Project: /workspace/yans2@xiaopeng.com/agentic_rl_pipeline
Base model: runs/phase6_qwen25_coder7b_sqltool_lora_ppu16_lr6e7_seed20260627_noswan_20260627_105305/merged_hf
Run: runs/phase9_swift_mixed_sql_tool_grpo_sync16_20260629_183720
Log: logs/phase9_swift_mixed_sql_tool_grpo_sync16_20260629_183720.log
Framework: ms-swift rlhf --rlhf_type grpo
Tracking: SwanLab local mode
```

## Files

```text
scripts/remote/prepare_phase9_mixed_grpo.py
scripts/remote/swift_mixed_sql_tool_reward_plugin.py
scripts/remote/run_swift_mixed_sql_tool_grpo_ppu16.sh
datasets/processed/phase9_mixed_sql_tool_grpo_20260629/train.jsonl
datasets/processed/phase9_mixed_sql_tool_grpo_20260629/manifest.json
```

## Data

The training set has 7373 examples:

| Type | Count | Source |
|---|---:|---|
| SQL execution | 4096 | Phase8 WikiSQL executable GRPO data |
| tool_call | 2048 | Phase5 unified train split |
| no_tool | 593 | Phase5 unified train split |
| clarify | 636 | Phase5 unified train split |

The intended mix was 60% SQL, 25% tool-call, and 15% boundary tasks. The final count is lower than 8192 because the currently available executable SQL GRPO set has 4096 examples.

## Reward

SQL samples use execution-based reward:

- SQL extraction.
- Safe `SELECT` only.
- SQLite execution success.
- Result exact match.
- Normalized SQL exact match as a small bonus.

Agent-action samples use structured reward:

- JSON object parse.
- `action` match.
- Tool call count.
- Tool name match.
- Argument exact match.
- `no_tool` empty-call discipline.
- `clarify` missing-field overlap and non-empty clarification message.

## Launch

```bash
REPORT_TO=swanlab SWANLAB_MODE=local MAX_STEPS=2000 \
  nohup bash scripts/remote/run_swift_mixed_sql_tool_grpo_ppu16.sh \
  phase9_swift_mixed_sql_tool_grpo_sync16_20260629_183720 \
  > logs/phase9_swift_mixed_sql_tool_grpo_sync16_20260629_183720.log 2>&1 &
```

Key hyperparameters:

```text
LoRA rank: 16
LoRA alpha: 32
LoRA dropout: 0.05
Learning rate: 2e-7
Max steps: 2000
num_generations: 4
beta: 0.03
DeepSpeed: ZeRO-1
```

## Initial Status

At `2026-06-29 18:37`, the run started with PID `1700020`.

At `2026-06-29 18:38`, all 16 `PPU-ZW810E` devices were detected and distributed/vLLM initialization was in progress. No step metrics had been emitted yet.

## Evaluation Plan

After training:

| Metric group | Dataset | Label |
|---|---|---|
| SQL execution | Internal rebased WikiSQL 256 probe | Internal probe, not official WikiSQL benchmark |
| Tool-call | `phase5_unified_20260626/validation.jsonl` | Internal unified validation |
| General retention | GSM8K fixed 256 subset | Internal subset |
| Instruction following | IFEval if runtime permits | Public benchmark rerun |

Success criteria:

- SQL execution accuracy should approach or exceed Phase8 `62.11%`.
- Tool action/tool-name exact should stay near Phase6 and clearly above Phase8-from-Phase5.
- GSM8K fixed subset should not regress by more than 1 percentage point.

## Runtime Update: SwanLab Fix and First Metrics

The first launch `phase9_swift_mixed_sql_tool_grpo_sync16_20260629_183720` failed before training because SwanLab local mode required `swanboard`:

```text
ImportError: Please install swanboard to use 'local' mode: pip install 'swanlab[dashboard]'
```

Fix applied on the remote job:

```bash
/opt/ac2/bin/python -m pip install 'swanlab[dashboard]' --no-cache-dir
```

This installed:

```text
swanboard-0.1.9b3
peewee-3.19.0
ujson-5.13.0
```

The successful restarted run is:

```text
Run: runs/phase9_swift_mixed_sql_tool_grpo_sync16_20260629_184129
PID: 1711045
Log: logs/phase9_swift_mixed_sql_tool_grpo_sync16_20260629_184129.log
Metrics: runs/phase9_swift_mixed_sql_tool_grpo_sync16_20260629_184129/v0-20260629-184156/logging.jsonl
```

Initial metrics at `99/2000`:

| Step | Reward | Reward std | Grad norm | KL | Mean length | Memory | Speed | ETA |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 99/2000 | 0.9750 | 0.0000 | 0.00038 | 0.0000315 | 19.81 | 36.18 GiB | 2.34 s/it | ~1h14m |

Recent-step reward is intentionally volatile because the mixed stream alternates SQL execution and tool/no-tool/clarify tasks. The healthier indicators at this stage are: no NaN, nonzero gradients on variable-reward groups, near-zero KL, no completion clipping, and steady step progress.

Current known non-blocking logs:

- `triton.language.target_info` warnings from vLLM/Triton compatibility.
- ModelScope connection retries for `unknown` model metadata. Local model loading continues and training steps are being written.

## Final Evaluation

Phase9 completed at `2000/2000` and was merged into:

```text
evals/phase9_swift_mixed_sql_tool_grpo_sync16_20260629/merged/phase9_mixed_grpo_sync16_step2000
```

Training summary:

```text
Runtime: 3603s
Train loss: 1.462e-05
Final checkpoint: checkpoint-2000
Trainable params: 40.37M, 0.5273%
```

Evaluation results, all internal fixed probes unless explicitly noted:

| Model | SQL exec acc | SQL exec rate | Tool action exact | Tool name exact | JSON exact | GSM8K fixed-256 acc |
|---|---:|---:|---:|---:|---:|---:|
| Phase5 SFT | 55.47% | 79.30% | 81.41% | 79.74% | 37.73% | 76.56% |
| Phase6 SQL/tool SFT | 51.56% | 73.44% | 95.72% | 95.35% | 50.56% | 76.17% |
| Phase8 SQL-only GRPO from Phase5 | 62.11% | 88.67% | 81.23% | 79.55% | 37.73% | 76.17% |
| Phase9 mixed GRPO from Phase6 | 55.86% | 80.08% | 95.72% | 95.35% | 50.56% | 75.00% |

Interpretation:

- Phase9 successfully preserved Phase6 tool-call capability. Tool action exact and tool name exact are effectively unchanged from Phase6 and much better than Phase8-from-Phase5.
- Phase9 improved SQL execution accuracy over Phase6 by `+4.30 pp`, but only slightly over Phase5 by `+0.39 pp` and clearly below Phase8 by `-6.25 pp`.
- GSM8K fixed-256 dropped to `75.00%`, around `-1.17 pp` versus Phase6/Phase8 and `-1.56 pp` versus Phase5. This is still small but should be treated as a regression signal.
- MMLU-Pro was not rerun because the required parquet file is still absent on the job. These numbers must not be treated as official full benchmark results.

Conclusion:

Phase9 achieved the intended tool-retention goal but did not achieve the SQL-improvement goal. The mixed GRPO reward likely under-optimized SQL because many tool/no-tool/clarify groups already receive high or low-variance rewards, while the SQL execution signal is harder and gets diluted. The next run should keep Phase6 as the base but use a staged schedule: short SQL-only GRPO from Phase6, then a lower-LR mixed tool-retention GRPO, or increase SQL sampling ratio and use per-task reward weighting.
