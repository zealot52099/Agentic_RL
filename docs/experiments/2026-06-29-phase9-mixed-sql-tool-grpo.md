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
