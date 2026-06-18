#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/workspace/yans2@xiaopeng.com/agentic_rl}"
QUEUE="$ROOT/runs/optimization_24h_queue_20260611"

date
echo "=== queue ==="
tail -40 "$QUEUE/queue.log" 2>/dev/null || true
echo "=== latest metrics ==="
for run in \
  qwen2.5_coder_3b_sft_v3_instruction_lr8e7_seed20 \
  qwen2.5_coder_3b_v3_grpo_verifiable_seed21 \
  qwen3_4b_sft_v3_instruction_lr6e7_seed22 \
  qwen2.5_3b_sft_v3_instruction_lr8e7_seed23; do
  test -f "$ROOT/runs/$run/train_metrics.jsonl" && {
    echo "--- $run"
    tail -3 "$ROOT/runs/$run/train_metrics.jsonl"
  }
done
echo "=== processes ==="
ps -eo pid,etimes,%cpu,%mem,cmd | grep -E \
  'optimization_24h|train_assistant_only|train_verifiable_grpo' | grep -v grep || true
echo "=== gpu ==="
nvidia-smi
