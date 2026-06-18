#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/workspace/yans2@xiaopeng.com/agentic_rl}"
QUEUE="$ROOT/runs/ppu16_sft_ablation_20260612"

date
echo "=== queue ==="
cat "$QUEUE/queue.log" 2>/dev/null || true
echo "=== workers ==="
for log in "$QUEUE"/*_worker.log; do
  [[ -f "$log" ]] || continue
  echo "--- $(basename "$log")"
  tail -5 "$log"
done
echo "=== latest training metrics ==="
for run in \
  qwen2.5_coder_3b_v3_short_2k_lr3e7_seed30 \
  qwen2.5_coder_3b_v3_short_4k_lr3e7_seed31 \
  qwen2.5_coder_3b_v3_short_4k_lr5e7_seed32 \
  qwen2.5_coder_3b_v3_short_6k_lr3e7_seed33; do
  metrics="$ROOT/runs/$run/train_metrics.jsonl"
  [[ -f "$metrics" ]] || continue
  echo "--- $run"
  tail -2 "$metrics"
done
echo "=== processes ==="
ps -eo pid,etimes,%cpu,%mem,cmd | grep -E \
  'run_16ppu_sft_ablation|train_assistant_only_sft|evaluate_(xlam|general|mmlu|wikisql)|generate_ifeval' |
  grep -v grep || true
echo "=== accelerators ==="
nvidia-smi
