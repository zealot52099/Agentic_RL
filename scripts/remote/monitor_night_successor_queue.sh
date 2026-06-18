#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/workspace/yans2@xiaopeng.com/agentic_rl}"
QUEUE_DIR="$ROOT/runs/night_successor_queue_20260610"

echo "=== time ==="
date
echo "=== queue ==="
if [[ -f "$QUEUE_DIR/queue.log" ]]; then
  tail -30 "$QUEUE_DIR/queue.log"
else
  echo "queue log not created"
fi
echo "=== process ==="
if [[ -f "$QUEUE_DIR/queue.pid" ]]; then
  pid="$(cat "$QUEUE_DIR/queue.pid")"
  ps -fp "$pid" || true
fi
echo "=== training metrics ==="
for run in \
  qwen2.5_coder_3b_sft_v2_hard90_lr1p5e6_seed18 \
  qwen2.5_3b_sft_v2_hard90_lr1p5e6_seed19; do
  metrics="$ROOT/runs/$run/train_metrics.jsonl"
  if [[ -f "$metrics" ]]; then
    echo "--- $run"
    tail -3 "$metrics"
  fi
done
echo "=== gpu ==="
nvidia-smi
