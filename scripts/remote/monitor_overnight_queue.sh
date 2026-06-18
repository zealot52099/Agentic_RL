#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/yans2@xiaopeng.com/agentic_rl}"
QUEUE_DIR="$ROOT/runs/overnight_sft_queue_20260609"

if [[ -f "$QUEUE_DIR/queue.pid" ]]; then
  pid="$(cat "$QUEUE_DIR/queue.pid")"
  ps -p "$pid" -o pid,etime,cmd --no-headers || echo "Queue PID $pid is not running"
fi

echo
echo "Queue log:"
tail -n 20 "$QUEUE_DIR/queue.log" 2>/dev/null || true

echo
echo "Active training metrics:"
for metrics in "$ROOT"/runs/qwen3_1.7b_sft_v2*/train_metrics.jsonl; do
  [[ -f "$metrics" ]] || continue
  echo "== $metrics =="
  tail -n 2 "$metrics"
done

echo
echo "GPU:"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
