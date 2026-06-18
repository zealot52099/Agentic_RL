#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/yans2@xiaopeng.com/agentic_rl}"
QUEUE_DIR="$ROOT/runs/qwen_model_comparison_queue_20260610"

if [[ -f "$QUEUE_DIR/queue.pid" ]]; then
  pid="$(cat "$QUEUE_DIR/queue.pid")"
  ps -p "$pid" -o pid,etime,cmd --no-headers || echo "Queue PID $pid is not running"
fi

echo
tail -n 25 "$QUEUE_DIR/queue.log" 2>/dev/null || true

echo
for metrics in \
  "$ROOT"/runs/qwen2.5_1.5b_sft_v2_hard77_lr2e6_seed13/train_metrics.jsonl \
  "$ROOT"/runs/qwen2.5_coder_1.5b_sft_v2_hard77_lr2e6_seed13/train_metrics.jsonl; do
  [[ -f "$metrics" ]] || continue
  echo "== $metrics =="
  tail -n 3 "$metrics"
done

echo
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
