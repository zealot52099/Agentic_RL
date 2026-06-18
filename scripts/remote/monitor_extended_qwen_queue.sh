#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/yans2@xiaopeng.com/agentic_rl}"
QUEUE_DIR="$ROOT/runs/extended_qwen_queue_20260610"

if [[ -f "$QUEUE_DIR/queue.pid" ]]; then
  pid="$(cat "$QUEUE_DIR/queue.pid")"
  ps -p "$pid" -o pid,etime,cmd --no-headers || echo "Queue PID $pid is not running"
fi

echo
tail -n 25 "$QUEUE_DIR/queue.log" 2>/dev/null || true

echo
for metrics in "$ROOT"/runs/qwen{2.5,3}_*sft_v2*/train_metrics.jsonl; do
  [[ -f "$metrics" ]] || continue
  case "$metrics" in
    *seed14*|*seed15*|*seed16*|*seed17*)
      echo "== $metrics =="
      tail -n 2 "$metrics"
      ;;
  esac
done

echo
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
