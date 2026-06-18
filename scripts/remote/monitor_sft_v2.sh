#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/yans2@xiaopeng.com/agentic_rl}"
RUN_NAME="${RUN_NAME:-qwen3_1.7b_assistant_sft_v2_step500}"
RUN_DIR="$ROOT/runs/$RUN_NAME"

if [[ ! -d "$RUN_DIR" ]]; then
  echo "Missing run directory: $RUN_DIR"
  exit 1
fi

if [[ -f "$RUN_DIR/launcher.pid" ]]; then
  pid="$(cat "$RUN_DIR/launcher.pid")"
  ps -p "$pid" -o pid,etime,cmd --no-headers || echo "Launcher PID $pid is not running"
fi

echo
echo "Latest metrics:"
tail -n 10 "$RUN_DIR/train_metrics.jsonl" 2>/dev/null || true

echo
echo "Checkpoints:"
find "$RUN_DIR/checkpoints" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort -V || true

echo
echo "GPU:"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
