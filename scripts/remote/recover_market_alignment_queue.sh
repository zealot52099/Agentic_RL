#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/workspace/yans2@xiaopeng.com/agentic_rl}"
QUEUE_DIR="$ROOT/runs/market_alignment_queue_20260611"

if [[ -f "$QUEUE_DIR/queue.pid" ]]; then
  prior_pid="$(cat "$QUEUE_DIR/queue.pid")"
  while kill -0 "$prior_pid" 2>/dev/null; do
    sleep 30
  done
fi

rm -f "$QUEUE_DIR/COMPLETED"
exec bash "$ROOT/scripts/remote/run_market_alignment_queue.sh"
