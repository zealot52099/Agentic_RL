#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/yans2@xiaopeng.com/agentic_rl}"
QUEUE_DIR="$ROOT/runs/wikisql_eval_queue_20260610"
OUTPUT_DIR="$ROOT/evals/wikisql_20260610"

if [[ -f "$QUEUE_DIR/queue.pid" ]]; then
  pid="$(cat "$QUEUE_DIR/queue.pid")"
  ps -p "$pid" -o pid,etime,cmd --no-headers || echo "Queue PID $pid is not running"
fi
echo
tail -n 25 "$QUEUE_DIR/queue.log" 2>/dev/null || true
echo
python - "$OUTPUT_DIR" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
for path in sorted(root.glob("*_metrics.json")):
    data = json.loads(path.read_text())
    print(
        path.name,
        f"extract={data['sql_extraction_rate']:.2%}",
        f"execute={data['execution_rate']:.2%}",
        f"exec_acc={data['execution_accuracy']:.2%}",
        f"sql_exact={data['normalized_sql_exact']:.2%}",
    )
PY
