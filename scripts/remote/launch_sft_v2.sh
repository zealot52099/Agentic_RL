#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/yans2@xiaopeng.com/agentic_rl}"
MODEL="${MODEL:-/tmp/agentic_rl_models/Qwen3-1.7B-Base}"
TRAIN_DATA="${TRAIN_DATA:-$ROOT/datasets/processed/sft_v2_mixture/train_sft.jsonl}"
RUN_NAME="${RUN_NAME:-qwen3_1.7b_assistant_sft_v2_step500}"
RUN_DIR="$ROOT/runs/$RUN_NAME"
STEPS="${STEPS:-500}"
SAVE_EVERY="${SAVE_EVERY:-100}"

cd "$ROOT"
mkdir -p "$RUN_DIR"

if [[ -f "$RUN_DIR/launcher.pid" ]]; then
  old_pid="$(cat "$RUN_DIR/launcher.pid")"
  if kill -0 "$old_pid" 2>/dev/null; then
    echo "Training is already running: PID $old_pid"
    exit 1
  fi
fi

python - "$TRAIN_DATA" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
rows = 0
with path.open(encoding="utf-8") as handle:
    for rows, line in enumerate(handle, start=1):
        json.loads(line)
if rows == 0:
    raise SystemExit(f"Empty training data: {path}")
print(f"Validated {rows} JSONL rows: {path}")
PY

if [[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^$/d' | wc -l)" -ne 0 ]]; then
  echo "At least one GPU already has a compute process; refusing to overlap."
  exit 1
fi

command=(
  torchrun --standalone --nproc-per-node=4
  scripts/remote/train_assistant_only_sft.py
  --model "$MODEL"
  --train-data "$TRAIN_DATA"
  --output-dir "$RUN_DIR"
  --steps "$STEPS"
  --seq-len 2048
  --micro-batch-size 1
  --grad-accum-steps 8
  --learning-rate 5e-6
  --warmup-steps 25
  --log-every 5
  --save-every "$SAVE_EVERY"
  --seed 20260609
)

printf '%q ' CUDA_VISIBLE_DEVICES=0,1,2,3 "${command[@]}" > "$RUN_DIR/launch_command.txt"
printf '\n' >> "$RUN_DIR/launch_command.txt"
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 "${command[@]}" \
  > "$RUN_DIR/train.log" 2>&1 < /dev/null &
echo "$!" > "$RUN_DIR/launcher.pid"
echo "Started $RUN_NAME with PID $(cat "$RUN_DIR/launcher.pid")"
