#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/workspace/yans2@xiaopeng.com/agentic_rl}"
PRIOR_QUEUE="$ROOT/runs/extended_qwen_queue_20260610"
QUEUE_DIR="$ROOT/runs/wikisql_eval_queue_20260610"
OUTPUT_DIR="$ROOT/evals/wikisql_20260610"
DATASET="$ROOT/datasets/eval/wikisql/test_256.jsonl"
DATABASE="$ROOT/datasets/sources/wikisql/extracted/data/test.db"

mkdir -p "$QUEUE_DIR" "$OUTPUT_DIR"
exec >> "$QUEUE_DIR/queue.log" 2>&1

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*"; }

gpus_idle() {
  [[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader |
    sed '/^$/d' | wc -l)" -eq 0 ]]
}

valid_model() {
  local path="$1"
  [[ -s "$path/model.safetensors" ||
    -s "$path/model.safetensors.index.json" ]]
}

evaluate() {
  local label="$1"
  local model="$2"
  if ! valid_model "$model"; then
    log "Skipping missing or incomplete model: $label at $model"
    return
  fi
  log "Evaluating WikiSQL: $label"
  CUDA_VISIBLE_DEVICES=0 python scripts/remote/evaluate_wikisql.py \
    --model "$model" \
    --model-label "$label" \
    --dataset "$DATASET" \
    --database "$DATABASE" \
    --output-dir "$OUTPUT_DIR" \
    > "$QUEUE_DIR/${label}.log" 2>&1
  log "WikiSQL completed: $label status=$?"
}

cd "$ROOT"
log "WikiSQL evaluation queue started PID $$"
while [[ ! -f "$PRIOR_QUEUE/COMPLETED" ]]; do
  if [[ -f "$PRIOR_QUEUE/queue.pid" ]]; then
    pid="$(cat "$PRIOR_QUEUE/queue.pid")"
    if ! kill -0 "$pid" 2>/dev/null; then
      log "Prior queue exited without COMPLETED; continuing after GPU idle"
      break
    fi
  fi
  sleep 60
done
while ! gpus_idle; do sleep 20; done

evaluate \
  "qwen3_1.7b_sft_v2b_hard90_lr2e6_seed11" \
  "$ROOT/runs/qwen3_1.7b_sft_v2b_hard90_lr2e6_seed11/hf"
evaluate \
  "qwen2.5_1.5b_sft_v2_hard77_lr2e6_seed13" \
  "$ROOT/runs/qwen2.5_1.5b_sft_v2_hard77_lr2e6_seed13/hf"
evaluate \
  "qwen2.5_coder_1.5b_sft_v2_hard77_lr2e6_seed13" \
  "$ROOT/runs/qwen2.5_coder_1.5b_sft_v2_hard77_lr2e6_seed13/hf"
evaluate \
  "qwen3_0.6b_sft_v2_hard77_lr3e6_seed14" \
  "$ROOT/runs/qwen3_0.6b_sft_v2_hard77_lr3e6_seed14/hf"
evaluate \
  "qwen2.5_0.5b_sft_v2_hard77_lr3e6_seed15" \
  "$ROOT/runs/qwen2.5_0.5b_sft_v2_hard77_lr3e6_seed15/hf"
evaluate \
  "qwen2.5_3b_sft_v2_hard77_lr1p5e6_seed16" \
  "$ROOT/runs/qwen2.5_3b_sft_v2_hard77_lr1p5e6_seed16/hf"
evaluate \
  "qwen3_4b_sft_v2_hard77_lr1e6_seed17" \
  "$ROOT/runs/qwen3_4b_sft_v2_hard77_lr1e6_seed17/hf"

log "WikiSQL evaluation queue completed"
touch "$QUEUE_DIR/COMPLETED"
