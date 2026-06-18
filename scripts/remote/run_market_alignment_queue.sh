#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/workspace/yans2@xiaopeng.com/agentic_rl}"
QUEUE_DIR="$ROOT/runs/market_alignment_queue_20260611"
EVAL_ROOT="$ROOT/evals/market_alignment_20260611"
IFEVAL_ROOT="$ROOT/third_party/google-research"
IFEVAL_INPUT="$ROOT/datasets/eval_suite/huggingface/google__IFEval/ifeval_input_data.jsonl"

mkdir -p "$QUEUE_DIR" "$EVAL_ROOT/ifeval"
exec >> "$QUEUE_DIR/queue.log" 2>&1

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*"; }

run_ifeval() {
  local gpu="$1"
  local label="$2"
  local model="$3"
  local output_dir="$EVAL_ROOT/ifeval/$label"
  mkdir -p "$output_dir"

  if [[ -s "$output_dir/metrics.json" ]]; then
    log "Skipping completed IFEval: $label"
    return
  fi
  log "Starting IFEval: $label"
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/remote/generate_ifeval_responses.py \
    --model "$model" --model-label "$label" --input-data "$IFEVAL_INPUT" \
    --output "$output_dir/responses.jsonl" \
    > "$QUEUE_DIR/${label}_generation.log" 2>&1
  local status=$?
  if [[ "$status" -ne 0 ]]; then
    log "IFEval generation failed: $label status=$status"
    return
  fi
  PYTHONPATH="$IFEVAL_ROOT" python -m instruction_following_eval.evaluation_main \
    --input_data="$IFEVAL_INPUT" \
    --input_response_data="$output_dir/responses.jsonl" \
    --output_dir="$output_dir" \
    > "$QUEUE_DIR/${label}_scoring.log" 2>&1
  status=$?
  if [[ "$status" -eq 0 ]]; then
    python scripts/remote/summarize_ifeval.py \
      --model-label "$label" \
      --strict "$output_dir/eval_results_strict.jsonl" \
      --loose "$output_dir/eval_results_loose.jsonl" \
      --output "$output_dir/metrics.json" \
      >> "$QUEUE_DIR/${label}_scoring.log" 2>&1
  fi
  log "IFEval completed: $label status=$status"
}

cd "$ROOT"
log "Market alignment queue started PID $$"

run_ifeval 0 \
  "ours_qwen2.5_coder_3b_hard90" \
  "$ROOT/runs/qwen2.5_coder_3b_sft_v2_hard90_lr1p5e6_seed18/hf" &
p0=$!
run_ifeval 1 \
  "ours_qwen3_4b_hard77" \
  "$ROOT/runs/qwen3_4b_sft_v2_hard77_lr1e6_seed17/hf" &
p1=$!
run_ifeval 2 \
  "official_qwen2.5_3b_instruct" \
  "/publicdata/huggingface.co/Qwen/Qwen2.5-3B-Instruct" &
p2=$!
run_ifeval 3 \
  "official_qwen3_4b" \
  "/publicdata/huggingface.co/Qwen/Qwen3-4B" &
p3=$!

status=0
wait "$p0" || status=1
wait "$p1" || status=1
wait "$p2" || status=1
wait "$p3" || status=1
echo "$status" > "$QUEUE_DIR/exit_code"
touch "$QUEUE_DIR/COMPLETED"
log "Market alignment queue completed status=$status"
