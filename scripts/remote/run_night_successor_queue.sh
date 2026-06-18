#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/workspace/yans2@xiaopeng.com/agentic_rl}"
PUBLIC_ROOT="/publicdata/huggingface.co/Qwen"
EXTENDED_QUEUE="$ROOT/runs/extended_qwen_queue_20260610"
WIKISQL_QUEUE="$ROOT/runs/wikisql_eval_queue_20260610"
QUEUE_DIR="$ROOT/runs/night_successor_queue_20260610"
EVAL_ROOT="$ROOT/evals/night_successor_queue_20260610"
TRAIN_DATA="$ROOT/datasets/processed/sft_v2_mixture_hard90_seed11/train_sft.jsonl"
XLAM_EVAL="$ROOT/datasets/processed/xlam_tool_family_v1/eval.jsonl"
GSM8K_EVAL="$ROOT/datasets/eval/gsm8k/test.parquet"
MMLU_EVAL="$ROOT/datasets/eval/mmlu_pro/test.parquet"
WIKISQL_EVAL="$ROOT/datasets/eval/wikisql/test_256.jsonl"
WIKISQL_DB="$ROOT/datasets/sources/wikisql/extracted/data/test.db"

mkdir -p "$QUEUE_DIR" "$EVAL_ROOT"
exec >> "$QUEUE_DIR/queue.log" 2>&1

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*"; }

gpus_idle() {
  [[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader |
    sed '/^$/d' | wc -l)" -eq 0 ]]
}

valid_hf_model() {
  local model_dir="$1"
  [[ -s "$model_dir/model.safetensors" ||
    -s "$model_dir/model.safetensors.index.json" ]]
}

wait_for_queue() {
  local name="$1"
  local directory="$2"
  log "Waiting for $name"
  while [[ ! -f "$directory/COMPLETED" ]]; do
    if [[ -f "$directory/queue.pid" ]]; then
      local pid
      pid="$(cat "$directory/queue.pid")"
      if ! kill -0 "$pid" 2>/dev/null; then
        log "$name exited without COMPLETED; continuing once GPUs are idle"
        break
      fi
    fi
    sleep 60
  done
}

stage_model() {
  local name="$1"
  local source="$PUBLIC_ROOT/$name"
  local destination="/tmp/agentic_rl_models/$name"
  if [[ ! -s "$destination/config.json" ]]; then
    log "Staging $name to node-local storage" >&2
    mkdir -p "$destination"
    cp -a "$source/." "$destination/"
  fi
  echo "$destination"
}

evaluate_model() {
  local run_name="$1"
  local model="$2"
  mkdir -p \
    "$EVAL_ROOT/xlam" \
    "$EVAL_ROOT/general" \
    "$EVAL_ROOT/mmlu_logprob" \
    "$EVAL_ROOT/wikisql"
  log "Evaluating $run_name"

  CUDA_VISIBLE_DEVICES=0 python scripts/remote/evaluate_xlam_tool_calls.py \
    --model "$model" --dataset "$XLAM_EVAL" \
    --output-dir "$EVAL_ROOT/xlam" --model-label "$run_name" \
    > "$QUEUE_DIR/${run_name}_xlam.log" 2>&1 &
  local p0=$!
  CUDA_VISIBLE_DEVICES=1 python scripts/remote/evaluate_general_regression.py \
    --model "$model" --model-label "$run_name" \
    --gsm8k "$GSM8K_EVAL" --mmlu-pro "$MMLU_EVAL" \
    --output-dir "$EVAL_ROOT/general" \
    > "$QUEUE_DIR/${run_name}_general.log" 2>&1 &
  local p1=$!
  CUDA_VISIBLE_DEVICES=2 python scripts/remote/evaluate_mmlu_logprob.py \
    --model "$model" --model-label "$run_name" --dataset "$MMLU_EVAL" \
    --output-dir "$EVAL_ROOT/mmlu_logprob" \
    > "$QUEUE_DIR/${run_name}_mmlu.log" 2>&1 &
  local p2=$!
  CUDA_VISIBLE_DEVICES=3 python scripts/remote/evaluate_wikisql.py \
    --model "$model" --model-label "$run_name" \
    --dataset "$WIKISQL_EVAL" --database "$WIKISQL_DB" \
    --output-dir "$EVAL_ROOT/wikisql" \
    > "$QUEUE_DIR/${run_name}_wikisql.log" 2>&1 &
  local p3=$!

  local status=0
  wait "$p0" || status=1
  wait "$p1" || status=1
  wait "$p2" || status=1
  wait "$p3" || status=1
  log "Evaluation completed for $run_name status=$status"
}

run_experiment() {
  local model_name="$1"
  local run_name="$2"
  local steps="$3"
  local learning_rate="$4"
  local seed="$5"
  local run_dir="$ROOT/runs/$run_name"

  if valid_hf_model "$run_dir/hf"; then
    log "Skipping completed training for $run_name"
    evaluate_model "$run_name" "$run_dir/hf"
    return
  fi

  local model
  model="$(stage_model "$model_name")"
  mkdir -p "$run_dir"
  log "Starting $run_name model=$model_name steps=$steps lr=$learning_rate"
  env CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 \
    scripts/remote/train_assistant_only_sft.py \
    --model "$model" --train-data "$TRAIN_DATA" --output-dir "$run_dir" \
    --steps "$steps" --seq-len 2048 --micro-batch-size 1 \
    --grad-accum-steps 8 --learning-rate "$learning_rate" \
    --warmup-steps 100 --log-every 20 --save-every 2000 \
    --seed "$seed" > "$run_dir/train.log" 2>&1
  local status=$?
  echo "$status" > "$run_dir/exit_code"
  if [[ "$status" -eq 0 ]] && valid_hf_model "$run_dir/hf"; then
    log "Training completed for $run_name"
    evaluate_model "$run_name" "$run_dir/hf"
  else
    log "Training failed for $run_name status=$status; continuing"
  fi
  while ! gpus_idle; do sleep 20; done
}

cd "$ROOT"
log "Night successor queue started PID $$"
wait_for_queue "extended Qwen queue" "$EXTENDED_QUEUE"
wait_for_queue "WikiSQL queue" "$WIKISQL_QUEUE"
while ! gpus_idle; do sleep 20; done

run_experiment \
  "Qwen2.5-Coder-3B" \
  "qwen2.5_coder_3b_sft_v2_hard90_lr1p5e6_seed18" \
  6000 1.5e-6 20260618

run_experiment \
  "Qwen2.5-3B" \
  "qwen2.5_3b_sft_v2_hard90_lr1p5e6_seed19" \
  6000 1.5e-6 20260619

log "Night successor queue completed"
touch "$QUEUE_DIR/COMPLETED"
