#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/workspace/yans2@xiaopeng.com/agentic_rl}"
PUBLIC_ROOT="/publicdata/huggingface.co/Qwen"
PRIOR_QUEUE="$ROOT/runs/qwen_model_comparison_queue_20260610"
QUEUE_DIR="$ROOT/runs/extended_qwen_queue_20260610"
EVAL_ROOT="$ROOT/evals/extended_qwen_queue_20260610"
TRAIN_DATA="$ROOT/datasets/processed/sft_v2_mixture_seed12/train_sft.jsonl"
XLAM_EVAL="$ROOT/datasets/processed/xlam_tool_family_v1/eval.jsonl"
GSM8K_EVAL="$ROOT/datasets/eval/gsm8k/test.parquet"
MMLU_EVAL="$ROOT/datasets/eval/mmlu_pro/test.parquet"

mkdir -p "$QUEUE_DIR" "$EVAL_ROOT"
exec >> "$QUEUE_DIR/queue.log" 2>&1

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*"; }

gpus_idle() {
  [[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader |
    sed '/^$/d' | wc -l)" -eq 0 ]]
}

wait_for_prior_queue() {
  log "Waiting for model comparison queue"
  while [[ ! -f "$PRIOR_QUEUE/COMPLETED" ]]; do
    if [[ -f "$PRIOR_QUEUE/queue.pid" ]]; then
      local pid
      pid="$(cat "$PRIOR_QUEUE/queue.pid")"
      if ! kill -0 "$pid" 2>/dev/null; then
        log "Prior queue exited without COMPLETED; continuing after GPU idle"
        break
      fi
    fi
    sleep 60
  done
  while ! gpus_idle; do sleep 20; done
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

valid_hf_model() {
  local model_dir="$1"
  [[ -s "$model_dir/model.safetensors" ||
    -s "$model_dir/model.safetensors.index.json" ]]
}

evaluate_model() {
  local run_name="$1"
  local model="$2"
  mkdir -p "$EVAL_ROOT/xlam" "$EVAL_ROOT/general" "$EVAL_ROOT/mmlu_logprob"
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

  local status=0
  wait "$p0" || status=1
  wait "$p1" || status=1
  wait "$p2" || status=1
  log "Evaluation completed for $run_name status=$status"
}

run_experiment() {
  local model_name="$1"
  local run_name="$2"
  local steps="$3"
  local learning_rate="$4"
  local seed="$5"
  local model
  model="$(stage_model "$model_name")"
  local run_dir="$ROOT/runs/$run_name"
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
log "Extended Qwen queue started PID $$"
wait_for_prior_queue

run_experiment \
  "Qwen3-0.6B-Base" \
  "qwen3_0.6b_sft_v2_hard77_lr3e6_seed14" \
  10000 3e-6 20260614

run_experiment \
  "Qwen2.5-0.5B" \
  "qwen2.5_0.5b_sft_v2_hard77_lr3e6_seed15" \
  10000 3e-6 20260615

run_experiment \
  "Qwen2.5-3B" \
  "qwen2.5_3b_sft_v2_hard77_lr1p5e6_seed16" \
  6000 1.5e-6 20260616

run_experiment \
  "Qwen3-4B-Base" \
  "qwen3_4b_sft_v2_hard77_lr1e6_seed17" \
  4000 1e-6 20260617

log "Extended Qwen queue completed"
touch "$QUEUE_DIR/COMPLETED"
