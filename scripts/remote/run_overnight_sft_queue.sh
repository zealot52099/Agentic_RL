#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/workspace/yans2@xiaopeng.com/agentic_rl}"
QUEUE_DIR="$ROOT/runs/overnight_sft_queue_20260609"
CURRENT_RUN="$ROOT/runs/qwen3_1.7b_assistant_sft_v2_step500"
BASE_MODEL="/tmp/agentic_rl_models/Qwen3-1.7B-Base"
XLAM_EVAL="$ROOT/datasets/processed/xlam_tool_family_v1/eval.jsonl"
GSM8K_EVAL="$ROOT/datasets/eval/gsm8k/test.parquet"
MMLU_EVAL="$ROOT/datasets/eval/mmlu_pro/test.parquet"

mkdir -p "$QUEUE_DIR"
exec >> "$QUEUE_DIR/queue.log" 2>&1

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  echo "[$(timestamp)] $*"
}

launcher_running() {
  local pid="$1"
  local state
  state="$(ps -p "$pid" -o stat= 2>/dev/null | tr -d ' ')"
  [[ -n "$state" && "$state" != Z* ]]
}

wait_for_current_run() {
  local pid
  pid="$(cat "$CURRENT_RUN/launcher.pid")"
  log "Waiting for current run PID $pid"
  while launcher_running "$pid"; do
    sleep 30
  done
  if [[ ! -s "$CURRENT_RUN/hf/model.safetensors" ]]; then
    log "Current run did not produce final HF weights"
    return 1
  fi
  log "Current run completed"
}

gpus_idle() {
  [[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader |
    sed '/^$/d' | wc -l)" -eq 0 ]]
}

wait_for_idle_gpus() {
  while ! gpus_idle; do
    log "Waiting for GPU compute processes to exit"
    sleep 20
  done
}

run_training() {
  local run_name="$1"
  local train_data="$2"
  local learning_rate="$3"
  local seed="$4"
  local run_dir="$ROOT/runs/$run_name"

  mkdir -p "$run_dir"
  log "Starting $run_name lr=$learning_rate seed=$seed"
  env CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 \
    scripts/remote/train_assistant_only_sft.py \
    --model "$BASE_MODEL" \
    --train-data "$train_data" \
    --output-dir "$run_dir" \
    --steps 6000 \
    --seq-len 2048 \
    --micro-batch-size 1 \
    --grad-accum-steps 8 \
    --learning-rate "$learning_rate" \
    --warmup-steps 100 \
    --log-every 20 \
    --save-every 1000 \
    --seed "$seed" \
    > "$run_dir/train.log" 2>&1
  local status=$?
  echo "$status" > "$run_dir/exit_code"
  if [[ "$status" -ne 0 || ! -s "$run_dir/hf/model.safetensors" ]]; then
    log "Training failed for $run_name with status $status"
    return 1
  fi
  log "Training completed for $run_name"
}

run_evaluations() {
  local run_name="$1"
  local model="$ROOT/runs/$run_name/hf"
  local output_root="$ROOT/evals/overnight_sft_queue_20260609"
  mkdir -p "$output_root/xlam" "$output_root/general" "$output_root/mmlu_logprob"

  log "Starting parallel evaluations for $run_name"
  CUDA_VISIBLE_DEVICES=0 python scripts/remote/evaluate_xlam_tool_calls.py \
    --model "$model" \
    --dataset "$XLAM_EVAL" \
    --output-dir "$output_root/xlam" \
    --model-label "$run_name" \
    > "$QUEUE_DIR/${run_name}_xlam.log" 2>&1 &
  local xlam_pid=$!

  CUDA_VISIBLE_DEVICES=1 python scripts/remote/evaluate_general_regression.py \
    --model "$model" \
    --model-label "$run_name" \
    --gsm8k "$GSM8K_EVAL" \
    --mmlu-pro "$MMLU_EVAL" \
    --output-dir "$output_root/general" \
    > "$QUEUE_DIR/${run_name}_general.log" 2>&1 &
  local general_pid=$!

  CUDA_VISIBLE_DEVICES=2 python scripts/remote/evaluate_mmlu_logprob.py \
    --model "$model" \
    --model-label "$run_name" \
    --dataset "$MMLU_EVAL" \
    --output-dir "$output_root/mmlu_logprob" \
    > "$QUEUE_DIR/${run_name}_mmlu_logprob.log" 2>&1 &
  local mmlu_pid=$!

  local status=0
  wait "$xlam_pid" || status=1
  wait "$general_pid" || status=1
  wait "$mmlu_pid" || status=1
  log "Evaluations finished for $run_name status=$status"
  return "$status"
}

cd "$ROOT"
log "Overnight queue started with PID $$"
wait_for_current_run || log "Continuing despite current-run validation failure"
wait_for_idle_gpus

declare -a experiments=(
  "qwen3_1.7b_sft_v2a_hard77_lr2e6_seed10|$ROOT/datasets/processed/sft_v2_mixture_seed10/train_sft.jsonl|2e-6|20260610"
  "qwen3_1.7b_sft_v2b_hard90_lr2e6_seed11|$ROOT/datasets/processed/sft_v2_mixture_hard90_seed11/train_sft.jsonl|2e-6|20260611"
  "qwen3_1.7b_sft_v2c_hard77_lr1e6_seed12|$ROOT/datasets/processed/sft_v2_mixture_seed12/train_sft.jsonl|1e-6|20260612"
)

for experiment in "${experiments[@]}"; do
  IFS='|' read -r run_name train_data learning_rate seed <<< "$experiment"
  if run_training "$run_name" "$train_data" "$learning_rate" "$seed"; then
    run_evaluations "$run_name" || true
  fi
  wait_for_idle_gpus
done

log "Overnight queue completed"
touch "$QUEUE_DIR/COMPLETED"
