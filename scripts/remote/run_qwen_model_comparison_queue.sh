#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/workspace/yans2@xiaopeng.com/agentic_rl}"
PUBLIC_ROOT="/publicdata/huggingface.co/Qwen"
PRIOR_QUEUE="$ROOT/runs/overnight_sft_queue_20260609"
QUEUE_DIR="$ROOT/runs/qwen_model_comparison_queue_20260610"
EVAL_ROOT="$ROOT/evals/qwen_model_comparison_20260610"
TRAIN_DATA="$ROOT/datasets/processed/sft_v2_mixture_seed10/train_sft.jsonl"
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
  log "Waiting for prior overnight queue"
  while [[ ! -f "$PRIOR_QUEUE/COMPLETED" ]]; do
    if [[ -f "$PRIOR_QUEUE/queue.pid" ]]; then
      local pid
      pid="$(cat "$PRIOR_QUEUE/queue.pid")"
      if ! kill -0 "$pid" 2>/dev/null; then
        log "Prior queue exited without COMPLETED marker; continuing after GPU idle"
        break
      fi
    fi
    sleep 60
  done
  while ! gpus_idle; do sleep 20; done
}

evaluate_model() {
  local name="$1"
  local model="$2"
  local label="${name//./_}"
  label="${label//-/_}"
  mkdir -p "$EVAL_ROOT/xlam" "$EVAL_ROOT/general" "$EVAL_ROOT/mmlu_logprob"
  log "Evaluating $name from $model"

  CUDA_VISIBLE_DEVICES=0 python scripts/remote/evaluate_xlam_tool_calls.py \
    --model "$model" --dataset "$XLAM_EVAL" \
    --output-dir "$EVAL_ROOT/xlam" --model-label "$label" \
    > "$QUEUE_DIR/${label}_xlam.log" 2>&1 &
  local p0=$!
  CUDA_VISIBLE_DEVICES=1 python scripts/remote/evaluate_general_regression.py \
    --model "$model" --model-label "$label" \
    --gsm8k "$GSM8K_EVAL" --mmlu-pro "$MMLU_EVAL" \
    --output-dir "$EVAL_ROOT/general" \
    > "$QUEUE_DIR/${label}_general.log" 2>&1 &
  local p1=$!
  CUDA_VISIBLE_DEVICES=2 python scripts/remote/evaluate_mmlu_logprob.py \
    --model "$model" --model-label "$label" --dataset "$MMLU_EVAL" \
    --output-dir "$EVAL_ROOT/mmlu_logprob" \
    > "$QUEUE_DIR/${label}_mmlu.log" 2>&1 &
  local p2=$!

  local status=0
  wait "$p0" || status=1
  wait "$p1" || status=1
  wait "$p2" || status=1
  log "Evaluation completed for $name status=$status"
  return "$status"
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

train_comparison() {
  local name="$1"
  local model="$2"
  local run_name="$3"
  local run_dir="$ROOT/runs/$run_name"
  mkdir -p "$run_dir"
  log "Starting fair SFT comparison $run_name from $name"
  env CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 \
    scripts/remote/train_assistant_only_sft.py \
    --model "$model" --train-data "$TRAIN_DATA" --output-dir "$run_dir" \
    --steps 6000 --seq-len 2048 --micro-batch-size 1 \
    --grad-accum-steps 8 --learning-rate 2e-6 --warmup-steps 100 \
    --log-every 20 --save-every 1000 --seed 20260613 \
    > "$run_dir/train.log" 2>&1
  local status=$?
  echo "$status" > "$run_dir/exit_code"
  if [[ "$status" -eq 0 && -s "$run_dir/hf/model.safetensors" ]]; then
    evaluate_model "$run_name" "$run_dir/hf" || true
  else
    log "Training failed for $run_name status=$status"
  fi
}

cd "$ROOT"
log "Qwen model comparison queue started PID $$"
wait_for_prior_queue

declare -a zero_shot_models=(
  "Qwen3-0.6B-Base"
  "Qwen3-1.7B"
  "Qwen2.5-1.5B"
  "Qwen2.5-1.5B-Instruct"
  "Qwen2.5-Coder-1.5B"
  "Qwen2.5-Math-1.5B"
)

for name in "${zero_shot_models[@]}"; do
  evaluate_model "$name" "$PUBLIC_ROOT/$name" || true
  while ! gpus_idle; do sleep 10; done
done

general_model="$(stage_model Qwen2.5-1.5B)"
train_comparison \
  "Qwen2.5-1.5B" "$general_model" \
  "qwen2.5_1.5b_sft_v2_hard77_lr2e6_seed13"
while ! gpus_idle; do sleep 20; done

coder_model="$(stage_model Qwen2.5-Coder-1.5B)"
train_comparison \
  "Qwen2.5-Coder-1.5B" "$coder_model" \
  "qwen2.5_coder_1.5b_sft_v2_hard77_lr2e6_seed13"

log "Qwen model comparison queue completed"
touch "$QUEUE_DIR/COMPLETED"
