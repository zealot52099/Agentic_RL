#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/workspace/yans2@xiaopeng.com/agentic_rl}"
QUEUE_DIR="$ROOT/runs/optimization_24h_queue_20260611"
EVAL_ROOT="$ROOT/evals/optimization_24h_20260611"
TRAIN_DATA="$ROOT/datasets/processed/sft_v3_instruction_balanced_seed20/train_sft.jsonl"
RL_DATA="$ROOT/datasets/processed/sft_v3_instruction_balanced_seed20/rl_verifiable.jsonl"
XLAM_EVAL="$ROOT/datasets/processed/xlam_tool_family_v1/eval.jsonl"
GSM8K_EVAL="$ROOT/datasets/eval/gsm8k/test.parquet"
MMLU_EVAL="$ROOT/datasets/eval/mmlu_pro/test.parquet"
WIKISQL_EVAL="$ROOT/datasets/eval/wikisql/test_256.jsonl"
WIKISQL_DB="$ROOT/datasets/sources/wikisql/extracted/data/test.db"
IFEVAL_INPUT="$ROOT/datasets/eval_suite/huggingface/google__IFEval/ifeval_input_data.jsonl"
IFEVAL_ROOT="$ROOT/third_party/google-research"

mkdir -p "$QUEUE_DIR" "$EVAL_ROOT"
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

wait_for_all_gpus() {
  log "Waiting for all GPUs to become idle"
  while ! gpus_idle; do
    sleep 30
  done
  log "All GPUs are idle"
}

evaluate_model() {
  local label="$1"
  local model="$2"
  local output="$EVAL_ROOT/$label"
  mkdir -p "$output"/{xlam,general,mmlu,wikisql,ifeval}
  log "Starting evaluation: $label"

  CUDA_VISIBLE_DEVICES=0 python scripts/remote/evaluate_xlam_tool_calls.py \
    --model "$model" --dataset "$XLAM_EVAL" \
    --output-dir "$output/xlam" --model-label "$label" \
    > "$QUEUE_DIR/${label}_xlam.log" 2>&1 &
  p0=$!
  CUDA_VISIBLE_DEVICES=1 python scripts/remote/evaluate_general_regression.py \
    --model "$model" --model-label "$label" \
    --gsm8k "$GSM8K_EVAL" --mmlu-pro "$MMLU_EVAL" \
    --output-dir "$output/general" \
    > "$QUEUE_DIR/${label}_general.log" 2>&1 &
  p1=$!
  CUDA_VISIBLE_DEVICES=2 python scripts/remote/evaluate_mmlu_logprob.py \
    --model "$model" --model-label "$label" --dataset "$MMLU_EVAL" \
    --output-dir "$output/mmlu" \
    > "$QUEUE_DIR/${label}_mmlu.log" 2>&1 &
  p2=$!
  CUDA_VISIBLE_DEVICES=3 python scripts/remote/evaluate_wikisql.py \
    --model "$model" --model-label "$label" \
    --dataset "$WIKISQL_EVAL" --database "$WIKISQL_DB" \
    --output-dir "$output/wikisql" \
    > "$QUEUE_DIR/${label}_wikisql.log" 2>&1 &
  p3=$!
  status=0
  wait "$p0" || status=1
  wait "$p1" || status=1
  wait "$p2" || status=1
  wait "$p3" || status=1

  CUDA_VISIBLE_DEVICES=0 python scripts/remote/generate_ifeval_responses.py \
    --model "$model" --model-label "$label" --input-data "$IFEVAL_INPUT" \
    --output "$output/ifeval/responses.jsonl" \
    > "$QUEUE_DIR/${label}_ifeval_generation.log" 2>&1
  if [[ "$?" -eq 0 ]]; then
    PYTHONPATH="$IFEVAL_ROOT" python -m instruction_following_eval.evaluation_main \
      --input_data="$IFEVAL_INPUT" \
      --input_response_data="$output/ifeval/responses.jsonl" \
      --output_dir="$output/ifeval" \
      > "$QUEUE_DIR/${label}_ifeval_scoring.log" 2>&1
    if [[ "$?" -eq 0 ]]; then
      python scripts/remote/summarize_ifeval.py \
        --model-label "$label" \
        --strict "$output/ifeval/eval_results_strict.jsonl" \
        --loose "$output/ifeval/eval_results_loose.jsonl" \
        --output "$output/ifeval/metrics.json" \
        >> "$QUEUE_DIR/${label}_ifeval_scoring.log" 2>&1
    else
      status=1
    fi
  else
    status=1
  fi
  log "Evaluation completed: $label status=$status"
}

run_sft() {
  local input_model="$1"
  local label="$2"
  local steps="$3"
  local lr="$4"
  local seed="$5"
  local run_dir="$ROOT/runs/$label"
  if valid_model "$run_dir/hf"; then
    log "Skipping completed SFT: $label"
  else
    mkdir -p "$run_dir"
    log "Starting SFT: $label steps=$steps lr=$lr"
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 \
      scripts/remote/train_assistant_only_sft.py \
      --model "$input_model" --train-data "$TRAIN_DATA" \
      --output-dir "$run_dir" --steps "$steps" --seq-len 2048 \
      --micro-batch-size 1 --grad-accum-steps 8 \
      --learning-rate "$lr" --warmup-steps 100 \
      --log-every 20 --save-every 4000 --seed "$seed" \
      > "$run_dir/train.log" 2>&1
    status=$?
    echo "$status" > "$run_dir/exit_code"
    log "SFT finished: $label status=$status"
  fi
  if valid_model "$run_dir/hf"; then
    evaluate_model "$label" "$run_dir/hf"
  fi
  wait_for_all_gpus
}

run_rlvr() {
  local input_model="$1"
  local label="$2"
  local run_dir="$ROOT/runs/$label"
  if valid_model "$run_dir/hf"; then
    log "Skipping completed RLVR: $label"
  else
    mkdir -p "$run_dir"
    log "Starting verifiable GRPO-style RLVR: $label"
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 \
      scripts/remote/train_verifiable_grpo.py \
      --model "$input_model" --dataset "$RL_DATA" --output-dir "$run_dir" \
      --steps 1200 --group-size 4 --max-new-tokens 160 \
      --temperature 0.8 --top-p 0.95 --learning-rate 1e-7 \
      --warmup-steps 50 --log-every 10 --save-every 400 \
      --seed 20260621 > "$run_dir/train.log" 2>&1
    status=$?
    echo "$status" > "$run_dir/exit_code"
    log "RLVR finished: $label status=$status"
  fi
  if valid_model "$run_dir/hf"; then
    evaluate_model "$label" "$run_dir/hf"
  fi
  wait_for_all_gpus
}

cd "$ROOT"
log "24-hour optimization queue started PID $$"
wait_for_all_gpus

run_sft \
  "$ROOT/runs/qwen2.5_coder_3b_sft_v2_hard90_lr1p5e6_seed18/hf" \
  "qwen2.5_coder_3b_sft_v3_instruction_lr8e7_seed20" \
  12000 8e-7 20260620

run_rlvr \
  "$ROOT/runs/qwen2.5_coder_3b_sft_v3_instruction_lr8e7_seed20/hf" \
  "qwen2.5_coder_3b_v3_grpo_verifiable_seed21"

run_sft \
  "$ROOT/runs/qwen3_4b_sft_v2_hard77_lr1e6_seed17/hf" \
  "qwen3_4b_sft_v3_instruction_lr6e7_seed22" \
  10000 6e-7 20260622

run_sft \
  "$ROOT/runs/qwen2.5_3b_sft_v2_hard77_lr1p5e6_seed16/hf" \
  "qwen2.5_3b_sft_v3_instruction_lr8e7_seed23" \
  12000 8e-7 20260623

touch "$QUEUE_DIR/COMPLETED"
log "24-hour optimization queue completed"
