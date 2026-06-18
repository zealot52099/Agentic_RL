#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/workspace/yans2@xiaopeng.com/agentic_rl}"
QUEUE_DIR="$ROOT/runs/ppu16_sft_ablation_20260612"
EVAL_ROOT="$ROOT/evals/ppu16_sft_ablation_20260612"
TRAIN_DATA="$ROOT/datasets/processed/sft_v3_instruction_balanced_seed20/train_sft.jsonl"
INPUT_MODEL="$ROOT/runs/qwen2.5_coder_3b_sft_v2_hard90_lr1p5e6_seed18/hf"
XLAM_EVAL="$ROOT/datasets/processed/xlam_tool_family_v1/eval.jsonl"
GSM8K_EVAL="$ROOT/datasets/eval/gsm8k/test.parquet"
MMLU_EVAL="$ROOT/datasets/eval/mmlu_pro/test.parquet"
WIKISQL_EVAL="$ROOT/datasets/eval/wikisql/test_256.jsonl"
WIKISQL_DB="$ROOT/datasets/sources/wikisql/extracted/data/test.db"
IFEVAL_INPUT="$ROOT/datasets/eval_suite/huggingface/google__IFEval/ifeval_input_data.jsonl"
IFEVAL_ROOT="$ROOT/third_party/google-research"

mkdir -p "$QUEUE_DIR" "$EVAL_ROOT"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

valid_model() {
  local path="$1"
  [[ -s "$path/model.safetensors" ||
    -s "$path/model.safetensors.index.json" ]]
}

evaluate_model() {
  local label="$1"
  local model="$2"
  local g0="$3"
  local g1="$4"
  local g2="$5"
  local g3="$6"
  local output="$EVAL_ROOT/$label"
  local status=0
  mkdir -p "$output"/{xlam,general,mmlu,wikisql,ifeval}

  CUDA_VISIBLE_DEVICES="$g0" python scripts/remote/evaluate_xlam_tool_calls.py \
    --model "$model" --dataset "$XLAM_EVAL" \
    --output-dir "$output/xlam" --model-label "$label" \
    > "$QUEUE_DIR/${label}_xlam.log" 2>&1 &
  local p0=$!
  CUDA_VISIBLE_DEVICES="$g1" python scripts/remote/evaluate_general_regression.py \
    --model "$model" --model-label "$label" \
    --gsm8k "$GSM8K_EVAL" --mmlu-pro "$MMLU_EVAL" \
    --output-dir "$output/general" \
    > "$QUEUE_DIR/${label}_general.log" 2>&1 &
  local p1=$!
  CUDA_VISIBLE_DEVICES="$g2" python scripts/remote/evaluate_mmlu_logprob.py \
    --model "$model" --model-label "$label" --dataset "$MMLU_EVAL" \
    --output-dir "$output/mmlu" \
    > "$QUEUE_DIR/${label}_mmlu.log" 2>&1 &
  local p2=$!
  CUDA_VISIBLE_DEVICES="$g3" python scripts/remote/evaluate_wikisql.py \
    --model "$model" --model-label "$label" \
    --dataset "$WIKISQL_EVAL" --database "$WIKISQL_DB" \
    --output-dir "$output/wikisql" \
    > "$QUEUE_DIR/${label}_wikisql.log" 2>&1 &
  local p3=$!

  wait "$p0" || status=1
  wait "$p1" || status=1
  wait "$p2" || status=1
  wait "$p3" || status=1

  CUDA_VISIBLE_DEVICES="$g0" python scripts/remote/generate_ifeval_responses.py \
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
        >> "$QUEUE_DIR/${label}_ifeval_scoring.log" 2>&1 || status=1
    else
      status=1
    fi
  else
    status=1
  fi
  return "$status"
}

train_one() {
  local label="$1"
  local steps="$2"
  local lr="$3"
  local seed="$4"
  local run_dir="$ROOT/runs/$label"
  local log_file="$QUEUE_DIR/${label}_train_worker.log"

  {
    echo "[$(timestamp)] training start label=$label gpus=0-15 steps=$steps lr=$lr"
    if valid_model "$run_dir/hf"; then
      echo "[$(timestamp)] existing model found; skipping training"
      return 0
    else
      mkdir -p "$run_dir"
      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
        torchrun --standalone --nproc-per-node=16 \
        scripts/remote/train_assistant_only_sft.py \
        --model "$INPUT_MODEL" --train-data "$TRAIN_DATA" \
        --output-dir "$run_dir" --steps "$steps" --seq-len 2048 \
        --micro-batch-size 1 --grad-accum-steps 2 \
        --learning-rate "$lr" --warmup-steps 100 \
        --log-every 20 --save-every 0 --seed "$seed" \
        > "$run_dir/train.log" 2>&1
      local train_status=$?
      echo "$train_status" > "$run_dir/exit_code"
      echo "[$(timestamp)] training exit=$train_status"
      return "$train_status"
    fi
  } >> "$log_file" 2>&1
}

evaluate_one() {
  local label="$1"
  local g0="$2"
  local g1="$3"
  local g2="$4"
  local g3="$5"
  local run_dir="$ROOT/runs/$label"
  local log_file="$QUEUE_DIR/${label}_eval_worker.log"

  {
    echo "[$(timestamp)] evaluation start label=$label gpus=$g0,$g1,$g2,$g3"
    if valid_model "$run_dir/hf"; then
      evaluate_model "$label" "$run_dir/hf" "$g0" "$g1" "$g2" "$g3"
      local status=$?
      echo "$status" > "$run_dir/eval_exit_code"
      echo "[$(timestamp)] evaluation complete status=$status"
      touch "$QUEUE_DIR/${label}.completed"
      return "$status"
    else
      echo "[$(timestamp)] no valid HF export; evaluation skipped"
      touch "$QUEUE_DIR/${label}.completed"
      return 1
    fi
  } >> "$log_file" 2>&1
}

cd "$ROOT"
rm -f "$QUEUE_DIR/COMPLETED" "$QUEUE_DIR"/*.completed
{
  echo "[$(timestamp)] 16-PPU SFT ablation started PID $$"
  echo "input_model=$INPUT_MODEL"
  echo "train_data=$TRAIN_DATA"
} >> "$QUEUE_DIR/queue.log"

if ! valid_model "$INPUT_MODEL"; then
  echo "[$(timestamp)] missing input model: $INPUT_MODEL" >> "$QUEUE_DIR/queue.log"
  exit 2
fi

status=0
train_one qwen2.5_coder_3b_v3_short_2k_lr3e7_seed30 \
  2000 3e-7 20260630 || status=1
train_one qwen2.5_coder_3b_v3_short_4k_lr3e7_seed31 \
  4000 3e-7 20260631 || status=1
train_one qwen2.5_coder_3b_v3_short_4k_lr5e7_seed32 \
  4000 5e-7 20260632 || status=1
train_one qwen2.5_coder_3b_v3_short_6k_lr3e7_seed33 \
  6000 3e-7 20260633 || status=1

evaluate_one qwen2.5_coder_3b_v3_short_2k_lr3e7_seed30 0 1 2 3 &
p0=$!
evaluate_one qwen2.5_coder_3b_v3_short_4k_lr3e7_seed31 4 5 6 7 &
p1=$!
evaluate_one qwen2.5_coder_3b_v3_short_4k_lr5e7_seed32 8 9 10 11 &
p2=$!
evaluate_one qwen2.5_coder_3b_v3_short_6k_lr3e7_seed33 12 13 14 15 &
p3=$!

wait "$p0" || status=1
wait "$p1" || status=1
wait "$p2" || status=1
wait "$p3" || status=1
echo "$status" > "$QUEUE_DIR/exit_code"
touch "$QUEUE_DIR/COMPLETED"
echo "[$(timestamp)] 16-PPU SFT ablation complete status=$status" >> "$QUEUE_DIR/queue.log"
exit "$status"
