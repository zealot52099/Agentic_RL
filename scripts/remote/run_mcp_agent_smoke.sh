#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/yans2@xiaopeng.com/agentic_rl}"
MODEL="${MODEL:-/publicdata/huggingface.co/Qwen/Qwen3-4B-Instruct-2507}"
RUN_NAME="${RUN_NAME:-qwen3_4b_instruct_2507_mcp_sft_smoke}"
DATA_DIR="${DATA_DIR:-$ROOT/datasets/processed/mcp_agent_v1_smoke}"
RUN_DIR="$ROOT/runs/$RUN_NAME"
EVAL_DIR="$ROOT/evals/$RUN_NAME/mcp_internal"
GPU_COUNT="${GPU_COUNT:-4}"

cd "$ROOT"
mkdir -p "$DATA_DIR" "$RUN_DIR" "$EVAL_DIR"

python scripts/mcp_agent_pipeline.py build-smoke \
  --output-dir "$DATA_DIR" \
  --count "${SMOKE_ROWS:-20000}" \
  --seed "${SEED:-20260615}" \
  --holdout-percent 20

torchrun --standalone --nproc_per_node "$GPU_COUNT" \
  scripts/remote/train_assistant_only_sft.py \
  --model "$MODEL" \
  --train-data "$DATA_DIR/train_sft.jsonl" \
  --output-dir "$RUN_DIR" \
  --steps "${SFT_STEPS:-200}" \
  --seq-len "${SEQ_LEN:-8192}" \
  --micro-batch-size 1 \
  --grad-accum-steps "${GRAD_ACCUM_STEPS:-4}" \
  --learning-rate "${LEARNING_RATE:-5e-6}" \
  --warmup-steps "${WARMUP_STEPS:-10}" \
  --save-every "${SAVE_EVERY:-100}"

CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}" python scripts/remote/evaluate_mcp_internal.py \
  --model "$RUN_DIR/hf" \
  --traces "$DATA_DIR/traces/ood_eval.jsonl" \
  --output-dir "$EVAL_DIR" \
  --model-label "$RUN_NAME"
