#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/yans2@xiaopeng.com/agentic_rl}"
DATA_DIR="${DATA_DIR:-$ROOT/datasets/processed/mcp_agent_v1_smoke}"
EVAL_ROOT="${EVAL_ROOT:-$ROOT/evals/mcp_agent_baselines_20260615}"

declare -a LABELS=(
  "qwen3.5_4b"
  "qwen3_4b_instruct_2507"
  "xlam2_3b_fc_r"
)
declare -a MODELS=(
  "${QWEN35_MODEL:-/publicdata/huggingface.co/Qwen/Qwen3.5-4B}"
  "${QWEN3_MODEL:-/publicdata/huggingface.co/Qwen/Qwen3-4B-Instruct-2507}"
  "${XLAM_MODEL:-/publicdata/huggingface.co/Salesforce/xLAM-2-3b-fc-r}"
)

cd "$ROOT"
if [[ ! -f "$DATA_DIR/traces/ood_eval.jsonl" ]]; then
  python scripts/mcp_agent_pipeline.py build-smoke \
    --output-dir "$DATA_DIR" --count 20000 --seed 20260615 --holdout-percent 20
fi

for index in "${!LABELS[@]}"; do
  label="${LABELS[$index]}"
  model="${MODELS[$index]}"
  if [[ ! -d "$model" ]]; then
    echo "SKIP $label: model directory does not exist: $model" >&2
    continue
  fi
  CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}" python scripts/remote/evaluate_mcp_internal.py \
    --model "$model" \
    --traces "$DATA_DIR/traces/ood_eval.jsonl" \
    --output-dir "$EVAL_ROOT/$label" \
    --model-label "$label"
done
