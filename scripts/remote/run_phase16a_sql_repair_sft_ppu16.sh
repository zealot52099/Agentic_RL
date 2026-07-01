#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/yans2@xiaopeng.com/agentic_rl_pipeline"
cd "$ROOT"

STAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="$ROOT/runs/phase16a_sql_repair_sft_${STAMP}"
EVAL_ROOT="$ROOT/evals/phase16a_sql_repair_sft_${STAMP}"
mkdir -p "$RUN_ROOT" "$EVAL_ROOT/logs"

export LD_LIBRARY_PATH=/usr/local/PPU_SDK/CUDA_SDK/targets/x86_64-linux/lib:/usr/local/PPU_SDK/CUDA_SDK/lib64:/usr/local/PPU_SDK/targets/x86_64-linux/lib:/usr/local/PPU_SDK/lib:${LD_LIBRARY_PATH:-}
export PPU_SDK=/usr/local/PPU_SDK
export PPU_HOME=/usr/local/PPU_SDK
export NCCL_SOCKET_IFNAME=hpn0
export NCCL_IB_HCA=
export PATH=/opt/ac2/bin:${PATH}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export TOKENIZERS_PARALLELISM=false

DATA_DIR="$ROOT/datasets/processed/phase16_sql_repair_20260701"
TRAIN_DATA="$DATA_DIR/train_sql_mixture_sft.jsonl"
REPAIR_DATA="$DATA_DIR/train_sql_repair_sft.jsonl"
DPO_DATA="$DATA_DIR/train_sql_dpo_pairs.jsonl"
GRPO_DATA="$DATA_DIR/train_sql_grpo.jsonl"
REPAIR_EVAL="$DATA_DIR/eval_sql_repair_probe.jsonl"
PHASE10_STAMP="${PHASE10_STAMP:-20260630_094959}"
PHASE10_EVAL="$ROOT/evals/phase10_staged_sql_then_mixed_${PHASE10_STAMP}"
BASE_MODEL="${BASE_MODEL:-}"

if [[ -z "$BASE_MODEL" ]]; then
  if [[ -s "$PHASE10_EVAL/final_model_path.txt" ]]; then
    BASE_MODEL="$(cat "$PHASE10_EVAL/final_model_path.txt")"
  else
    BASE_MODEL="$ROOT/evals/phase10_staged_sql_then_mixed_${PHASE10_STAMP}/merged/phase10b_mixed_retention_step0800"
  fi
fi

for path in "$BASE_MODEL" "$TRAIN_DATA" "$REPAIR_DATA" "$DPO_DATA" "$GRPO_DATA" "$REPAIR_EVAL"; do
  if [[ ! -e "$path" ]]; then
    echo "missing required path: $path" | tee -a "$EVAL_ROOT/queue.log"
    exit 2
  fi
done

RUN_NAME="phase16a_sql_repair_sft_7b_lora_${STAMP}"
TRAIN_DIR="$RUN_ROOT/$RUN_NAME"
mkdir -p "$TRAIN_DIR"

cat > "$EVAL_ROOT/phase16a_manifest.json" <<JSON
{
  "stamp": "$STAMP",
  "stage": "Phase16a SQL repair focused LoRA SFT",
  "base_model": "$BASE_MODEL",
  "train_data": "$TRAIN_DATA",
  "repair_data": "$REPAIR_DATA",
  "dpo_data": "$DPO_DATA",
  "grpo_data": "$GRPO_DATA",
  "repair_eval": "$REPAIR_EVAL",
  "output_dir": "$TRAIN_DIR",
  "planned_next": ["Phase16b DPO/SimPO on train_sql_dpo_pairs.jsonl", "Phase16c SQL-only GRPO on train_sql_grpo.jsonl"],
  "label": "Internal SQL repair SFT, not official benchmark"
}
JSON

/opt/ac2/bin/python - <<'PY' "$TRAIN_DATA" "$REPAIR_EVAL" | tee -a "$EVAL_ROOT/queue.log"
import json, sys
for path in sys.argv[1:]:
    rows = 0
    sources = {}
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        rows += 1
        sources[row.get("source", "unknown")] = sources.get(row.get("source", "unknown"), 0) + 1
    print(json.dumps({"validated": path, "rows": rows, "sources": sources}, ensure_ascii=False))
PY

echo "$$" > "$RUN_ROOT/queue.pid"
echo "starting $RUN_NAME at $(date)" | tee -a "$EVAL_ROOT/queue.log"
echo "base_model=$BASE_MODEL" | tee -a "$EVAL_ROOT/queue.log"

torchrun --standalone --nproc-per-node=16 \
  scripts/remote/train_lora_sft.py \
  --model "$BASE_MODEL" \
  --train-data "$TRAIN_DATA" \
  --output-dir "$TRAIN_DIR" \
  --steps "${STEPS:-1000}" \
  --seq-len 4096 \
  --micro-batch-size 1 \
  --grad-accum-steps 2 \
  --learning-rate "${LR:-4e-7}" \
  --warmup-steps "${WARMUP_STEPS:-60}" \
  --max-grad-norm 0.5 \
  --lora-r 32 \
  --lora-alpha 64 \
  --lora-dropout 0.05 \
  --seed "${SEED:-20260701}" \
  --log-every 5 \
  --save-every 250 \
  --swanlab-project agentic-rl-sql-tool \
  --swanlab-run-name "$RUN_NAME" \
  --swanlab-mode local \
  --swanlab-tags phase16a,sql-repair,sft,7b,qwen2.5-coder \
  > "$TRAIN_DIR/train.log" 2>&1

echo "$TRAIN_DIR/adapter" > "$EVAL_ROOT/final_adapter_path.txt"
echo "phase16a training completed: $TRAIN_DIR/adapter" | tee -a "$EVAL_ROOT/queue.log"
