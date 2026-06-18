#!/usr/bin/env bash
set -uo pipefail

set +u
source /usr/local/PPU_SDK/envsetup.sh
set -u

ROOT="${ROOT:-/workspace/yans2@xiaopeng.com/agentic_rl}"
QUEUE="$ROOT/runs/mcp_ppu16_queue_20260615"
DATA="$ROOT/datasets/processed/mcp_sft_v1_seed50"
INPUT="$ROOT/runs/qwen3_4b_sft_v4_sota_2k_lr2e7_seed40/hf"
SMOKE="$ROOT/runs/qwen3_4b_mcp_v1_ppu16_smoke"
TRAIN="$ROOT/runs/qwen3_4b_mcp_v1_6k_lr1e7_seed50"
PYTHON="/opt/ac2/bin/python3"
TORCHRUN="/opt/ac2/bin/torchrun"
LAUNCH_COMMAND="$TORCHRUN --standalone --nproc-per-node=16 scripts/remote/train_assistant_only_sft.py --model $INPUT --train-data $DATA/train_sft.jsonl --output-dir $TRAIN --steps 6000 --seq-len 2048 --micro-batch-size 1 --grad-accum-steps 2 --learning-rate 1e-7 --warmup-steps 180 --log-every 20 --save-every 2000 --seed 20260650"

mkdir -p "$QUEUE"
exec >> "$QUEUE/queue.log" 2>&1

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
valid_model() {
  [[ -s "$1/model.safetensors" || -s "$1/model.safetensors.index.json" ]]
}

cd "$ROOT"
echo $$ > "$QUEUE/queue.pid"
rm -f "$QUEUE/COMPLETED"
log "MCP PPU16 queue started PID $$"

if [[ ! -s "$DATA/train_sft.jsonl" ]]; then
  log "Preparing MCP SFT v1 mixture"
  "$PYTHON" scripts/remote/prepare_mcp_sft_v1.py \
    --xlam datasets/processed/xlam_tool_family_v1/train_sft.jsonl \
    --replay datasets/processed/sft_v4_sota_balanced_seed40/train_sft.jsonl \
    --output-dir "$DATA" --total-rows 96000 --seed 20260650 \
    > "$QUEUE/data_prepare.log" 2>&1
  status=$?
  echo "$status" > "$QUEUE/data_exit_code"
  [[ "$status" -eq 0 ]] || exit "$status"
fi

if ! valid_model "$INPUT"; then
  log "Missing input model: $INPUT"
  exit 2
fi

log "Starting 2-step 16-PPU communication and loss smoke"
mkdir -p "$SMOKE"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  "$TORCHRUN" --standalone --nproc-per-node=16 \
  scripts/remote/train_assistant_only_sft.py \
  --model "$INPUT" --train-data "$DATA/train_sft.jsonl" \
  --output-dir "$SMOKE" --steps 2 --seq-len 2048 \
  --micro-batch-size 1 --grad-accum-steps 1 \
  --learning-rate 1e-7 --warmup-steps 1 \
  --log-every 1 --seed 20260649 \
  > "$SMOKE/train.log" 2>&1
smoke_status=$?
echo "$smoke_status" > "$SMOKE/exit_code"
log "Smoke complete status=$smoke_status"
if [[ "$smoke_status" -ne 0 ]]; then
  exit "$smoke_status"
fi

log "Starting MCP SFT steps=6000 lr=1e-7 global_batch=32"
mkdir -p "$TRAIN"
"$PYTHON" scripts/remote/capture_run_provenance.py \
  --output-dir "$TRAIN" --model "$INPUT" \
  --dataset "$DATA/train_sft.jsonl" \
  --scripts scripts/remote/train_assistant_only_sft.py \
    scripts/remote/prepare_mcp_sft_v1.py \
    scripts/remote/run_mcp_ppu16_queue_20260615.sh \
  --launch-command "$LAUNCH_COMMAND" \
  > "$TRAIN/provenance_capture.log" 2>&1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  "$TORCHRUN" --standalone --nproc-per-node=16 \
  scripts/remote/train_assistant_only_sft.py \
  --model "$INPUT" --train-data "$DATA/train_sft.jsonl" \
  --output-dir "$TRAIN" --steps 6000 --seq-len 2048 \
  --micro-batch-size 1 --grad-accum-steps 2 \
  --learning-rate 1e-7 --warmup-steps 180 \
  --log-every 20 --save-every 2000 --seed 20260650 \
  > "$TRAIN/train.log" 2>&1
train_status=$?
echo "$train_status" > "$TRAIN/exit_code"
log "MCP SFT complete status=$train_status"

(
  cd "$TRAIN" || exit 1
  find . -maxdepth 2 -type f \
    \( -name '*.log' -o -name '*.json' -o -name '*.jsonl' -o -name 'exit_code' \) \
    -print0 | sort -z | xargs -0 sha256sum > logs.sha256
  tar -czf logs_bundle.tar.gz \
    train.log train_metrics.jsonl training_config.json provenance.json \
    provenance_capture.log exit_code logs.sha256 2>/dev/null || true
)

touch "$QUEUE/COMPLETED"
exit "$train_status"
