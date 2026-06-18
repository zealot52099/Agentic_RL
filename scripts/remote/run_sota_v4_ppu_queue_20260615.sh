#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/workspace/yans2@xiaopeng.com/agentic_rl}"
QUEUE="$ROOT/runs/sota_v4_ppu_queue_20260615"
DATA="$ROOT/datasets/processed/sft_v4_sota_balanced_seed40"
INPUT="$ROOT/runs/qwen3_4b_sft_v3_instruction_lr6e7_seed22/hf"
SFT_RUN="$ROOT/runs/qwen3_4b_sft_v4_sota_2k_lr2e7_seed40"
RL_RUN="$ROOT/runs/qwen3_4b_v4_hard_rlvr_300_seed41"

mkdir -p "$QUEUE"
exec >> "$QUEUE/queue.log" 2>&1

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*"; }
valid_model() {
  [[ -s "$1/model.safetensors" || -s "$1/model.safetensors.index.json" ]]
}

cd "$ROOT"
rm -f "$QUEUE/COMPLETED"
log "SOTA-inspired v4 queue started PID $$"

if [[ ! -s "$DATA/train_sft.jsonl" ]]; then
  log "Preparing v4 mixture"
  python scripts/remote/prepare_sft_v4_agent_mixture.py \
    --v3-data datasets/processed/sft_v3_instruction_balanced_seed20/train_sft.jsonl \
    --tool-data datasets/processed/xlam_tool_family_v1/train_sft.jsonl \
    --swe-data datasets/processed/swe-gym-openhands-sft/train.jsonl \
    --output-dir "$DATA" --total-rows 96000 --seed 20260640 \
    > "$QUEUE/data_prepare.log" 2>&1
  status=$?
  echo "$status" > "$QUEUE/data_exit_code"
  if [[ "$status" -ne 0 ]]; then
    log "Data preparation failed status=$status"
    exit "$status"
  fi
fi

if ! valid_model "$INPUT"; then
  log "Missing input model: $INPUT"
  exit 2
fi

if ! valid_model "$SFT_RUN/hf"; then
  mkdir -p "$SFT_RUN"
  log "Starting 16-PPU SFT steps=2000 lr=2e-7"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
    torchrun --standalone --nproc-per-node=16 \
    scripts/remote/train_assistant_only_sft.py \
    --model "$INPUT" --train-data "$DATA/train_sft.jsonl" \
    --output-dir "$SFT_RUN" --steps 2000 --seq-len 2048 \
    --micro-batch-size 1 --grad-accum-steps 2 \
    --learning-rate 2e-7 --warmup-steps 100 \
    --log-every 20 --save-every 1000 --seed 20260640 \
    > "$SFT_RUN/train.log" 2>&1
  sft_status=$?
  echo "$sft_status" > "$SFT_RUN/exit_code"
  log "SFT complete status=$sft_status"
else
  log "Existing SFT export found"
  sft_status=0
fi

if [[ "$sft_status" -eq 0 ]] && valid_model "$SFT_RUN/hf"; then
  if ! valid_model "$RL_RUN/hf"; then
    mkdir -p "$RL_RUN"
    log "Starting 16-PPU hard-constraint RLVR steps=300"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
      torchrun --standalone --nproc-per-node=16 \
      scripts/remote/train_verifiable_grpo.py \
      --model "$SFT_RUN/hf" \
      --dataset "$DATA/rl_verifiable_hard.jsonl" \
      --output-dir "$RL_RUN" --steps 300 --group-size 4 \
      --max-new-tokens 192 --temperature 0.9 --top-p 0.95 \
      --learning-rate 5e-8 --warmup-steps 20 \
      --log-every 5 --save-every 150 --seed 20260641 \
      > "$RL_RUN/train.log" 2>&1
    rl_status=$?
    echo "$rl_status" > "$RL_RUN/exit_code"
    log "RLVR complete status=$rl_status"
  fi
fi

touch "$QUEUE/COMPLETED"
log "SOTA-inspired v4 queue completed"
