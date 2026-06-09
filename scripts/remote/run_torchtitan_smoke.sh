#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/yans2@xiaopeng.com/agentic_rl
CONFIG="$ROOT/configs/torchtitan/qwen3_0.6b_smoke.toml"
ADAPTER_DIR="$ROOT/scripts/remote"
RUN_DIR="$ROOT/runs/qwen3_0.6b_smoke"

mkdir -p "$RUN_DIR"

echo "Checking GPU 1 before launch"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$ADAPTER_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

torchrun \
  --standalone \
  --nproc_per_node=1 \
  -m torchtitan.train \
  --job.config-file "$CONFIG" \
  2>&1 | tee "$RUN_DIR/console.log"
