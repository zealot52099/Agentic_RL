#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/yans2@xiaopeng.com/agentic_rl_pipeline"
cd "$ROOT"

STAMP="${1:-20260701_1138_phase16a}"
RUN_ROOT="$ROOT/runs/phase16a_sql_repair_sft_${STAMP}"
TRAIN_DIR="$RUN_ROOT/phase16a_sql_repair_sft_7b_lora_${STAMP}"
EVAL_ROOT="$ROOT/evals/phase16a_sql_repair_sft_${STAMP}"
FOLLOW="$ROOT/datasets/processed/phase16_followup_assets_20260701"
BASE_MODEL="$ROOT/evals/phase10_staged_sql_then_mixed_20260630_094959/merged/phase10b_mixed_retention_step0800"
ADAPTER="$TRAIN_DIR/adapter"
MERGED="$EVAL_ROOT/merged/phase16a_sql_repair_sft_merged"

mkdir -p "$EVAL_ROOT/logs" "$EVAL_ROOT/merged"

export LD_LIBRARY_PATH=/usr/local/PPU_SDK/CUDA_SDK/targets/x86_64-linux/lib:/usr/local/PPU_SDK/CUDA_SDK/lib64:/usr/local/PPU_SDK/targets/x86_64-linux/lib:/usr/local/PPU_SDK/lib:${LD_LIBRARY_PATH:-}
export PPU_SDK=/usr/local/PPU_SDK
export PPU_HOME=/usr/local/PPU_SDK
export NCCL_SOCKET_IFNAME=hpn0
export NCCL_IB_HCA=
export PATH=/opt/ac2/bin:${PATH}
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$EVAL_ROOT/post_eval_queue.log"
}

merge_adapter() {
  if [[ -s "$MERGED/model.safetensors.index.json" ]]; then
    log "merged model already exists: $MERGED"
    return
  fi
  log "merging adapter"
  /opt/ac2/bin/python - <<PY > "$EVAL_ROOT/logs/merge_phase16a.log" 2>&1
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base = "$BASE_MODEL"
adapter = "$ADAPTER"
out = Path("$MERGED")
print("loading tokenizer", flush=True)
tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
print("loading base", base, flush=True)
model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True, torch_dtype=torch.bfloat16, device_map="cpu")
print("loading adapter", adapter, flush=True)
model = PeftModel.from_pretrained(model, adapter, local_files_only=True)
print("merging", flush=True)
model = model.merge_and_unload()
out.mkdir(parents=True, exist_ok=True)
print("saving", out, flush=True)
model.save_pretrained(out, safe_serialization=True, max_shard_size="2GB")
tok.save_pretrained(out)
(out / "merge_metadata.json").write_text(json.dumps({"base": base, "adapter": adapter}, indent=2) + "\\n", encoding="utf-8")
print("done", flush=True)
PY
  echo "$MERGED" > "$EVAL_ROOT/final_merged_model_path.txt"
  log "merge complete: $MERGED"
}

run_eval() {
  local name="$1"
  shift
  log "starting $name"
  "$@" > "$EVAL_ROOT/logs/${name}.log" 2>&1
  log "completed $name"
}

main() {
  for path in "$BASE_MODEL" "$ADAPTER" "$FOLLOW"; do
    if [[ ! -e "$path" ]]; then
      log "missing required path: $path"
      exit 2
    fi
  done

  merge_adapter

  run_eval wikisql_phase16a \
    /opt/ac2/bin/python scripts/remote/evaluate_wikisql.py \
      --model "$MERGED" \
      --model-label phase16a_sql_repair_sft \
      --dataset "$FOLLOW/wikisql_eval_256.jsonl" \
      --database "$FOLLOW/wikisql_eval_256.sqlite" \
      --output-dir "$EVAL_ROOT/wikisql"

  run_eval sql_repair_phase16a \
    /opt/ac2/bin/python scripts/remote/evaluate_sql_repair_probe.py \
      --model "$MERGED" \
      --model-label phase16a_sql_repair_sft \
      --dataset "$FOLLOW/sql_repair_eval.jsonl" \
      --output-dir "$EVAL_ROOT/sql_repair"

  mkdir -p "$FOLLOW/sql_repair_execution_eval"
  run_eval prepare_sql_repair_execution_phase16a \
    /opt/ac2/bin/python scripts/remote/prepare_sql_repair_execution_eval.py \
      --wikisql-eval "$FOLLOW/wikisql_eval_256.jsonl" \
      --database "$FOLLOW/wikisql_eval_256.sqlite" \
      --output "$FOLLOW/sql_repair_execution_eval/sql_repair_execution_eval_128.jsonl" \
      --manifest "$FOLLOW/sql_repair_execution_eval/manifest.json" \
      --limit 128

  run_eval sql_repair_execution_phase16a \
    /opt/ac2/bin/python scripts/remote/evaluate_sql_repair_execution.py \
      --model "$MERGED" \
      --model-label phase16a_sql_repair_sft \
      --dataset "$FOLLOW/sql_repair_execution_eval/sql_repair_execution_eval_128.jsonl" \
      --database "$FOLLOW/wikisql_eval_256.sqlite" \
      --output-dir "$EVAL_ROOT/sql_repair_execution"

  run_eval multiturn_phase16a \
    /opt/ac2/bin/python scripts/remote/evaluate_data_agent_multiturn.py \
      --model "$MERGED" \
      --model-label phase16a_sql_repair_sft \
      --traces "$FOLLOW/data_agent_multiturn_eval_500.jsonl" \
      --database "$FOLLOW/data_agent_eval.sqlite" \
      --output-dir "$EVAL_ROOT/data_agent_multiturn" \
      --limit 300

  run_eval mcp_array_smoke_phase16a \
    /opt/ac2/bin/python scripts/remote/evaluate_xlam_tool_calls.py \
      --model "$MERGED" \
      --model-label phase16a_sql_repair_sft \
      --dataset "$FOLLOW/mcp_xlam_array_tool_probe.jsonl" \
      --output-dir "$EVAL_ROOT/mcp_array_smoke" \
      --gpu-memory-utilization 0.55

  run_eval general_phase16a \
    /opt/ac2/bin/python scripts/remote/evaluate_general_regression.py \
      --model "$MERGED" \
      --model-label phase16a_sql_repair_sft \
      --gsm8k datasets/eval_suite/huggingface/openai__gsm8k/main/test-00000-of-00001.parquet \
      --mmlu-pro datasets/eval_suite/huggingface/TIGER-Lab__MMLU-Pro/data/test-00000-of-00001.parquet \
      --output-dir "$EVAL_ROOT/general" \
      --samples-per-benchmark 256

  log "all post eval tasks completed"
}

main "$@"
