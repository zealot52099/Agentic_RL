#!/usr/bin/env bash
set -Eeuo pipefail

set +u
source /usr/local/PPU_SDK/envsetup.sh
set -u

ROOT="${ROOT:-/workspace/yans2@xiaopeng.com/agentic_rl}"
QUEUE="$ROOT/runs/mcp_lora_v2_queue_20260616"
DATA="$ROOT/datasets/processed/mcp_sft_v2_seed60"
BASE="$ROOT/runs/qwen3_4b_sft_v4_sota_2k_lr2e7_seed40/hf"
GATE="$ROOT/runs/qwen3_4b_mcp_lora_v2_gate_200_seed60"
TRAIN_ROOT="$ROOT/runs/mcp_lora_v2_parallel_20260616"
EVAL="$ROOT/evals/mcp_lora_v2_20260616"
PYTHON="/opt/ac2/bin/python3"
TORCHRUN="/opt/ac2/bin/torchrun"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-agentic-rl-mcp}"
SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
SWANLAB_WORKSPACE="${SWANLAB_WORKSPACE:-}"

mkdir -p "$QUEUE" "$EVAL"
exec >> "$QUEUE/queue.log" 2>&1

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
finish() {
  status=$?
  echo "$status" > "$QUEUE/exit_code"
  log "Queue exiting status=$status"
  if [[ "$status" -eq 0 ]]; then
    touch "$QUEUE/COMPLETED"
  else
    touch "$QUEUE/FAILED"
  fi
}
trap finish EXIT

cd "$ROOT"
echo $$ > "$QUEUE/queue.pid"
rm -f "$QUEUE/COMPLETED" "$QUEUE/FAILED"
log "Starting MCP LoRA v2 queue PID=$$"

if [[ ! -s "$DATA/train_sft.jsonl" || ! -s "$DATA/validation_sft.jsonl" ]]; then
  log "Preparing repaired MCP SFT v2 data"
  "$PYTHON" scripts/remote/prepare_mcp_sft_v2.py \
    --input datasets/processed/mcp_sft_v1_seed50/train_sft.jsonl \
    --model "$BASE" --output-dir "$DATA" \
    --seq-len 2048 --validation-per-source 256 --seed 20260660 \
    > "$QUEUE/data_prepare.log" 2>&1
fi

log "Evaluating frozen baseline on fixed validation split"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" scripts/remote/evaluate_sft_loss.py \
  --model "$BASE" --train-data "$DATA/validation_sft.jsonl" \
  --output "$EVAL/baseline_loss.json" --samples-per-source 64 \
  --seq-len 2048 --seed 20260662 > "$EVAL/baseline_loss.log" 2>&1

rm -rf "$GATE"
mkdir -p "$GATE"
log "Starting 100-step single-PPU FP32 LoRA learning gate"
CUDA_VISIBLE_DEVICES=0 "$TORCHRUN" --standalone --nproc-per-node=1 \
  scripts/remote/train_lora_sft.py \
  --model "$BASE" --train-data "$DATA/train_sft.jsonl" \
  --output-dir "$GATE" --steps 100 --seq-len 2048 \
  --micro-batch-size 1 --grad-accum-steps 16 \
  --learning-rate 2e-5 --warmup-steps 10 \
  --lora-r 32 --lora-alpha 64 --lora-dropout 0.05 \
  --max-grad-norm 1.0 --log-every 5 --save-every 50 \
  --swanlab-project "$SWANLAB_PROJECT" \
  --swanlab-run-name "gate_lora_v2_lr2e-5_r32_seed20260660" \
  --swanlab-workspace "$SWANLAB_WORKSPACE" \
  --swanlab-mode "$SWANLAB_MODE" \
  --swanlab-tags "mcp,lora,gate,qwen3-4b" \
  --seed 20260660 > "$GATE/train.log" 2>&1
echo 0 > "$GATE/exit_code"

log "Evaluating gate adapter"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" scripts/remote/evaluate_sft_loss.py \
  --model "$BASE" --adapter "$GATE/adapter" \
  --train-data "$DATA/validation_sft.jsonl" \
  --output "$EVAL/gate_loss.json" --samples-per-source 64 \
  --seq-len 2048 --seed 20260662 > "$EVAL/gate_loss.log" 2>&1

log "Checking learning gate"
"$PYTHON" - <<'PY' > "$EVAL/gate_decision.json"
import json
from pathlib import Path
from safetensors import safe_open

root = Path("/workspace/yans2@xiaopeng.com/agentic_rl")
baseline = json.loads((root / "evals/mcp_lora_v2_20260616/baseline_loss.json").read_text())
gate = json.loads((root / "evals/mcp_lora_v2_20260616/gate_loss.json").read_text())
adapter = root / "runs/qwen3_4b_mcp_lora_v2_gate_200_seed60/adapter/adapter_model.safetensors"
with safe_open(adapter, framework="pt", device="cpu") as handle:
    norms = {
        key: float(handle.get_tensor(key).float().norm())
        for key in handle.keys()
    }
    b_norms = [value for key, value in norms.items() if "lora_B" in key]

checks = {
    "lora_B_updated": max(b_norms, default=0.0) > 0.0,
    "overall_improved_2pct": gate["overall_token_mean_loss"] < baseline["overall_token_mean_loss"] * 0.98,
    "no_tool_improved_10pct": gate["by_source"]["mcp_no_tool"]["token_mean_loss"] < baseline["by_source"]["mcp_no_tool"]["token_mean_loss"] * 0.90,
    "clarify_improved_5pct": gate["by_source"]["mcp_clarify"]["token_mean_loss"] < baseline["by_source"]["mcp_clarify"]["token_mean_loss"] * 0.95,
    "positive_regression_under_10pct": gate["by_source"]["mcp_positive"]["token_mean_loss"] < baseline["by_source"]["mcp_positive"]["token_mean_loss"] * 1.10,
}
result = {
    "baseline": baseline,
    "gate": gate,
    "adapter_max_tensor_norm": max(norms.values(), default=0.0),
    "adapter_max_lora_B_norm": max(b_norms, default=0.0),
    "checks": checks,
    "passed": all(checks.values()),
}
print(json.dumps(result, ensure_ascii=False, indent=2))
if not result["passed"]:
    raise SystemExit(3)
PY

rm -rf "$TRAIN_ROOT"
mkdir -p "$TRAIN_ROOT"
log "Gate passed; starting 16 independent single-PPU FP32 LoRA experiments"

lrs=(1e-5 1e-5 1e-5 1e-5 2e-5 2e-5 2e-5 2e-5 4e-5 4e-5 4e-5 4e-5 2e-5 2e-5 2e-5 2e-5)
ranks=(32 32 32 32 32 32 32 32 32 32 32 32 16 16 16 16)
alphas=(64 64 64 64 64 64 64 64 64 64 64 64 32 32 32 32)
pids=()
labels=()

for gpu in $(seq 0 15); do
  seed=$((20260670 + gpu))
  lr="${lrs[$gpu]}"
  rank="${ranks[$gpu]}"
  alpha="${alphas[$gpu]}"
  label="gpu${gpu}_lr${lr}_r${rank}_seed${seed}"
  run="$TRAIN_ROOT/$label"
  labels+=("$label")
  mkdir -p "$run"
  "$PYTHON" scripts/remote/capture_run_provenance.py \
    --output-dir "$run" --model "$BASE" \
    --dataset "$DATA/train_sft.jsonl" \
    --scripts scripts/remote/train_lora_sft.py \
      scripts/remote/prepare_mcp_sft_v2.py \
      scripts/remote/run_mcp_lora_v2_queue_20260616.sh \
    --launch-command "single PPU=$gpu LoRA steps=1500 lr=$lr r=$rank seed=$seed" \
    > "$run/provenance_capture.log" 2>&1
  (
    set +e
    CUDA_VISIBLE_DEVICES="$gpu" "$TORCHRUN" --standalone --nproc-per-node=1 \
      scripts/remote/train_lora_sft.py \
      --model "$BASE" --train-data "$DATA/train_sft.jsonl" \
      --output-dir "$run" --steps 1500 --seq-len 2048 \
      --micro-batch-size 1 --grad-accum-steps 16 \
      --learning-rate "$lr" --warmup-steps 50 \
      --lora-r "$rank" --lora-alpha "$alpha" --lora-dropout 0.05 \
      --max-grad-norm 1.0 --log-every 10 --save-every 500 \
      --swanlab-project "$SWANLAB_PROJECT" \
      --swanlab-run-name "$label" \
      --swanlab-workspace "$SWANLAB_WORKSPACE" \
      --swanlab-mode "$SWANLAB_MODE" \
      --swanlab-tags "mcp,lora,parallel,lr-$lr,r-$rank,qwen3-4b" \
      --seed "$seed" > "$run/train.log" 2>&1
    status=$?
    echo "$status" > "$run/exit_code"
    exit "$status"
  ) &
  pids+=("$!")
done

parallel_status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    log "Experiment complete ${labels[$index]}"
  else
    log "Experiment failed ${labels[$index]}"
    parallel_status=1
  fi
done
[[ "$parallel_status" -eq 0 ]]

log "Evaluating all 16 adapters on fixed validation split"
for index in "${!labels[@]}"; do
  label="${labels[$index]}"
  gpu=$((index % 16))
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/remote/evaluate_sft_loss.py \
    --model "$BASE" --adapter "$TRAIN_ROOT/$label/adapter" \
    --train-data "$DATA/validation_sft.jsonl" \
    --output "$EVAL/${label}_loss.json" --samples-per-source 64 \
    --seq-len 2048 --seed 20260662 > "$EVAL/${label}_loss.log" 2>&1 &
  pids[$index]=$!
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

"$PYTHON" - <<'PY' > "$EVAL/parallel_summary.json"
import json
from pathlib import Path

root = Path("/workspace/yans2@xiaopeng.com/agentic_rl/evals/mcp_lora_v2_20260616")
rows = []
for path in sorted(root.glob("gpu*_loss.json")):
    item = json.loads(path.read_text())
    rows.append({
        "label": path.name.removesuffix("_loss.json"),
        "overall": item["overall_token_mean_loss"],
        **{
            source: metrics["token_mean_loss"]
            for source, metrics in item["by_source"].items()
        },
    })
rows.sort(key=lambda row: row["overall"])
print(json.dumps({"experiments": rows, "best": rows[0]}, indent=2))
PY

for label in "${labels[@]}"; do
  run="$TRAIN_ROOT/$label"
  (
    cd "$run"
    find . -maxdepth 3 -type f \
      \( -name '*.log' -o -name '*.json' -o -name '*.jsonl' \
         -o -name '*.safetensors' -o -name 'exit_code' \) \
      -print0 | sort -z | xargs -0 sha256sum > logs.sha256
  )
done
log "MCP LoRA v2 queue completed"
