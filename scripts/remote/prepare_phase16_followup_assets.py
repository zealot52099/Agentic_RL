#!/usr/bin/env python3
"""Prepare follow-up training/evaluation assets for Phase16.

The script does not train or evaluate. It creates stable splits and manifests so
Phase16b/16c and post-Phase16 evaluation can run without rediscovering paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return len(rows)


def stable_key(row: dict[str, Any]) -> str:
    payload = row.get("id") or row.get("prompt") or row
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def stable_sample(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return sorted(rows, key=stable_key)[: min(limit, len(rows))]


def split_holdout(rows: list[dict[str, Any]], holdout: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(rows, key=stable_key)
    eval_rows = ordered[: min(holdout, len(ordered))]
    eval_ids = {stable_key(row) for row in eval_rows}
    train_rows = [row for row in rows if stable_key(row) not in eval_ids]
    return train_rows, eval_rows


def parse_json_completion(row: dict[str, Any]) -> Any | None:
    text = row.get("completion") or row.get("chosen") or ""
    try:
        return json.loads(text)
    except Exception:
        return None


def to_data_agent_probe(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    out = []
    seen = set()
    for row in stable_sample(rows, limit * 3):
        expected = parse_json_completion(row)
        if not isinstance(expected, dict):
            continue
        key = (row["prompt"], json.dumps(expected, ensure_ascii=False, sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "id": row.get("id") or stable_key(row)[:16],
                "source": row.get("source", "unknown"),
                "prompt": row["prompt"],
                "expected": expected,
                "format": "data_agent_json_object",
            }
        )
        if len(out) >= limit:
            break
    return out


def to_xlam_array_probe(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    out = []
    seen = set()
    for row in stable_sample(rows, limit * 3):
        value = parse_json_completion(row)
        if not isinstance(value, list):
            continue
        calls = []
        ok = True
        for item in value:
            if not isinstance(item, dict):
                ok = False
                break
            name = item.get("name") or item.get("tool_name")
            args = item.get("arguments", {})
            if "tool_name" in item and "server_id" in item:
                name = f"{item['server_id']}.{item['tool_name']}"
            if not isinstance(name, str) or not isinstance(args, dict):
                ok = False
                break
            calls.append({"name": name, "arguments": args})
        if not ok:
            continue
        key = (row["prompt"], json.dumps(calls, ensure_ascii=False, sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "id": row.get("id") or stable_key(row)[:16],
                "family": row.get("source", "mcp_array_tool_call"),
                "prompt": row["prompt"],
                "expected_calls": calls,
                "format": "xlam_json_array_compatible",
            }
        )
        if len(out) >= limit:
            break
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="datasets/processed/phase16_followup_assets_20260701")
    parser.add_argument("--phase16-dir", default="datasets/processed/phase16_sql_repair_20260701")
    parser.add_argument("--dpo-holdout", type=int, default=256)
    parser.add_argument("--grpo-smoke", type=int, default=256)
    parser.add_argument("--tool-probe-limit", type=int, default=512)
    parser.add_argument("--multiturn-limit", type=int, default=500)
    parser.add_argument("--wikisql-eval-jsonl", default="evals/phase7_wikisql_exec_grpo_independent16_20260628/wikisql_eval_256.jsonl")
    parser.add_argument("--wikisql-eval-db", default="evals/phase7_wikisql_exec_grpo_independent16_20260628/wikisql_eval_256.sqlite")
    parser.add_argument("--data-agent-eval", default="datasets/processed/phase15_multiturn_clean_v4_20260630/eval_traces_clean.jsonl")
    parser.add_argument("--data-agent-db", default="datasets/processed/phase15_data_agent_multiturn_eval_v4_20260630/data_agent_eval.sqlite")
    parser.add_argument("--tool-validation", default="datasets/processed/phase5_unified_20260626/validation.jsonl")
    parser.add_argument("--mcp-array-validation", default="datasets/processed/mcp_lora_sft_v3_20260618/sft_v2/validation_sft.jsonl")
    parser.add_argument("--gsm8k", default="datasets/eval_suite/huggingface/openai__gsm8k/main/test-00000-of-00001.parquet")
    parser.add_argument("--mmlu-pro", default="datasets/eval_suite/huggingface/TIGER-Lab__MMLU-Pro/data/test-00000-of-00001.parquet")
    parser.add_argument("--ifeval", default="datasets/eval_suite/huggingface/google__IFEval/ifeval_input_data.jsonl")
    parser.add_argument("--bfcl-dir", default="datasets/eval_suite/huggingface/gorilla-llm__Berkeley-Function-Calling-Leaderboard")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    phase16 = Path(args.phase16_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dpo_rows = read_jsonl(phase16 / "train_sql_dpo_pairs.jsonl")
    dpo_train, dpo_eval = split_holdout(dpo_rows, args.dpo_holdout)
    grpo_rows = read_jsonl(phase16 / "train_sql_grpo.jsonl")
    repair_eval = read_jsonl(phase16 / "eval_sql_repair_probe.jsonl")
    multiturn_rows = stable_sample(read_jsonl(Path(args.data_agent_eval)), args.multiturn_limit)
    tool_rows = to_data_agent_probe(read_jsonl(Path(args.tool_validation)), args.tool_probe_limit)
    xlam_rows = to_xlam_array_probe(read_jsonl(Path(args.mcp_array_validation)), args.tool_probe_limit)

    counts = {
        "phase16b_dpo_train": write_jsonl(out_dir / "phase16b_dpo_train.jsonl", dpo_train),
        "phase16b_dpo_eval_holdout": write_jsonl(out_dir / "phase16b_dpo_eval_holdout.jsonl", dpo_eval),
        "phase16c_grpo_train": write_jsonl(out_dir / "phase16c_grpo_train.jsonl", grpo_rows),
        "phase16c_grpo_smoke_256": write_jsonl(out_dir / "phase16c_grpo_smoke_256.jsonl", stable_sample(grpo_rows, args.grpo_smoke)),
        "sql_repair_eval": write_jsonl(out_dir / "sql_repair_eval.jsonl", repair_eval),
        "data_agent_multiturn_eval_500": write_jsonl(out_dir / "data_agent_multiturn_eval_500.jsonl", multiturn_rows),
        "data_agent_tool_action_probe": write_jsonl(out_dir / "data_agent_tool_action_probe.jsonl", tool_rows),
        "mcp_xlam_array_tool_probe": write_jsonl(out_dir / "mcp_xlam_array_tool_probe.jsonl", xlam_rows),
    }

    copied = {}
    for src, name in [
        (Path(args.wikisql_eval_jsonl), "wikisql_eval_256.jsonl"),
        (Path(args.wikisql_eval_db), "wikisql_eval_256.sqlite"),
        (Path(args.data_agent_db), "data_agent_eval.sqlite"),
    ]:
        if src.exists():
            dst = out_dir / name
            shutil.copy2(src, dst)
            copied[name] = str(src)

    public_eval_paths = {
        "gsm8k": args.gsm8k,
        "mmlu_pro": args.mmlu_pro,
        "ifeval": args.ifeval,
        "bfcl_dir": args.bfcl_dir,
    }
    public_eval_available = {key: Path(value).exists() for key, value in public_eval_paths.items()}
    bfcl_files = sorted(str(p) for p in Path(args.bfcl_dir).glob("BFCL_v3_*.json")) if Path(args.bfcl_dir).exists() else []

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(out_dir),
        "purpose": "Stable follow-up assets for Phase16b DPO, Phase16c GRPO, and post-Phase16 evaluation.",
        "counts": counts,
        "copied_files": copied,
        "training_assets": {
            "phase16a_sft": str(phase16 / "train_sql_mixture_sft.jsonl"),
            "phase16b_dpo_train": str(out_dir / "phase16b_dpo_train.jsonl"),
            "phase16b_dpo_eval_holdout": str(out_dir / "phase16b_dpo_eval_holdout.jsonl"),
            "phase16c_grpo_train": str(out_dir / "phase16c_grpo_train.jsonl"),
            "phase16c_grpo_smoke_256": str(out_dir / "phase16c_grpo_smoke_256.jsonl"),
        },
        "evaluation_assets": {
            "wikisql_internal_256": {
                "jsonl": str(out_dir / "wikisql_eval_256.jsonl"),
                "sqlite": str(out_dir / "wikisql_eval_256.sqlite"),
                "label": "internal fixed WikiSQL probe, not official benchmark",
            },
            "sql_repair_probe": {
                "jsonl": str(out_dir / "sql_repair_eval.jsonl"),
                "label": "held-out real Phase10 failure repair probe",
            },
            "data_agent_multiturn_internal_500": {
                "jsonl": str(out_dir / "data_agent_multiturn_eval_500.jsonl"),
                "sqlite": str(out_dir / "data_agent_eval.sqlite"),
                "label": "internal executable multi-turn Data Agent probe",
            },
            "data_agent_tool_action_probe": {
                "jsonl": str(out_dir / "data_agent_tool_action_probe.jsonl"),
                "label": "internal prompt/completion JSON action probe",
            },
            "mcp_xlam_array_tool_probe": {
                "jsonl": str(out_dir / "mcp_xlam_array_tool_probe.jsonl"),
                "label": "internal MCP array probe compatible with evaluate_xlam_tool_calls.py",
            },
            "public_regression": public_eval_paths,
            "public_regression_available": public_eval_available,
            "bfcl_v3_files": bfcl_files,
        },
        "notes": [
            "DPO holdout is split before Phase16b and should not be used for DPO training.",
            "Spider DB files are still absent; Spider remains SFT-only until DBs are uploaded.",
            "BFCL files are present, but current in-repo evaluator is internal/xLAM-style rather than official BFCL scoring.",
            "IFEval data is present; use the existing response generation/summarization scripts for regression.",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "README.md").write_text(
        "# Phase16 Follow-up Assets\n\n"
        "This directory contains stable splits and evaluation probes for Phase16b/16c and post-Phase16 evaluation.\n\n"
        f"See `manifest.json` for full paths and counts.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
