#!/usr/bin/env python3
"""Repair MCP SFT labels and create deterministic train/validation splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


CATALOG_RE = re.compile(r"MCP_SERVER_CATALOG:\n(.*?)\n\nUSER:\n", re.DOTALL)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_bucket(value: str, modulo: int = 10_000) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulo


def clarification_target(row: dict[str, Any]) -> str:
    prompt = row["prompt"]
    match = CATALOG_RE.search(prompt)
    tool_name_match = re.search(r"Use ([A-Za-z0-9_.:-]+) for me", prompt)
    missing: list[str] = []
    if match and tool_name_match:
        catalog = json.loads(match.group(1))
        target_name = tool_name_match.group(1)
        for server in catalog:
            for tool in server.get("tools", []):
                if tool.get("name") == target_name:
                    missing = tool.get("input_schema", {}).get("required", [])
                    break
    if missing:
        fields = ", ".join(missing)
        message = f"Please provide the required information: {fields}."
    else:
        message = "Please provide the required information before I call the tool."
    return json.dumps(
        {"action": "clarify", "missing": missing, "message": message},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def repair_prompt(row: dict[str, Any]) -> str:
    old = (
        "Return [] when no tool applies or required information is missing."
    )
    new = (
        "Return [] when no tool applies. When required information is missing, "
        'return one JSON object with action=\"clarify\", a missing field-name '
        "array, and a concise message. Do not guess required arguments."
    )
    return row["prompt"].replace(old, new)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--validation-per-source", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260616)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    rows = read_jsonl(args.input)
    repaired: list[dict[str, Any]] = []
    dropped = Counter()
    for row in rows:
        item = dict(row)
        source = item.get("mixture_source", "unknown")
        if source in {"mcp_positive", "mcp_no_tool", "mcp_clarify"}:
            item["prompt"] = repair_prompt(item)
        if source == "mcp_clarify":
            item["completion"] = clarification_target(item)
            verifier = dict(item.get("verifier", {}))
            verifier["expected_action"] = "clarify"
            item["verifier"] = verifier
        item["loss_weight"] = {
            "mcp_positive": 1.0,
            "mcp_no_tool": 16.0,
            "mcp_clarify": 2.0,
            "general_replay": 0.5,
        }.get(source, 1.0)

        prompt_tokens = len(tokenizer.encode(item["prompt"], add_special_tokens=False))
        completion_tokens = len(
            tokenizer.encode(item["completion"], add_special_tokens=False)
        ) + 1
        if prompt_tokens + completion_tokens > args.seq_len:
            dropped[source] += 1
            continue
        item["prompt_tokens"] = prompt_tokens
        item["completion_tokens"] = completion_tokens
        repaired.append(item)

    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in repaired:
        by_source.setdefault(row["mixture_source"], []).append(row)

    validation: list[dict[str, Any]] = []
    training: list[dict[str, Any]] = []
    for source, source_rows in sorted(by_source.items()):
        source_rows.sort(key=lambda row: stable_bucket(str(row["id"])))
        count = min(args.validation_per_source, max(1, len(source_rows) // 20))
        validation.extend(source_rows[:count])
        training.extend(source_rows[count:])

    rng = random.Random(args.seed)
    rng.shuffle(training)
    validation.sort(key=lambda row: (row["mixture_source"], str(row["id"])))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train_sft.jsonl", training)
    write_jsonl(args.output_dir / "validation_sft.jsonl", validation)
    manifest = {
        "format_version": 2,
        "seed": args.seed,
        "seq_len": args.seq_len,
        "train_rows": len(training),
        "validation_rows": len(validation),
        "train_counts": dict(Counter(row["mixture_source"] for row in training)),
        "validation_counts": dict(
            Counter(row["mixture_source"] for row in validation)
        ),
        "dropped_overlength": dict(dropped),
        "repairs": [
            "clarification targets are explicit JSON actions instead of []",
            "MCP system prompt distinguishes no-tool from clarification",
            "samples exceeding sequence length are removed",
            "fixed deterministic validation split is excluded from training",
            "source loss weights compensate for very short no-tool targets",
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
