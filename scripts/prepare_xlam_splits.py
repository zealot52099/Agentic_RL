#!/usr/bin/env python3
"""Create deterministic tool-family-held-out xLAM splits and SFT records."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROMPT_TEMPLATE_VERSION = "xlam_tool_json_v1"
SYSTEM_PROMPT = (
    "You are a tool-calling assistant. Select only tools from the provided "
    "definitions. Return only a JSON array. Each item must have keys "
    '"name" and "arguments". The arguments value must be a JSON object.'
)


def parse_tools(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise ValueError("tools must be a JSON list")
    return value


def expected_calls(row: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for message in row.get("messages") or []:
        if message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            calls.append(
                {
                    "name": function.get("name"),
                    "arguments": arguments,
                }
            )
    return calls


def user_text(row: dict[str, Any]) -> str:
    texts = [
        message.get("content", "")
        for message in row.get("messages") or []
        if message.get("role") == "user"
    ]
    return "\n".join(text for text in texts if text)


def tool_family(calls: list[dict[str, Any]]) -> str:
    return "|".join(sorted({str(call["name"]) for call in calls}))


def family_bucket(family: str, modulus: int) -> int:
    digest = hashlib.sha256(family.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulus


def render_prompt(row: dict[str, Any]) -> str:
    tools = parse_tools(row["tools"])
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"TOOLS:\n{json.dumps(tools, ensure_ascii=False, sort_keys=True)}\n\n"
        f"USER:\n{user_text(row)}\n\n"
        "ASSISTANT:\n"
    )


def canonical_completion(calls: list[dict[str, Any]]) -> str:
    return json.dumps(calls, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def stratified_eval(
    rows: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)

    selected: list[dict[str, Any]] = []
    depth = 0
    families = sorted(by_family)
    while len(selected) < limit:
        added = False
        for family in families:
            family_rows = by_family[family]
            if depth < len(family_rows):
                selected.append(family_rows[depth])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        depth += 1
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--holdout-modulus", type=int, default=10)
    parser.add_argument("--holdout-bucket", type=int, default=0)
    parser.add_argument("--eval-limit", type=int, default=512)
    args = parser.parse_args()

    train_rows: list[dict[str, Any]] = []
    heldout_rows: list[dict[str, Any]] = []
    skipped = Counter()
    family_counts = Counter()

    with args.input.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            try:
                calls = expected_calls(row)
                if not calls or any(not call["name"] for call in calls):
                    skipped["missing_calls"] += 1
                    continue
                family = tool_family(calls)
                record = {
                    "id": (row.get("extra") or {}).get("id", line_number - 1),
                    "family": family,
                    "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                    "prompt": render_prompt(row),
                    "completion": canonical_completion(calls),
                    "expected_calls": calls,
                    "messages": row.get("messages"),
                    "tools": parse_tools(row["tools"]),
                }
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                skipped["invalid_record"] += 1
                continue

            family_counts[family] += 1
            is_heldout = (
                family_bucket(family, args.holdout_modulus) == args.holdout_bucket
            )
            if is_heldout:
                heldout_rows.append(record)
            else:
                train_rows.append(record)

    heldout_rows.sort(key=lambda row: (row["family"], str(row["id"])))
    eval_rows = stratified_eval(heldout_rows, args.eval_limit)

    write_jsonl(args.output_dir / "train_sft.jsonl", train_rows)
    write_jsonl(args.output_dir / "heldout_all.jsonl", heldout_rows)
    write_jsonl(args.output_dir / "eval.jsonl", eval_rows)

    manifest = {
        "format_version": 1,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "source": str(args.input),
        "holdout": {
            "unit": "sorted unique expected tool-name set",
            "hash": "sha256 first 8 bytes",
            "modulus": args.holdout_modulus,
            "bucket": args.holdout_bucket,
        },
        "counts": {
            "train": len(train_rows),
            "heldout_all": len(heldout_rows),
            "eval": len(eval_rows),
            "families_total": len(family_counts),
            "families_train": len({row["family"] for row in train_rows}),
            "families_heldout": len({row["family"] for row in heldout_rows}),
            "skipped": dict(skipped),
        },
        "largest_families": family_counts.most_common(20),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
