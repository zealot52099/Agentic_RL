#!/usr/bin/env python3
"""Build a deterministic SFT v2 mixture for tool use and math replay."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def argument_size(row: dict) -> int:
    return sum(
        len(json.dumps(call["arguments"], ensure_ascii=False, sort_keys=True))
        for call in row["expected_calls"]
    )


def sample(rows: list[dict], count: int, rng: random.Random) -> list[dict]:
    if not rows:
        raise ValueError("Cannot sample from an empty bucket")
    if count <= len(rows):
        return rng.sample(rows, count)
    return [rng.choice(rows) for _ in range(count)]


def math_rows(path: Path) -> list[dict]:
    frame = pd.read_parquet(path)
    rows = []
    for index, item in frame.iterrows():
        rows.append(
            {
                "id": f"gsm8k-{index}",
                "source": "gsm8k_math_replay",
                "prompt": (
                    "Solve the following math problem. Show concise reasoning and "
                    "end with the final numeric answer.\n\n"
                    f"PROBLEM:\n{item['question']}\n\nASSISTANT:\n"
                ),
                "completion": str(item["answer"]),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlam-train", type=Path, required=True)
    parser.add_argument("--gsm8k-train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--total-rows", type=int, default=64000)
    parser.add_argument("--hard-argument-threshold", type=int, default=77)
    parser.add_argument("--seed", type=int, default=20260609)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    xlam = load_jsonl(args.xlam_train)
    hard_arguments = [
        row for row in xlam if argument_size(row) >= args.hard_argument_threshold
    ]
    multi_tool = [row for row in xlam if len(row["expected_calls"]) >= 3]
    ordinary = [
        row
        for row in xlam
        if len(row["expected_calls"]) <= 2
        and argument_size(row) < args.hard_argument_threshold
    ]
    math = math_rows(args.gsm8k_train)

    targets = {
        "hard_arguments": round(args.total_rows * 0.40),
        "multi_tool_3plus": round(args.total_rows * 0.25),
        "ordinary_tool": round(args.total_rows * 0.20),
    }
    targets["gsm8k_math_replay"] = args.total_rows - sum(targets.values())

    mixture = []
    for source, rows in (
        ("hard_arguments", hard_arguments),
        ("multi_tool_3plus", multi_tool),
        ("ordinary_tool", ordinary),
        ("gsm8k_math_replay", math),
    ):
        selected = sample(rows, targets[source], rng)
        for row in selected:
            record = dict(row)
            record["mixture_source"] = source
            mixture.append(record)
    rng.shuffle(mixture)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in mixture:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "format_version": 1,
        "seed": args.seed,
        "total_rows": len(mixture),
        "hard_argument_threshold": args.hard_argument_threshold,
        "target_counts": targets,
        "actual_counts": dict(Counter(row["mixture_source"] for row in mixture)),
        "source_pool_counts": {
            "hard_arguments": len(hard_arguments),
            "multi_tool_3plus": len(multi_tool),
            "ordinary_tool": len(ordinary),
            "gsm8k_math_replay": len(math),
        },
        "notes": [
            "Sampling with replacement is used when a target exceeds its source pool.",
            "Hard arguments are selected by canonical JSON character count.",
            "The 15% replay slice is GSM8K math, not a broad instruction mixture.",
        ],
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
