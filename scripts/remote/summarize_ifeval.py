#!/usr/bin/env python3
"""Summarize official IFEval JSONL outputs into a compact metrics file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(path: Path) -> dict:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    prompt_correct = sum(bool(row["follow_all_instructions"]) for row in rows)
    instruction_total = sum(len(row["follow_instruction_list"]) for row in rows)
    instruction_correct = sum(
        sum(bool(value) for value in row["follow_instruction_list"])
        for row in rows
    )
    return {
        "prompts": len(rows),
        "prompt_accuracy": prompt_correct / len(rows),
        "instructions": instruction_total,
        "instruction_accuracy": instruction_correct / instruction_total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--strict", type=Path, required=True)
    parser.add_argument("--loose", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metrics = {
        "model_label": args.model_label,
        "benchmark": "IFEval",
        "scorer": "google-research/instruction_following_eval",
        "strict": summarize(args.strict),
        "loose": summarize(args.loose),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
