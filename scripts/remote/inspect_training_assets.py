#!/usr/bin/env python3
"""Inspect local datasets and possible embedded Qwen3 weights."""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset
from safetensors import safe_open


DATASET_PATH = Path(
    "/workspace/yans2@xiaopeng.com/agentic_rl/datasets/"
    "processed/xlam-function-calling-60k/train.jsonl"
)
WEIGHT_PATH = Path(
    "/workspace/yans2@xiaopeng.com/models/"
    "Qwen3-ForcedAligner-0.6B/model.safetensors"
)


def main() -> None:
    dataset = load_dataset(
        "json",
        data_files={"train": str(DATASET_PATH)},
        split="train",
    )
    print("DATASET", dataset)
    print("COLUMNS", dataset.column_names)
    print("FIRST_ROW", json.dumps(dataset[0], ensure_ascii=False)[:2000])

    if not WEIGHT_PATH.exists():
        print("WEIGHTS missing", WEIGHT_PATH)
        return

    with safe_open(WEIGHT_PATH, framework="pt", device="cpu") as weights:
        keys = list(weights.keys())
    print("WEIGHT_KEY_COUNT", len(keys))
    print("WEIGHT_KEYS")
    for key in keys[:160]:
        print(key)


if __name__ == "__main__":
    main()

