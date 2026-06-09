#!/usr/bin/env python3
"""Run fixed GSM8K and MMLU-Pro regression subsets with vLLM."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from vllm import LLM, SamplingParams


def hash_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def gsm8k_subset(path: Path, limit: int) -> list[dict[str, Any]]:
    rows = pq.read_table(path).to_pylist()
    rows.sort(key=lambda row: hash_key(row["question"]))
    selected = rows[:limit]
    for row in selected:
        row["benchmark"] = "gsm8k"
        row["id"] = hash_key(row["question"])[:16]
        row["prompt"] = (
            "Solve the following math problem step by step. End your response "
            "with exactly '#### <answer>'.\n\n"
            f"Problem: {row['question']}\n\nSolution:"
        )
    return selected


def stratified_mmlu(path: Path, limit: int) -> list[dict[str, Any]]:
    rows = pq.read_table(path).to_pylist()
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
    for category_rows in by_category.values():
        category_rows.sort(key=lambda row: int(row["question_id"]))

    selected = []
    depth = 0
    categories = sorted(by_category)
    while len(selected) < limit:
        added = False
        for category in categories:
            if depth < len(by_category[category]):
                selected.append(by_category[category][depth])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        depth += 1

    letters = "ABCDEFGHIJ"
    for row in selected:
        options = "\n".join(
            f"{letters[index]}. {option}"
            for index, option in enumerate(row["options"])
        )
        row["benchmark"] = "mmlu_pro"
        row["id"] = str(row["question_id"])
        row["prompt"] = (
            "Answer the multiple-choice question. Think carefully, then end "
            "with exactly 'Answer: <letter>'.\n\n"
            f"Question: {row['question']}\n{options}\n\nReasoning:"
        )
    return selected


def normalize_number(value: str) -> str:
    return value.replace(",", "").replace("$", "").strip().rstrip(".")


def gsm_answer(text: str) -> str | None:
    matches = re.findall(r"####\s*([-+]?\$?[\d,]+(?:\.\d+)?)", text)
    if matches:
        return normalize_number(matches[-1])
    numbers = re.findall(r"[-+]?\$?[\d,]+(?:\.\d+)?", text)
    return normalize_number(numbers[-1]) if numbers else None


def mmlu_answer(text: str) -> str | None:
    matches = re.findall(r"Answer\s*:\s*([A-J])\b", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    stripped = text.strip().upper()
    return stripped if stripped in set("ABCDEFGHIJ") else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--gsm8k", type=Path, required=True)
    parser.add_argument("--mmlu-pro", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-benchmark", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=384)
    args = parser.parse_args()

    rows = gsm8k_subset(args.gsm8k, args.samples_per_benchmark)
    rows.extend(stratified_mmlu(args.mmlu_pro, args.samples_per_benchmark))

    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=8192,
        gpu_memory_utilization=0.75,
        trust_remote_code=False,
        enforce_eager=True,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    started = time.time()
    outputs = llm.generate(
        [row["prompt"] for row in rows],
        sampling,
        use_tqdm=True,
    )
    elapsed = time.time() - started

    counters: dict[str, Counter] = defaultdict(Counter)
    predictions = []
    for row, output in zip(rows, outputs, strict=True):
        text = output.outputs[0].text
        benchmark = row["benchmark"]
        if benchmark == "gsm8k":
            predicted = gsm_answer(text)
            expected_match = re.search(r"####\s*(.+?)\s*$", row["answer"])
            expected = normalize_number(expected_match.group(1))
        else:
            predicted = mmlu_answer(text)
            expected = row["answer"]
        correct = predicted == expected
        counters[benchmark]["total"] += 1
        counters[benchmark]["parsed"] += int(predicted is not None)
        counters[benchmark]["correct"] += int(correct)
        predictions.append(
            {
                "benchmark": benchmark,
                "id": row["id"],
                "category": row.get("category"),
                "expected": expected,
                "predicted": predicted,
                "correct": correct,
                "raw_response": text,
                "prompt_tokens": len(output.prompt_token_ids),
                "completion_tokens": len(output.outputs[0].token_ids),
            }
        )

    metrics = {
        "model": args.model,
        "model_label": args.model_label,
        "elapsed_seconds": elapsed,
        "benchmarks": {
            name: {
                "samples": values["total"],
                "parse_rate": values["parsed"] / values["total"],
                "accuracy": values["correct"] / values["total"],
            }
            for name, values in counters.items()
        },
        "selection": {
            "gsm8k": "lowest SHA256(question), fixed 256",
            "mmlu_pro": "round-robin by sorted category, then question_id",
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.model_label}_predictions.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in predictions
        ),
        encoding="utf-8",
    )
    (args.output_dir / f"{args.model_label}_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
