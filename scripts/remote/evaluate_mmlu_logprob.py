#!/usr/bin/env python3
"""Evaluate a fixed MMLU-Pro subset by direct A-J token log probabilities."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tokenizer(model_path: str):
    try:
        return AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    except AttributeError as error:
        if "'list' object has no attribute 'keys'" not in str(error):
            raise
        return AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            extra_special_tokens={},
        )


def stratified_rows(path: Path, limit: int) -> list[dict]:
    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in pq.read_table(path).to_pylist():
        by_category[row["category"]].append(row)
    for rows in by_category.values():
        rows.sort(key=lambda row: int(row["question_id"]))

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
    return selected


def render(row: dict) -> str:
    letters = "ABCDEFGHIJ"
    options = "\n".join(
        f"{letters[index]}. {option}" for index, option in enumerate(row["options"])
    )
    return (
        "Choose the correct option. Return only the option letter.\n\n"
        f"Question: {row['question']}\n{options}\n\nAnswer:"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    letters = "ABCDEFGHIJ"
    candidate_ids = []
    for letter in letters:
        ids = tokenizer.encode(" " + letter, add_special_tokens=False)
        if len(ids) != 1:
            ids = tokenizer.encode(letter, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Option {letter} is not a single token: {ids}")
        candidate_ids.append(ids[0])

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).cuda()
    model.eval()

    rows = stratified_rows(args.dataset, args.samples)
    predictions = []
    category_counts: dict[str, Counter] = defaultdict(Counter)
    correct = 0
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        tokens = tokenizer(
            [render(row) for row in batch_rows],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to("cuda")
        with torch.inference_mode():
            logits = model(**tokens).logits[:, -1, candidate_ids].float()
        predicted_indices = logits.argmax(dim=-1).tolist()
        for row, predicted_index, scores in zip(
            batch_rows,
            predicted_indices,
            logits.cpu().tolist(),
            strict=True,
        ):
            predicted = letters[predicted_index]
            is_correct = predicted == row["answer"]
            correct += int(is_correct)
            category_counts[row["category"]]["total"] += 1
            category_counts[row["category"]]["correct"] += int(is_correct)
            predictions.append(
                {
                    "benchmark": "mmlu_pro_direct_logprob",
                    "id": str(row["question_id"]),
                    "category": row["category"],
                    "expected": row["answer"],
                    "predicted": predicted,
                    "correct": is_correct,
                    "option_logits": dict(zip(letters, scores, strict=True)),
                }
            )

    metrics = {
        "model": args.model,
        "model_label": args.model_label,
        "samples": len(rows),
        "accuracy": correct / len(rows),
        "candidate_token_ids": dict(zip(letters, candidate_ids, strict=True)),
        "category_accuracy": {
            category: values["correct"] / values["total"]
            for category, values in sorted(category_counts.items())
        },
        "selection": "round-robin by sorted category, then question_id",
        "scoring": "single-token conditional logits for A-J after direct-choice prompt",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.model_label}_mmlu_logprob_predictions.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in predictions
        ),
        encoding="utf-8",
    )
    (args.output_dir / f"{args.model_label}_mmlu_logprob_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
