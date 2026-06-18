#!/usr/bin/env python3
"""Evaluate fixed completion-only SFT loss by mixture source."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from peft import PeftModel

from train_assistant_only_sft import SftCollator, load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-source", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260652)
    args = parser.parse_args()

    rows_by_source: dict[str, list[dict]] = defaultdict(list)
    with args.train_data.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows_by_source[row.get("mixture_source", "unknown")].append(row)

    rng = random.Random(args.seed)
    selected = {
        source: rng.sample(rows, min(args.samples_per_source, len(rows)))
        for source, rows in sorted(rows_by_source.items())
    }

    tokenizer = load_tokenizer(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    collator = SftCollator(tokenizer, args.seq_len)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).cuda().eval()
    if args.adapter:
        model = PeftModel.from_pretrained(
            model,
            args.adapter,
            local_files_only=True,
        ).cuda().eval()

    metrics = {}
    with torch.inference_mode():
        for source, rows in selected.items():
            nll_sum = 0.0
            token_count = 0
            example_losses = []
            for row in rows:
                batch = {
                    key: value.cuda(non_blocking=True)
                    for key, value in collator([row]).items()
                }
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                ).logits[:, :-1, :]
                labels = batch["labels"][:, 1:]
                nll = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                tokens = int((labels != -100).sum())
                nll_value = float(nll)
                nll_sum += nll_value
                token_count += tokens
                example_losses.append(nll_value / tokens)
            metrics[source] = {
                "examples": len(rows),
                "supervised_tokens": token_count,
                "token_mean_loss": nll_sum / token_count,
                "example_mean_loss": sum(example_losses) / len(example_losses),
            }

    total_nll = sum(
        item["token_mean_loss"] * item["supervised_tokens"]
        for item in metrics.values()
    )
    total_tokens = sum(item["supervised_tokens"] for item in metrics.values())
    result = {
        "model": args.model,
        "adapter": args.adapter,
        "data": str(args.train_data),
        "samples_per_source": args.samples_per_source,
        "seed": args.seed,
        "overall_token_mean_loss": total_nll / total_tokens,
        "by_source": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
