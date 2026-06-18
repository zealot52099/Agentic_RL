#!/usr/bin/env python3
"""DPO entry point for MCP route, no-tool, clarification, and safety pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

try:
    from trl import DPOConfig, DPOTrainer
except ImportError as error:
    raise SystemExit("train_mcp_dpo.py requires a compatible `trl` installation") from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    dataset = Dataset.from_list(
        [
            {
                "prompt": row["prompt"],
                "chosen": row["chosen"],
                "rejected": row["rejected"],
            }
            for row in rows
        ]
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = DPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        beta=args.beta,
        num_train_epochs=args.epochs,
        max_length=args.max_length,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="steps",
        save_steps=200,
        report_to="none",
    )
    trainer = DPOTrainer(
        model=args.model,
        ref_model=None,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
