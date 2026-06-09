#!/usr/bin/env python3
"""Copy HF assets into a TorchTitan export and validate Transformers loading."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ASSET_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "merges.txt",
    "vocab.json",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-assets", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    args = parser.parse_args()

    for name in ASSET_FILES:
        shutil.copy2(args.source_assets / name, args.export_dir / name)

    tokenizer = AutoTokenizer.from_pretrained(
        args.export_dir,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.export_dir,
        local_files_only=True,
        dtype=torch.bfloat16,
    )
    inputs = tokenizer("The capital of France is", return_tensors="pt")
    with torch.inference_mode():
        logits = model(**inputs).logits

    result = {
        "model_class": type(model).__name__,
        "parameters": model.num_parameters(),
        "tokenizer_size": len(tokenizer),
        "tie_word_embeddings": model.config.tie_word_embeddings,
        "logits_shape": list(logits.shape),
        "logits_finite": bool(torch.isfinite(logits).all()),
    }
    (args.export_dir / "export_validation.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
