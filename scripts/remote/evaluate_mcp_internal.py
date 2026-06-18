#!/usr/bin/env python3
"""Generate model responses and score the internal held-out MCP probe."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agentic_rl.pipeline import evaluate_predictions, read_jsonl, render_prompt, write_jsonl


def load_tokenizer(model: str):
    try:
        return AutoTokenizer.from_pretrained(model, local_files_only=True)
    except AttributeError as error:
        if "'list' object has no attribute 'keys'" not in str(error):
            raise
        return AutoTokenizer.from_pretrained(
            model, local_files_only=True, extra_special_tokens={}
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    traces = read_jsonl(args.traces)
    if args.limit:
        traces = traces[: args.limit]
    tokenizer = load_tokenizer(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    model.eval()
    predictions = []
    for index, trace in enumerate(traces, start=1):
        prompt = render_prompt(trace)
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=16384
        ).to(model.device)
        started = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        elapsed_ms = (time.perf_counter() - started) * 1000
        completion = output[0, inputs["input_ids"].shape[1] :]
        predictions.append(
            {
                "task_id": trace["task_id"],
                "response": tokenizer.decode(completion, skip_special_tokens=True),
                "latency_ms": elapsed_ms,
                "output_tokens": int(completion.shape[0]),
            }
        )
        if index % 20 == 0:
            print(json.dumps({"completed": index, "total": len(traces)}), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "predictions.jsonl", predictions)
    result = evaluate_predictions(traces, predictions)
    result["model"] = args.model_label
    result["model_path"] = args.model
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
