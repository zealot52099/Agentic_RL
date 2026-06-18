#!/usr/bin/env python3
"""Generate deterministic responses for the official IFEval scorer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--input-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful assistant. Follow all user instructions precisely.",
    )
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.input_data.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tokenizer = load_tokenizer(args.model)
    prompts = []
    for row in rows:
        messages = [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": row["prompt"]},
        ]
        if tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = f"{args.system_prompt}\n\nUser: {row['prompt']}\n\nAssistant:"
        prompts.append(prompt)

    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=False,
        enforce_eager=True,
    )
    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
        use_tqdm=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row, output in zip(rows, outputs, strict=True):
            handle.write(
                json.dumps(
                    {
                        "prompt": row["prompt"],
                        "response": output.outputs[0].text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
