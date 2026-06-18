#!/usr/bin/env python3
"""Build an instruction-balanced Agent SFT mixture with verifiable constraints."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sample(rows: list[dict], count: int, rng: random.Random) -> list[dict]:
    if not rows:
        raise ValueError("Cannot sample from an empty source")
    if count <= len(rows):
        return rng.sample(rows, count)
    return [rng.choice(rows) for _ in range(count)]


def message_rows(path: Path, source: str) -> list[dict]:
    rows = []
    for index, row in enumerate(pq.read_table(path).to_pylist()):
        messages = row.get("messages") or row.get("conversations")
        if not isinstance(messages, list):
            continue
        user_parts = []
        assistant = None
        for message in messages:
            role = message.get("role") or message.get("from")
            content = message.get("content") or message.get("value")
            if not isinstance(content, str):
                continue
            if role in {"user", "human"}:
                user_parts.append(content)
            elif role in {"assistant", "gpt"}:
                assistant = content
        if not user_parts or not assistant:
            continue
        rows.append(
            {
                "id": f"{source}-{index}",
                "source": source,
                "prompt": (
                    "Follow the user's instructions precisely.\n\n"
                    f"USER:\n{user_parts[-1]}\n\nASSISTANT:\n"
                ),
                "completion": assistant,
            }
        )
    return rows


def verifiable_rows(count: int, rng: random.Random) -> list[dict]:
    topics = [
        "reliable APIs", "database migrations", "incident response",
        "unit testing", "data validation", "tool selection",
        "model evaluation", "secure automation", "query optimization",
        "distributed training",
    ]
    phrases = [
        "verify first", "measure twice", "keep it reversible",
        "record the evidence", "prefer deterministic checks",
    ]
    rows = []
    for index in range(count):
        topic = rng.choice(topics)
        phrase = rng.choice(phrases)
        kind = index % 5
        if kind == 0:
            prompt = (
                f"Explain {topic} in exactly three bullet points. Each bullet "
                f"must start with '- '. Include the exact phrase '{phrase}' "
                "once. Do not use the word 'simply'."
            )
            completion = (
                f"- Define the goal and constraints for {topic}.\n"
                f"- Apply the rule: {phrase}.\n"
                "- Check the outcome and retain rollback evidence."
            )
            verifier = {
                "kind": "bullets_phrase_forbidden",
                "bullet_count": 3,
                "required_phrase": phrase,
                "forbidden": "simply",
            }
        elif kind == 1:
            prompt = (
                f"Return only JSON about {topic}, with exactly the keys "
                "'goal', 'risk', and 'check'. All values must be strings."
            )
            completion = json.dumps(
                {
                    "goal": f"Improve {topic}",
                    "risk": "Unverified behavior",
                    "check": "Run a deterministic validation",
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            verifier = {
                "kind": "json_keys",
                "keys": ["goal", "risk", "check"],
            }
        elif kind == 2:
            prompt = (
                f"Write two sentences about {topic}. Start the first sentence "
                "with 'First,' and end the second sentence with 'Done.'"
            )
            completion = (
                f"First, define measurable requirements for {topic}. "
                "Then validate the result against those requirements. Done."
            )
            verifier = {
                "kind": "prefix_suffix_sentences",
                "prefix": "First,",
                "suffix": "Done.",
                "sentence_count": 2,
            }
        elif kind == 3:
            prompt = (
                f"Give a numbered list of exactly four checks for {topic}. "
                "Use the labels 1., 2., 3., and 4. and no introductory text."
            )
            completion = (
                "1. Confirm inputs and assumptions.\n"
                "2. Validate the main operation.\n"
                "3. Test failure and rollback behavior.\n"
                "4. Record metrics and evidence."
            )
            verifier = {"kind": "numbered_list", "count": 4}
        else:
            prompt = (
                f"Answer the question 'What matters most in {topic}?' using "
                f"between 18 and 28 words, and include '{phrase}'."
            )
            completion = (
                f"The key is to {phrase}, define measurable outcomes, test "
                f"failure cases, and preserve evidence while improving {topic}."
            )
            verifier = {
                "kind": "word_range_phrase",
                "min_words": 18,
                "max_words": 28,
                "required_phrase": phrase,
            }
        rows.append(
            {
                "id": f"verifiable-constraint-{index}",
                "source": "synthetic_verifiable_constraints",
                "prompt": f"{prompt}\n\nASSISTANT:\n",
                "completion": completion,
                "verifier": verifier,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool-data", type=Path, required=True)
    parser.add_argument("--gsm8k-data", type=Path, required=True)
    parser.add_argument("--tulu-if", type=Path, required=True)
    parser.add_argument("--no-robots", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rl-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--total-rows", type=int, default=128000)
    parser.add_argument("--seed", type=int, default=20260620)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    tool = read_jsonl(args.tool_data)
    gsm = [
        row
        for row in read_jsonl(args.gsm8k_data)
        if row.get("mixture_source") == "gsm8k_math_replay"
        or row.get("source") == "gsm8k_math_replay"
    ]
    tulu = message_rows(args.tulu_if, "tulu_personas_instruction_following")
    no_robots = message_rows(args.no_robots, "no_robots")
    synthetic = verifiable_rows(24000, rng)

    targets = {
        "tool": round(args.total_rows * 0.35),
        "tulu_if": round(args.total_rows * 0.25),
        "no_robots": round(args.total_rows * 0.15),
        "verifiable": round(args.total_rows * 0.15),
    }
    targets["gsm8k"] = args.total_rows - sum(targets.values())
    mixture = []
    for label, source_rows in [
        ("tool", tool),
        ("tulu_if", tulu),
        ("no_robots", no_robots),
        ("verifiable", synthetic),
        ("gsm8k", gsm),
    ]:
        for row in sample(source_rows, targets[label], rng):
            record = dict(row)
            record["mixture_source"] = label
            mixture.append(record)
    rng.shuffle(mixture)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in mixture:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with args.rl_output.open("w", encoding="utf-8") as handle:
        for row in synthetic:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "format_version": 1,
        "seed": args.seed,
        "total_rows": len(mixture),
        "counts": dict(Counter(row["mixture_source"] for row in mixture)),
        "source_pool_counts": {
            "tool": len(tool),
            "gsm8k": len(gsm),
            "tulu_if": len(tulu),
            "no_robots": len(no_robots),
            "verifiable": len(synthetic),
        },
        "notes": [
            "IFEval benchmark prompts and responses are excluded.",
            "Synthetic constraints have deterministic verifiers for RLVR.",
            "General instruction replay prevents tool-only catastrophic forgetting.",
        ],
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
