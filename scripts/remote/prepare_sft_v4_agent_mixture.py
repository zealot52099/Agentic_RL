#!/usr/bin/env python3
"""Build a balanced Agent SFT/RLVR mixture inspired by public SOTA recipes."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sample(rows: list[dict], count: int, rng: random.Random) -> list[dict]:
    if not rows:
        raise ValueError("Cannot sample an empty source")
    if count <= len(rows):
        return rng.sample(rows, count)
    return [rng.choice(rows) for _ in range(count)]


def swe_rows(path: Path) -> list[dict]:
    output = []
    for index, row in enumerate(read_jsonl(path)):
        messages = row.get("messages")
        if not isinstance(messages, list):
            continue
        for position, message in enumerate(messages):
            completion = message.get("content")
            if message.get("role") != "assistant" or not completion:
                continue
            rendered = []
            for prior in messages[:position]:
                role = str(prior.get("role", "unknown")).upper()
                content = prior.get("content")
                if content:
                    rendered.append(f"{role}:\n{content}")
            if not rendered or len(completion) > 12000:
                continue
            prompt = "\n\n".join(rendered)
            prompt = prompt[-28000:] + "\n\nASSISTANT:\n"
            output.append(
                {
                    "id": f"swe-trajectory-{index}-turn-{position}",
                    "source": "swe_gym_openhands",
                    "prompt": prompt,
                    "completion": completion,
                }
            )
    return output


def no_tool_rows(tool_rows: list[dict], count: int, rng: random.Random) -> list[dict]:
    requests = [
        "Explain why backups should be tested regularly.",
        "Write a short thank-you note to a colleague.",
        "What is the difference between latency and throughput?",
        "Give three general tips for clearer technical documentation.",
        "Summarize the benefits of unit testing without using any external tools.",
    ]
    rows = []
    for index, source in enumerate(sample(tool_rows, count, rng)):
        tools_block = source["prompt"].split("\n\nUSER:\n", 1)[0]
        request = requests[index % len(requests)]
        rows.append(
            {
                "id": f"tool-negative-{index}",
                "source": "synthetic_tool_negative",
                "prompt": (
                    f"{tools_block}\n\nUSER:\n{request}\n\n"
                    "No provided tool is relevant. Return only an empty JSON array.\n\n"
                    "ASSISTANT:\n"
                ),
                "completion": "[]",
            }
        )
    return rows


def hard_constraints(count: int, rng: random.Random) -> list[dict]:
    topics = [
        "API migration",
        "incident response",
        "database rollback",
        "distributed training",
        "repository debugging",
        "tool selection",
    ]
    phrases = ["verify first", "preserve evidence", "measure the outcome"]
    rows = []
    for index in range(count):
        topic = rng.choice(topics)
        phrase = rng.choice(phrases)
        if index % 3 == 0:
            prompt = (
                f"Return only a JSON object about {topic}. It must have exactly "
                "the keys plan, risks, and checks. Each value must be an array "
                f"of exactly two strings. Include the exact phrase '{phrase}' "
                "in exactly one string and never use the word 'easy'."
            )
            completion = json.dumps(
                {
                    "plan": [f"Define scope for {topic}", phrase],
                    "risks": ["Unexpected behavior", "Incomplete rollback"],
                    "checks": ["Run deterministic tests", "Review recorded evidence"],
                },
                separators=(",", ":"),
            )
            verifier = {
                "kind": "nested_json_constraints",
                "keys": ["plan", "risks", "checks"],
                "array_length": 2,
                "required_phrase": phrase,
                "forbidden": "easy",
            }
        elif index % 3 == 1:
            prompt = (
                f"Write exactly four numbered lines about {topic}, labelled 1. "
                f"through 4. Include '{phrase}' exactly once. Line 4 must end "
                "with 'Validated.' Do not add an introduction."
            )
            completion = (
                f"1. Define the intended outcome for {topic}.\n"
                f"2. Apply the rule: {phrase}.\n"
                "3. Test failure and rollback behavior.\n"
                "4. Record the result. Validated."
            )
            verifier = {
                "kind": "numbered_phrase_suffix",
                "count": 4,
                "required_phrase": phrase,
                "suffix": "Validated.",
            }
        else:
            prompt = (
                f"Explain {topic} in exactly three sentences and 35 to 55 words. "
                f"The first sentence must start with 'First,'. Include '{phrase}' "
                "exactly once, and the final sentence must end with 'Done.'"
            )
            completion = (
                f"First, define measurable requirements and boundaries for {topic}. "
                f"Next, {phrase} while testing normal behavior, failures, and rollback "
                "paths with deterministic checks. Finally, record the result, unresolved "
                "risks, and evidence needed for review. Done."
            )
            verifier = {
                "kind": "sentence_word_phrase",
                "sentence_count": 3,
                "min_words": 35,
                "max_words": 55,
                "prefix": "First,",
                "suffix": "Done.",
                "required_phrase": phrase,
            }
        rows.append(
            {
                "id": f"hard-constraint-{index}",
                "source": "synthetic_hard_constraints",
                "prompt": prompt + "\n\nASSISTANT:\n",
                "completion": completion,
                "verifier": verifier,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3-data", type=Path, required=True)
    parser.add_argument("--tool-data", type=Path, required=True)
    parser.add_argument("--swe-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-rows", type=int, default=96000)
    parser.add_argument("--seed", type=int, default=20260640)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    v3 = read_jsonl(args.v3_data)
    tool = read_jsonl(args.tool_data)
    swe = swe_rows(args.swe_data)
    broad = [
        row for row in v3 if row.get("mixture_source") in {"tulu_if", "no_robots"}
    ]
    math = [row for row in v3 if row.get("mixture_source") == "gsm8k"]
    hard = hard_constraints(max(12000, args.total_rows // 8), rng)
    negatives = no_tool_rows(tool, args.total_rows // 10, rng)

    targets = {
        "tool_positive": round(args.total_rows * 0.25),
        "tool_negative": round(args.total_rows * 0.10),
        "broad_instruction": round(args.total_rows * 0.35),
        "swe_trajectory": round(args.total_rows * 0.10),
        "math": round(args.total_rows * 0.10),
    }
    targets["hard_constraint"] = args.total_rows - sum(targets.values())
    sources = {
        "tool_positive": tool,
        "tool_negative": negatives,
        "broad_instruction": broad,
        "swe_trajectory": swe,
        "math": math,
        "hard_constraint": hard,
    }
    mixture = []
    for label, count in targets.items():
        for row in sample(sources[label], count, rng):
            record = dict(row)
            record["mixture_source"] = label
            mixture.append(record)
    rng.shuffle(mixture)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train_sft.jsonl"
    rl_path = args.output_dir / "rl_verifiable_hard.jsonl"
    with train_path.open("w", encoding="utf-8") as handle:
        for row in mixture:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with rl_path.open("w", encoding="utf-8") as handle:
        for row in hard:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "format_version": 1,
        "seed": args.seed,
        "total_rows": len(mixture),
        "counts": dict(Counter(row["mixture_source"] for row in mixture)),
        "source_pool_counts": {key: len(value) for key, value in sources.items()},
        "design": [
            "APIGen-style positive calls plus explicit irrelevant-tool negatives.",
            "Broad instruction replay and SWE trajectories limit specialization loss.",
            "RLVR uses harder compositional constraints to avoid zero reward variance.",
            "No IFEval, BFCL, SWE-bench test, or WikiSQL test examples are included.",
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
