#!/usr/bin/env python3
"""Paired bootstrap comparison for general regression predictions."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def load(path: Path) -> dict[tuple[str, str], dict]:
    return {
        (row["benchmark"], str(row["id"])): row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[int(q * (len(values) - 1))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260609)
    args = parser.parse_args()

    baseline = load(args.baseline)
    candidate = load(args.candidate)
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in sorted(set(baseline) & set(candidate)):
        grouped[key[0]].append(key)

    rng = random.Random(args.seed)
    result = {
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "benchmarks": {},
    }
    for benchmark, keys in grouped.items():
        base = [int(bool(baseline[key]["correct"])) for key in keys]
        cand = [int(bool(candidate[key]["correct"])) for key in keys]
        deltas = [right - left for left, right in zip(base, cand, strict=True)]
        bootstraps = [
            sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
            for _ in range(args.bootstrap_samples)
        ]
        result["benchmarks"][benchmark] = {
            "samples": len(keys),
            "baseline_accuracy": sum(base) / len(base),
            "candidate_accuracy": sum(cand) / len(cand),
            "delta": sum(deltas) / len(deltas),
            "bootstrap_95_ci": [
                percentile(bootstraps, 0.025),
                percentile(bootstraps, 0.975),
            ],
            "improved": sum(delta > 0 for delta in deltas),
            "regressed": sum(delta < 0 for delta in deltas),
            "unchanged": sum(delta == 0 for delta in deltas),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
