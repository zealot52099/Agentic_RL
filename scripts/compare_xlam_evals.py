#!/usr/bin/env python3
"""Compare paired xLAM prediction files and bootstrap score deltas."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


METRICS = (
    "ordered_exact",
    "unordered_exact",
    "names_exact",
    "count_exact",
)


def load(path: Path) -> dict[str, dict]:
    return {
        str(row["id"]): row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(q * (len(ordered) - 1))))
    return ordered[index]


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
    ids = sorted(set(baseline) & set(candidate))
    rng = random.Random(args.seed)

    result = {
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "paired_samples": len(ids),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "metrics": {},
    }

    for metric in METRICS:
        base = [int(bool(baseline[item][metric])) for item in ids]
        cand = [int(bool(candidate[item][metric])) for item in ids]
        deltas = [right - left for left, right in zip(base, cand, strict=True)]
        samples = []
        for _ in range(args.bootstrap_samples):
            samples.append(
                sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
            )
        result["metrics"][metric] = {
            "baseline": sum(base) / len(base),
            "candidate": sum(cand) / len(cand),
            "delta": sum(deltas) / len(deltas),
            "bootstrap_95_ci": [
                percentile(samples, 0.025),
                percentile(samples, 0.975),
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

