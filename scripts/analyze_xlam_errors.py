#!/usr/bin/env python3
"""Compare two xLAM evaluation files and summarize failure modes."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load(path: Path) -> dict[int, dict]:
    rows = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[int(row["id"])] = row
    return rows


def bucket(row: dict) -> str:
    if row["parse_status"] != "ok":
        return "parse_failure"
    if not row["count_exact"]:
        return "wrong_call_count"
    if not row["names_exact"]:
        return "wrong_tool_names"
    if not row["unordered_exact"]:
        return "wrong_arguments"
    if not row["ordered_exact"]:
        return "wrong_call_order_only"
    if not row["strict_format"]:
        return "correct_calls_extra_text"
    return "fully_correct_strict"


def summarize(rows: dict[int, dict]) -> dict:
    failures = Counter(bucket(row) for row in rows.values())
    by_expected_count: dict[str, Counter] = defaultdict(Counter)
    for row in rows.values():
        count = len(row["expected_calls"])
        count_group = "1" if count == 1 else "2" if count == 2 else "3+"
        by_expected_count[count_group]["samples"] += 1
        by_expected_count[count_group]["ordered_exact"] += int(row["ordered_exact"])
        by_expected_count[count_group]["unordered_exact"] += int(row["unordered_exact"])

    return {
        "samples": len(rows),
        "failure_buckets": dict(failures),
        "failure_rates": {
            name: count / len(rows) for name, count in failures.items()
        },
        "by_expected_call_count": {
            name: {
                "samples": values["samples"],
                "ordered_exact": values["ordered_exact"] / values["samples"],
                "unordered_exact": values["unordered_exact"] / values["samples"],
            }
            for name, values in sorted(by_expected_count.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = load(args.baseline)
    candidate = load(args.candidate)
    shared = sorted(set(baseline) & set(candidate))
    if len(shared) != len(baseline) or len(shared) != len(candidate):
        raise ValueError("Baseline and candidate IDs do not match")

    transitions = Counter(
        f"{bucket(baseline[sample_id])} -> {bucket(candidate[sample_id])}"
        for sample_id in shared
    )
    ordered_transitions = {
        "improved": sum(
            not baseline[sample_id]["ordered_exact"]
            and candidate[sample_id]["ordered_exact"]
            for sample_id in shared
        ),
        "regressed": sum(
            baseline[sample_id]["ordered_exact"]
            and not candidate[sample_id]["ordered_exact"]
            for sample_id in shared
        ),
        "unchanged": sum(
            baseline[sample_id]["ordered_exact"]
            == candidate[sample_id]["ordered_exact"]
            for sample_id in shared
        ),
    }

    result = {
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "baseline_summary": summarize(baseline),
        "candidate_summary": summarize(candidate),
        "ordered_exact_transitions": ordered_transitions,
        "failure_bucket_transitions": dict(transitions.most_common()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
