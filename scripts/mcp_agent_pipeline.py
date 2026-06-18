#!/usr/bin/env python3
"""CLI for MCP trace generation, validation, export, and internal evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentic_rl.fingerprint import contamination_report
from agentic_rl.pipeline import (
    build_dataset,
    evaluate_predictions,
    read_jsonl,
)
from agentic_rl.schema import AgentTrace


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build-smoke")
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--count", type=int, default=20000)
    build.add_argument("--seed", type=int, default=20260615)
    build.add_argument("--holdout-percent", type=int, default=20)

    package = commands.add_parser("package-traces")
    package.add_argument("--input", type=Path, required=True)
    package.add_argument("--output-dir", type=Path, required=True)
    package.add_argument("--seed", type=int, default=20260615)
    package.add_argument("--holdout-percent", type=int, default=20)

    validate = commands.add_parser("validate")
    validate.add_argument("--traces", type=Path, required=True)

    contamination = commands.add_parser("check-contamination")
    contamination.add_argument("--train", type=Path, required=True)
    contamination.add_argument("--eval", type=Path, required=True)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--traces", type=Path, required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "build-smoke":
        result = build_dataset(args.output_dir, args.count, args.seed, args.holdout_percent)
    elif args.command == "package-traces":
        from agentic_rl.pipeline import package_traces

        result = package_traces(
            read_jsonl(args.input),
            args.output_dir,
            args.seed,
            args.holdout_percent,
            source=str(args.input),
        )
    elif args.command == "validate":
        rows = read_jsonl(args.traces)
        for row in rows:
            AgentTrace.from_dict(row)
        result = {"valid": True, "count": len(rows)}
    elif args.command == "check-contamination":
        result = contamination_report(read_jsonl(args.train), read_jsonl(args.eval))
        if not result["clean"]:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            raise SystemExit(2)
    else:
        result = evaluate_predictions(read_jsonl(args.traces), read_jsonl(args.predictions))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
