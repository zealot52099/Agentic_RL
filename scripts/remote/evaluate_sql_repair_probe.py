#!/usr/bin/env python3
"""Evaluate held-out SQL repair prompts with deterministic generation."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from vllm import LLM, SamplingParams


def normalize_sql(sql: str | None) -> str:
    if not sql:
        return ""
    sql = sql.strip()
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", sql, flags=re.I | re.S)
    if fenced:
        sql = fenced.group(1).strip()
    match = re.search(r"\bSELECT\b.*", sql, flags=re.I | re.S)
    if match:
        sql = match.group(0)
    sql = sql.split("```", 1)[0].strip().rstrip(";")
    sql = re.sub(r"\s+", " ", sql)
    return sql.casefold()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    started = time.time()
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=8192,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=False,
        enforce_eager=True,
    )
    outputs = llm.generate(
        [row["prompt"] for row in rows],
        SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
        use_tqdm=True,
    )
    predictions = []
    exact = 0
    extracted = 0
    by_failure: dict[str, dict[str, int]] = {}
    for row, output in zip(rows, outputs, strict=True):
        raw = output.outputs[0].text
        pred = normalize_sql(raw)
        gold = normalize_sql(row.get("completion") or row.get("gold_sql"))
        ok = bool(pred) and pred == gold
        extracted += int(bool(pred))
        exact += int(ok)
        bucket = row.get("failure_type", "unknown")
        by_failure.setdefault(bucket, {"total": 0, "exact": 0})
        by_failure[bucket]["total"] += 1
        by_failure[bucket]["exact"] += int(ok)
        predictions.append(
            {
                "id": row.get("id"),
                "failure_type": bucket,
                "raw_response": raw,
                "predicted_sql_normalized": pred,
                "gold_sql": row.get("completion") or row.get("gold_sql"),
                "gold_sql_normalized": gold,
                "normalized_exact": ok,
            }
        )

    total = len(rows)
    metrics = {
        "model": args.model,
        "model_label": args.model_label,
        "dataset": str(args.dataset),
        "samples": total,
        "elapsed_seconds": time.time() - started,
        "sql_extraction_rate": extracted / total if total else 0.0,
        "normalized_repair_exact": exact / total if total else 0.0,
        "by_failure_type": {
            key: {
                "samples": value["total"],
                "normalized_repair_exact": value["exact"] / value["total"] if value["total"] else 0.0,
            }
            for key, value in sorted(by_failure.items())
        },
        "scoring": "normalized SQL exact on held-out repair prompts; no execution because prompts do not contain all table payloads",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / args.model_label
    Path(f"{prefix}_predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions),
        encoding="utf-8",
    )
    Path(f"{prefix}_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
