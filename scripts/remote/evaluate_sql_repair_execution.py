#!/usr/bin/env python3
"""Evaluate SQL repair by executing repaired queries against SQLite."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

from vllm import LLM, SamplingParams


def extract_sql(text: str) -> str | None:
    value = text.strip()
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", value, re.I | re.S)
    if fenced:
        value = fenced.group(1).strip()
    match = re.search(r"\bSELECT\b.*", value, re.I | re.S)
    if not match:
        return None
    value = match.group(0).strip()
    value = value.split("```", 1)[0].strip()
    if ";" in value:
        value = value.split(";", 1)[0].strip() + ";"
    return value


def normalized_sql(value: str | None) -> str:
    if not value:
        return ""
    value = value.strip().rstrip(";")
    value = re.sub(r"\s+", " ", value)
    return value.casefold()


def normalize_result(rows: list[Any]) -> list[tuple[str, ...]]:
    normalized = []
    for row in rows:
        if not isinstance(row, (list, tuple)):
            row = [row]
        values = []
        for value in row:
            if value is None:
                values.append("<null>")
            elif isinstance(value, float):
                values.append(f"{value:.8g}")
            else:
                values.append(str(value))
        normalized.append(tuple(values))
    return sorted(normalized)


def safe_execute(connection: sqlite3.Connection, sql: str) -> list[tuple[Any, ...]]:
    if not re.match(r"^\s*SELECT\b", sql, re.I):
        raise sqlite3.OperationalError("Only SELECT is allowed")
    connection.set_progress_handler(
        lambda: 1 if time.monotonic() > safe_execute.deadline else 0,
        1000,
    )
    safe_execute.deadline = time.monotonic() + 2.0
    try:
        return connection.execute(sql).fetchall()
    finally:
        connection.set_progress_handler(None, 0)


safe_execute.deadline = 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
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

    connection = sqlite3.connect(f"file:{args.database}?mode=ro", uri=True)
    counters = {
        "extracted": 0,
        "executed": 0,
        "execution_exact": 0,
        "sql_exact": 0,
        "previous_execution_exact": 0,
    }
    by_failure: dict[str, dict[str, int]] = {}
    predictions = []
    try:
        for row, output in zip(rows, outputs, strict=True):
            raw = output.outputs[0].text
            sql = extract_sql(raw)
            error = None
            result = None
            if sql is not None:
                counters["extracted"] += 1
                try:
                    result = safe_execute(connection, sql)
                    counters["executed"] += 1
                except sqlite3.Error as exc:
                    error = str(exc)
            gold_result = row["gold_result"]
            execution_exact = result is not None and normalize_result(result) == normalize_result(gold_result)
            sql_exact = sql is not None and normalized_sql(sql) == normalized_sql(row["gold_sql"])
            previous_exact = normalize_result(row.get("previous_result") or []) == normalize_result(gold_result)
            counters["execution_exact"] += int(execution_exact)
            counters["sql_exact"] += int(sql_exact)
            counters["previous_execution_exact"] += int(previous_exact)
            bucket = row.get("failure_type", "unknown")
            by_failure.setdefault(bucket, {"total": 0, "execution_exact": 0, "executed": 0})
            by_failure[bucket]["total"] += 1
            by_failure[bucket]["execution_exact"] += int(execution_exact)
            by_failure[bucket]["executed"] += int(result is not None)
            predictions.append(
                {
                    "id": row.get("id"),
                    "failure_type": bucket,
                    "question": row.get("question"),
                    "raw_response": raw,
                    "predicted_sql": sql,
                    "gold_sql": row.get("gold_sql"),
                    "previous_sql": row.get("previous_sql"),
                    "execution_error": error,
                    "predicted_result": result,
                    "gold_result": gold_result,
                    "execution_exact": execution_exact,
                    "normalized_sql_exact": sql_exact,
                    "previous_execution_exact": previous_exact,
                }
            )
    finally:
        connection.close()

    total = len(rows)
    metrics = {
        "model": args.model,
        "model_label": args.model_label,
        "dataset": str(args.dataset),
        "database": str(args.database),
        "samples": total,
        "elapsed_seconds": time.time() - started,
        "sql_extraction_rate": counters["extracted"] / total if total else 0.0,
        "execution_rate": counters["executed"] / total if total else 0.0,
        "execution_repair_accuracy": counters["execution_exact"] / total if total else 0.0,
        "normalized_sql_exact": counters["sql_exact"] / total if total else 0.0,
        "previous_sql_execution_accuracy_baseline": counters["previous_execution_exact"] / total if total else 0.0,
        "by_failure_type": {
            key: {
                "samples": value["total"],
                "execution_rate": value["executed"] / value["total"] if value["total"] else 0.0,
                "execution_repair_accuracy": value["execution_exact"] / value["total"] if value["total"] else 0.0,
            }
            for key, value in sorted(by_failure.items())
        },
        "scoring": "read-only SQLite execution; repaired result rows compared order-insensitively against gold result",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / args.model_label
    with Path(f"{prefix}_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    Path(f"{prefix}_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
