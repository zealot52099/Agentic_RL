#!/usr/bin/env python3
"""Build an executable SQL repair evaluation set from WikiSQL probe rows.

The older Phase16 repair probe used real failed predictions, but those prompts
did not include enough schema/table context for executable scoring. This script
constructs a deterministic repair probe from the fixed WikiSQL execution probe:
each example contains schema/sample rows, a corrupted previous SQL query,
execution feedback, the expected result, and the gold SQL. The companion
evaluator can then judge repaired SQL by read-only SQLite execution result.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any


def normalized_sql(value: str | None) -> str:
    if not value:
        return ""
    value = value.strip().rstrip(";")
    value = re.sub(r"\s+", " ", value)
    return value.casefold()


def normalize_result(rows: list[tuple[Any, ...]]) -> list[tuple[str, ...]]:
    normalized = []
    for row in rows:
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


def table_context(row: dict[str, Any]) -> str:
    columns = ", ".join(
        f'col{index} = "{name}" ({kind})'
        for index, (name, kind) in enumerate(
            zip(row["header"], row["types"], strict=True)
        )
    )
    samples = "\n".join(
        json.dumps(
            {
                f"col{index}": value
                for index, value in enumerate(values)
            },
            ensure_ascii=False,
        )
        for values in row["sample_rows"]
    )
    return (
        f'Table: "{row["sqlite_table"]}"\n'
        f"Columns: {columns}\n"
        f"Sample rows:\n{samples}"
    )


def repair_prompt(row: dict[str, Any], previous_sql: str, feedback: str) -> str:
    return (
        "You are repairing a SQLite SELECT query for a data agent. "
        "Use only the table and columns shown. Return one corrected SQL query "
        "only, without explanation or Markdown.\n\n"
        f"{table_context(row)}\n\n"
        f"Question: {row['question']}\n\n"
        f"Previous SQL:\n{previous_sql}\n\n"
        f"Execution feedback:\n{feedback}\n\n"
        f"Expected execution result:\n{json.dumps(row['gold_result'], ensure_ascii=False)}\n\n"
        "Corrected SQL:"
    )


def corruptions(gold_sql: str, column_count: int) -> list[tuple[str, str]]:
    variants: list[tuple[str, str]] = []
    sql = gold_sql.strip().rstrip(";")
    variants.append(("drop_identifier_quotes", sql.replace('"', "")))
    variants.append(("sum_to_count", re.sub(r"\bSUM\s*\(", "COUNT(", sql, flags=re.I)))
    variants.append(("avg_to_count", re.sub(r"\bAVG\s*\(", "COUNT(", sql, flags=re.I)))
    variants.append(("max_to_min", re.sub(r"\bMAX\s*\(", "MIN(", sql, flags=re.I)))
    variants.append(("min_to_max", re.sub(r"\bMIN\s*\(", "MAX(", sql, flags=re.I)))

    where_match = re.search(r"\bWHERE\b(.+)$", sql, flags=re.I | re.S)
    if where_match:
        variants.append(("remove_where_clause", sql[: where_match.start()].strip()))

    selected = re.search(r"\bSELECT\s+(?:\w+\s*\()?\s*\"?col(\d+)\"?", sql, flags=re.I)
    if selected and column_count > 1:
        old_index = int(selected.group(1))
        new_index = (old_index + 1) % column_count
        variants.append(("wrong_selected_column", re.sub(rf"\bcol{old_index}\b", f"col{new_index}", sql, count=1)))

    seen = {normalized_sql(sql)}
    out = []
    for kind, candidate in variants:
        candidate = candidate.strip().rstrip(";")
        key = normalized_sql(candidate)
        if not candidate or key in seen:
            continue
        seen.add(key)
        out.append((kind, candidate))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wikisql-eval", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=128)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.wikisql_eval.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    connection = sqlite3.connect(f"file:{args.database}?mode=ro", uri=True)
    prepared = []
    try:
        for row in rows:
            gold_result = normalize_result([tuple(x) for x in row["gold_result"]])
            for failure_type, previous_sql in corruptions(row["gold_sql"], len(row["header"])):
                try:
                    previous_result = safe_execute(connection, previous_sql)
                    previous_exact = normalize_result(previous_result) == gold_result
                    if previous_exact:
                        continue
                    feedback = (
                        "The SQL executed but returned a result different from the expected answer. "
                        f"Observed result: {json.dumps(previous_result, ensure_ascii=False)}"
                    )
                    previous_error = None
                except sqlite3.Error as exc:
                    previous_result = None
                    previous_error = str(exc)
                    feedback = previous_error
                prepared.append(
                    {
                        "id": f"exec-repair-{row['id']}-{len(prepared):04d}",
                        "source": "wikisql_executable_repair_eval",
                        "failure_type": failure_type,
                        "sqlite_table": row["sqlite_table"],
                        "question": row["question"],
                        "header": row["header"],
                        "types": row["types"],
                        "sample_rows": row["sample_rows"],
                        "prompt": repair_prompt(row, previous_sql, feedback),
                        "previous_sql": previous_sql,
                        "previous_execution_error": previous_error,
                        "previous_result": previous_result,
                        "gold_sql": row["gold_sql"],
                        "gold_result": row["gold_result"],
                    }
                )
                break
            if len(prepared) >= args.limit:
                break
    finally:
        connection.close()

    if not prepared:
        raise RuntimeError("No executable repair examples could be prepared")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in prepared:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "format_version": 1,
        "dataset": "internal executable WikiSQL repair probe",
        "source": str(args.wikisql_eval),
        "database": str(args.database),
        "samples": len(prepared),
        "selection": "fixed WikiSQL probe order; first executable wrong corruption per row",
        "metrics": [
            "SQL extraction rate",
            "SQLite execution rate",
            "execution repair accuracy",
            "normalized SQL exact",
            "previous SQL execution accuracy baseline",
        ],
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
