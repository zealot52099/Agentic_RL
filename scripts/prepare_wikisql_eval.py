#!/usr/bin/env python3
"""Prepare a deterministic WikiSQL evaluation subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path


AGGREGATIONS = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
OPERATORS = ["=", ">", "<", "OP"]


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def literal(value: object) -> str:
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value).replace("'", "''")
    return f"'{text}'"


def render_gold_sql(sql: dict, table_name: str) -> str:
    selected = quote_identifier(f"col{sql['sel']}")
    aggregate = AGGREGATIONS[sql["agg"]]
    expression = f"{aggregate}({selected})" if aggregate else selected
    query = f"SELECT {expression} FROM {quote_identifier(table_name)}"
    conditions = []
    for column, operator, value in sql["conds"]:
        condition = (
            f"{quote_identifier(f'col{column}')} "
            f"{OPERATORS[operator]} {literal(value)}"
        )
        conditions.append(condition)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    return query


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=256)
    args = parser.parse_args()

    tables = {}
    with (args.data_dir / "test.tables.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            tables[row["id"]] = row

    questions = []
    with (args.data_dir / "test.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = hashlib.sha256(
                f"{row['table_id']}\n{row['question']}".encode()
            ).hexdigest()
            questions.append((key, row))
    questions.sort(key=lambda item: item[0])

    connection = sqlite3.connect(args.data_dir / "test.db")
    prepared = []
    for key, row in questions:
        table = tables[row["table_id"]]
        sqlite_table = "table_" + row["table_id"].replace("-", "_")
        gold_sql = render_gold_sql(row["sql"], sqlite_table)
        try:
            gold_result = connection.execute(gold_sql).fetchall()
        except sqlite3.Error:
            continue
        if not gold_result:
            continue
        prepared.append(
            {
                "id": key[:16],
                "table_id": row["table_id"],
                "sqlite_table": sqlite_table,
                "question": row["question"],
                "header": table["header"],
                "types": table["types"],
                "sample_rows": table["rows"][:3],
                "gold_sql": gold_sql,
                "gold_result": gold_result,
            }
        )
        if len(prepared) == args.limit:
            break
    connection.close()

    if len(prepared) != args.limit:
        raise RuntimeError(f"Only prepared {len(prepared)} valid rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in prepared:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "format_version": 1,
        "dataset": "WikiSQL official test",
        "selection": "lowest SHA256(table_id + newline + question)",
        "samples": len(prepared),
        "source_questions": len(questions),
        "metrics": [
            "SQL extraction rate",
            "SQLite execution rate",
            "execution result exact match",
            "normalized SQL exact match",
        ],
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
