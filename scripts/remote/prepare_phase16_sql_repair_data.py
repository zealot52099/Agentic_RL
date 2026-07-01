#!/usr/bin/env python3
"""Prepare Phase16 SQL repair, preference, and GRPO data.

This script is intentionally data-only.  It converts already available
WikiSQL/Spider/SQL-Create-Context/Data-Agent assets into a set of verifiable
SQL-focused files for the next training stage:

* repair SFT rows from previous failed predictions;
* DPO pairs: gold SQL vs failed/executable-wrong SQL;
* GRPO prompts with executable WikiSQL table payloads;
* a mixed SFT file with SQL repair + schema/value-grounded SQL + replay rows;
* a failure taxonomy report for Phase10 WikiSQL predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SQL_KEYWORDS = {
    "select",
    "from",
    "where",
    "group",
    "by",
    "order",
    "limit",
    "and",
    "or",
    "as",
    "sum",
    "count",
    "min",
    "max",
    "avg",
    "distinct",
    "having",
    "join",
    "on",
}


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def stable_id(*parts: Any) -> str:
    h = hashlib.sha1()
    for part in parts:
        h.update(json.dumps(part, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]


def norm_sql(sql: str | None) -> str:
    if not sql:
        return ""
    return re.sub(r"\s+", " ", sql.strip().rstrip(";")).strip()


def tokenize_question(text: str) -> set[str]:
    return {t.lower() for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-./]*", text)}


def parse_solution(raw: str | dict[str, Any] | None) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return {}


def extract_table_block(query: str) -> str:
    marker = "\n\nQuestion:"
    text = query.split(marker, 1)[0].strip() if marker in query else query.strip()
    return normalize_context(text)


def normalize_context(text: str) -> str:
    """Remove inherited task instructions, keeping only schema/table context."""
    if not text:
        return ""
    lines = text.strip().splitlines()
    while lines and (
        lines[0].startswith("Write one SQLite SELECT query")
        or lines[0].startswith("Use only the")
        or lines[0].startswith("Return SQL only")
    ):
        lines.pop(0)
        while lines and not lines[0].strip():
            lines.pop(0)
    return "\n".join(lines).strip()


def value_grounding(question: str, payload: dict[str, Any], max_values: int = 12) -> list[dict[str, Any]]:
    tokens = tokenize_question(question)
    headers = payload.get("header") or []
    rows = payload.get("rows") or []
    hits: list[dict[str, Any]] = []
    seen = set()
    for row in rows:
        if not isinstance(row, list):
            continue
        for idx, value in enumerate(row):
            text = str(value)
            low = text.lower()
            value_tokens = tokenize_question(text)
            matched = bool(tokens & value_tokens) or any(tok and tok in low for tok in tokens if len(tok) >= 4)
            if not matched:
                continue
            key = (idx, low)
            if key in seen:
                continue
            seen.add(key)
            hits.append(
                {
                    "column": f"col{idx}",
                    "display_name": headers[idx] if idx < len(headers) else f"col{idx}",
                    "value": text,
                }
            )
            if len(hits) >= max_values:
                return hits
    return hits


def infer_failure(row: dict[str, Any]) -> str:
    pred = norm_sql(row.get("predicted_sql") or row.get("raw_response"))
    gold = norm_sql(row.get("gold_sql"))
    err = str(row.get("execution_error") or "").lower()
    if row.get("execution_exact"):
        return "correct_execution"
    if not pred:
        return "empty_prediction"
    if err:
        if "no such column" in err:
            return "wrong_column_or_unquoted_display_name"
        if "no such table" in err:
            return "wrong_table"
        if "syntax" in err:
            return "sql_syntax_error"
        return "execution_error_other"
    if gold and re.sub(r"\blimit\s+\d+\b", "", pred, flags=re.I) != re.sub(r"\blimit\s+\d+\b", "", gold, flags=re.I):
        if any(fn in pred.lower() + gold.lower() for fn in ("sum(", "count(", "avg(", "min(", "max(")):
            return "wrong_aggregation_or_selection"
        if " where " in pred.lower() or " where " in gold.lower():
            return "wrong_filter_or_value"
        if " order by " in pred.lower() or " limit " in pred.lower():
            return "wrong_order_or_limit"
    return "executable_wrong_result"


def make_repair_prompt(row: dict[str, Any], original_query: str | None = None) -> str:
    question = row.get("question") or ""
    pred = norm_sql(row.get("predicted_sql") or row.get("raw_response"))
    err = row.get("execution_error")
    gold_result = row.get("gold_result")
    context = original_query or f"Question: {question}"
    return (
        "You are improving a SQLite query for a data agent. Return one corrected SQLite SELECT query only.\n\n"
        f"{context.strip()}\n\n"
        f"Previous SQL:\n{pred}\n\n"
        f"Execution feedback:\n{err if err else 'The SQL executed but returned a result different from the expected answer.'}\n\n"
        f"Expected execution result:\n{json.dumps(gold_result, ensure_ascii=False)}\n\n"
        "Corrected SQL:"
    )


def make_sql_sft_prompt(question: str, context: str, value_hints: list[dict[str, Any]] | None = None) -> str:
    context = normalize_context(context)
    hints = ""
    if value_hints:
        hints = "\nRelevant cell values:\n" + json.dumps(value_hints, ensure_ascii=False)
    return (
        "Write one SQLite SELECT query that answers the question. "
        "Use only the provided schema/table context. Return SQL only, without explanation or Markdown.\n\n"
        f"{context.strip()}{hints}\n\nQuestion: {question}\nSQL:"
    )


def dedupe_rows(rows: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for row in rows:
        if mode == "sft":
            key = (row.get("prompt"), row.get("completion"))
        elif mode == "dpo":
            key = (row.get("prompt"), row.get("chosen"), row.get("rejected"))
        elif mode == "grpo":
            key = (row.get("query"), row.get("solution"))
        else:
            key = (json.dumps(row, ensure_ascii=False, sort_keys=True),)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def try_sqlite_result(payload: dict[str, Any], sql: str) -> tuple[bool, Any]:
    table = payload.get("table_name")
    headers = payload.get("header") or []
    rows = payload.get("rows") or []
    if not table or not headers:
        return False, "missing_table_payload"
    conn = sqlite3.connect(":memory:")
    try:
        cols = ", ".join([f'"col{i}" TEXT' for i in range(len(headers))])
        conn.execute(f'CREATE TABLE "{table}" ({cols})')
        placeholders = ", ".join(["?"] * len(headers))
        conn.executemany(f'INSERT INTO "{table}" VALUES ({placeholders})', [r[: len(headers)] for r in rows if isinstance(r, list)])
        cur = conn.execute(sql)
        return True, [list(r) for r in cur.fetchall()]
    except Exception as exc:
        return False, str(exc)
    finally:
        conn.close()


def corrupt_sql(sql: str) -> list[tuple[str, str]]:
    variants: list[tuple[str, str]] = []
    s = norm_sql(sql)
    if not s:
        return variants
    variants.append(("drop_quotes", s.replace('"', "")))
    variants.append(("remove_limit", re.sub(r"\s+LIMIT\s+\d+\b", "", s, flags=re.I)))
    variants.append(("wrong_agg_sum_to_count", re.sub(r"\bSUM\s*\(", "COUNT(", s, flags=re.I)))
    variants.append(("wrong_agg_min_to_max", re.sub(r"\bMIN\s*\(", "MAX(", s, flags=re.I)))
    variants.append(("wrong_agg_max_to_min", re.sub(r"\bMAX\s*\(", "MIN(", s, flags=re.I)))
    out = []
    seen = {s}
    for kind, cand in variants:
        cand = norm_sql(cand)
        if cand and cand not in seen:
            seen.add(cand)
            out.append((kind, cand))
    return out


def build_phase16(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wikisql_grpo = read_jsonl(Path(args.wikisql_grpo), args.max_wikisql)
    predictions = read_jsonl(Path(args.predictions), args.max_predictions)
    spider_train = read_jsonl(Path(args.spider_train), args.max_spider)
    spider_val = read_jsonl(Path(args.spider_val), args.max_spider_val)
    sql_context = read_jsonl(Path(args.sql_create_context), args.max_sql_context)
    replay = read_jsonl(Path(args.replay_sft), args.max_replay)

    question_to_wikisql: dict[str, dict[str, Any]] = {}
    for row in wikisql_grpo:
        question_to_wikisql.setdefault(str(row.get("question", "")), row)

    failure_rows = []
    repair_sft = []
    synthetic_repair_sft = []
    dpo_pairs = []
    grpo_rows = []
    eval_repair = []
    failure_counts = Counter()
    source_counts = Counter()

    for pred in predictions:
        failure = infer_failure(pred)
        failure_counts[failure] += 1
        if failure == "correct_execution":
            continue
        question = str(pred.get("question", ""))
        src = question_to_wikisql.get(question)
        payload = parse_solution(src.get("solution")) if src else {}
        original_query = src.get("query") if src else None
        gold_sql = norm_sql(pred.get("gold_sql") or payload.get("gold_sql"))
        rejected_sql = norm_sql(pred.get("predicted_sql") or pred.get("raw_response"))
        if not gold_sql:
            continue
        value_hints = value_grounding(question, payload)
        repair_row = {
            "id": "phase16-repair-" + stable_id(question, rejected_sql, gold_sql),
            "source": "phase10_wikisql_failed_prediction",
            "failure_type": failure,
            "prompt": make_repair_prompt(pred, original_query),
            "completion": gold_sql,
            "loss_weight": 2.0 if "wrong_column" in failure else 1.6,
            "question": question,
            "gold_sql": gold_sql,
            "rejected_sql": rejected_sql,
            "execution_error": pred.get("execution_error"),
            "gold_result": pred.get("gold_result"),
            "value_hints": value_hints,
        }
        repair_sft.append(repair_row)
        dpo_pairs.append(
            {
                "id": "phase16-dpo-" + stable_id(question, rejected_sql, gold_sql),
                "source": "phase10_wikisql_failed_prediction",
                "failure_type": failure,
                "prompt": make_sql_sft_prompt(question, extract_table_block(original_query or ""), value_hints),
                "chosen": gold_sql,
                "rejected": rejected_sql,
                "chosen_result": pred.get("gold_result"),
                "rejected_result": pred.get("predicted_result"),
            }
        )
        failure_rows.append(
            {
                "id": pred.get("id") or stable_id(question, rejected_sql),
                "question": question,
                "failure_type": failure,
                "predicted_sql": rejected_sql,
                "gold_sql": gold_sql,
                "execution_error": pred.get("execution_error"),
                "execution_exact": bool(pred.get("execution_exact")),
                "value_hints": value_hints,
            }
        )

    # WikiSQL executable GRPO and synthetic DPO candidates.
    for row in wikisql_grpo:
        payload = parse_solution(row.get("solution"))
        gold_sql = norm_sql(payload.get("gold_sql"))
        question = str(row.get("question", ""))
        if not gold_sql:
            continue
        hints = value_grounding(question, payload)
        grpo_rows.append(
            {
                "id": "phase16-grpo-" + stable_id(question, gold_sql),
                "source": "wikisql_executable",
                "query": make_sql_sft_prompt(question, extract_table_block(row.get("query", "")), hints),
                "solution": json.dumps(
                    {
                        **payload,
                        "gold_sql": gold_sql,
                        "value_hints": hints,
                        "reward_mode": "execution_exact_readonly_sqlite",
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "question": question,
                "gold_sql": gold_sql,
                "value_hints": hints,
            }
        )
        for kind, bad_sql in corrupt_sql(gold_sql):
            ok, result = try_sqlite_result(payload, bad_sql)
            gold_ok, gold_result = try_sqlite_result(payload, gold_sql)
            if bad_sql == gold_sql:
                continue
            if len(synthetic_repair_sft) < args.synthetic_repair_keep:
                synthetic_repair_sft.append(
                    {
                        "id": "phase16-synth-repair-" + stable_id(question, kind, bad_sql, gold_sql),
                        "source": "wikisql_synthetic_sql_repair",
                        "failure_type": kind if not ok else "synthetic_executable_or_semantic_negative",
                        "prompt": (
                            "You are improving a SQLite query for a data agent. Return one corrected SQLite SELECT query only.\n\n"
                            f"{make_sql_sft_prompt(question, extract_table_block(row.get('query', '')), hints)}\n\n"
                            f"Previous SQL:\n{bad_sql}\n\n"
                            "Execution feedback:\n"
                            f"{result if not ok else 'The SQL executed but may use the wrong aggregation, quoting, filter, or limit compared with the expected result.'}\n\n"
                            f"Expected execution result:\n{json.dumps(gold_result if gold_ok else payload.get('gold_result'), ensure_ascii=False)}\n\n"
                            "Corrected SQL:"
                        ),
                        "completion": gold_sql,
                        "loss_weight": 1.4,
                        "question": question,
                        "gold_sql": gold_sql,
                        "rejected_sql": bad_sql,
                        "rejected_error": None if ok else result,
                        "gold_result": gold_result if gold_ok else payload.get("gold_result"),
                        "value_hints": hints,
                    }
                )
            # Keep both execution-error and executable-wrong variants; both teach useful boundaries.
            dpo_pairs.append(
                {
                    "id": "phase16-synth-dpo-" + stable_id(question, kind, bad_sql),
                    "source": "wikisql_synthetic_negative",
                    "failure_type": kind if not ok else "synthetic_executable_or_semantic_negative",
                    "prompt": make_sql_sft_prompt(question, extract_table_block(row.get("query", "")), hints),
                    "chosen": gold_sql,
                    "rejected": bad_sql,
                    "chosen_result": gold_result if gold_ok else None,
                    "rejected_result": result if ok else None,
                    "rejected_error": None if ok else result,
                }
            )

    # Spider and SQL-Create-Context are SFT-only here because DB files are not present.
    schema_sft = []
    for row in spider_train + spider_val:
        question = str(row.get("question", ""))
        sql = norm_sql(row.get("query"))
        db_id = row.get("db_id")
        if not question or not sql:
            continue
        prompt = make_sql_sft_prompt(
            question,
            f"Database id: {db_id}\nSchema: unavailable in this snapshot. Use the table and column names implied by the SQL benchmark metadata.",
        )
        schema_sft.append(
            {
                "id": "phase16-spider-" + stable_id(db_id, question, sql),
                "source": "spider_sft_no_db_validation",
                "prompt": prompt,
                "completion": sql,
                "loss_weight": 1.0,
                "db_id": db_id,
            }
        )
    for row in sql_context:
        question = str(row.get("question", ""))
        sql = norm_sql(row.get("answer"))
        context = str(row.get("context", ""))
        if not question or not sql or not context:
            continue
        schema_sft.append(
            {
                "id": "phase16-sqlctx-" + stable_id(context, question, sql),
                "source": "sql_create_context_sft",
                "prompt": make_sql_sft_prompt(question, context),
                "completion": sql,
                "loss_weight": 1.1,
            }
        )

    # Keep a small amount of tool/multi-turn replay, but with lower SQL priority.
    replay_rows = []
    for row in replay:
        if "prompt" in row and "completion" in row:
            replay_rows.append(
                {
                    **row,
                    "id": row.get("id") or "phase16-replay-" + stable_id(row.get("prompt"), row.get("completion")),
                    "source": row.get("source", "phase15_clean_v4_replay"),
                    "phase16_role": "tool_and_multiturn_retention",
                    "loss_weight": min(float(row.get("loss_weight", 1.0)), 1.0),
                }
            )

    random.shuffle(repair_sft)
    random.shuffle(synthetic_repair_sft)
    random.shuffle(schema_sft)
    random.shuffle(replay_rows)
    random.shuffle(dpo_pairs)
    random.shuffle(grpo_rows)

    eval_repair = repair_sft[: min(len(repair_sft), args.eval_repair_size)]
    train_repair = repair_sft[len(eval_repair) :]
    train_repair = train_repair + synthetic_repair_sft
    random.shuffle(train_repair)
    train_schema = schema_sft[: args.schema_sft_keep]
    train_replay = replay_rows[: args.replay_keep]
    train_repair = dedupe_rows(train_repair, "sft")
    train_schema = dedupe_rows(train_schema, "sft")
    train_replay = dedupe_rows(train_replay, "sft")
    dpo_pairs = dedupe_rows(dpo_pairs, "dpo")
    grpo_rows = dedupe_rows(grpo_rows, "grpo")
    mixed_sft = dedupe_rows(train_repair + train_schema + train_replay, "sft")
    random.shuffle(mixed_sft)

    counts = {
        "train_sql_repair_sft": write_jsonl(out_dir / "train_sql_repair_sft.jsonl", train_repair),
        "eval_sql_repair_probe": write_jsonl(out_dir / "eval_sql_repair_probe.jsonl", eval_repair),
        "train_sql_dpo_pairs": write_jsonl(out_dir / "train_sql_dpo_pairs.jsonl", dpo_pairs),
        "train_sql_grpo": write_jsonl(out_dir / "train_sql_grpo.jsonl", grpo_rows),
        "train_sql_mixture_sft": write_jsonl(out_dir / "train_sql_mixture_sft.jsonl", mixed_sft),
        "failure_taxonomy": write_jsonl(out_dir / "failure_taxonomy.jsonl", failure_rows),
    }
    for row in mixed_sft:
        source_counts[row.get("source", "unknown")] += 1

    examples = {
        "repair_sft": train_repair[:2],
        "dpo_pair": dpo_pairs[:2],
        "grpo": grpo_rows[:1],
        "mixed_sft": mixed_sft[:2],
    }
    (out_dir / "examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(out_dir),
        "seed": args.seed,
        "inputs": {
            "wikisql_grpo": args.wikisql_grpo,
            "predictions": args.predictions,
            "spider_train": args.spider_train,
            "spider_val": args.spider_val,
            "sql_create_context": args.sql_create_context,
            "replay_sft": args.replay_sft,
        },
        "counts": counts,
        "failure_counts": dict(failure_counts),
        "mixed_source_counts": dict(source_counts),
        "notes": [
            "WikiSQL rows are executable and should be used for SQL GRPO/rewarded repair.",
            "Spider rows are SFT-only because this snapshot does not include Spider DB files.",
            "Repair SFT includes real Phase10 failures plus synthetic WikiSQL corruptions with execution feedback.",
            "DPO pairs include both real Phase10 failures and synthetic corrupted SQL negatives.",
            "Value hints are derived from question/table cell overlap and should be treated as retrieval features, not labels.",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "failure_report.json").write_text(
        json.dumps(
            {
                "failure_counts": dict(failure_counts),
                "top_failures": failure_counts.most_common(),
                "mixed_source_counts": dict(source_counts),
                "counts": counts,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="datasets/processed/phase16_sql_repair_20260701")
    parser.add_argument("--wikisql-grpo", default="datasets/processed/phase8_swift_wikisql_grpo_20260629/train.jsonl")
    parser.add_argument(
        "--predictions",
        default="evals/phase10_phase15_post_eval_20260701_102944/wikisql_phase10/phase10b_mixed_retention_step0800_predictions.jsonl",
    )
    parser.add_argument("--spider-train", default="datasets/modelscope/spider_train.jsonl")
    parser.add_argument("--spider-val", default="datasets/modelscope/spider_val.jsonl")
    parser.add_argument("--sql-create-context", default="datasets/modelscope/sql_create_context.jsonl")
    parser.add_argument("--replay-sft", default="datasets/processed/phase15_multiturn_clean_v4_20260630/train_sft_mixture_clean.jsonl")
    parser.add_argument("--max-wikisql", type=int, default=4096)
    parser.add_argument("--max-predictions", type=int, default=4096)
    parser.add_argument("--max-spider", type=int, default=7000)
    parser.add_argument("--max-spider-val", type=int, default=200)
    parser.add_argument("--max-sql-context", type=int, default=500)
    parser.add_argument("--max-replay", type=int, default=20000)
    parser.add_argument("--schema-sft-keep", type=int, default=5000)
    parser.add_argument("--replay-keep", type=int, default=3000)
    parser.add_argument("--synthetic-repair-keep", type=int, default=4096)
    parser.add_argument("--eval-repair-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260701)
    args = parser.parse_args()
    manifest = build_phase16(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
