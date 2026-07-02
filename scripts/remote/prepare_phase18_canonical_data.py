#!/usr/bin/env python3
"""Build Phase18 canonical Data Agent SFT data.

Phase18 fixes a data governance issue found after auditing the real processed
JSONL files: SQL, tool-call, and multi-turn rows used different prompt markers
and sometimes different output protocols. This script re-renders rows into one
Data Agent action format:

  prompt: canonical system + AVAILABLE_TOOLS + task context + USER + ASSISTANT
  completion: {"action":"tool_call|clarify|refuse|final","...":...}

The script keeps upstream metadata, but training consumption is always
prompt/completion/source/mixture_source/loss_weight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


CANONICAL_SYSTEM = (
    "You are the main model of a Data Agent. Use the provided tools when they "
    "are needed, ask for clarification when required information is missing, "
    "refuse unsafe requests, or provide a final answer when no tool is needed. "
    "Return exactly one JSON object and no extra text.\n\n"
    "Allowed actions:\n"
    "{\"action\":\"tool_call\",\"calls\":[{\"name\":\"tool_name\",\"arguments\":{...}}]}\n"
    "{\"action\":\"clarify\",\"missing\":[\"field\"],\"message\":\"question\"}\n"
    "{\"action\":\"refuse\",\"message\":\"reason\"}\n"
    "{\"action\":\"final\",\"answer\":\"answer\"}"
)


SQL_TOOLS = [
    {
        "name": "list_tables",
        "description": "List available database tables.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "describe_table",
        "description": "Return columns and types for one table.",
        "input_schema": {
            "type": "object",
            "required": ["table"],
            "properties": {"table": {"type": "string"}},
            "additionalProperties": False,
        },
    },
    {
        "name": "run_sql",
        "description": "Execute one read-only SQLite SELECT query.",
        "input_schema": {
            "type": "object",
            "required": ["sql"],
            "properties": {"sql": {"type": "string"}},
            "additionalProperties": False,
        },
    },
    {
        "name": "final_answer",
        "description": "Return the final natural-language answer to the user.",
        "input_schema": {
            "type": "object",
            "required": ["answer"],
            "properties": {"answer": {"type": "string"}},
            "additionalProperties": False,
        },
    },
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_id(*parts: Any) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def safe_json_loads(text: str) -> Any | None:
    try:
        return json.loads(text)
    except Exception:
        return None


def normalize_tools_block(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    parsed = safe_json_loads(text)
    if parsed is None:
        return text
    return canonical_json(parsed)


def extract_available_and_user(prompt: str) -> tuple[str, str, str]:
    """Return available block, user request, and extra context from old prompts."""
    available = ""
    user = ""
    extra = ""
    markers = ["AVAILABLE_TOOLS:", "AVAILABLE FUNCTIONS:", "MCP_SERVER_CATALOG:"]
    rest = prompt
    for marker in markers:
        if marker in prompt:
            before, after = prompt.split(marker, 1)
            if "\n\nUSER:" in after:
                available, rest = after.split("\n\nUSER:", 1)
                extra = before.split("Allowed actions:")[-1].strip()
                break
    if not available and "\n\nUSER:" in prompt:
        rest = prompt.split("\n\nUSER:", 1)[1]
    elif not available and "USER:" in prompt:
        rest = prompt.split("USER:", 1)[1]
    if "ASSISTANT:" in rest:
        # Keep intermediate transcript assistant/tool turns for multi-turn rows,
        # and remove only the final assistant generation marker.
        user = rest.rsplit("ASSISTANT:", 1)[0].strip()
    else:
        user = rest.strip()
    return normalize_tools_block(available), user, extra


def render_prompt(tools: Any, user: str, context: str = "") -> str:
    if not isinstance(tools, str):
        tools = canonical_json(tools)
    parts = [CANONICAL_SYSTEM, "AVAILABLE_TOOLS:\n" + tools.strip()]
    if context.strip():
        parts.append(context.strip())
    parts.append("USER:\n" + user.strip())
    return "\n\n".join(parts) + "\n\nASSISTANT:\n"


def parse_action_completion(text: str) -> dict[str, Any] | None:
    obj = safe_json_loads((text or "").strip())
    if obj is None:
        return None
    if isinstance(obj, dict) and obj.get("action"):
        return obj
    if isinstance(obj, list):
        return {"action": "no_tool", "calls": []} if not obj else {
            "action": "tool_call",
            "calls": [
                {
                    "name": item.get("name") or item.get("tool_name") or item.get("function"),
                    "arguments": item.get("arguments") or item.get("args") or {},
                }
                for item in obj
                if isinstance(item, dict)
            ],
        }
    return None


def is_select_sql(text: str) -> bool:
    return bool(re.match(r"^\s*SELECT\b", text or "", re.I))


def sql_action(sql: str) -> dict[str, Any]:
    return {"action": "tool_call", "calls": [{"name": "run_sql", "arguments": {"sql": sql.strip()}}]}


def extract_question(prompt: str, row: dict[str, Any]) -> str:
    if row.get("question"):
        return str(row["question"]).strip()
    match = re.search(r"\nQuestion:\s*(.*?)\n(?:SQL:|$)", prompt, flags=re.S)
    if match:
        return match.group(1).strip()
    _, user, _ = extract_available_and_user(prompt)
    return user.strip() or "Answer the SQL question using the provided context."


def extract_sql_context(prompt: str, row: dict[str, Any]) -> str:
    chunks: list[str] = []
    for marker in ["Table:", "Columns:", "Sample rows:", "Relevant cell values:", "DATABASE CONTEXT:"]:
        if marker in prompt:
            # Keep from the first useful SQL marker up to Question/Previous SQL.
            start = prompt.find(marker)
            end_candidates = [
                pos
                for token in [
                    "\n\nQuestion:",
                    "\n\nPrevious SQL:",
                    "\nSQL:\n\nPrevious SQL:",
                    "\n\nUSER:",
                ]
                if (pos := prompt.find(token, start)) != -1
            ]
            end = min(end_candidates) if end_candidates else len(prompt)
            chunk = prompt[start:end].strip()
            if chunk and chunk not in chunks:
                chunks.append(chunk)
            break
    if row.get("value_hints") and "Relevant cell values:" not in "\n".join(chunks):
        chunks.append("Relevant cell values:\n" + json.dumps(row["value_hints"], ensure_ascii=False))
    if row.get("db_id") and not chunks:
        chunks.append(
            "DATABASE CONTEXT:\n"
            f"database: {row['db_id']}\n"
            "schema: unavailable in this source snapshot; this row is low-weight SQL format replay."
        )
    return "\n\n".join(chunks)


def make_sft_row(
    *,
    id_: str,
    prompt: str,
    completion: dict[str, Any],
    source: str,
    mixture_source: str,
    loss_weight: float,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "id": id_,
        "prompt": prompt,
        "completion": canonical_json(completion),
        "source": source,
        "mixture_source": mixture_source,
        "loss_weight": round(float(loss_weight), 4),
        "render_template_version": "phase18_data_agent_action_v1",
        "action_schema_version": "data_agent_action_v1",
    }
    if metadata:
        row["metadata"] = metadata
    return row


def canonicalize_existing_action(row: dict[str, Any], source_prefix: str) -> dict[str, Any] | None:
    action = parse_action_completion(str(row.get("completion", "")))
    if not action:
        return None
    available, user, _ = extract_available_and_user(str(row.get("prompt", "")))
    if not available or not user:
        return None
    prompt = render_prompt(available, user)
    return make_sft_row(
        id_=f"phase18-{source_prefix}-{row.get('id') or stable_id(row)}",
        prompt=prompt,
        completion=action,
        source=f"phase18_{source_prefix}",
        mixture_source=str(row.get("mixture_source") or action.get("action") or "tool_action"),
        loss_weight=float(row.get("loss_weight", 1.0)),
        metadata={"upstream_id": row.get("id"), "upstream_source": row.get("source")},
    )


def canonicalize_multiturn(row: dict[str, Any]) -> dict[str, Any] | None:
    action = parse_action_completion(str(row.get("completion", "")))
    if not action:
        return None
    available, user, _ = extract_available_and_user(str(row.get("prompt", "")))
    if not available or not user:
        return None
    # Preserve transcript inside user block because it contains tool observations.
    prompt = render_prompt(available, user)
    return make_sft_row(
        id_=f"phase18-multiturn-{row.get('id') or stable_id(row)}",
        prompt=prompt,
        completion=action,
        source="phase18_multiturn_canonical",
        mixture_source=str(row.get("mixture_source") or row.get("scenario") or "multiturn"),
        loss_weight=float(row.get("loss_weight", 1.0)),
        metadata={
            "upstream_id": row.get("id"),
            "trace_id": row.get("trace_id"),
            "scenario": row.get("scenario"),
            "phase15_bucket": row.get("phase15_bucket"),
        },
    )


def canonicalize_sql(row: dict[str, Any], source_prefix: str, default_weight: float) -> dict[str, Any] | None:
    completion = str(row.get("completion") or row.get("gold_sql") or "").strip()
    action = parse_action_completion(completion)
    if action and action.get("action") == "tool_call":
        calls = []
        for call in action.get("calls") or []:
            if not isinstance(call, dict):
                continue
            args = call.get("arguments") or {}
            if call.get("name") in {"execute_sql", "run_sql"} and isinstance(args, dict) and args.get("sql"):
                calls.append({"name": "run_sql", "arguments": {"sql": str(args["sql"])}})
        if calls:
            action = {"action": "tool_call", "calls": calls}
        else:
            action = None
    elif is_select_sql(completion):
        action = sql_action(completion)
    else:
        action = None
    if not action:
        return None
    original_prompt = str(row.get("prompt") or row.get("query") or "")
    question = extract_question(original_prompt, row)
    context = extract_sql_context(original_prompt, row)
    prompt = render_prompt(SQL_TOOLS, question, context)
    weight = float(row.get("loss_weight", default_weight))
    if "schema: unavailable" in context or "schema: not provided" in original_prompt:
        weight = min(weight, 0.35)
    return make_sft_row(
        id_=f"phase18-sql-{source_prefix}-{row.get('id') or stable_id(row)}",
        prompt=prompt,
        completion=action,
        source=f"phase18_{source_prefix}",
        mixture_source=str(row.get("mixture_source") or row.get("source") or "sql_grounding"),
        loss_weight=weight,
        metadata={
            "upstream_id": row.get("id"),
            "upstream_source": row.get("source"),
            "db_id": row.get("db_id"),
            "failure_type": row.get("failure_type"),
            "has_value_hints": bool(row.get("value_hints")),
        },
    )


def dedup(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for row in rows:
        key = hashlib.sha256((row["prompt"] + "\n" + row["completion"]).encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def sample_rows(rows: list[dict[str, Any]], limit: int, rng: random.Random) -> list[dict[str, Any]]:
    rows = list(rows)
    rng.shuffle(rows)
    return rows[:limit] if limit > 0 else rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase5-train", type=Path, required=True)
    parser.add_argument("--phase15-clean", type=Path, required=True)
    parser.add_argument("--phase17-train", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--max-tool", type=int, default=4500)
    parser.add_argument("--max-multiturn", type=int, default=12000)
    parser.add_argument("--max-sql", type=int, default=9000)
    parser.add_argument("--validation", type=int, default=768)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    counters = Counter()
    rows: list[dict[str, Any]] = []

    phase5 = read_jsonl(args.phase5_train)
    tool_candidates = []
    spider_low_weight = []
    for row in phase5:
        if row.get("source") == "spider_train":
            sql_row = canonicalize_sql(row, "phase5_spider_low_weight", 0.3)
            if sql_row:
                spider_low_weight.append(sql_row)
            continue
        item = canonicalize_existing_action(row, "phase5_tool")
        if item:
            tool_candidates.append(item)
    rows.extend(sample_rows(tool_candidates, args.max_tool, rng))
    rows.extend(sample_rows(spider_low_weight, min(1200, len(spider_low_weight)), rng))
    counters["phase5_tool"] = min(len(tool_candidates), args.max_tool)
    counters["phase5_spider_low_weight"] = min(len(spider_low_weight), 1200)

    phase15 = read_jsonl(args.phase15_clean)
    multiturn = [item for row in phase15 if (item := canonicalize_multiturn(row))]
    rows.extend(sample_rows(multiturn, args.max_multiturn, rng))
    counters["phase15_multiturn"] = min(len(multiturn), args.max_multiturn)

    phase17 = read_jsonl(args.phase17_train)
    sql_rows = []
    replay_actions = []
    for row in phase17:
        item = canonicalize_sql(row, "phase17_sql", 1.0)
        if item:
            sql_rows.append(item)
            continue
        item = canonicalize_existing_action(row, "phase17_replay_action")
        if item:
            replay_actions.append(item)
    rows.extend(sample_rows(sql_rows, args.max_sql, rng))
    rows.extend(sample_rows(replay_actions, 2500, rng))
    counters["phase17_sql"] = min(len(sql_rows), args.max_sql)
    counters["phase17_replay_action"] = min(len(replay_actions), 2500)

    rows = dedup(rows)
    rng.shuffle(rows)
    validation = rows[: args.validation]
    train = rows[args.validation :]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train_phase18_canonical_sft.jsonl", train)
    write_jsonl(args.output_dir / "validation_phase18_canonical_sft.jsonl", validation)
    write_jsonl(args.output_dir / "audit_examples.jsonl", rows[:80])

    source_counts = Counter(row["source"] for row in rows)
    mixture_counts = Counter(row["mixture_source"] for row in rows)
    audit = {
        "dataset_id": "phase18_canonical_data_agent_sft",
        "created_at": "2026-07-02",
        "render_template_version": "phase18_data_agent_action_v1",
        "action_schema_version": "data_agent_action_v1",
        "inputs": {
            "phase5_train": str(args.phase5_train),
            "phase15_clean": str(args.phase15_clean),
            "phase17_train": str(args.phase17_train),
        },
        "counts": {
            "total": len(rows),
            "train": len(train),
            "validation": len(validation),
            "pre_dedup_buckets": dict(counters),
            "source_counts": dict(source_counts),
            "top_mixture_counts": dict(mixture_counts.most_common(30)),
        },
        "policy": [
            "All rows use prompt/completion for SFT consumption.",
            "All prompts use AVAILABLE_TOOLS marker.",
            "SQL rows are expressed as run_sql tool_call actions.",
            "Spider rows without schema are retained only as low-weight SQL format replay.",
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
