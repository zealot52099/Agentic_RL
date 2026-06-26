#!/usr/bin/env python3
"""Build Phase5 unified-format Data Agent SFT data.

All completions use one production-style action schema:
  tool_call: {"action":"tool_call","calls":[{"name":...,"arguments":{...}}]}
  no_tool:   {"action":"no_tool","calls":[]}
  clarify:   {"action":"clarify","missing":[...],"message":"..."}

The goal is to avoid mixing benchmark/MCP-specific output protocols during SFT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SYSTEM = (
    "You are the main model of a data agent. Decide whether to call a tool, "
    "ask for clarification, or answer without tools. Return exactly one JSON "
    "object and no extra text. Use this schema: "
    '{"action":"tool_call","calls":[{"name":"tool_name","arguments":{...}}]} '
    "for tool calls; "
    '{"action":"no_tool","calls":[]} when no tool is needed; '
    '{"action":"clarify","missing":["field"],"message":"question"} when '
    "required information is missing."
)

EXECUTE_SQL_TOOL = {
    "name": "execute_sql",
    "description": "Execute a SQL query against the specified database.",
    "parameters": {
        "type": "object",
        "properties": {
            "database": {
                "type": "string",
                "description": "Database id or logical database name.",
            },
            "sql": {"type": "string", "description": "SQL query to execute."},
        },
        "required": ["database", "sql"],
    },
}


def read_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def stable_bucket(value: str, modulo: int = 10000) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:12], 16) % modulo


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def extract_block(prompt: str) -> tuple[str, str]:
    """Return available-functions/catalog block and user text from older prompts."""
    available = ""
    user = prompt
    for marker in ["AVAILABLE FUNCTIONS:", "MCP_SERVER_CATALOG:"]:
        if marker in prompt:
            after = prompt.split(marker, 1)[1]
            if "\n\nUSER:" in after:
                available, rest = after.split("\n\nUSER:", 1)
                user = rest.split("\n\nASSISTANT", 1)[0].split("ASSISTANT:", 1)[0].strip()
                return available.strip(), user.strip()
    if "USER:" in prompt:
        user = prompt.split("USER:", 1)[1].split("ASSISTANT", 1)[0].strip()
    return available.strip(), user.strip()


def render_prompt(available: Any, user: str, extra_context: str = "") -> str:
    if not isinstance(available, str):
        available = json.dumps(available, ensure_ascii=False, sort_keys=True)
    parts = [SYSTEM]
    if available.strip():
        parts.append("AVAILABLE FUNCTIONS:\n" + available.strip())
    if extra_context.strip():
        parts.append(extra_context.strip())
    parts.append("USER:\n" + user.strip())
    return "\n\n".join(parts) + "\n\nASSISTANT:\n"


def normalize_call(item: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    name = item.get("name") or item.get("tool_name") or item.get("function")
    args = item.get("arguments") or item.get("args") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {"value": args}
    if not name:
        return None
    return {"name": str(name), "arguments": args if isinstance(args, dict) else {"value": args}}


def normalize_completion(text: str) -> tuple[dict[str, Any] | None, str]:
    stripped = (text or "").strip()
    if not stripped:
        return None, "empty"
    try:
        obj = json.loads(stripped)
    except Exception:
        return None, "non_json"
    if isinstance(obj, list):
        if not obj:
            return {"action": "no_tool", "calls": []}, "no_tool"
        calls = [call for call in (normalize_call(item) for item in obj) if call]
        if calls:
            return {"action": "tool_call", "calls": calls}, "tool_call"
        return None, "bad_list"
    if isinstance(obj, dict):
        action = obj.get("action")
        if action == "clarify":
            return {
                "action": "clarify",
                "missing": obj.get("missing") if isinstance(obj.get("missing"), list) else [],
                "message": str(obj.get("message") or "Please provide the required information."),
            }, "clarify"
        if action == "no_tool":
            return {"action": "no_tool", "calls": []}, "no_tool"
        if action == "tool_call" and isinstance(obj.get("calls"), list):
            calls = [call for call in (normalize_call(item) for item in obj["calls"]) if call]
            return {"action": "tool_call", "calls": calls}, "tool_call" if calls else "bad_calls"
        call = normalize_call(obj)
        if call:
            return {"action": "tool_call", "calls": [call]}, "tool_call"
    return None, "unknown_json"


def make_record(
    id_: str,
    prompt: str,
    completion_obj: dict[str, Any],
    mixture: str,
    source: str,
    weight: float = 1.0,
    meta: dict[str, Any] | None = None,
):
    row = {
        "id": id_,
        "prompt": prompt,
        "completion": canonical_json(completion_obj),
        "mixture_source": mixture,
        "source": source,
        "loss_weight": weight,
    }
    if meta:
        row.update(meta)
    return row


def from_existing_sft(path: Path, source: str, limit: int, rng: random.Random):
    rows = []
    skipped = Counter()
    for row in read_jsonl(path) or []:
        completion, kind = normalize_completion(str(row.get("completion", "")))
        if not completion:
            skipped[kind] += 1
            continue
        available, user = extract_block(str(row.get("prompt", "")))
        if not user:
            skipped["no_user"] += 1
            continue
        mixture = row.get("mixture_source") or row.get("mixture") or kind
        weight = 1.0 if kind == "tool_call" else 2.0 if kind == "clarify" else 3.0
        rows.append(
            make_record(
                str(row.get("id", f"{source}-{len(rows)}")),
                render_prompt(available, user),
                completion,
                str(mixture),
                source,
                weight,
            )
        )
    rng.shuffle(rows)
    return rows[:limit], skipped


def from_spider(path: Path, limit: int, rng: random.Random):
    rows = []
    for index, row in enumerate(read_jsonl(path) or []):
        question = row.get("question")
        sql = row.get("query")
        database = row.get("db_id", "unknown")
        if not question or not sql:
            continue
        context = (
            "DATABASE CONTEXT:\n"
            f"database: {database}\n"
            "schema: not provided in this local Spider JSONL snapshot. Use the "
            "question and database id; do not invent non-SQL text."
        )
        prompt = render_prompt([EXECUTE_SQL_TOOL], str(question), context)
        completion = {
            "action": "tool_call",
            "calls": [
                {
                    "name": "execute_sql",
                    "arguments": {"database": str(database), "sql": str(sql)},
                }
            ],
        }
        rows.append(
            make_record(
                f"spider-{index}",
                prompt,
                completion,
                "sql_spider",
                "spider_train",
                1.3,
                {"db_id": database},
            )
        )
    rng.shuffle(rows)
    return rows[:limit]


def write_jsonl(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260626)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    plan = {
        "phase3": 6500,
        "mcp": 3500,
        "parallel": 2000,
        "spider": 3500,
    }
    phase3, skip3 = from_existing_sft(
        Path("datasets/processed/phase3_final/train_phase3.jsonl"),
        "phase3_final",
        plan["phase3"],
        rng,
    )
    mcp, skipm = from_existing_sft(
        Path("datasets/processed/mcp_lora_sft_v3_20260618/sft_v2_all/train_sft.jsonl"),
        "mcp_sft_v2_all",
        plan["mcp"],
        rng,
    )
    parallel, skipp = from_existing_sft(
        Path("datasets/processed/mcp_lora_sft_v3_20260618/sft_clean_parallel/train_clean_parallel.jsonl"),
        "mcp_parallel",
        plan["parallel"],
        rng,
    )
    spider = from_spider(Path("datasets/modelscope/spider_train.jsonl"), plan["spider"], rng)
    all_rows = phase3 + mcp + parallel + spider
    rng.shuffle(all_rows)

    train, validation = [], []
    for row in all_rows:
        if stable_bucket(str(row["id"])) < 500:
            validation.append(row)
        else:
            train.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "validation.jsonl", validation)

    by_mix = defaultdict(list)
    for row in all_rows:
        if len(by_mix[row["mixture_source"]]) < 3:
            by_mix[row["mixture_source"]].append(row)
    audit = []
    for key in sorted(by_mix):
        audit.extend(by_mix[key])
    write_jsonl(args.output_dir / "audit_examples.jsonl", audit)

    counts = Counter(row["mixture_source"] for row in all_rows)
    actions = Counter(json.loads(row["completion"]).get("action") for row in all_rows)
    manifest = {
        "format_version": 1,
        "schema": "data_agent_action_v1",
        "seed": args.seed,
        "counts": {"all": len(all_rows), "train": len(train), "validation": len(validation)},
        "mixture_counts": dict(counts),
        "action_counts": dict(actions),
        "source_plan": plan,
        "skipped": {
            "phase3": dict(skip3),
            "mcp": dict(skipm),
            "parallel": dict(skipp),
        },
        "output_schema": {
            "tool_call": {"action": "tool_call", "calls": [{"name": "...", "arguments": {}}]},
            "no_tool": {"action": "no_tool", "calls": []},
            "clarify": {"action": "clarify", "missing": ["..."], "message": "..."},
        },
        "limitations": [
            "Spider local JSONL lacks table schemas; SQL samples train execute_sql formatting and SQL expression but not full schema linking.",
            "Official benchmark test files are not included in training.",
            "This dataset targets production-format evaluation, not official BFCL raw-output leaderboard format.",
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
