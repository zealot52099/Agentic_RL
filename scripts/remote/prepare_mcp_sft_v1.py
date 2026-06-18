#!/usr/bin/env python3
"""Build an MCP-routing SFT mixture from verified xLAM and replay data."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


SYSTEM = (
    "You are an MCP agent. Select only tools from the supplied server catalog. "
    "Return only a JSON array. Each call must contain server_id, tool_name, and "
    "arguments. Return [] when no tool applies or required information is missing."
)

DOMAIN_RULES = {
    "calendar": ("calendar", "event", "meeting", "schedule"),
    "mail": ("email", "mail", "message", "contact"),
    "files": ("file", "folder", "document", "pdf", "storage"),
    "database": ("sql", "database", "record", "query", "table"),
    "search": ("search", "scrape", "web", "browser", "url"),
    "weather": ("weather", "forecast", "temperature", "climate"),
    "finance": ("stock", "finance", "currency", "price", "trading"),
    "travel": ("travel", "flight", "hotel", "route", "map"),
    "commerce": ("product", "order", "shop", "payment", "giveaway"),
    "media": ("image", "video", "audio", "music", "photo"),
    "developer": ("git", "code", "repository", "issue", "deploy"),
    "social": ("social", "post", "tweet", "profile", "user"),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def server_for(name: str) -> str:
    lowered = name.lower()
    for server, keywords in DOMAIN_RULES.items():
        if any(keyword in lowered for keyword in keywords):
            return server
    shard = int(hashlib.sha256(lowered.encode()).hexdigest()[:4], 16) % 16
    return f"services_{shard:02d}"


def convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    servers: dict[str, list[dict[str, Any]]] = {}
    for wrapper in tools:
        function = wrapper["function"]
        name = function["name"]
        server = server_for(name)
        servers.setdefault(server, []).append(
            {
                "name": name,
                "description": function.get("description", ""),
                "input_schema": function.get("parameters", {"type": "object"}),
            }
        )
    return [
        {"server_id": server, "tools": values}
        for server, values in sorted(servers.items())
    ]


def user_text(row: dict[str, Any]) -> str:
    return "\n".join(
        message.get("content", "")
        for message in row.get("messages", [])
        if message.get("role") == "user" and message.get("content")
    )


def prompt(catalog: list[dict[str, Any]], user: str) -> str:
    return (
        f"{SYSTEM}\n\nMCP_SERVER_CATALOG:\n"
        f"{json.dumps(catalog, ensure_ascii=False, sort_keys=True)}\n\n"
        f"USER:\n{user}\n\nASSISTANT:\n"
    )


def convert_positive(row: dict[str, Any]) -> dict[str, Any]:
    catalog = convert_tools(row["tools"])
    calls = [
        {
            "server_id": server_for(call["name"]),
            "tool_name": call["name"],
            "arguments": call["arguments"],
        }
        for call in row["expected_calls"]
    ]
    verifier = {
        "kind": "mcp_tool_call",
        "decision": "tool",
        "expected_calls": calls,
        "execution_success": True,
        "task_success": True,
        "recovery_success": True,
        "safety_success": True,
    }
    return {
        "id": f"mcp-xlam-{row['id']}",
        "source": "xlam_verified_mcp_conversion",
        "prompt": prompt(catalog, user_text(row)),
        "completion": json.dumps(calls, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        "verifier": verifier,
    }


def no_tool_rows(
    positives: list[dict[str, Any]], count: int, rng: random.Random
) -> list[dict[str, Any]]:
    requests = [
        "Explain latency versus throughput without using tools.",
        "Write a brief thank-you message. No external action is needed.",
        "Give three general code review tips without calling any tool.",
        "Summarize why backups should be tested. Do not use tools.",
        "What is the purpose of unit testing? Answer directly.",
    ]
    rows = []
    for index in range(count):
        source = rng.choice(positives)
        catalog_block = source["prompt"].split("\n\nUSER:\n", 1)[0]
        rows.append(
            {
                "id": f"mcp-no-tool-{index}",
                "source": "mcp_no_tool",
                "prompt": f"{catalog_block}\n\nUSER:\n{requests[index % len(requests)]}\n\nASSISTANT:\n",
                "completion": "[]",
                "verifier": {
                    "kind": "mcp_tool_call",
                    "decision": "no_tool",
                    "expected_calls": [],
                    "execution_success": True,
                    "task_success": True,
                    "recovery_success": True,
                    "safety_success": True,
                },
            }
        )
    return rows


def clarification_rows(
    positives: list[dict[str, Any]], count: int, rng: random.Random
) -> list[dict[str, Any]]:
    rows = []
    for index in range(count):
        source = rng.choice(positives)
        catalog_block = source["prompt"].split("\n\nUSER:\n", 1)[0]
        tool = source["verifier"]["expected_calls"][0]
        request = (
            f"Use {tool['tool_name']} for me, but I have not provided the required "
            "details yet. Ask for the missing information instead of guessing."
        )
        rows.append(
            {
                "id": f"mcp-clarify-{index}",
                "source": "mcp_missing_argument",
                "prompt": f"{catalog_block}\n\nUSER:\n{request}\n\nASSISTANT:\n",
                "completion": "[]",
                "verifier": {
                    "kind": "mcp_tool_call",
                    "decision": "clarify",
                    "expected_calls": [],
                    "execution_success": True,
                    "task_success": True,
                    "recovery_success": True,
                    "safety_success": True,
                },
            }
        )
    return rows


def sample(rows: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    if count <= len(rows):
        return rng.sample(rows, count)
    return [rng.choice(rows) for _ in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlam", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-rows", type=int, default=96000)
    parser.add_argument("--seed", type=int, default=20260650)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    positives = [convert_positive(row) for row in read_jsonl(args.xlam)]
    replay_pool = [
        row
        for row in read_jsonl(args.replay)
        if row.get("mixture_source")
        in {"broad_instruction", "swe_trajectory", "math", "hard_constraint"}
    ]
    targets = {
        "mcp_positive": round(args.total_rows * 0.55),
        "mcp_no_tool": round(args.total_rows * 0.15),
        "mcp_clarify": round(args.total_rows * 0.10),
    }
    targets["general_replay"] = args.total_rows - sum(targets.values())
    pools = {
        "mcp_positive": positives,
        "mcp_no_tool": no_tool_rows(positives, targets["mcp_no_tool"], rng),
        "mcp_clarify": clarification_rows(positives, targets["mcp_clarify"], rng),
        "general_replay": replay_pool,
    }
    mixture = []
    for label, count in targets.items():
        for row in sample(pools[label], count, rng):
            record = dict(row)
            record["mixture_source"] = label
            mixture.append(record)
    rng.shuffle(mixture)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train_sft.jsonl", mixture)
    write_jsonl(
        args.output_dir / "train_rlvr.jsonl",
        [row for row in mixture if row.get("verifier", {}).get("kind") == "mcp_tool_call"],
    )
    manifest = {
        "format_version": 1,
        "seed": args.seed,
        "total_rows": len(mixture),
        "counts": dict(Counter(row["mixture_source"] for row in mixture)),
        "server_strategy": "semantic domains plus 16 stable misc shards",
        "benchmark_exclusion": "source uses xLAM train split and replay only; no BFCL/MCP benchmark test data",
        "limitations": [
            "xLAM calls are schema-verified but not executed against real MCP servers.",
            "No-tool and clarification prompts are synthetic and require later diversification.",
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
