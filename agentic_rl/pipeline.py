"""Synthetic trace generation, dataset export, and internal MCP evaluation."""

from __future__ import annotations

import copy
import json
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .fingerprint import assign_split, contamination_report, trace_group_ids
from .reward import parse_tool_calls, score_tool_response
from .sandbox import MCPStateSandbox, build_office_sandbox
from .schema import TRACE_FORMAT_VERSION, AgentTrace


SYSTEM_PROMPT = (
    "You are an MCP agent. Select only tools from the supplied server catalog. "
    "Return a JSON array of calls with server_id, tool_name, and arguments. "
    "Return [] when no tool is needed or when required information is missing."
)


TASK_TEMPLATES = [
    {
        "family": "filesystem.read",
        "user": "Read /notes/todo.txt.",
        "calls": [{"server_id": "files", "tool_name": "read_file", "arguments": {"path": "/notes/todo.txt"}}],
        "terminal": "success",
    },
    {
        "family": "filesystem.write",
        "user": "Write 'release approved' to /notes/release.txt.",
        "calls": [{"server_id": "files", "tool_name": "write_file", "arguments": {"path": "/notes/release.txt", "content": "release approved"}}],
        "terminal": "success",
    },
    {
        "family": "calendar.create",
        "user": "Create an event named MCP review starting at 2026-06-16T09:00:00+08:00.",
        "calls": [{"server_id": "calendar", "tool_name": "create_event", "arguments": {"title": "MCP review", "start": "2026-06-16T09:00:00+08:00"}}],
        "terminal": "success",
    },
    {
        "family": "mail.send",
        "user": "Send a status email to dev@example.com with subject MCP and body Tests passed.",
        "calls": [{"server_id": "mail", "tool_name": "send_email", "arguments": {"to": "dev@example.com", "subject": "MCP", "body": "Tests passed."}}],
        "terminal": "success",
    },
    {
        "family": "database.query",
        "user": "Find database records whose project is atlas.",
        "calls": [{"server_id": "database", "tool_name": "query_records", "arguments": {"field": "project", "value": "atlas"}}],
        "terminal": "success",
    },
    {
        "family": "git.issue.update",
        "user": "Set issue 7 to closed.",
        "calls": [{"server_id": "git", "tool_name": "update_issue", "arguments": {"issue_id": 7, "status": "closed"}}],
        "terminal": "success",
    },
    {
        "family": "no_tool.general",
        "user": "Explain the difference between throughput and latency. Do not use tools.",
        "calls": [],
        "terminal": "success",
        "decision": "no_tool",
    },
    {
        "family": "clarification.calendar",
        "user": "Schedule a review meeting tomorrow.",
        "calls": [],
        "terminal": "clarification",
        "decision": "clarify",
    },
]


def canonical_calls(calls: list[dict[str, Any]]) -> str:
    return json.dumps(calls, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def render_prompt(trace: dict[str, Any]) -> str:
    user_messages = [
        message["content"] for message in trace["messages"] if message["role"] == "user"
    ]
    return (
        f"{SYSTEM_PROMPT}\n\nMCP_SERVER_CATALOG:\n"
        f"{json.dumps(trace['server_catalog'], ensure_ascii=False, sort_keys=True)}\n\n"
        f"USER:\n{user_messages[-1]}\n\nASSISTANT:\n"
    )


def _execute_calls(
    env: MCPStateSandbox, calls: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], bool]:
    results = []
    success = True
    for index, call in enumerate(calls):
        result = env.execute(call["server_id"], call["tool_name"], call["arguments"])
        results.append({"call_id": f"call-{index}", **result})
        success = success and bool(result["ok"])
    return results, success


def synthesize_traces(count: int, seed: int = 20260615) -> list[dict[str, Any]]:
    """Create deterministic smoke traces. Production generation replaces templates."""
    rng = random.Random(seed)
    rows = []
    for index in range(count):
        template = copy.deepcopy(TASK_TEMPLATES[index % len(TASK_TEMPLATES)])
        env = build_office_sandbox()
        calls = template["calls"]
        results, execution_success = _execute_calls(env, calls)
        tool_calls = [
            {"call_id": f"call-{position}", **call}
            for position, call in enumerate(calls)
        ]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": template["user"]},
            {
                "role": "assistant",
                "content": canonical_calls(calls),
                "tool_calls": tool_calls,
            },
        ]
        for result in results:
            messages.append(
                {
                    "role": "tool",
                    "content": json.dumps(result, ensure_ascii=False, sort_keys=True),
                    "tool_call_id": result["call_id"],
                }
            )
        verifier = {
            "kind": "mcp_tool_call",
            "decision": template.get("decision", "tool"),
            "expected_calls": calls,
            "execution_success": execution_success,
            "task_success": execution_success,
            "recovery_success": True,
            "safety_success": True,
        }
        score = score_tool_response(canonical_calls(calls), verifier)
        trace = {
            "format_version": TRACE_FORMAT_VERSION,
            "task_id": f"mcp-smoke-{seed}-{index:06d}",
            "split_group_ids": [template["family"]],
            "server_catalog": env.catalog(),
            "tool_schemas": [
                {"server_id": server["server_id"], **tool}
                for server in env.catalog()
                for tool in server["tools"]
            ],
            "messages": messages,
            "assistant_thought": None,
            "tool_calls": tool_calls,
            "tool_results": results,
            "environment_state": copy.deepcopy(env.state),
            "permissions": {
                "granted": sorted(env.permissions),
                "high_risk_requires_confirmation": True,
            },
            "terminal_state": template["terminal"],
            "verifier_results": verifier,
            "reward_components": score,
            "generation_metadata": {
                "generator": "deterministic_template",
                "seed": rng.randrange(2**31),
                "benchmark_contamination_checked": False,
            },
        }
        AgentTrace.from_dict(trace)
        rows.append(trace)
    return rows


def split_traces(
    traces: list[dict[str, Any]], holdout_percent: int = 20
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        key = "|".join(sorted(trace_group_ids(trace)))
        grouped[key].append(trace)
    train, evaluation = [], []
    for key, rows in sorted(grouped.items()):
        target = evaluation if assign_split({key}, holdout_percent) == "ood_eval" else train
        target.extend(rows)
    return train, evaluation


def export_sft(traces: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for trace in traces:
        verifier = trace["verifier_results"]
        rows.append(
            {
                "id": trace["task_id"],
                "source": "mcp_trace_v1",
                "prompt": render_prompt(trace),
                "completion": canonical_calls(verifier["expected_calls"]),
                "verifier": verifier,
                "split_group_ids": sorted(trace_group_ids(trace)),
            }
        )
    return rows


def _wrong_call(trace: dict[str, Any]) -> list[dict[str, Any]]:
    expected = trace["verifier_results"]["expected_calls"]
    if not expected:
        first_server = trace["server_catalog"][0]
        first_tool = first_server["tools"][0]
        return [{"server_id": first_server["server_id"], "tool_name": first_tool["name"], "arguments": {}}]
    wrong = copy.deepcopy(expected)
    available = [
        (server["server_id"], tool["name"])
        for server in trace["server_catalog"]
        for tool in server["tools"]
        if (server["server_id"], tool["name"])
        != (wrong[0]["server_id"], wrong[0]["tool_name"])
    ]
    wrong[0]["server_id"], wrong[0]["tool_name"] = available[0]
    return wrong


def export_preferences(traces: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": trace["task_id"],
            "prompt": render_prompt(trace),
            "chosen": canonical_calls(trace["verifier_results"]["expected_calls"]),
            "rejected": canonical_calls(_wrong_call(trace)),
            "preference_type": (
                "no_tool_vs_unnecessary_call"
                if not trace["verifier_results"]["expected_calls"]
                else "correct_route_vs_distractor"
            ),
        }
        for trace in traces
    ]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def bootstrap_interval(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = statistics.mean(values)
    if len(values) == 1:
        return mean, mean
    error = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return max(0.0, mean - error), min(1.0, mean + error)


def evaluate_predictions(
    traces: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> dict[str, Any]:
    by_id = {row["task_id"]: row for row in traces}
    per_task = []
    latencies = []
    token_counts = []
    confusion = Counter()
    for prediction in predictions:
        task_id = prediction["task_id"]
        if task_id not in by_id:
            continue
        trace = by_id[task_id]
        score = score_tool_response(prediction["response"], trace["verifier_results"])
        expected = trace["verifier_results"]["expected_calls"]
        actual, _ = parse_tool_calls(prediction["response"])
        expected_tool = bool(expected)
        actual_tool = bool(actual)
        confusion[(expected_tool, actual_tool)] += 1
        per_task.append({"task_id": task_id, **score})
        if "latency_ms" in prediction:
            latencies.append(float(prediction["latency_ms"]))
        if "output_tokens" in prediction:
            token_counts.append(float(prediction["output_tokens"]))

    def mean_component(name: str) -> float:
        values = [row["components"][name] for row in per_task]
        return statistics.mean(values) if values else 0.0

    successes = [float(row["success"]) for row in per_task]
    tp = confusion[(True, True)]
    tn = confusion[(False, False)]
    fp = confusion[(False, True)]
    fn = confusion[(True, False)]
    no_tool_precision = tn / max(1, tn + fn)
    no_tool_recall = tn / max(1, tn + fp)
    result = {
        "format_version": "mcp_internal_eval_v1",
        "evaluation_type": "internal_probe_not_official_benchmark",
        "counts": {"traces": len(traces), "predictions": len(per_task)},
        "metrics": {
            "json_schema_valid_rate": statistics.mean([float(row["format_valid"]) for row in per_task]) if per_task else 0.0,
            "server_tool_exact_accuracy": mean_component("route"),
            "argument_exact_accuracy": mean_component("arguments"),
            "execution_success_rate": mean_component("execution"),
            "task_success_rate": statistics.mean(successes) if successes else 0.0,
            "task_success_ci95": bootstrap_interval(successes),
            "no_tool_precision": no_tool_precision,
            "no_tool_recall": no_tool_recall,
            "no_tool_f1": 2 * no_tool_precision * no_tool_recall / max(1e-12, no_tool_precision + no_tool_recall),
            "tool_hallucination_rate": fp / max(1, fp + tn),
            "safety_success_rate": mean_component("safety"),
            "mean_calls": statistics.mean([len(row["parsed_calls"]) for row in per_task]) if per_task else 0.0,
            "mean_output_tokens": statistics.mean(token_counts) if token_counts else None,
            "latency_p50_ms": statistics.median(latencies) if latencies else None,
            "latency_p95_ms": sorted(latencies)[min(len(latencies) - 1, int(0.95 * len(latencies)))] if latencies else None,
        },
        "per_task": per_task,
        "generated_at_unix": time.time(),
    }
    return result


def package_traces(
    traces: list[dict[str, Any]],
    output_dir: Path,
    seed: int,
    holdout_percent: int,
    source: str,
) -> dict[str, Any]:
    for trace in traces:
        AgentTrace.from_dict(trace)
    train, evaluation = split_traces(traces, holdout_percent)
    report = contamination_report(train, evaluation)
    if not report["clean"]:
        raise RuntimeError(f"contamination detected: {report}")
    counts = {
        "traces": write_jsonl(output_dir / "traces" / "all.jsonl", traces),
        "train_traces": write_jsonl(output_dir / "traces" / "train.jsonl", train),
        "ood_eval_traces": write_jsonl(output_dir / "traces" / "ood_eval.jsonl", evaluation),
        "sft": write_jsonl(output_dir / "train_sft.jsonl", export_sft(train)),
        "preferences": write_jsonl(output_dir / "train_preferences.jsonl", export_preferences(train)),
        "rlvr": write_jsonl(output_dir / "train_rlvr.jsonl", export_sft(train)),
    }
    manifest = {
        "format_version": 1,
        "seed": seed,
        "source": source,
        "holdout_percent": holdout_percent,
        "counts": counts,
        "contamination": report,
        "limitations": [
            "Template-generated smoke corpus; replace generator with teacher and real MCP adapters for production.",
            "Internal OOD split is not an official public benchmark.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def build_dataset(output_dir: Path, count: int, seed: int, holdout_percent: int) -> dict[str, Any]:
    return package_traces(
        synthesize_traces(count, seed),
        output_dir,
        seed,
        holdout_percent,
        source="deterministic_smoke_templates",
    )
