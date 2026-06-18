"""Deterministic, gated rewards for tool routing and MCP task completion."""

from __future__ import annotations

import json
from typing import Any


DEFAULT_REWARD_WEIGHTS = {
    "route": 0.10,
    "arguments": 0.15,
    "execution": 0.20,
    "task": 0.35,
    "recovery": 0.10,
    "safety": 0.10,
}


def parse_tool_calls(text: str) -> tuple[list[dict[str, Any]], bool]:
    try:
        value = json.loads(text.strip())
    except (json.JSONDecodeError, TypeError):
        return [], False
    if isinstance(value, dict) and "tool_calls" in value:
        value = value["tool_calls"]
    if not isinstance(value, list):
        return [], False
    calls = []
    for item in value:
        if not isinstance(item, dict):
            return [], False
        server_id = item.get("server_id")
        tool_name = item.get("tool_name", item.get("name"))
        arguments = item.get("arguments")
        if not isinstance(server_id, str) or not isinstance(tool_name, str):
            return [], False
        if not isinstance(arguments, dict):
            return [], False
        calls.append(
            {"server_id": server_id, "tool_name": tool_name, "arguments": arguments}
        )
    return calls, True


def _canonical_call(call: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(call.get("server_id", "")),
        str(call.get("tool_name", call.get("name", ""))),
        json.dumps(call.get("arguments", {}), sort_keys=True, separators=(",", ":")),
    )


def score_tool_response(
    text: str,
    verifier: dict[str, Any],
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Score a proposed next action.

    Format is a gate: malformed output receives no positive component reward.
    Environment-derived execution/task fields are supplied by online rollouts;
    static datasets may set them to their verified expected values.
    """
    weights = weights or DEFAULT_REWARD_WEIGHTS
    calls, format_valid = parse_tool_calls(text)
    expected = verifier.get("expected_calls", [])
    no_tool_expected = verifier.get("decision") in {"no_tool", "clarify", "refuse"}
    expected_routes = {
        (str(call["server_id"]), str(call.get("tool_name", call.get("name"))))
        for call in expected
    }
    actual_routes = {(call["server_id"], call["tool_name"]) for call in calls}
    route_ok = actual_routes == expected_routes
    arguments_ok = sorted(map(_canonical_call, calls)) == sorted(
        map(_canonical_call, expected)
    )
    if no_tool_expected:
        route_ok = not calls
        arguments_ok = not calls

    components = {
        "route": float(route_ok),
        "arguments": float(arguments_ok),
        "execution": float(verifier.get("execution_success", arguments_ok)),
        "task": float(verifier.get("task_success", arguments_ok)),
        "recovery": float(verifier.get("recovery_success", True)),
        "safety": float(verifier.get("safety_success", True)),
    }
    if not format_valid:
        components = {key: 0.0 for key in components}

    redundant = max(0, len(calls) - len(expected))
    repeated = len(calls) - len({_canonical_call(call) for call in calls})
    penalties = {
        "redundant_calls": min(0.20, redundant * 0.05),
        "repeated_calls": min(0.20, repeated * 0.05),
        "irreversible_error": 0.25 if verifier.get("irreversible_error") else 0.0,
    }
    total = sum(weights[key] * components[key] for key in weights)
    total -= sum(penalties.values())
    return {
        "format_valid": format_valid,
        "components": components,
        "penalties": penalties,
        "reward": max(-1.0, min(1.0, total)),
        "success": bool(format_valid and components["task"] == 1.0 and components["safety"] == 1.0),
        "parsed_calls": calls,
    }
