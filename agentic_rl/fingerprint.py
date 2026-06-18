"""Canonical fingerprints and leakage-resistant split assignment."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


DESCRIPTION_KEYS = {"description", "title", "examples", "default"}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _shape(value: Any) -> Any:
    if isinstance(value, dict):
        output = {}
        for key, item in sorted(value.items()):
            if key in DESCRIPTION_KEYS:
                continue
            output[key] = _shape(item)
        return output
    if isinstance(value, list):
        return [_shape(item) for item in value]
    return value


def schema_fingerprint(schema: dict[str, Any]) -> str:
    """Fingerprint schema structure while ignoring prose and defaults."""
    payload = canonical_json(_shape(schema)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def tool_family_id(server: dict[str, Any], tool: dict[str, Any]) -> str:
    explicit = tool.get("family_id") or server.get("family_id")
    if explicit:
        return normalized_name(str(explicit))
    operation = tool.get("operation") or tool.get("name", "")
    return f"{normalized_name(str(operation))}:{schema_fingerprint(tool['input_schema'])[:16]}"


def trace_group_ids(trace: dict[str, Any]) -> set[str]:
    explicit = trace.get("split_group_ids")
    if isinstance(explicit, list) and explicit:
        return {normalized_name(str(item)) for item in explicit}
    called: set[str] = set()
    lookup = {}
    for server in trace["server_catalog"]:
        for tool in server["tools"]:
            lookup[(server["server_id"], tool["name"])] = tool_family_id(server, tool)
    for message in trace.get("messages", []):
        for call in message.get("tool_calls", []):
            family = lookup.get((call.get("server_id"), call.get("tool_name")))
            if family:
                called.add(family)
    if called:
        return called
    groups: set[str] = set()
    for server in trace["server_catalog"]:
        for tool in server["tools"]:
            groups.add(tool_family_id(server, tool))
    return groups


def assign_split(group_ids: set[str], holdout_percent: int = 20) -> str:
    """Assign the complete connected tool group to train or OOD evaluation."""
    if not 1 <= holdout_percent <= 50:
        raise ValueError("holdout_percent must be between 1 and 50")
    key = "|".join(sorted(group_ids))
    bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "ood_eval" if bucket < holdout_percent else "train"


def contamination_report(
    train: list[dict[str, Any]], evaluation: list[dict[str, Any]]
) -> dict[str, Any]:
    train_groups = set().union(*(trace_group_ids(row) for row in train)) if train else set()
    eval_groups = set().union(*(trace_group_ids(row) for row in evaluation)) if evaluation else set()
    train_tasks = {row["task_id"] for row in train}
    eval_tasks = {row["task_id"] for row in evaluation}
    return {
        "train_trace_count": len(train),
        "eval_trace_count": len(evaluation),
        "overlapping_group_ids": sorted(train_groups & eval_groups),
        "overlapping_task_ids": sorted(train_tasks & eval_tasks),
        "clean": not (train_groups & eval_groups or train_tasks & eval_tasks),
    }
