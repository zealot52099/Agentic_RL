"""Versioned, dependency-free schema validation for MCP agent traces."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


TRACE_FORMAT_VERSION = "mcp_agent_trace_v1"
TERMINAL_STATES = {"success", "failure", "clarification", "refusal", "in_progress"}
MESSAGE_ROLES = {"system", "user", "assistant", "tool"}


class TraceValidationError(ValueError):
    """Raised when a trace cannot be used safely for training or evaluation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TraceValidationError(message)


def _validate_json_schema(schema: dict[str, Any], path: str) -> None:
    _require(isinstance(schema, dict), f"{path} must be an object")
    _require(schema.get("type", "object") == "object", f"{path}.type must be object")
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    _require(isinstance(properties, dict), f"{path}.properties must be an object")
    _require(isinstance(required, list), f"{path}.required must be a list")
    _require(set(required) <= set(properties), f"{path}.required contains unknown keys")


def validate_trace(data: dict[str, Any]) -> None:
    _require(isinstance(data, dict), "trace must be an object")
    _require(data.get("format_version") == TRACE_FORMAT_VERSION, "unsupported format_version")
    _require(bool(data.get("task_id")), "task_id is required")

    catalog = data.get("server_catalog")
    _require(isinstance(catalog, list) and catalog, "server_catalog must be non-empty")
    server_ids: set[str] = set()
    tool_names: set[str] = set()
    for index, server in enumerate(catalog):
        path = f"server_catalog[{index}]"
        _require(isinstance(server, dict), f"{path} must be an object")
        server_id = server.get("server_id")
        _require(isinstance(server_id, str) and server_id, f"{path}.server_id is required")
        _require(server_id not in server_ids, f"duplicate server_id: {server_id}")
        server_ids.add(server_id)
        tools = server.get("tools")
        _require(isinstance(tools, list) and tools, f"{path}.tools must be non-empty")
        for tool_index, tool in enumerate(tools):
            tool_path = f"{path}.tools[{tool_index}]"
            _require(isinstance(tool, dict), f"{tool_path} must be an object")
            name = tool.get("name")
            _require(isinstance(name, str) and name, f"{tool_path}.name is required")
            qualified = f"{server_id}.{name}"
            _require(qualified not in tool_names, f"duplicate tool: {qualified}")
            tool_names.add(qualified)
            _validate_json_schema(tool.get("input_schema", {}), f"{tool_path}.input_schema")

    messages = data.get("messages")
    _require(isinstance(messages, list) and messages, "messages must be non-empty")
    call_ids: set[str] = set()
    result_ids: set[str] = set()
    for index, message in enumerate(messages):
        path = f"messages[{index}]"
        _require(isinstance(message, dict), f"{path} must be an object")
        role = message.get("role")
        _require(role in MESSAGE_ROLES, f"{path}.role is invalid")
        _require(isinstance(message.get("content", ""), str), f"{path}.content must be text")
        calls = message.get("tool_calls", [])
        _require(isinstance(calls, list), f"{path}.tool_calls must be a list")
        for call in calls:
            _require(role == "assistant", "only assistant messages may contain tool_calls")
            call_id = call.get("call_id")
            qualified = f"{call.get('server_id')}.{call.get('tool_name')}"
            _require(isinstance(call_id, str) and call_id, "tool call_id is required")
            _require(call_id not in call_ids, f"duplicate call_id: {call_id}")
            _require(qualified in tool_names, f"unknown tool call: {qualified}")
            _require(isinstance(call.get("arguments"), dict), "tool arguments must be an object")
            call_ids.add(call_id)
        if role == "tool":
            call_id = message.get("tool_call_id")
            _require(call_id in call_ids, f"orphan tool result: {call_id}")
            _require(call_id not in result_ids, f"duplicate tool result: {call_id}")
            result_ids.add(call_id)

    permissions = data.get("permissions")
    _require(isinstance(permissions, dict), "permissions must be an object")
    _require(
        data.get("terminal_state") in TERMINAL_STATES,
        f"terminal_state must be one of {sorted(TERMINAL_STATES)}",
    )
    _require(isinstance(data.get("environment_state"), dict), "environment_state must be an object")
    _require(isinstance(data.get("verifier_results"), dict), "verifier_results must be an object")
    _require(isinstance(data.get("reward_components"), dict), "reward_components must be an object")


@dataclass(frozen=True)
class AgentTrace:
    """Validated wrapper used at data-pipeline boundaries."""

    data: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentTrace":
        validate_trace(data)
        return cls(data=data)

    @classmethod
    def from_json(cls, text: str) -> "AgentTrace":
        value = json.loads(text)
        if not isinstance(value, dict):
            raise TraceValidationError("trace JSON must be an object")
        return cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.data)

    def to_json(self) -> str:
        return json.dumps(self.data, ensure_ascii=False, sort_keys=True)
