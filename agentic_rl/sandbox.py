"""Small deterministic MCP-like state environment for data and verifier tests."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable


class ToolExecutionError(RuntimeError):
    code = "tool_error"


class PermissionDenied(ToolExecutionError):
    code = "permission_denied"


class SimulatedTimeout(ToolExecutionError):
    code = "timeout"


Handler = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


@dataclass
class ToolSpec:
    server_id: str
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Handler
    permission: str = "read"
    family_id: str = ""
    irreversible: bool = False

    def public_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "permission": self.permission,
            "family_id": self.family_id or f"{self.server_id}.{self.name}",
            "irreversible": self.irreversible,
        }


@dataclass
class MCPStateSandbox:
    state: dict[str, Any]
    permissions: set[str] = field(default_factory=lambda: {"read"})
    tools: dict[tuple[str, str], ToolSpec] = field(default_factory=dict)
    failure_plan: dict[str, list[str]] = field(default_factory=dict)
    call_log: list[dict[str, Any]] = field(default_factory=list)

    def register(self, spec: ToolSpec) -> None:
        key = (spec.server_id, spec.name)
        if key in self.tools:
            raise ValueError(f"duplicate tool {key}")
        self.tools[key] = spec

    def catalog(self) -> list[dict[str, Any]]:
        servers: dict[str, list[dict[str, Any]]] = {}
        for (server_id, _), spec in sorted(self.tools.items()):
            servers.setdefault(server_id, []).append(spec.public_dict())
        return [
            {"server_id": server_id, "tools": tools}
            for server_id, tools in sorted(servers.items())
        ]

    def execute(
        self, server_id: str, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        key = (server_id, tool_name)
        if key not in self.tools:
            return {"ok": False, "error": {"code": "unknown_tool"}}
        spec = self.tools[key]
        missing = [
            name
            for name in spec.input_schema.get("required", [])
            if name not in arguments
        ]
        if missing:
            return {
                "ok": False,
                "error": {"code": "invalid_arguments", "missing": missing},
            }
        if spec.permission not in self.permissions:
            return {"ok": False, "error": {"code": PermissionDenied.code}}

        qualified = f"{server_id}.{tool_name}"
        planned = self.failure_plan.get(qualified, [])
        if planned:
            code = planned.pop(0)
            result = {"ok": False, "error": {"code": code}}
        else:
            before = copy.deepcopy(self.state)
            try:
                result = {"ok": True, "result": spec.handler(self.state, arguments)}
            except ToolExecutionError as error:
                self.state = before
                result = {"ok": False, "error": {"code": error.code, "message": str(error)}}
            except Exception:
                self.state = before
                raise
        self.call_log.append(
            {
                "server_id": server_id,
                "tool_name": tool_name,
                "arguments": copy.deepcopy(arguments),
                "result": copy.deepcopy(result),
            }
        )
        return result


def object_schema(
    properties: dict[str, dict[str, Any]], required: list[str]
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def build_office_sandbox() -> MCPStateSandbox:
    state = {
        "files": {"/notes/todo.txt": "review release checklist"},
        "events": [],
        "emails": [],
        "issues": [{"id": 7, "title": "Parser regression", "status": "open"}],
        "records": [{"project": "atlas", "owner": "Lin", "status": "active"}],
    }
    env = MCPStateSandbox(state=state, permissions={"read", "write", "send"})

    def read_file(state: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
        path = args["path"]
        return {"path": path, "content": state["files"].get(path), "found": path in state["files"]}

    def write_file(state: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
        state["files"][args["path"]] = args["content"]
        return {"written": True, "path": args["path"]}

    def create_event(state: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
        event = {"title": args["title"], "start": args["start"]}
        state["events"].append(event)
        return event

    def send_email(state: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
        message = {"to": args["to"], "subject": args["subject"], "body": args["body"]}
        state["emails"].append(message)
        return {"message_id": f"msg-{len(state['emails'])}"}

    def query_records(state: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
        rows = [row for row in state["records"] if row.get(args["field"]) == args["value"]]
        return {"rows": rows}

    def update_issue(state: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
        for issue in state["issues"]:
            if issue["id"] == args["issue_id"]:
                issue["status"] = args["status"]
                return issue
        return {"found": False}

    specs = [
        ToolSpec("files", "read_file", "Read a text file.", object_schema({"path": {"type": "string"}}, ["path"]), read_file, family_id="filesystem.read"),
        ToolSpec("files", "write_file", "Write a text file.", object_schema({"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]), write_file, "write", "filesystem.write"),
        ToolSpec("calendar", "create_event", "Create a calendar event.", object_schema({"title": {"type": "string"}, "start": {"type": "string"}}, ["title", "start"]), create_event, "write", "calendar.create"),
        ToolSpec("mail", "send_email", "Send an email.", object_schema({"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}}, ["to", "subject", "body"]), send_email, "send", "mail.send", True),
        ToolSpec("database", "query_records", "Query records by one field.", object_schema({"field": {"type": "string"}, "value": {"type": "string"}}, ["field", "value"]), query_records, family_id="database.query"),
        ToolSpec("git", "update_issue", "Update an issue status.", object_schema({"issue_id": {"type": "integer"}, "status": {"type": "string"}}, ["issue_id", "status"]), update_issue, "write", "git.issue.update"),
    ]
    for spec in specs:
        env.register(spec)
    return env
