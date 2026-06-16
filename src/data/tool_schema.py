"""
Tool Schema Definitions
Defines standard tool schemas for function calling in OpenAI-compatible format.
Supports: search, calculator, database, api_call, file_ops, code_exec, email, calendar
"""
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ToolParameter:
    name: str
    type: str
    description: str
    required: bool = True
    enum: Optional[List[str]] = None


@dataclass
class ToolSchema:
    name: str
    description: str
    category: str
    parameters: List[ToolParameter]

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function-calling format"""
        properties = {}
        required = []
        for p in self.parameters:
            prop = {"type": p.type, "description": p.description}
            if p.enum:
                prop["enum"] = p.enum
            properties[p.name] = prop
            if p.required:
                required.append(p.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


# ============================================================
# Standard Tool Library
# ============================================================

TOOL_LIBRARY: Dict[str, ToolSchema] = {
    # ---- Search ----
    "web_search": ToolSchema(
        name="web_search",
        description="Search the web for information on a given query",
        category="search",
        parameters=[
            ToolParameter("query", "string", "The search query"),
            ToolParameter("num_results", "integer", "Number of results to return", required=False),
        ],
    ),
    "knowledge_lookup": ToolSchema(
        name="knowledge_lookup",
        description="Look up factual information from the knowledge base",
        category="search",
        parameters=[
            ToolParameter("topic", "string", "The topic to look up"),
            ToolParameter("filters", "object", "Optional filters for the search", required=False),
        ],
    ),

    # ---- Calculator ----
    "calculate": ToolSchema(
        name="calculate",
        description="Perform a mathematical calculation",
        category="calculator",
        parameters=[
            ToolParameter("expression", "string", "The mathematical expression to evaluate"),
        ],
    ),
    "convert_units": ToolSchema(
        name="convert_units",
        description="Convert between different units of measurement",
        category="calculator",
        parameters=[
            ToolParameter("value", "number", "The value to convert"),
            ToolParameter("from_unit", "string", "Source unit"),
            ToolParameter("to_unit", "string", "Target unit"),
        ],
    ),

    # ---- Database ----
    "db_query": ToolSchema(
        name="db_query",
        description="Execute a read-only SQL query against the database",
        category="database",
        parameters=[
            ToolParameter("query", "string", "SQL SELECT query to execute"),
            ToolParameter("limit", "integer", "Maximum rows to return", required=False),
        ],
    ),
    "db_insert": ToolSchema(
        name="db_insert",
        description="Insert a new record into the database",
        category="database",
        parameters=[
            ToolParameter("table", "string", "Target table name"),
            ToolParameter("data", "object", "Record data to insert"),
        ],
    ),
    "db_update": ToolSchema(
        name="db_update",
        description="Update existing records in the database",
        category="database",
        parameters=[
            ToolParameter("table", "string", "Target table name"),
            ToolParameter("data", "object", "Fields to update"),
            ToolParameter("condition", "string", "WHERE condition to select rows"),
        ],
    ),

    # ---- API Call ----
    "api_get": ToolSchema(
        name="api_get",
        description="Make an HTTP GET request to an external API",
        category="api_call",
        parameters=[
            ToolParameter("url", "string", "The API endpoint URL"),
            ToolParameter("headers", "object", "HTTP headers", required=False),
            ToolParameter("params", "object", "Query parameters", required=False),
        ],
    ),
    "api_post": ToolSchema(
        name="api_post",
        description="Make an HTTP POST request to an external API",
        category="api_call",
        parameters=[
            ToolParameter("url", "string", "The API endpoint URL"),
            ToolParameter("body", "object", "Request body"),
            ToolParameter("headers", "object", "HTTP headers", required=False),
        ],
    ),

    # ---- File Operations ----
    "read_file": ToolSchema(
        name="read_file",
        description="Read the contents of a file",
        category="file_ops",
        parameters=[
            ToolParameter("path", "string", "Path to the file"),
            ToolParameter("encoding", "string", "File encoding", required=False, enum=["utf-8", "gbk", "ascii"]),
        ],
    ),
    "write_file": ToolSchema(
        name="write_file",
        description="Write content to a file",
        category="file_ops",
        parameters=[
            ToolParameter("path", "string", "Path to the file"),
            ToolParameter("content", "string", "Content to write"),
            ToolParameter("mode", "string", "Write mode", required=False, enum=["w", "a"]),
        ],
    ),
    "list_directory": ToolSchema(
        name="list_directory",
        description="List files and directories in a given path",
        category="file_ops",
        parameters=[
            ToolParameter("path", "string", "Directory path to list"),
            ToolParameter("pattern", "string", "Optional glob pattern filter", required=False),
        ],
    ),

    # ---- Code Execution ----
    "execute_python": ToolSchema(
        name="execute_python",
        description="Execute Python code in a sandboxed environment",
        category="code_exec",
        parameters=[
            ToolParameter("code", "string", "Python code to execute"),
            ToolParameter("timeout", "integer", "Timeout in seconds", required=False),
        ],
    ),
    "execute_shell": ToolSchema(
        name="execute_shell",
        description="Execute a shell command",
        category="code_exec",
        parameters=[
            ToolParameter("command", "string", "Shell command to execute"),
            ToolParameter("working_dir", "string", "Working directory", required=False),
        ],
    ),

    # ---- Email ----
    "send_email": ToolSchema(
        name="send_email",
        description="Send an email",
        category="email",
        parameters=[
            ToolParameter("to", "string", "Recipient email address"),
            ToolParameter("subject", "string", "Email subject"),
            ToolParameter("body", "string", "Email body content"),
            ToolParameter("cc", "string", "CC recipients", required=False),
        ],
    ),

    # ---- Calendar ----
    "create_event": ToolSchema(
        name="create_event",
        description="Create a calendar event",
        category="calendar",
        parameters=[
            ToolParameter("title", "string", "Event title"),
            ToolParameter("start_time", "string", "Start time (ISO 8601 format)"),
            ToolParameter("end_time", "string", "End time (ISO 8601 format)"),
            ToolParameter("attendees", "array", "List of attendee emails", required=False),
        ],
    ),
    "query_calendar": ToolSchema(
        name="query_calendar",
        description="Query calendar events in a time range",
        category="calendar",
        parameters=[
            ToolParameter("start_date", "string", "Start date (YYYY-MM-DD)"),
            ToolParameter("end_date", "string", "End date (YYYY-MM-DD)"),
        ],
    ),
}


def get_tools_by_category(category: str) -> List[ToolSchema]:
    """Get all tools in a given category."""
    return [t for t in TOOL_LIBRARY.values() if t.category == category]


def get_all_openai_schemas() -> List[Dict[str, Any]]:
    """Get all tools in OpenAI-compatible format."""
    return [t.to_openai_format() for t in TOOL_LIBRARY.values()]


def validate_tool_call(tool_name: str, arguments: Dict[str, Any]) -> tuple[bool, str]:
    """
    Validate a tool call against the schema.
    Returns (is_valid, error_message).
    """
    tool = TOOL_LIBRARY.get(tool_name)
    if tool is None:
        return False, f"Unknown tool: {tool_name}"

    for param in tool.parameters:
        if param.required and param.name not in arguments:
            return False, f"Missing required parameter: {param.name}"

        if param.name in arguments:
            value = arguments[param.name]
            # Type check
            type_map = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}
            expected = type_map.get(param.type)
            if expected and not isinstance(value, expected):
                return False, f"Parameter '{param.name}' should be {param.type}, got {type(value).__name__}"

            # Enum check
            if param.enum and value not in param.enum:
                return False, f"Parameter '{param.name}' should be one of {param.enum}, got '{value}'"

    return True, "OK"


def get_tool_description_prompt() -> str:
    """Generate a system prompt describing all available tools."""
    lines = ["You have access to the following tools:\n"]
    for tool in TOOL_LIBRARY.values():
        params_desc = ", ".join(
            f"{p.name}: {p.type}" + (" (required)" if p.required else " (optional)")
            for p in tool.parameters
        )
        lines.append(f"- {tool.name}: {tool.description}")
        lines.append(f"  Parameters: {params_desc}")
    lines.append("\nTo call a tool, respond with a JSON object:")
    lines.append('{"tool_calls": [{"name": "<tool_name>", "arguments": {...}}]}')
    return "\n".join(lines)
