"""
Synthetic Training Data Generator for Tool-Calling Scenarios

Generates diverse tool-calling trajectories covering:
  - Single-tool simple queries
  - Multi-tool orchestrated workflows
  - Complex task decomposition with error recovery
"""
import json
import random
import uuid
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

from .tool_schema import TOOL_LIBRARY, ToolSchema, validate_tool_call


# ============================================================
# Scenario Templates (Task Orchestration Patterns)
# ============================================================

SCENARIO_TEMPLATES = {
    "easy": [
        {
            "pattern": "single_lookup",
            "description": "User asks a factual question requiring one tool call",
            "tool_categories": ["search"],
            "num_tool_calls": 1,
            "user_prompts": [
                "What's the weather like in {city} today?",
                "Find information about {topic}",
                "Search for the latest news on {query}",
                "What is the capital of {country}?",
            ],
        },
        {
            "pattern": "simple_calc",
            "description": "User needs a simple calculation",
            "tool_categories": ["calculator"],
            "num_tool_calls": 1,
            "user_prompts": [
                "Calculate {expression}",
                "What is {a} times {b}?",
                "Convert {value} {from_unit} to {to_unit}",
            ],
        },
        {
            "pattern": "file_read",
            "description": "User asks to read a file",
            "tool_categories": ["file_ops"],
            "num_tool_calls": 1,
            "user_prompts": [
                "Read the contents of {path}",
                "Show me what's in {path}",
            ],
        },
    ],
    "medium": [
        {
            "pattern": "search_then_summarize",
            "description": "Search for information then process results",
            "tool_categories": ["search", "file_ops"],
            "num_tool_calls": [2, 3],
            "user_prompts": [
                "Research {topic} and save the findings to {path}",
                "Find the top 5 {items}, calculate the average {metric}, and save the report",
            ],
        },
        {
            "pattern": "db_query_then_report",
            "description": "Query database then format and send report",
            "tool_categories": ["database", "email"],
            "num_tool_calls": [2, 4],
            "user_prompts": [
                "Query the {table} for {condition}, generate a summary, and email it to {email}",
                "Get all records from {table} where {condition}, calculate statistics, send to {email}",
            ],
        },
        {
            "pattern": "api_chain",
            "description": "Chain multiple API calls together",
            "tool_categories": ["api_call", "file_ops"],
            "num_tool_calls": [2, 3],
            "user_prompts": [
                "Fetch data from {url}, transform it, and save to {path}",
                "Call {api1} to get user info, then use that to query {api2}",
            ],
        },
        {
            "pattern": "code_exec_workflow",
            "description": "Execute code to process data",
            "tool_categories": ["code_exec", "file_ops"],
            "num_tool_calls": [2, 3],
            "user_prompts": [
                "Read the data from {path}, run analysis with Python, and save results",
                "Execute {script} on the input file and report the output",
            ],
        },
    ],
    "hard": [
        {
            "pattern": "full_orchestration",
            "description": "Complex multi-step task requiring multiple tool categories",
            "tool_categories": ["search", "database", "calculator", "email", "file_ops"],
            "num_tool_calls": [4, 6],
            "user_prompts": [
                "Analyze {topic}: search for latest data, query relevant DB records, run statistical analysis, generate a comprehensive report, and email it to the team",
                "Plan and execute a {project_type} project: research requirements, estimate costs, create a schedule, and notify stakeholders",
            ],
        },
        {
            "pattern": "error_recovery",
            "description": "Task requiring error handling and retry logic",
            "tool_categories": ["search", "api_call", "file_ops"],
            "num_tool_calls": [3, 5],
            "user_prompts": [
                "Try to fetch data from {primary_source}. If that fails, try {fallback_source}. Save whatever data you get to {path}.",
                "Query {api_endpoint}. If the response is empty, search the web instead. If still no results, log the failure to {log_path}.",
            ],
        },
        {
            "pattern": "multi_dependency",
            "description": "Tasks where later steps depend on earlier outputs",
            "tool_categories": ["database", "calculator", "code_exec", "email"],
            "num_tool_calls": [4, 6],
            "user_prompts": [
                "Get sales data from DB for {period}, calculate QoQ growth, run a predictive model, generate a forecast chart, and email the executive summary",
                "For each customer in {segment}, retrieve their purchase history, calculate lifetime value, segment them into tiers, and update the CRM",
            ],
        },
    ],
}


# ============================================================
# Data Generation Engine
# ============================================================

@dataclass
class ToolCallRecord:
    """A single tool call in a trajectory."""
    turn: int
    tool_name: str
    arguments: Dict[str, Any]
    result: str
    is_valid: bool


@dataclass
class ToolCallSample:
    """A complete training sample."""
    id: str
    difficulty: str
    pattern: str
    system_prompt: str
    user_prompt: str
    tool_calls: List[Dict[str, Any]]        # The tool call trajectory
    final_response: str                      # Final assistant response
    num_turns: int
    tools_used: List[str]


class SyntheticDataGenerator:
    """Generates synthetic tool-calling training data."""

    FILLER_VALUES = {
        "city": ["Beijing", "Shanghai", "Tokyo", "London", "New York", "Paris", "Berlin", "Sydney"],
        "country": ["China", "Japan", "UK", "USA", "France", "Germany", "Australia"],
        "topic": ["machine learning", "renewable energy", "blockchain", "quantum computing", "climate change"],
        "query": ["AI breakthroughs 2026", "best programming languages", "stock market trends"],
        "expression": ["2 + 2 * 3", "(100 - 25) / 5", "sqrt(144) + 3^2"],
        "a": list(map(str, range(10, 100))),
        "value": list(map(str, range(1, 1000))),
        "from_unit": ["km", "miles", "kg", "pounds", "celsius", "fahrenheit"],
        "to_unit": ["miles", "km", "pounds", "kg", "fahrenheit", "celsius"],
        "path": ["/data/report.txt", "/home/user/config.json", "/tmp/output.csv", "/var/log/app.log"],
        "table": ["users", "orders", "products", "inventory", "sales"],
        "condition": ["status='active'", "created_at > '2026-01-01'", "amount > 100", "category='premium'"],
        "email": ["manager@company.com", "team@org.com", "admin@example.com"],
        "url": ["https://api.example.com/v1/users", "https://data.source.com/records"],
        "items": ["smartphones", "laptops", "cars", "restaurants", "stocks"],
        "metric": ["price", "rating", "sales_volume", "growth_rate"],
        "api1": ["user-service", "auth-api", "product-catalog"],
        "api2": ["recommendation-engine", "analytics-api", "payment-gateway"],
        "script": ["analyze.py", "process_data.py", "generate_report.py"],
        "project_type": ["marketing campaign", "product launch", "system migration"],
        "primary_source": ["primary-db", "main-api", "core-service"],
        "fallback_source": ["backup-db", "cache-layer", "cdn"],
        "api_endpoint": ["/v2/analytics", "/v1/reports", "/latest/metrics"],
        "log_path": ["/var/log/errors.log", "/tmp/failures.txt"],
        "period": ["Q1 2026", "last month", "fiscal year 2025"],
        "segment": ["premium", "enterprise", "SMB"],
        "items_single": ["smartphone", "laptop", "car", "restaurant", "stock"],
    }

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def _fill_template(self, template: str) -> str:
        """Replace placeholders like {city} with random values."""
        result = template
        for key, values in self.FILLER_VALUES.items():
            placeholder = "{" + key + "}"
            while placeholder in result:
                result = result.replace(placeholder, random.choice(values), 1)
        return result

    def _generate_tool_call(self, tool: ToolSchema) -> Dict[str, Any]:
        """Generate a plausible tool call with arguments."""
        args = {}
        for param in tool.parameters:
            if not param.required and random.random() < 0.3:
                continue
            if param.type == "string":
                if param.enum:
                    args[param.name] = random.choice(param.enum)
                else:
                    args[param.name] = f"example_{param.name}_{random.randint(1, 100)}"
            elif param.type == "integer":
                args[param.name] = random.randint(1, 100)
            elif param.type == "number":
                args[param.name] = round(random.uniform(1.0, 100.0), 2)
            elif param.type == "boolean":
                args[param.name] = random.choice([True, False])
            elif param.type == "array":
                args[param.name] = [f"item_{i}" for i in range(random.randint(1, 3))]
            elif param.type == "object":
                args[param.name] = {"key": f"value_{random.randint(1, 10)}"}
        return {"name": tool.name, "arguments": args}

    def _generate_result(self, tool_name: str) -> str:
        """Generate a plausible simulation result."""
        results = [
            f"Successfully executed {tool_name}. Result: {{'status': 'ok', 'data': 'sample_output_{random.randint(1, 100)}'}}",
            f"{tool_name} completed. Returned {random.randint(1, 50)} records.",
            f"Execution of {tool_name} successful. Output: 'processed_{random.randint(1, 1000)}_items'",
        ]
        return random.choice(results)

    def generate_sample(self, difficulty: str) -> Optional[ToolCallSample]:
        """Generate a single training sample at the specified difficulty."""
        templates = SCENARIO_TEMPLATES.get(difficulty, [])
        if not templates:
            return None

        template = random.choice(templates)

        # Pick tools for this scenario
        categories = template["tool_categories"]
        available_tools = []
        for cat in categories:
            cat_tools = [t for t in TOOL_LIBRARY.values() if t.category == cat]
            if cat_tools:
                available_tools.append(random.choice(cat_tools))

        if not available_tools:
            return None

        # Determine number of tool calls
        num_calls = template["num_tool_calls"]
        if isinstance(num_calls, list):
            num_calls = random.randint(num_calls[0], num_calls[1])

        # Ensure we have enough tools (allow reuse if needed)
        selected_tools = []
        for i in range(num_calls):
            selected_tools.append(random.choice(available_tools))

        # Generate tool calls
        tool_calls = []
        for turn, tool in enumerate(selected_tools):
            call = self._generate_tool_call(tool)
            call["turn"] = turn + 1
            call["result"] = self._generate_result(tool.name)
            tool_calls.append(asdict(ToolCallRecord(
                turn=turn + 1,
                tool_name=tool.name,
                arguments=call["arguments"],
                result=call["result"],
                is_valid=True,
            )))

        # Generate user prompt
        user_prompt = self._fill_template(random.choice(template["user_prompts"]))

        # Generate final response
        final_lines = [f"Task completed. I used {len(tool_calls)} tool call(s):"]
        for tc in tool_calls:
            final_lines.append(f"- {tc['tool_name']}: {tc['result']}")

        return ToolCallSample(
            id=str(uuid.uuid4())[:8],
            difficulty=difficulty,
            pattern=template["pattern"],
            system_prompt="You are an AI assistant with access to tools. Use them to complete user tasks accurately.",
            user_prompt=user_prompt,
            tool_calls=tool_calls,
            final_response="\n".join(final_lines),
            num_turns=num_calls,
            tools_used=[t.name for t in selected_tools],
        )

    def generate_dataset(self, num_samples: int,
                         difficulty_dist: Dict[str, float]) -> List[ToolCallSample]:
        """Generate a full dataset with difficulty distribution."""
        samples = []
        for diff, ratio in difficulty_dist.items():
            n = int(num_samples * ratio)
            for _ in range(n):
                sample = self.generate_sample(diff)
                if sample:
                    samples.append(sample)
        random.shuffle(samples)
        return samples

    def to_jsonl(self, samples: List[ToolCallSample], output_path: Path) -> None:
        """Save samples to JSONL format."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")

    def to_chat_format(self, samples: List[ToolCallSample]) -> List[Dict[str, Any]]:
        """Convert samples to chat-format messages for SFT training."""
        chat_samples = []
        for s in samples:
            messages = [
                {"role": "system", "content": s.system_prompt},
                {"role": "user", "content": s.user_prompt},
            ]
            # Add tool call turns
            for tc in s.tool_calls:
                tool_call_json = json.dumps(
                    {"tool_calls": [{"name": tc["tool_name"], "arguments": tc["arguments"]}]}
                )
                messages.append({"role": "assistant", "content": tool_call_json})
                messages.append({"role": "tool", "content": tc["result"]})
            # Final response
            messages.append({"role": "assistant", "content": s.final_response})
            chat_samples.append({"messages": messages})
        return chat_samples


# ============================================================
# Data Validation
# ============================================================

def validate_sample(sample: ToolCallSample) -> Tuple[bool, List[str]]:
    """Validate a tool-call sample for correctness."""
    errors = []

    if not sample.user_prompt:
        errors.append("Empty user prompt")
    if not sample.tool_calls:
        errors.append("No tool calls")
    if sample.num_turns != len(sample.tool_calls):
        errors.append(f"Mismatched turns: {sample.num_turns} vs {len(sample.tool_calls)}")

    for tc in sample.tool_calls:
        is_valid, err = validate_tool_call(tc["tool_name"], tc["arguments"])
        if not is_valid:
            errors.append(f"Invalid tool call at turn {tc['turn']}: {err}")

    return len(errors) == 0, errors
