"""
Tool Call Accuracy Evaluation

Metrics:
  1. Function Name Accuracy — correct tool selected
  2. Parameter Accuracy — correct parameters passed
  3. JSON Validity Rate — well-formed output
  4. Task Success Rate — overall task completed
  5. Efficiency Score — optimal number of tool calls
"""
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter

from rapidfuzz import fuzz

# Import from parent package
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.tool_schema import TOOL_LIBRARY, validate_tool_call


@dataclass
class ToolAccuracyMetrics:
    """Aggregate metrics for tool-calling evaluation."""
    total_samples: int = 0
    function_name_accuracy: float = 0.0    # % correct tool selection
    parameter_accuracy: float = 0.0        # % correct parameters
    json_validity_rate: float = 0.0        # % well-formed JSON output
    task_success_rate: float = 0.0         # % overall task success
    efficiency_score: float = 0.0          # tool call count efficiency
    avg_tool_calls: float = 0.0            # average tool calls per sample
    exact_match_rate: float = 0.0          # % perfect tool call matches

    # Detailed breakdown
    per_difficulty: Dict[str, Dict[str, float]] = field(default_factory=dict)
    per_tool_category: Dict[str, Dict[str, float]] = field(default_factory=dict)
    error_types: Counter = field(default_factory=Counter)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "function_name_accuracy": round(self.function_name_accuracy, 4),
            "parameter_accuracy": round(self.parameter_accuracy, 4),
            "json_validity_rate": round(self.json_validity_rate, 4),
            "task_success_rate": round(self.task_success_rate, 4),
            "efficiency_score": round(self.efficiency_score, 4),
            "avg_tool_calls": round(self.avg_tool_calls, 2),
            "exact_match_rate": round(self.exact_match_rate, 4),
            "error_types": dict(self.error_types.most_common()),
        }

    def print_report(self):
        print("=" * 60)
        print("TOOL CALL ACCURACY REPORT")
        print("=" * 60)
        print(f"Total Samples:        {self.total_samples}")
        print(f"Function Name Accuracy: {self.function_name_accuracy:.2%}")
        print(f"Parameter Accuracy:    {self.parameter_accuracy:.2%}")
        print(f"JSON Validity Rate:    {self.json_validity_rate:.2%}")
        print(f"Task Success Rate:     {self.task_success_rate:.2%}")
        print(f"Efficiency Score:      {self.efficiency_score:.2%}")
        print(f"Exact Match Rate:      {self.exact_match_rate:.2%}")
        print(f"Avg Tool Calls:        {self.avg_tool_calls:.2f}")
        print()
        if self.error_types:
            print("Top Error Types:")
            for error, count in self.error_types.most_common(5):
                print(f"  - {error}: {count}")


def extract_tool_calls_from_response(text: str) -> List[Dict[str, Any]]:
    """Extract tool calls from model output, handling various formats."""
    tool_calls = []

    # Format 1: {"tool_calls": [{"name": "...", "arguments": {...}}]}
    try:
        for match in re.finditer(r'\{[^{}]*"tool_calls"[^{}]*\[.*?\][^{}]*\}', text, re.DOTALL):
            obj = json.loads(match.group())
            for tc in obj.get("tool_calls", []):
                tool_calls.append({"name": tc.get("name", ""), "arguments": tc.get("arguments", {})})
    except (json.JSONDecodeError, KeyError, AttributeError):
        pass

    # Format 2: Direct {"name": "...", "arguments": {...}}
    if not tool_calls:
        for match in re.finditer(r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]+\})\s*\}', text):
            try:
                obj = json.loads(match.group())
                tool_calls.append({"name": obj.get("name", ""), "arguments": obj.get("arguments", {})})
            except json.JSONDecodeError:
                pass

    # Format 3: Function-call style <tool_call>{"name": "...", ...}</tool_call>
    if not tool_calls:
        for match in re.finditer(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL):
            try:
                obj = json.loads(match.group(1))
                tool_calls.append({"name": obj.get("name", ""), "arguments": obj.get("arguments", {})})
            except json.JSONDecodeError:
                pass

    return tool_calls


def compare_tool_calls(predicted: List[Dict], expected: List[Dict]) -> Tuple[bool, bool, bool, str]:
    """
    Compare predicted tool calls against expected.

    Returns:
      - name_match: tool names match
      - params_match: parameters match (fuzzy)
      - exact_match: perfect match
      - error_type: error category if any
    """
    if not predicted and not expected:
        return True, True, True, "no_tools_expected"
    if not predicted:
        return False, False, False, "no_tool_calls_found"
    if not expected:
        return False, False, False, "unexpected_tool_calls"

    # Compare tool names
    pred_names = [tc.get("name", "") for tc in predicted]
    exp_names = [tc.get("name", "") for tc in expected]

    name_match = set(pred_names) == set(exp_names)

    # Compare parameters (fuzzy match on argument keys + values)
    params_ok = True
    for i, (pred, exp) in enumerate(zip(predicted, expected)):
        pred_args = pred.get("arguments", {})
        exp_args = exp.get("arguments", {})
        if set(pred_args.keys()) != set(exp_args.keys()):
            params_ok = False
            break

    exact = name_match and params_ok

    # Error classification
    if not name_match:
        missing = set(exp_names) - set(pred_names)
        extra = set(pred_names) - set(exp_names)
        if missing and extra:
            error_type = "wrong_tools"
        elif missing:
            error_type = "missing_tools"
        else:
            error_type = "extra_tools"
    elif not params_ok:
        error_type = "wrong_parameters"
    else:
        error_type = "none"

    return name_match, params_ok, exact, error_type


def is_valid_json(text: str) -> bool:
    """Check if the response contains valid JSON tool calls."""
    tool_calls = extract_tool_calls_from_response(text)
    if not tool_calls:
        return False
    return all(
        isinstance(tc.get("name"), str) and isinstance(tc.get("arguments"), dict)
        for tc in tool_calls
    )


def evaluate_tool_accuracy(
    predictions: List[str],
    expected_calls: List[List[Dict[str, Any]]],
    difficulties: Optional[List[str]] = None,
    tool_categories: Optional[List[str]] = None,
) -> ToolAccuracyMetrics:
    """
    Evaluate tool-calling accuracy across multiple samples.

    Args:
        predictions: List of model-generated responses
        expected_calls: List of expected tool calls per sample
        difficulties: Optional difficulty labels per sample
        tool_categories: Optional tool category labels per sample

    Returns:
        ToolAccuracyMetrics with detailed breakdown
    """
    metrics = ToolAccuracyMetrics()
    metrics.total_samples = len(predictions)

    total_name_correct = 0
    total_params_correct = 0
    total_json_valid = 0
    total_exact_match = 0
    total_tool_calls = 0
    per_diff = {}
    per_cat = {}
    error_counter = Counter()

    for i, (pred, exp) in enumerate(zip(predictions, expected_calls)):
        # Extract predicted tool calls
        pred_calls = extract_tool_calls_from_response(pred)
        total_tool_calls += len(pred_calls)

        # JSON validity
        json_ok = is_valid_json(pred)
        if json_ok:
            total_json_valid += 1

        # Compare
        name_ok, params_ok, exact_ok, error = compare_tool_calls(pred_calls, exp)

        if name_ok:
            total_name_correct += 1
        if params_ok:
            total_params_correct += 1
        if exact_ok:
            total_exact_match += 1
        if error != "none":
            error_counter[error] += 1

        # Per-difficulty tracking
        diff = difficulties[i] if difficulties else "unknown"
        if diff not in per_diff:
            per_diff[diff] = {"total": 0, "success": 0}
        per_diff[diff]["total"] += 1
        if exact_ok:
            per_diff[diff]["success"] += 1

    n = metrics.total_samples
    metrics.function_name_accuracy = total_name_correct / n if n else 0
    metrics.parameter_accuracy = total_params_correct / n if n else 0
    metrics.json_validity_rate = total_json_valid / n if n else 0
    metrics.exact_match_rate = total_exact_match / n if n else 0
    metrics.avg_tool_calls = total_tool_calls / n if n else 0

    # Task success = exact match (all tools + params correct)
    metrics.task_success_rate = total_exact_match / n if n else 0

    # Efficiency = penalize over-calling
    metrics.efficiency_score = metrics.exact_match_rate  # Simplified

    # Per-difficulty rates
    for d in per_diff:
        per_diff[d]["rate"] = per_diff[d]["success"] / per_diff[d]["total"] if per_diff[d]["total"] else 0
    metrics.per_difficulty = per_diff

    metrics.error_types = error_counter
    return metrics
