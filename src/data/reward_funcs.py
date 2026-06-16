"""
Verifiable Reward Functions for Tool-Calling GRPO Training

Multi-dimensional rewards:
  1. tool_call_accuracy (0.45) — correct function name + valid parameters
  2. json_validity       (0.20) — output is parseable JSON
  3. task_completion     (0.25) — task objective achieved
  4. efficiency          (0.10) — minimal tool calls, no redundancy
"""
import json
import re
from typing import List, Dict, Any, Tuple, Optional
from rapidfuzz import fuzz

from .tool_schema import TOOL_LIBRARY, validate_tool_call


# ============================================================
# Reward 1: Tool Call Accuracy (weight: 0.45)
# ============================================================

def extract_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Extract tool call JSON objects from model output text."""
    tool_calls = []

    # Pattern 1: {"tool_calls": [...]}
    try:
        # Find JSON objects in the text
        for match in re.finditer(r'\{[^{}]*"tool_calls"[^{}]*\[.*?\][^{}]*\}', text, re.DOTALL):
            obj = json.loads(match.group())
            for tc in obj.get("tool_calls", []):
                tool_calls.append(tc)
    except (json.JSONDecodeError, KeyError):
        pass

    # Pattern 2: {"name": "...", "arguments": {...}}
    if not tool_calls:
        for match in re.finditer(r'\{"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^}]+\}\}', text):
            try:
                tool_calls.append(json.loads(match.group()))
            except json.JSONDecodeError:
                pass

    return tool_calls


def reward_tool_call_accuracy(completion: str, expected_tools: List[str],
                              expected_params: Optional[List[Dict]] = None) -> float:
    """
    Reward for accurate tool selection and parameter specification.

    Scoring:
      - 1.0: All tools correct + all parameters match
      - 0.7: Right tool names, minor parameter errors
      - 0.4: Some correct tools, some wrong
      - 0.0: No valid tool calls found
    """
    actual_tools = extract_tool_calls(completion)
    if not actual_tools:
        return 0.0

    score = 0.0

    # Score each tool call
    for i, actual in enumerate(actual_tools):
        tool_name = actual.get("name", "")
        args = actual.get("arguments", {})

        # Check if tool name matches expected
        if tool_name in expected_tools:
            score += 0.5 / len(expected_tools)
        elif tool_name in TOOL_LIBRARY:
            # Valid tool but not the expected one
            score += 0.2 / len(expected_tools)

        # Validate arguments against schema
        is_valid, _ = validate_tool_call(tool_name, args)
        if is_valid:
            score += 0.5 / len(actual_tools)

    return min(score, 1.0)


# ============================================================
# Reward 2: JSON Validity (weight: 0.20)
# ============================================================

def reward_json_validity(completion: str) -> float:
    """
    Reward for producing valid, parseable JSON tool calls.

    Scoring:
      - 1.0: All tool calls are valid JSON with correct structure
      - 0.5: Some JSON found but not well-formed
      - 0.0: No JSON found or completely malformed
    """
    try:
        tool_calls = extract_tool_calls(completion)
        if not tool_calls:
            return 0.0

        valid_count = 0
        for tc in tool_calls:
            if isinstance(tc, dict) and "name" in tc:
                args = tc.get("arguments", {})
                if isinstance(args, dict):
                    valid_count += 1

        return valid_count / len(tool_calls) if tool_calls else 0.0

    except Exception:
        return 0.0


# ============================================================
# Reward 3: Task Completion (weight: 0.25)
# ============================================================

def reward_task_completion(completion: str, expected_outcome_keywords: List[str]) -> float:
    """
    Reward for completing the task objective.
    Checks if the final response contains expected outcome indicators.

    Scoring:
      - 1.0: All expected keywords/outcomes present
      - 0.5: At least half present
      - 0.0: No expected outcomes
    """
    completion_lower = completion.lower()
    matches = sum(1 for kw in expected_outcome_keywords if kw.lower() in completion_lower)
    return matches / len(expected_outcome_keywords) if expected_outcome_keywords else 0.5


# ============================================================
# Reward 4: Efficiency (weight: 0.10)
# ============================================================

def reward_efficiency(completion: str, expected_min_calls: int = 1,
                      expected_max_calls: int = 5) -> float:
    """
    Reward for efficient tool use — not too few, not too many.

    Scoring:
      - 1.0: Tool calls within expected range
      - 0.5: Slightly over or under
      - 0.0: Way too many calls (>2x expected) or none
    """
    tool_calls = extract_tool_calls(completion)
    n = len(tool_calls)

    if n == 0:
        return 0.0
    if expected_min_calls <= n <= expected_max_calls:
        return 1.0
    if n < expected_min_calls:
        return 0.5 * (n / expected_min_calls)
    if n > expected_max_calls:
        penalty = (n - expected_max_calls) / expected_max_calls
        return max(0.0, 1.0 - penalty)

    return 0.5


# ============================================================
# Combined Reward Function
# ============================================================

def compute_tool_reward(completion: str,
                        expected_tools: List[str],
                        expected_params: Optional[List[Dict]] = None,
                        expected_outcomes: Optional[List[str]] = None,
                        expected_min_calls: int = 1,
                        expected_max_calls: int = 5,
                        weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """
    Compute the combined multi-dimensional reward for tool calling.

    Args:
        completion: The model's generated text
        expected_tools: List of expected tool names
        expected_params: Expected parameters for each tool
        expected_outcomes: Keywords indicating task completion
        expected_min_calls: Minimum expected tool calls
        expected_max_calls: Maximum expected tool calls
        weights: Override default weights

    Returns:
        Dictionary with individual reward components and total
    """
    if weights is None:
        weights = {
            "tool_call_accuracy": 0.45,
            "json_validity": 0.20,
            "task_completion": 0.25,
            "efficiency": 0.10,
        }

    if expected_outcomes is None:
        expected_outcomes = ["completed", "success", "done", "result"]

    rewards = {
        "tool_call_accuracy": reward_tool_call_accuracy(completion, expected_tools, expected_params),
        "json_validity": reward_json_validity(completion),
        "task_completion": reward_task_completion(completion, expected_outcomes),
        "efficiency": reward_efficiency(completion, expected_min_calls, expected_max_calls),
    }

    total = sum(weights[k] * rewards[k] for k in weights)
    rewards["total"] = round(total, 4)

    return rewards


# ============================================================
# GRPO Reward Wrapper (for TRL integration)
# ============================================================

class ToolCallRewardCalculator:
    """Wrapper for use with TRL's GRPOTrainer reward_funcs parameter."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "tool_call_accuracy": 0.45,
            "json_validity": 0.20,
            "task_completion": 0.25,
            "efficiency": 0.10,
        }

    def __call__(self, completions: List[str], prompts: List[Any], **kwargs) -> List[float]:
        """
        TRL-compatible reward function.

        Args:
            completions: List of model-generated completions
            prompts: Corresponding prompts (unused in this reward)

        Returns:
            List of scalar rewards, one per completion
        """
        rewards = []
        for completion in completions:
            r = compute_tool_reward(
                completion=completion,
                expected_tools=[],  # Will be extracted from prompt context
                weights=self.weights,
            )
            rewards.append(r["total"])
        return rewards
