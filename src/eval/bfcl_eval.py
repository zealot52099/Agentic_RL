"""
BFCL-style Evaluation Suite

Berkeley Function Calling Leaderboard (BFCL) compatible evaluation:
  - Single-turn function calling
  - Multi-turn function calling
  - Parallel function calling
  - Agentic scenarios
"""
import json
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class BFCLMetrics:
    """BFCL-style evaluation metrics."""
    # Overall
    overall_accuracy: float = 0.0

    # Per-category
    single_turn_acc: float = 0.0
    multi_turn_acc: float = 0.0
    parallel_acc: float = 0.0
    agentic_acc: float = 0.0

    # Structure
    ast_summary: Dict[str, int] = None

    def __post_init__(self):
        if self.ast_summary is None:
            self.ast_summary = {}

    def print_report(self):
        print("=" * 60)
        print("BFCL-STYLE EVALUATION REPORT")
        print("=" * 60)
        print(f"Overall Accuracy:        {self.overall_accuracy:.2%}")
        print(f"Single-Turn Accuracy:    {self.single_turn_acc:.2%}")
        print(f"Multi-Turn Accuracy:     {self.multi_turn_acc:.2%}")
        print(f"Parallel Call Accuracy:  {self.parallel_acc:.2%}")
        print(f"Agentic Scenario Acc:    {self.agentic_acc:.2%}")


class BFCLEvaluator:
    """
    BFCL-compatible evaluator for tool-calling models.

    Categories:
      1. Single-turn: One prompt → one tool call
      2. Multi-turn: Conversation with multiple sequential tool calls
      3. Parallel: Multiple independent tool calls in one response
      4. Agentic: Complex scenarios requiring planning
    """

    def __init__(self):
        self.results = {
            "single_turn": {"correct": 0, "total": 0},
            "multi_turn": {"correct": 0, "total": 0},
            "parallel": {"correct": 0, "total": 0},
            "agentic": {"correct": 0, "total": 0},
        }

    def categorize_sample(self, expected_calls: List[Dict]) -> str:
        """Categorize a sample based on expected tool calls."""
        n = len(expected_calls)
        if n == 0:
            return "single_turn"
        if n == 1:
            return "single_turn"

        # Check if calls are parallel (same turn) or sequential
        turns = set()
        for tc in expected_calls:
            turns.add(tc.get("turn", 0))

        if len(turns) == 1 and n > 1:
            return "parallel"
        elif len(turns) > 2:
            return "agentic"
        else:
            return "multi_turn"

    def evaluate_sample(self, prediction: str, expected_calls: List[Dict]) -> bool:
        """
        Evaluate a single sample — all tool names must match exactly.
        """
        from .tool_accuracy import extract_tool_calls_from_response

        pred_calls = extract_tool_calls_from_response(prediction)
        pred_names = set(tc.get("name", "") for tc in pred_calls)
        exp_names = set(tc.get("tool_name", tc.get("name", "")) for tc in expected_calls)

        return pred_names == exp_names and len(pred_names) > 0

    def evaluate(self, predictions: List[str],
                 expected_calls_list: List[List[Dict]]) -> BFCLMetrics:
        """Run full BFCL evaluation."""
        for pred, expected in zip(predictions, expected_calls_list):
            category = self.categorize_sample(expected)
            correct = self.evaluate_sample(pred, expected)

            self.results[category]["total"] += 1
            if correct:
                self.results[category]["correct"] += 1

        metrics = BFCLMetrics()
        total_correct = sum(r["correct"] for r in self.results.values())
        total_all = sum(r["total"] for r in self.results.values())

        metrics.overall_accuracy = total_correct / total_all if total_all else 0
        for cat in ["single_turn", "multi_turn", "parallel", "agentic"]:
            r = self.results[cat]
            setattr(metrics, f"{cat}_acc", r["correct"] / r["total"] if r["total"] else 0.0)

        return metrics
