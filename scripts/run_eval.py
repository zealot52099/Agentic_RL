#!/usr/bin/env python3
"""
Evaluation Script for Tool-Calling Agent

Usage:
  python scripts/run_eval.py --model_path ./output/grpo_model --eval_data ./data/test_raw.jsonl
"""
import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.benchmark_runner import run_benchmark_suite
from src.data.tool_schema import get_tool_description_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Tool-Calling Agent"
    )
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--eval_data", type=str, required=True,
                       help="Path to evaluation JSONL data")
    parser.add_argument("--output", type=str, default="./output/eval_report.json",
                       help="Output path for JSON evaluation report")
    parser.add_argument("--system_prompt", type=str, default="",
                       help="Override system prompt (default: auto from tool library)")

    args = parser.parse_args()

    system_prompt = args.system_prompt or get_tool_description_prompt()

    report = run_benchmark_suite(
        model_path=args.model_path,
        eval_data_path=args.eval_data,
        output_path=args.output,
        system_prompt=system_prompt,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    ta = report.tool_accuracy
    bm = report.bfcl_metrics
    print(f"Function Name Accuracy:  {ta.get('function_name_accuracy', 0):.2%}")
    print(f"Parameter Accuracy:      {ta.get('parameter_accuracy', 0):.2%}")
    print(f"JSON Validity Rate:      {ta.get('json_validity_rate', 0):.2%}")
    print(f"Exact Match Rate:        {ta.get('exact_match_rate', 0):.2%}")
    print(f"BFCL Overall:            {bm.get('overall_accuracy', 0):.2%}")
    print(f"BFCL Agentic:            {bm.get('agentic_acc', 0):.2%}")
    print(f"\nDetailed report: {args.output}")


if __name__ == "__main__":
    main()
