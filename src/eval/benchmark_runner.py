"""
Benchmark Runner — End-to-End Evaluation Pipeline

Runs the full evaluation suite:
  1. Tool Call Accuracy (function name + parameters + JSON validity)
  2. BFCL-style metrics (single/multi/parallel/agentic)
  3. Per-difficulty breakdown
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

from .tool_accuracy import evaluate_tool_accuracy, ToolAccuracyMetrics
from .bfcl_eval import BFCLEvaluator, BFCLMetrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkReport:
    """Complete evaluation report."""
    tool_accuracy: Dict[str, Any]
    bfcl_metrics: Dict[str, Any]
    model_info: Dict[str, str]


def load_model_and_tokenizer(model_path: str, use_lora: bool = True):
    """Load model and tokenizer for evaluation."""
    logger.info(f"Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.eval()

    return model, tokenizer


def generate_tool_calls(
    model, tokenizer,
    prompts: List[str],
    system_prompt: str = "",
    max_new_tokens: int = 2048,
    temperature: float = 0.1,
) -> List[str]:
    """Generate tool-call responses for a batch of prompts."""
    predictions = []

    for prompt in tqdm(prompts, desc="Generating tool calls"):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        predictions.append(response)

    return predictions


def load_eval_data(data_path: str) -> tuple:
    """Load evaluation data with expected tool calls."""
    prompts = []
    expected_calls = []
    difficulties = []

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            prompts.append(obj.get("user_prompt", ""))
            expected_calls.append(obj.get("tool_calls", []))
            difficulties.append(obj.get("difficulty", "unknown"))

    return prompts, expected_calls, difficulties


def run_benchmark_suite(
    model_path: str,
    eval_data_path: str,
    output_path: Optional[str] = None,
    system_prompt: str = "",
) -> BenchmarkReport:
    """
    Run the complete evaluation benchmark suite.

    Args:
        model_path: Path to model checkpoint
        eval_data_path: Path to evaluation JSONL data
        output_path: Optional path to save JSON report
        system_prompt: System prompt for tool-calling

    Returns:
        BenchmarkReport with all metrics
    """
    logger.info("=" * 60)
    logger.info("RUNNING BENCHMARK SUITE")
    logger.info(f"Model: {model_path}")
    logger.info(f"Eval Data: {eval_data_path}")
    logger.info("=" * 60)

    # Load model
    model, tokenizer = load_model_and_tokenizer(model_path)

    # Load eval data
    prompts, expected_calls, difficulties = load_eval_data(eval_data_path)
    logger.info(f"Loaded {len(prompts)} evaluation samples")

    # Generate predictions
    predictions = generate_tool_calls(
        model, tokenizer, prompts,
        system_prompt=system_prompt,
    )

    # ---- Evaluation 1: Tool Accuracy ----
    logger.info("Running tool accuracy evaluation...")
    tool_metrics = evaluate_tool_accuracy(
        predictions=predictions,
        expected_calls=expected_calls,
        difficulties=difficulties,
    )
    tool_metrics.print_report()

    # ---- Evaluation 2: BFCL Metrics ----
    logger.info("Running BFCL evaluation...")
    bfcl = BFCLEvaluator()
    bfcl_metrics = bfcl.evaluate(predictions, expected_calls)
    bfcl_metrics.print_report()

    # ---- Generate Report ----
    report = BenchmarkReport(
        tool_accuracy=tool_metrics.to_dict(),
        bfcl_metrics={
            "overall_accuracy": bfcl_metrics.overall_accuracy,
            "single_turn_acc": bfcl_metrics.single_turn_acc,
            "multi_turn_acc": bfcl_metrics.multi_turn_acc,
            "parallel_acc": bfcl_metrics.parallel_acc,
            "agentic_acc": bfcl_metrics.agentic_acc,
        },
        model_info={
            "model_path": model_path,
            "eval_samples": str(len(prompts)),
        },
    )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2, ensure_ascii=False)
        logger.info(f"Report saved to: {output_path}")

    return report


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python benchmark_runner.py <model_path> <eval_data.jsonl> [output.json]")
        sys.exit(1)

    report = run_benchmark_suite(
        model_path=sys.argv[1],
        eval_data_path=sys.argv[2],
        output_path=sys.argv[3] if len(sys.argv) > 3 else None,
    )
