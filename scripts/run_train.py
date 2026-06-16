#!/usr/bin/env python3
"""
Training Pipeline: SFT Cold-Start → GRPO Optimization

Usage:
  python scripts/run_train.py --stage sft     # SFT cold-start only
  python scripts/run_train.py --stage grpo    # GRPO training only
  python scripts/run_train.py --stage all     # Full pipeline
"""
import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train.sft_trainer import SFTConfig, train_sft
from src.train.grpo_trainer import GRPOToolCallConfig, train_grpo

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_sft(args):
    """Run SFT cold-start training."""
    config = SFTConfig(
        model_name_or_path=args.model_name_or_path,
        output_dir=args.sft_output_dir or "./output/sft_model",
        data_path=args.sft_data or "./data/train_sft.jsonl",
        val_data_path=args.sft_val_data or "./data/val_sft.jsonl",
        num_epochs=args.sft_epochs,
        per_device_batch_size=args.batch_size,
        learning_rate=args.sft_lr,
        max_seq_length=args.max_seq_length,
    )
    return train_sft(config)


def run_grpo(args):
    """Run GRPO training."""
    config = GRPOToolCallConfig(
        model_name_or_path=args.grpo_model or args.model_name_or_path,
        output_dir=args.grpo_output_dir or "./output/grpo_model",
        data_path=args.grpo_data or "./data/train_grpo.jsonl",
        num_generations=args.grpo_g,
        learning_rate=args.grpo_lr,
        beta=args.beta,
        num_train_epochs=args.grpo_epochs,
        per_device_batch_size=args.batch_size,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
    )
    return train_grpo(config)


def main():
    parser = argparse.ArgumentParser(
        description="Agentic RL Training Pipeline for Tool Calling"
    )

    # Stage selection
    parser.add_argument("--stage", choices=["sft", "grpo", "all"], default="all",
                       help="Training stage to run")
    parser.add_argument("--config", type=str, default=None,
                       help="YAML config file path (overrides CLI args)")

    # Model
    parser.add_argument("--model_name_or_path", type=str,
                       default="Qwen/Qwen2.5-7B-Instruct",
                       help="Base model path or HF name")

    # SFT args
    parser.add_argument("--sft_output_dir", type=str, default="./output/sft_model")
    parser.add_argument("--sft_data", type=str, default="./data/train_sft.jsonl")
    parser.add_argument("--sft_val_data", type=str, default=None)
    parser.add_argument("--sft_epochs", type=int, default=3)
    parser.add_argument("--sft_lr", type=float, default=2e-4)

    # GRPO args
    parser.add_argument("--grpo_output_dir", type=str, default="./output/grpo_model")
    parser.add_argument("--grpo_data", type=str, default="./data/train_grpo.jsonl")
    parser.add_argument("--grpo_model", type=str, default=None,
                       help="Model path for GRPO (defaults to SFT output if stage=all)")
    parser.add_argument("--grpo_epochs", type=int, default=3)
    parser.add_argument("--grpo_lr", type=float, default=2e-5)
    parser.add_argument("--grpo_g", type=int, default=8,
                       help="Group size G for GRPO")
    parser.add_argument("--beta", type=float, default=0.01,
                       help="KL penalty coefficient for GRPO")

    # Common args
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_completion_length", type=int, default=4096)

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Agentic RL Training Pipeline")
    logger.info(f"Stage: {args.stage}")
    logger.info(f"Base Model: {args.model_name_or_path}")
    logger.info("=" * 60)

    if args.stage in ("sft", "all"):
        logger.info("\n>>> STAGE 1: SFT Cold-Start Training")
        sft_path = run_sft(args)
        if args.stage == "all":
            args.grpo_model = sft_path

    if args.stage in ("grpo", "all"):
        logger.info("\n>>> STAGE 2: GRPO Reinforcement Learning")
        grpo_path = run_grpo(args)

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING PIPELINE COMPLETE!")
    if args.stage == "all":
        logger.info(f"SFT model: {args.sft_output_dir}")
        logger.info(f"GRPO model: {args.grpo_output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
