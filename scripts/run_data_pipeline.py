#!/usr/bin/env python3
"""
End-to-End Data Pipeline for Tool-Calling Agentic RL

Stages:
  1. Generate synthetic tool-call training data
  2. Clean & filter data
  3. Split into train/val/test
  4. Convert to chat-format for SFT
  5. Save GRPO-ready prompts
"""
import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.synthetic_data import SyntheticDataGenerator, validate_sample
from src.data.data_cleaner import ToolCallDataCleaner
from src.data.tool_schema import TOOL_LIBRARY, get_tool_description_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---- Config ----
OUTPUT_DIR = Path("./data")
NUM_SAMPLES = 10000
DIFFICULTY_DIST = {"easy": 0.2, "medium": 0.5, "hard": 0.3}
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
SEED = 42


def main():
    logger.info("=" * 60)
    logger.info("DATA PIPELINE: Tool-Calling Training Data Generation")
    logger.info("=" * 60)
    logger.info(f"Tools available: {len(TOOL_LIBRARY)}")
    logger.info(f"Target samples: {NUM_SAMPLES}")
    logger.info(f"Difficulty distribution: {DIFFICULTY_DIST}")

    # ---- Stage 1: Generate Synthetic Data ----
    logger.info("\n[Stage 1] Generating synthetic tool-call data...")
    gen = SyntheticDataGenerator(seed=SEED)
    samples = gen.generate_dataset(
        num_samples=NUM_SAMPLES,
        difficulty_dist=DIFFICULTY_DIST,
    )
    logger.info(f"  Generated {len(samples)} raw samples")

    # Validate samples
    valid_count = 0
    for s in samples:
        ok, errs = validate_sample(s)
        if ok:
            valid_count += 1
        elif errs:
            logger.debug(f"  Validation errors in {s.id}: {errs}")
    logger.info(f"  Valid samples: {valid_count}/{len(samples)}")

    # ---- Stage 2: Clean & Filter ----
    logger.info("\n[Stage 2] Cleaning and filtering data...")
    from dataclasses import asdict
    raw_dicts = [asdict(s) for s in samples]

    cleaner = ToolCallDataCleaner(
        min_tool_calls=1,
        max_tool_calls=10,
    )
    cleaned, stats = cleaner.clean(raw_dicts)
    logger.info(f"  {cleaner.get_stats_report()}")

    # ---- Stage 3: Split Data ----
    logger.info("\n[Stage 3] Splitting into train/val/test...")
    import random
    random.seed(SEED)
    random.shuffle(cleaned)

    n = len(cleaned)
    train_end = int(n * SPLIT_RATIOS["train"])
    val_end = train_end + int(n * SPLIT_RATIOS["val"])

    splits = {
        "train": cleaned[:train_end],
        "val": cleaned[train_end:val_end],
        "test": cleaned[val_end:],
    }
    for name, data in splits.items():
        logger.info(f"  {name}: {len(data)} samples")

    # ---- Stage 4: Save Data ----
    logger.info("\n[Stage 4] Saving data...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save raw JSONL (for transparency)
    for name, data in splits.items():
        path = OUTPUT_DIR / f"{name}_raw.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for s in data:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        logger.info(f"  Saved: {path}")

    # Save chat-format for SFT
    logger.info("\n[Stage 5] Converting to chat format for SFT...")
    tool_desc = get_tool_description_prompt()

    for name, data in splits.items():
        chat_samples = []
        for s in data:
            messages = [
                {"role": "system", "content": tool_desc},
                {"role": "user", "content": s.get("user_prompt", "")},
            ]
            for tc in s.get("tool_calls", []):
                tool_json = json.dumps({
                    "tool_calls": [{"name": tc["tool_name"], "arguments": tc.get("arguments", {})}]
                }, ensure_ascii=False)
                messages.append({"role": "assistant", "content": tool_json})
                messages.append({"role": "tool", "content": tc.get("result", "Success.")})
            messages.append({"role": "assistant", "content": s.get("final_response", "Task completed.")})
            chat_samples.append({"messages": messages})

        path = OUTPUT_DIR / f"{name}_sft.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for cs in chat_samples:
                f.write(json.dumps(cs, ensure_ascii=False) + "\n")
        logger.info(f"  SFT format: {path} ({len(chat_samples)} samples)")

    # Save GRPO prompts (for GRPO training)
    logger.info("\n[Stage 6] Saving GRPO prompts...")
    for name, data in splits.items():
        grpo_samples = []
        for s in data:
            grpo_samples.append({
                "prompt": s.get("user_prompt", ""),
                "expected_tools": [tc["tool_name"] for tc in s.get("tool_calls", [])],
                "expected_params": [tc.get("arguments", {}) for tc in s.get("tool_calls", [])],
                "num_turns": s.get("num_turns", 1),
                "difficulty": s.get("difficulty", "medium"),
                "system_prompt": tool_desc,
            })

        path = OUTPUT_DIR / f"{name}_grpo.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for gs in grpo_samples:
                f.write(json.dumps(gs, ensure_ascii=False) + "\n")
        logger.info(f"  GRPO format: {path} ({len(grpo_samples)} samples)")

    # Save tool library reference
    tools_path = OUTPUT_DIR / "tool_library.json"
    with open(tools_path, "w", encoding="utf-8") as f:
        json.dump(
            {name: {"description": t.description, "category": t.category,
                    "parameters": [{"name": p.name, "type": p.type, "required": p.required}
                                   for p in t.parameters]}
             for name, t in TOOL_LIBRARY.items()},
            f, indent=2, ensure_ascii=False,
        )
    logger.info(f"\n  Tool library: {tools_path}")

    logger.info("\n" + "=" * 60)
    logger.info("DATA PIPELINE COMPLETE!")
    logger.info(f"Output directory: {OUTPUT_DIR.absolute()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
