"""
GRPO Training for Tool-Calling Agent

Implements Group Relative Policy Optimization (GRPO) with:
  - Multi-dimensional verifiable rewards (accuracy, JSON validity, completion, efficiency)
  - vLLM-accelerated rollout generation
  - LoRA for memory efficiency
"""
import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import yaml
from datasets import Dataset, load_from_disk

from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.reward_funcs import compute_tool_reward

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class GRPOToolCallConfig:
    """Configuration for GRPO tool-call training."""

    def __init__(self, config_path: Optional[str] = None):
        # Defaults
        self.model_name_or_path = "Qwen/Qwen2.5-7B-Instruct"
        self.output_dir = "./output/grpo_model"
        self.data_path = "./data/grpo_train.jsonl"

        # GRPO params
        self.num_generations = 8          # Group size G
        self.max_prompt_length = 2048
        self.max_completion_length = 4096
        self.beta = 0.01                  # KL penalty coefficient
        self.epsilon = 0.2
        self.learning_rate = 2e-5

        # Training
        self.num_train_epochs = 3
        self.per_device_batch_size = 4
        self.gradient_accumulation_steps = 32
        self.logging_steps = 10
        self.save_steps = 200
        self.bf16 = True
        self.seed = 42

        # LoRA
        self.use_lora = True
        self.lora_r = 16
        self.lora_alpha = 32
        self.lora_dropout = 0.05

        # Reward weights
        self.reward_weights = {
            "tool_call_accuracy": 0.45,
            "json_validity": 0.20,
            "task_completion": 0.25,
            "efficiency": 0.10,
        }

        # vLLM
        self.use_vllm = True
        self.vllm_gpu_memory = 0.6

        if config_path:
            self._load_yaml(config_path)

    def _load_yaml(self, path: str):
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        model_cfg = data.get("model", {})
        self.model_name_or_path = model_cfg.get("name_or_path", self.model_name_or_path)

        train_cfg = data.get("training", {})
        for k, v in train_cfg.items():
            if hasattr(self, k):
                setattr(self, k, v)

        reward_cfg = data.get("rewards", {})
        if reward_cfg:
            self.reward_weights = reward_cfg

        lora_cfg = data.get("lora", {})
        if lora_cfg:
            for k, v in lora_cfg.items():
                setattr(self, f"lora_{k}" if k not in ("enabled",) else "use_lora", v)

        vllm_cfg = data.get("vllm", {})
        if vllm_cfg:
            self.vllm_gpu_memory = vllm_cfg.get("gpu_memory_utilization", self.vllm_gpu_memory)

    def to_grpo_config(self) -> GRPOConfig:
        return GRPOConfig(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            bf16=self.bf16,
            seed=self.seed,
            num_generations=self.num_generations,
            max_prompt_length=self.max_prompt_length,
            max_completion_length=self.max_completion_length,
            beta=self.beta,
            epsilon=self.epsilon,
            use_vllm=self.use_vllm,
            vllm_gpu_memory_utilization=self.vllm_gpu_memory,
            report_to=["tensorboard", "wandb"],
        )


def load_grpo_data(data_path: str) -> Dataset:
    """Load tool-calling data for GRPO training (prompts only)."""
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                samples.append({
                    "prompt": obj.get("user_prompt", ""),
                    "expected_tools": [tc["tool_name"] for tc in obj.get("tool_calls", [])],
                    "expected_params": [tc.get("arguments", {}) for tc in obj.get("tool_calls", [])],
                    "num_turns": obj.get("num_turns", 1),
                    "difficulty": obj.get("difficulty", "medium"),
                })
    return Dataset.from_list(samples)


def make_reward_function(config: GRPOToolCallConfig):
    """
    Create a reward function compatible with TRL GRPOTrainer.

    TRL's GRPOTrainer expects reward_funcs as a list of callables,
    each taking (completions, prompts, **kwargs) -> List[float].
    """

    def tool_calling_reward(completions: List[str], prompts: List[str] = None, **kwargs) -> List[float]:
        rewards = []
        for i, completion in enumerate(completions):
            # Extract expected values from the prompt's dataset info
            # (passed through kwargs by TRL)
            prompt_data = kwargs.get("prompts_data", [{}])[i] if "prompts_data" in kwargs else {}

            expected_tools = prompt_data.get("expected_tools", [])
            expected_params = prompt_data.get("expected_params", [])
            num_turns = prompt_data.get("num_turns", 1)

            r = compute_tool_reward(
                completion=completion,
                expected_tools=expected_tools,
                expected_params=expected_params,
                expected_min_calls=max(1, num_turns - 1),
                expected_max_calls=num_turns + 2,
                weights=config.reward_weights,
            )
            rewards.append(r["total"])
        return rewards

    return [tool_calling_reward]


def train_grpo(config: GRPOToolCallConfig):
    """Run GRPO training for tool-calling optimization."""
    logger.info("=" * 60)
    logger.info("GRPO Training for Tool-Calling Agent")
    logger.info(f"Model: {config.model_name_or_path}")
    logger.info(f"Data: {config.data_path}")
    logger.info(f"Group Size G: {config.num_generations}")
    logger.info(f"Reward Weights: {config.reward_weights}")
    logger.info("=" * 60)

    # ---- Load Tokenizer ----
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=True,
        padding_side="left",  # Left padding for generation
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Load Model with QLoRA ----
    logger.info("Loading model...")
    if config.use_lora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        model = prepare_model_for_kbit_training(model)
        model.config.use_cache = False

        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        model.config.use_cache = False

    # ---- Load Data ----
    logger.info("Loading training data...")
    train_dataset = load_grpo_data(config.data_path)
    logger.info(f"Loaded {len(train_dataset)} training prompts")

    # ---- GRPO Config ----
    grpo_config = config.to_grpo_config()

    # ---- Reward Functions ----
    reward_funcs = make_reward_function(config)

    # ---- Trainer ----
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        reward_funcs=reward_funcs,
        tokenizer=tokenizer,
    )

    logger.info("Starting GRPO training...")
    trainer.train()

    # ---- Save ----
    logger.info(f"Saving model to {config.output_dir}")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    logger.info("GRPO training complete!")
    return config.output_dir


if __name__ == "__main__":
    config = GRPOToolCallConfig()
    if len(sys.argv) > 1:
        config = GRPOToolCallConfig(sys.argv[1])
    train_grpo(config)
