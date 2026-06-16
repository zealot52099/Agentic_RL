"""
SFT (Supervised Fine-Tuning) Cold-Start Training Script

Trains the base model on tool-calling demonstrations before GRPO training.
Uses QLoRA for memory efficiency on single-GPU setups.
"""
import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

import torch
import yaml
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class SFTConfig:
    """SFT Training Configuration."""
    model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"
    output_dir: str = "./output/sft_model"
    data_path: str = "./data/train_sft.jsonl"
    val_data_path: Optional[str] = None

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Training
    num_epochs: int = 3
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    max_seq_length: int = 4096
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200

    # Quantization (QLoRA)
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"

    # System
    seed: int = 42
    bf16: bool = True
    gradient_checkpointing: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "SFTConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        sft_data = data.get("sft", data)
        model_data = data.get("model", {})
        return cls(
            model_name_or_path=model_data.get("name_or_path", cls.model_name_or_path),
            output_dir=sft_data.get("output_dir", cls.output_dir),
            **{k: v for k, v in sft_data.items() if k != "output_dir"},
        )


def load_jsonl(path: str) -> Dataset:
    """Load a JSONL file of chat-format messages into a HuggingFace Dataset."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return Dataset.from_list(samples)


def format_chat_messages(example: Dict, tokenizer: AutoTokenizer) -> Dict:
    """Format chat messages using the model's chat template and tokenize."""
    messages = example.get("messages", [])
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=4096,
        padding=False,
    )
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": tokenized["input_ids"].copy(),  # Standard LM loss
    }


def train_sft(config: SFTConfig):
    """Run SFT training for tool-calling cold start."""
    logger.info("=" * 60)
    logger.info("SFT Cold-Start Training for Tool Calling")
    logger.info(f"Model: {config.model_name_or_path}")
    logger.info(f"Data: {config.data_path}")
    logger.info("=" * 60)

    # ---- Load Model & Tokenizer ----
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # QLoRA quantization config
    if config.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        )
    else:
        bnb_config = None

    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # ---- LoRA ----
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- Load Data ----
    logger.info("Loading training data...")
    train_ds = load_jsonl(config.data_path)
    train_ds = train_ds.map(
        lambda x: format_chat_messages(x, tokenizer),
        remove_columns=train_ds.column_names,
    )

    val_ds = None
    if config.val_data_path:
        val_ds = load_jsonl(config.val_data_path)
        val_ds = val_ds.map(
            lambda x: format_chat_messages(x, tokenizer),
            remove_columns=val_ds.column_names,
        )

    # ---- Training Arguments ----
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps if val_ds else None,
        eval_strategy="steps" if val_ds else "no",
        bf16=config.bf16,
        seed=config.seed,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to=["tensorboard"],
        save_total_limit=2,
        load_best_model_at_end=True if val_ds else False,
    )

    # ---- Train ----
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
    )

    logger.info("Starting SFT training...")
    trainer.train()

    # ---- Save ----
    logger.info(f"Saving model to {config.output_dir}")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    logger.info("SFT training complete!")
    return config.output_dir


if __name__ == "__main__":
    config = SFTConfig()
    if len(sys.argv) > 1:
        config = SFTConfig.from_yaml(sys.argv[1])
    train_sft(config)
