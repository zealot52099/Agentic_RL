#!/usr/bin/env python3
"""Full-parameter SFT for Phase5 unified Data Agent format.

This is a fallback trainer that intentionally avoids PEFT/LoRA.  It is useful
when the remote PEFT stack is unstable, and gives us a clean signal that the
dataset, tokenizer, SwanLab logging, and optimizer loop are working.
"""
from __future__ import annotations

import json
import math
import os
import platform
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import swanlab
except Exception:
    swanlab = None


class PromptCompletionDataset(torch.utils.data.Dataset):
    def __init__(self, path: Path):
        self.samples = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class AssistantOnlyCollator:
    def __init__(self, tokenizer, seq_len: int):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def __call__(self, rows):
        texts = [row["prompt"] + row["completion"] for row in rows]
        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.seq_len,
            padding=True,
            return_tensors="pt",
        )
        labels = encoded["input_ids"].clone()
        for i, row in enumerate(rows):
            prompt_ids = self.tokenizer(
                row["prompt"], add_special_tokens=False
            )["input_ids"]
            labels[i, : len(prompt_ids)] = -100
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
            "sample_weights": torch.tensor(
                [float(row.get("loss_weight", 1.0)) for row in rows],
                dtype=torch.float32,
            ),
        }


def cosine_lr(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


def parse_dtype(name: str):
    normalized = name.lower()
    if normalized in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    if normalized in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "torch.float16"}:
        return torch.float16
    raise ValueError(f"Unsupported MODEL_DTYPE={name}")


@dataclass
class Config:
    model_path: str = os.environ.get(
        "MODEL_PATH",
        "/publicdata/huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct",
    )
    train_data: str = os.environ.get("TRAIN_DATA", "")
    output_dir: str = os.environ.get("OUTPUT_DIR", "output/phase5_full_sft")
    steps: int = int(os.environ.get("STEPS", "1000"))
    seq_len: int = int(os.environ.get("SEQ_LEN", "2048"))
    micro_batch_size: int = int(os.environ.get("MICRO_BATCH_SIZE", "2"))
    grad_accum_steps: int = int(os.environ.get("GRAD_ACCUM_STEPS", "8"))
    learning_rate: float = float(os.environ.get("LEARNING_RATE", "8e-7"))
    warmup_steps: int = int(os.environ.get("WARMUP_STEPS", "100"))
    max_grad_norm: float = float(os.environ.get("MAX_GRAD_NORM", "1.0"))
    seed: int = int(os.environ.get("SEED", "20260626"))
    log_every: int = int(os.environ.get("LOG_EVERY", "25"))
    save_every: int = int(os.environ.get("SAVE_EVERY", "500"))
    swanlab_project: str = os.environ.get("SWANLAB_PROJECT", "agentic-rl-pipeline")
    swanlab_run_name: str = os.environ.get("SWANLAB_RUN", "phase5_full_sft")
    swanlab_workspace: str = os.environ.get("SWANLAB_WORKSPACE", "yans2")
    swanlab_mode: str = os.environ.get("SWANLAB_MODE", "cloud")
    attn_implementation: str = os.environ.get("ATTN_IMPLEMENTATION", "sdpa")
    load_direct_to_gpu: bool = os.environ.get("LOAD_DIRECT_TO_GPU", "1") == "1"
    model_dtype: str = os.environ.get("MODEL_DTYPE", "float32")
    use_autocast: bool = os.environ.get("USE_AUTOCAST", "0") == "1"
    fused_optimizer: bool = os.environ.get("FUSED_OPTIMIZER", "0") == "1"


def main():
    cfg = Config()
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Loading tokenizer from {cfg.model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from {cfg.model_path}", flush=True)
    model_dtype = parse_dtype(cfg.model_dtype)
    load_kwargs = {
        "torch_dtype": model_dtype,
        "trust_remote_code": True,
        "attn_implementation": cfg.attn_implementation,
    }
    if cfg.load_direct_to_gpu and device.type == "cuda":
        load_kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(cfg.model_path, **load_kwargs)
    if not cfg.load_direct_to_gpu:
        model = model.to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.train()

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable full params: {trainable_count:,}", flush=True)

    dataset = PromptCompletionDataset(Path(cfg.train_data))
    loader = DataLoader(
        dataset,
        batch_size=cfg.micro_batch_size,
        shuffle=True,
        collate_fn=AssistantOnlyCollator(tokenizer, cfg.seq_len),
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )
    iterator = iter(loader)
    print(f"Training samples: {len(dataset)}", flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
        fused=cfg.fused_optimizer,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: cosine_lr(step, cfg.steps, cfg.warmup_steps)
    )

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": cfg.model_path,
        "train_data": cfg.train_data,
        "steps": cfg.steps,
        "seq_len": cfg.seq_len,
        "global_batch_size": cfg.micro_batch_size * cfg.grad_accum_steps,
        "learning_rate": cfg.learning_rate,
        "trainable_parameters": trainable_count,
        "runtime": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
    }
    (out / "training_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if swanlab and cfg.swanlab_project:
        swanlab.init(
            project=cfg.swanlab_project,
            experiment_name=cfg.swanlab_run_name,
            workspace=cfg.swanlab_workspace,
            mode=cfg.swanlab_mode,
            config=metadata,
            logdir=str(out / "swanlab"),
        )
        print(f"SwanLab: {cfg.swanlab_run_name}", flush=True)

    metrics_path = out / "train_metrics.jsonl"
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    tokens_seen = 0
    loss_ema = None

    for step in range(1, cfg.steps + 1):
        interval_loss = 0.0
        interval_tokens = 0
        total_weighted_tokens = 0.0
        for _ in range(cfg.grad_accum_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)

            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            weights = batch.pop("sample_weights")
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if cfg.use_autocast and device.type == "cuda"
                else torch.enable_grad()
            )
            with autocast_ctx:
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                ).logits[:, :-1, :]
                labels = batch["labels"][:, 1:]
                token_loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100,
                    reduction="none",
                ).view(labels.shape)
                mask = labels != -100
                per_sample_loss = (token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
                weighted_loss = (per_sample_loss * weights).sum() / weights.sum().clamp_min(1e-6)
                loss = weighted_loss / cfg.grad_accum_steps

            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")

            loss.backward()
            supervised_tokens = int(mask.sum().item())
            interval_loss += float(weighted_loss.detach().item())
            interval_tokens += supervised_tokens
            total_weighted_tokens += supervised_tokens

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        tokens_seen += interval_tokens
        avg_loss = interval_loss / cfg.grad_accum_steps
        loss_ema = avg_loss if loss_ema is None else 0.9 * loss_ema + 0.1 * avg_loss

        if step % cfg.log_every == 0 or step == 1:
            elapsed = max(1e-6, time.time() - started)
            metrics = {
                "step": step,
                "loss": avg_loss,
                "loss_ema": loss_ema,
                "grad_norm": float(grad_norm),
                "lr": scheduler.get_last_lr()[0],
                "tokens_per_second": tokens_seen / elapsed,
                "max_memory_gib": torch.cuda.max_memory_allocated() / (1024 ** 3)
                if torch.cuda.is_available()
                else 0.0,
            }
            print(json.dumps(metrics, ensure_ascii=False), flush=True)
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
            if swanlab:
                swanlab.log({f"train/{k}": v for k, v in metrics.items()})

        if step % cfg.save_every == 0:
            ckpt = out / f"checkpoint-{step}"
            model.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            print(f"Saved checkpoint: {ckpt}", flush=True)

    model.save_pretrained(out / "hf")
    tokenizer.save_pretrained(out / "hf")
    if swanlab:
        swanlab.finish()
    (out / "exit_code").write_text("0\n", encoding="utf-8")
    print(f"FULL_SFT_COMPLETE: {out / 'hf'}", flush=True)


if __name__ == "__main__":
    main()
