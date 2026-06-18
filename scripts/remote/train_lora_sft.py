#!/usr/bin/env python3
"""Distributed LoRA SFT with global supervised-token normalization."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
import platform
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM
import transformers

from train_assistant_only_sft import JsonlSftDataset, SftCollator, load_tokenizer

try:
    import swanlab
except Exception:  # pragma: no cover - optional runtime dependency
    swanlab = None


@dataclass
class WeightedSftCollator(SftCollator):
    def __call__(self, rows: list[dict]) -> dict[str, torch.Tensor]:
        batch = super().__call__(rows)
        batch["sample_weights"] = torch.tensor(
            [float(row.get("loss_weight", 1.0)) for row in rows],
            dtype=torch.float32,
        )
        return batch


def cosine_lr(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--swanlab-project")
    parser.add_argument("--swanlab-run-name")
    parser.add_argument("--swanlab-workspace")
    parser.add_argument("--swanlab-mode", default=os.environ.get("SWANLAB_MODE", "cloud"))
    parser.add_argument("--swanlab-tags", default="")
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    tokenizer = load_tokenizer(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    base_model.enable_input_require_grads()
    model = get_peft_model(
        base_model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
        autocast_adapter_dtype=True,
    ).to(device)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.dtype != torch.float32:
            parameter.data = parameter.data.float()
    model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        gradient_as_bucket_view=True,
        find_unused_parameters=False,
    )

    dataset = JsonlSftDataset(args.train_data)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        sampler=sampler,
        collate_fn=WeightedSftCollator(tokenizer, args.seq_len),
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )
    iterator = iter(loader)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay,
        fused=True,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_lr(step, args.steps, args.warmup_steps),
    )

    swanlab_enabled = bool(args.swanlab_project)
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        trainable_count = sum(parameter.numel() for parameter in trainable)
        metadata = {
            **vars(args),
            "train_data": str(args.train_data),
            "output_dir": str(args.output_dir),
            "world_size": world_size,
            "global_batch_size": (
                args.micro_batch_size * args.grad_accum_steps * world_size
            ),
            "loss_reduction": "global supervised-token mean",
            "trainable_parameters": trainable_count,
            "trainable_dtype": sorted({str(p.dtype) for p in trainable}),
            "runtime": {
                "hostname": platform.node(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
        }
        (args.output_dir / "training_config.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if swanlab_enabled:
            if swanlab is None:
                raise RuntimeError(
                    "SwanLab logging requested but package is not installed"
                )
            swanlab.init(
                project=args.swanlab_project,
                experiment_name=args.swanlab_run_name or args.output_dir.name,
                workspace=args.swanlab_workspace,
                mode=args.swanlab_mode,
                tags=[tag for tag in args.swanlab_tags.split(",") if tag],
                config=metadata,
                logdir=str(args.output_dir / "swanlab"),
            )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    tokens_seen = 0
    sampler_epoch = 0
    loss_ema = None
    interval_nll = 0.0
    interval_tokens = 0
    interval_grad_norm = 0.0
    interval_clipped = 0
    interval_steps = 0
    for step in range(1, args.steps + 1):
        batches = []
        local_weighted_tokens = torch.zeros((), device=device)
        for _ in range(args.grad_accum_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                sampler_epoch += 1
                sampler.set_epoch(sampler_epoch)
                iterator = iter(loader)
                batch = next(iterator)
            supervised_mask = batch["labels"][:, 1:] != -100
            local_weighted_tokens += (
                supervised_mask.sum(dim=1) * batch["sample_weights"]
            ).sum().to(device)
            batches.append(batch)
        global_weighted_tokens = local_weighted_tokens.clone()
        dist.all_reduce(global_weighted_tokens, op=dist.ReduceOp.SUM)
        if global_weighted_tokens.item() == 0:
            raise RuntimeError(f"No supervised tokens at step {step}")

        local_nll = torch.zeros((), device=device)
        for batch_index, batch in enumerate(batches):
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }
            sample_weights = batch.pop("sample_weights")
            context = (
                model.no_sync()
                if batch_index < len(batches) - 1
                else nullcontext()
            )
            with context:
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                ).logits[:, :-1, :]
                labels = batch["labels"][:, 1:]
                token_nll = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100,
                    reduction="none",
                ).view_as(labels)
                batch_nll = (
                    token_nll * sample_weights[:, None]
                ).sum()
                (batch_nll * world_size / global_weighted_tokens).backward()
                local_nll += batch_nll.detach()

        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_nll = local_nll.clone()
        dist.all_reduce(global_nll, op=dist.ReduceOp.SUM)

        step_weighted_tokens = float(global_weighted_tokens.item())
        step_loss = float(global_nll.item() / step_weighted_tokens)
        loss_ema = step_loss if loss_ema is None else 0.95 * loss_ema + 0.05 * step_loss
        tokens_seen += int(step_weighted_tokens)
        interval_nll += float(global_nll.item())
        interval_tokens += step_weighted_tokens
        interval_grad_norm += float(grad_norm)
        interval_clipped += int(float(grad_norm) > args.max_grad_norm)
        interval_steps += 1

        if rank == 0 and (step == 1 or step % args.log_every == 0):
            elapsed = time.time() - started
            record = {
                "step": step,
                "loss": interval_nll / interval_tokens,
                "step_loss": step_loss,
                "loss_ema": loss_ema,
                "grad_norm": float(grad_norm),
                "mean_grad_norm": interval_grad_norm / interval_steps,
                "clipped_step_rate": interval_clipped / interval_steps,
                "lr": scheduler.get_last_lr()[0],
                "supervised_tokens": tokens_seen,
                "supervised_tokens_per_second": tokens_seen / elapsed,
                "max_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
            print(json.dumps(record), flush=True)
            with (args.output_dir / "train_metrics.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(json.dumps(record) + "\n")
            if swanlab_enabled:
                swanlab.log(record, step=step)
            interval_nll = 0.0
            interval_tokens = 0
            interval_grad_norm = 0.0
            interval_clipped = 0
            interval_steps = 0

        if args.save_every > 0 and step % args.save_every == 0:
            dist.barrier()
            if rank == 0:
                checkpoint = args.output_dir / "checkpoints" / f"step-{step}"
                model.module.save_pretrained(checkpoint, safe_serialization=True)
                tokenizer.save_pretrained(checkpoint)
            dist.barrier()

    dist.barrier()
    if rank == 0:
        adapter_dir = args.output_dir / "adapter"
        model.module.save_pretrained(adapter_dir, safe_serialization=True)
        tokenizer.save_pretrained(adapter_dir)
        if swanlab_enabled:
            swanlab.finish()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
