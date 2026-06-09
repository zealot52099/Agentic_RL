#!/usr/bin/env python3
"""Small full-parameter DDP SFT pilot with prompt loss masking."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer


class JsonlSftDataset(Dataset):
    def __init__(self, path: Path) -> None:
        self.rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


@dataclass
class SftCollator:
    tokenizer: object
    seq_len: int

    def _truncate_prompt(self, prompt: list[int], available: int) -> list[int]:
        if len(prompt) <= available:
            return prompt
        prefix = min(128, available // 4)
        return prompt[:prefix] + prompt[-(available - prefix) :]

    def __call__(self, rows: list[dict]) -> dict[str, torch.Tensor]:
        encoded = []
        eos = self.tokenizer.eos_token_id
        for row in rows:
            prompt_ids = self.tokenizer.encode(
                row["prompt"],
                add_special_tokens=False,
            )
            completion_ids = self.tokenizer.encode(
                row["completion"],
                add_special_tokens=False,
            )
            if eos is not None:
                completion_ids.append(eos)
            completion_ids = completion_ids[: self.seq_len - 1]
            available = self.seq_len - len(completion_ids)
            prompt_ids = self._truncate_prompt(prompt_ids, available)
            input_ids = prompt_ids + completion_ids
            labels = [-100] * len(prompt_ids) + completion_ids
            encoded.append((input_ids, labels))

        max_len = max(len(item[0]) for item in encoded)
        pad_id = self.tokenizer.pad_token_id
        batch_input = []
        batch_labels = []
        batch_mask = []
        for input_ids, labels in encoded:
            padding = max_len - len(input_ids)
            batch_input.append(input_ids + [pad_id] * padding)
            batch_labels.append(labels + [-100] * padding)
            batch_mask.append([1] * len(input_ids) + [0] * padding)
        return {
            "input_ids": torch.tensor(batch_input, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "attention_mask": torch.tensor(batch_mask, dtype=torch.long),
        }


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
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--log-every", type=int, default=5)
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

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        gradient_as_bucket_view=True,
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
        collate_fn=SftCollator(tokenizer, args.seq_len),
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )
    iterator = iter(loader)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        fused=True,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_lr(step, args.steps, args.warmup_steps),
    )

    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            **vars(args),
            "train_data": str(args.train_data),
            "output_dir": str(args.output_dir),
            "world_size": world_size,
            "global_batch_size": (
                args.micro_batch_size * args.grad_accum_steps * world_size
            ),
            "loss_mask": "completion tokens only",
        }
        (args.output_dir / "training_config.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    tokens_seen = 0
    for step in range(1, args.steps + 1):
        loss_sum = torch.zeros((), device=device)
        supervised_tokens = torch.zeros((), device=device)
        for _ in range(args.grad_accum_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                sampler.set_epoch(step)
                iterator = iter(loader)
                batch = next(iterator)
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            output = model(**batch)
            loss = output.loss / args.grad_accum_steps
            loss.backward()
            loss_sum += output.loss.detach()
            supervised_tokens += (batch["labels"] != -100).sum()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(supervised_tokens, op=dist.ReduceOp.SUM)
        tokens_seen += int(supervised_tokens.item())
        if rank == 0 and (step == 1 or step % args.log_every == 0):
            elapsed = time.time() - started
            record = {
                "step": step,
                "loss": float(loss_sum.item() / world_size),
                "grad_norm": float(grad_norm),
                "lr": scheduler.get_last_lr()[0],
                "supervised_tokens": tokens_seen,
                "elapsed_seconds": elapsed,
                "supervised_tokens_per_second": tokens_seen / elapsed,
                "max_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
            print(json.dumps(record), flush=True)
            with (args.output_dir / "train_metrics.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(json.dumps(record) + "\n")

    dist.barrier()
    if rank == 0:
        hf_dir = args.output_dir / "hf"
        model.module.save_pretrained(
            hf_dir,
            safe_serialization=True,
            max_shard_size="5GB",
        )
        tokenizer.save_pretrained(hf_dir)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
