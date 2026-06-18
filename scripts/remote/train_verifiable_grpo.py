#!/usr/bin/env python3
"""Minimal multi-GPU GRPO-style RLVR for deterministic format constraints."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoModelForCausalLM, AutoTokenizer

from agentic_rl.reward import score_tool_response


def load_tokenizer(model_path: str):
    try:
        return AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    except AttributeError as error:
        if "'list' object has no attribute 'keys'" not in str(error):
            raise
        return AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            extra_special_tokens={},
        )


def load_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def constraint_scores(text: str, verifier: dict) -> list[float]:
    stripped = text.strip()
    kind = verifier["kind"]
    if kind == "mcp_tool_call":
        result = score_tool_response(text, verifier)
        components = result["components"]
        return [
            float(result["format_valid"]),
            components["route"],
            components["arguments"],
            components["execution"],
            components["task"],
            components["recovery"],
            components["safety"],
        ]
    if kind == "bullets_phrase_forbidden":
        bullets = [line for line in stripped.splitlines() if line.startswith("- ")]
        return [
            float(len(bullets) == verifier["bullet_count"]),
            float(stripped.count(verifier["required_phrase"]) == 1),
            float(verifier["forbidden"].lower() not in stripped.lower()),
        ]
    if kind == "json_keys":
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            return [0.0, 0.0, 0.0]
        return [
            float(isinstance(value, dict)),
            float(isinstance(value, dict) and set(value) == set(verifier["keys"])),
            float(
                isinstance(value, dict)
                and all(isinstance(item, str) for item in value.values())
            ),
        ]
    if kind == "prefix_suffix_sentences":
        sentence_count = len(re.findall(r"[.!?](?:\s|$)", stripped))
        return [
            float(stripped.startswith(verifier["prefix"])),
            float(stripped.endswith(verifier["suffix"])),
            float(sentence_count == verifier["sentence_count"]),
        ]
    if kind == "numbered_list":
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        expected = [f"{index}." for index in range(1, verifier["count"] + 1)]
        return [
            float(len(lines) == verifier["count"]),
            float(
                len(lines) == verifier["count"]
                and all(line.startswith(prefix) for line, prefix in zip(lines, expected))
            ),
            float(not lines or not any(line.startswith(("-", "*")) for line in lines)),
        ]
    if kind == "word_range_phrase":
        words = re.findall(r"\b[\w'-]+\b", stripped)
        return [
            float(verifier["min_words"] <= len(words) <= verifier["max_words"]),
            float(verifier["required_phrase"] in stripped),
        ]
    if kind == "nested_json_constraints":
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            return [0.0, 0.0, 0.0, 0.0]
        valid_object = isinstance(value, dict)
        keys_match = valid_object and set(value) == set(verifier["keys"])
        arrays_match = keys_match and all(
            isinstance(value[key], list)
            and len(value[key]) == verifier["array_length"]
            and all(isinstance(item, str) for item in value[key])
            for key in verifier["keys"]
        )
        return [
            float(valid_object),
            float(keys_match),
            float(arrays_match),
            float(stripped.count(verifier["required_phrase"]) == 1),
            float(verifier["forbidden"].lower() not in stripped.lower()),
        ]
    if kind == "numbered_phrase_suffix":
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        numbered = len(lines) == verifier["count"] and all(
            line.startswith(f"{index}.")
            for index, line in enumerate(lines, start=1)
        )
        return [
            float(len(lines) == verifier["count"]),
            float(numbered),
            float(stripped.count(verifier["required_phrase"]) == 1),
            float(stripped.endswith(verifier["suffix"])),
        ]
    if kind == "sentence_word_phrase":
        words = re.findall(r"\b[\w'-]+\b", stripped)
        sentences = len(re.findall(r"[.!?](?:\s|$)", stripped))
        return [
            float(sentences == verifier["sentence_count"]),
            float(verifier["min_words"] <= len(words) <= verifier["max_words"]),
            float(stripped.startswith(verifier["prefix"])),
            float(stripped.endswith(verifier["suffix"])),
            float(stripped.count(verifier["required_phrase"]) == 1),
        ]
    raise ValueError(f"Unknown verifier kind: {kind}")


def reward(text: str, verifier: dict) -> tuple[float, bool]:
    if verifier["kind"] == "mcp_tool_call":
        result = score_tool_response(text, verifier)
        return float(result["reward"]), bool(result["success"])
    scores = constraint_scores(text, verifier)
    success = all(score == 1.0 for score in scores)
    shaped = sum(scores) / len(scores)
    length_penalty = max(0, len(text.split()) - 160) / 320
    return shaped + float(success) - min(length_penalty, 0.25), success


def cosine_lr(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=1e-7)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=400)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    tokenizer = load_tokenizer(args.model)
    tokenizer.padding_side = "left"
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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
        fused=True,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_lr(step, args.steps, args.warmup_steps),
    )
    rows = load_rows(args.dataset)
    rng = random.Random(args.seed + rank)

    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "training_config.json").write_text(
            json.dumps(
                {**vars(args), "dataset": str(args.dataset), "world_size": world_size},
                ensure_ascii=False,
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )

    started = time.time()
    cumulative_success = 0.0
    cumulative_samples = 0
    skipped_groups = 0
    for step in range(1, args.steps + 1):
        row = rows[rng.randrange(len(rows))]
        prompt_tokens = tokenizer(
            row["prompt"],
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(device)
        prompt_length = prompt_tokens["input_ids"].shape[1]

        model.module.eval()
        model.module.config.use_cache = True
        with torch.no_grad():
            generated = model.module.generate(
                **prompt_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                num_return_sequences=args.group_size,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = generated.detach().clone()
        completions = generated[:, prompt_length:]
        texts = tokenizer.batch_decode(completions, skip_special_tokens=True)
        scored = [reward(text, row["verifier"]) for text in texts]
        rewards = torch.tensor([item[0] for item in scored], device=device)
        successes = sum(item[1] for item in scored)
        cumulative_success += successes
        cumulative_samples += len(scored)

        reward_std = rewards.std(unbiased=False)
        if reward_std < 1e-6:
            skipped_groups += 1
            loss_value = 0.0
            grad_norm_value = 0.0
        else:
            advantages = (rewards - rewards.mean()) / (reward_std + 1e-6)
            attention_mask = generated.ne(tokenizer.pad_token_id).long()
            labels = generated.clone()
            labels[:, :prompt_length] = -100
            labels[attention_mask == 0] = -100

            model.module.config.use_cache = False
            model.train()
            output = model(input_ids=generated, attention_mask=attention_mask)
            shift_logits = output.logits[:, :-1].float()
            shift_labels = labels[:, 1:]
            valid = shift_labels.ne(-100)
            gather_labels = shift_labels.masked_fill(~valid, 0)
            token_log_probs = torch.log_softmax(shift_logits, dim=-1).gather(
                -1, gather_labels.unsqueeze(-1)
            ).squeeze(-1)
            sequence_log_probs = (token_log_probs * valid).sum(-1) / valid.sum(
                -1
            ).clamp_min(1)
            loss = -(advantages.detach() * sequence_log_probs).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            loss_value = float(loss.detach())
            grad_norm_value = float(grad_norm)

        if rank == 0 and (step == 1 or step % args.log_every == 0):
            record = {
                "step": step,
                "loss": loss_value,
                "grad_norm": grad_norm_value,
                "lr": scheduler.get_last_lr()[0],
                "group_reward_mean": float(rewards.mean()),
                "group_reward_std": float(reward_std),
                "group_success_rate": successes / len(scored),
                "running_success_rate": cumulative_success / cumulative_samples,
                "skipped_group_rate": skipped_groups / step,
                "mean_completion_tokens": float(
                    completions.ne(tokenizer.pad_token_id).sum(-1).float().mean()
                ),
                "elapsed_seconds": time.time() - started,
            }
            print(json.dumps(record), flush=True)
            with (args.output_dir / "train_metrics.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(json.dumps(record) + "\n")

        if args.save_every and step % args.save_every == 0:
            dist.barrier()
            if rank == 0:
                checkpoint = args.output_dir / "checkpoints" / f"step-{step}"
                model.module.save_pretrained(
                    checkpoint, safe_serialization=True, max_shard_size="5GB"
                )
                tokenizer.save_pretrained(checkpoint)
            dist.barrier()

    dist.barrier()
    if rank == 0:
        hf_dir = args.output_dir / "hf"
        model.module.save_pretrained(
            hf_dir, safe_serialization=True, max_shard_size="5GB"
        )
        tokenizer.save_pretrained(hf_dir)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
