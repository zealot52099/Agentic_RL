# 2026-06-10 Night Successor Queue

## Goal

Keep all four H800 GPUs productively occupied overnight while adding two
controlled 3B-scale comparisons relevant to agentic and software-engineering
performance.

## Existing critical path

1. `Qwen2.5-3B`, hard77 mixture, 6,000 steps.
2. Existing xLAM, GSM8K, MMLU-Pro evaluation.
3. `Qwen3-4B-Base`, hard77 mixture, 4,000 steps.
4. Existing xLAM, GSM8K, MMLU-Pro evaluation.
5. WikiSQL evaluation queue for all completed models.

At 18:38 CST, the first job was at approximately step 1,500/6,000 with a
throughput of about 1,198 supervised tokens/s. The existing critical path was
estimated to cover GPU time until approximately 01:00-02:00 CST.

## Successor experiments

| Order | Model | Data | Steps | LR | Seed | Purpose |
|---|---|---|---:|---:|---:|---|
| 1 | Qwen2.5-Coder-3B | hard90 seed11 | 6,000 | 1.5e-6 | 20260618 | Test whether code prior improves tool and SQL behavior at 3B scale |
| 2 | Qwen2.5-3B | hard90 seed11 | 6,000 | 1.5e-6 | 20260619 | Controlled hard77 versus hard90 data-mixture comparison |

Each experiment uses four-GPU DDP, sequence length 2,048, micro batch 1,
gradient accumulation 8, global batch 32, bf16, FlashAttention 2, gradient
checkpointing, cosine decay, and checkpoints every 2,000 steps.

After each training run, four evaluations execute concurrently:

- GPU 0: internal xLAM tool-call probe
- GPU 1: GSM8K and generated MMLU-Pro regression probe
- GPU 2: direct-logprob MMLU-Pro probe
- GPU 3: internal WikiSQL execution probe

These internal probes are regression gates, not substitutes for official BFCL,
IFEval, tau-bench, or SWE-bench reporting.

## Scheduling and recovery

The successor queue waits for both the extended training queue and the WikiSQL
queue, then verifies that all GPUs are idle before starting. Failed experiments
record an exit code and do not prevent the next experiment from running.
Completed Hugging Face exports are detected so a restarted queue does not
repeat training.

Expected successor duration is roughly 6-7 hours, extending useful GPU work to
approximately 07:00-09:00 CST on 2026-06-11.

## Paths

- Queue log: `runs/night_successor_queue_20260610/queue.log`
- Queue PID: `runs/night_successor_queue_20260610/queue.pid`
- Evaluation output: `evals/night_successor_queue_20260610`
- Monitor: `scripts/remote/monitor_night_successor_queue.sh`
