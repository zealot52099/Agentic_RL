# Agentic RL Evaluation Suite Preparation

This document tracks the local, transferable evaluation assets used by the
project. Dataset files and evaluator source revisions are recorded in:

`datasets/eval_suite/manifests/eval_suite_manifest.json`

## Coverage

| Capability | Benchmark | Local asset | Execution requirement |
|---|---|---|---|
| Function calling | BFCL V4 | Dataset and official Gorilla evaluator | Model adapter; some agentic categories need network/tool services |
| Multi-turn tools | tau-bench | Official repository tasks and environments | User-simulator model/API and domain environment |
| Instruction following | IFEval | Official prompts and Google evaluator | Deterministic generation and response JSONL |
| Software engineering | SWE-bench Verified/Lite | Dataset and official harness | Linux Docker, repository images, substantial disk and CPU |
| Code generation | LiveCodeBench Lite | Dataset and official evaluator | Sandboxed code execution |
| Code regression | HumanEval, MBPP | Complete datasets | Sandboxed Python execution |
| General reasoning | GSM8K, MMLU-Pro | Complete datasets | Fixed prompt and decoding protocol |
| General agent | GAIA | Manifest entry only | Hugging Face license acceptance, browser/search/file tools |
| Environment agent | AgentBench | Official repository | Multiple Docker/services; evaluate selected domains first |

## Local preparation

```powershell
python scripts/prepare_eval_suite.py --download
```

Re-running the command updates Git repositories to their current upstream HEAD
and records exact commits and Hugging Face revisions. Preserve the generated
manifest with every reported result.

## Transfer to the job

Prefer transferring one compressed archive so that many small Git and metadata
files do not dominate SSH overhead:

```powershell
tar -caf agentic_eval_suite_20260610.tar.zst datasets/eval_suite
scp -F .ssh-bifrost-config agentic_eval_suite_20260610.tar.zst bifrost-agentic-rl:/workspace/yans2@xiaopeng.com/agentic_rl/datasets/
```

If Windows `tar` does not support zstd, use gzip:

```powershell
tar -czf agentic_eval_suite_20260610.tar.gz datasets/eval_suite
```

Do not include GAIA files in a generally shared archive. Its validation and
test data are gated and subject to non-redistribution terms.

## SWE-bench staging

The local download contains task metadata and the official harness, not the
full Docker image cache. Prepare the execution environment on the Linux job:

1. Confirm Docker access and at least 150 GB free storage for broad Verified
   coverage. A small smoke subset can run with much less.
2. Start with SWE-bench Lite or 10-20 stratified Verified instances.
3. Generate patches with a fixed agent scaffold, token budget, temperature,
   timeout, and retry count.
4. Run the official harness in isolated containers and retain per-instance
   predictions, logs, image revision, and resolved status.
5. Expand only after the smoke subset has no infrastructure failures.

SWE-bench measures the model plus the agent scaffold. Results from different
scaffolds, budgets, tool interfaces, or retry policies are not directly
comparable.

## Common pitfalls

- Do not compare internal xLAM exact match with official BFCL scores.
- Pin LiveCodeBench by revision and date range because it is continuously
  updated.
- Keep generation and scoring separate so decoding changes do not silently
  alter the benchmark.
- Run generated code only in resource-limited containers without host network
  or writable host mounts.
- Separate model failures from infrastructure failures in tau-bench,
  AgentBench, GAIA, and SWE-bench.
- Treat chat template, tool schema serialization, max tokens, retries, and
  reasoning mode as part of the evaluation protocol.
- Never train on benchmark test prompts, gold patches, hidden tests, or
  evaluation trajectories derived from them.
