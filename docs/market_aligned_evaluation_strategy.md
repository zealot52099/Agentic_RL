# Market-Aligned Evaluation Strategy

## Result labels

- `OUR-RERUN`: model trained by this project and evaluated by our pinned runner.
- `OFFICIAL-RERUN`: official open weights evaluated by the same pinned runner.
- `VENDOR-REPORTED`: score copied from an official model card or leaderboard.
- `INTERNAL-PROBE`: project-only regression set; never compare numerically with
  an official leaderboard.

Only `OUR-RERUN` versus `OFFICIAL-RERUN` is a strict apples-to-apples model
comparison. `VENDOR-REPORTED` scores are market targets, not controlled
baselines.

## Primary scorecard

| Capability | Primary benchmark | Development benchmark |
|---|---|---|
| Instruction following | Official IFEval strict prompt accuracy | Internal format checks |
| Function calling | BFCL V4 overall and category scores | Internal xLAM probe |
| Multi-turn agent | tau2-bench retail/airline/telecom | Internal tool trajectories |
| Math | Full GSM8K with pinned few-shot protocol | Fixed 256 regression subset |
| Knowledge/reasoning | Full MMLU-Pro with pinned protocol | Fixed 256 direct-logprob subset |
| Code generation | HumanEval+, MBPP+, LiveCodeBench date slice | Syntax and unit-test probes |
| Software engineering | SWE-bench Verified resolved rate | SWE-bench Lite smoke subset |
| Text-to-SQL | WikiSQL and Spider/BIRD execution metrics | Fixed 256 WikiSQL probe |

## Optimization gates

Training runs should not be promoted from development to expensive official
evaluation unless they:

1. improve the target internal probe by at least one percentage point;
2. avoid more than a one-point regression on IFEval, GSM8K, and MMLU-Pro;
3. retain valid output/tool syntax above 99%;
4. reproduce on at least two seeds for changes smaller than two points.

The initial market targets for a 3B-4B model are:

- IFEval: match the official same-family instruct checkpoint under our runner;
- BFCL V4: first match Qwen3-4B-Instruct-2507, then target the best open model
  below 5B in the pinned leaderboard snapshot;
- GSM8K and MMLU-Pro: remain within two points of the official same-size
  instruct checkpoint while improving agent metrics;
- SWE-bench Verified: report the agent scaffold, token budget, retries, tools,
  and container revision with the resolved percentage.

## Current limitations

- The previously uploaded Hugging Face BFCL snapshot contains v3 data. BFCL V4
  evaluator and data are pinned separately from the official Gorilla repo.
- The current job has no Docker command, so SWE-bench resolved scores cannot be
  produced there until a Docker-capable worker or preconfigured execution
  service is available.
- Existing xLAM, fixed GSM8K, fixed MMLU-Pro, and fixed WikiSQL scores remain
  `INTERNAL-PROBE` results.
