# Remote Agentic RL Dataset Inventory

Remote root:

`/workspace/yans2@xiaopeng.com/agentic_rl/datasets`

Prepared sources:

| Dataset | Purpose | Size / records | License |
|---|---|---:|---|
| xLAM function-calling 60k parsed | Function-calling SFT | 60,000 | CC-BY-4.0 |
| SWE-Gym OpenHands SFT trajectories | Terminal/code-agent SFT | 491 | MIT |
| tau2-bench | Multi-turn tools, policy and environment evaluation | Upstream task sets | Upstream |
| AgentTuning / AgentBench assets | Agent prompts, tasks and evaluation assets | Upstream repository | Upstream |
| WikiSQL | Text-to-SQL generation and SQLite execution evaluation | 15,878 test queries; fixed 256 probe | BSD-3-Clause |

Generated files:

- `processed/xlam-function-calling-60k/train.jsonl`
- `processed/swe-gym-openhands-sft/train.jsonl`
- `eval/wikisql/test_256.jsonl`
- `manifests/datasets.json`

WikiSQL evaluation:

- Official archive SHA256:
  `755c728ab188e364575705c8641f3fafd86fb089cb8b08e8c03f01832aae0881`
- Fixed probe SHA256:
  `4c3071d365c92f2af872d1c6bbbd62f1a8e24121dafbf39297e839e138e20da1`
- Metrics: SQL extraction rate, SQLite execution rate, execution result exact
  match, and normalized SQL exact match.
- The evaluator opens the database read-only, accepts only `SELECT`, and applies
  a two-second execution timeout per query.

Rebuild command on the job:

```bash
python3 /workspace/yans2@xiaopeng.com/agentic_rl/datasets/scripts/prepare_remote_datasets.py
```

Hugging Face is not directly reachable from this job as of 2026-06-09. The two
Parquet files were downloaded locally, SHA256 checked, and transferred over SSH.
