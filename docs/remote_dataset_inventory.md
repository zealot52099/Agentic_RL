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

Generated files:

- `processed/xlam-function-calling-60k/train.jsonl`
- `processed/swe-gym-openhands-sft/train.jsonl`
- `manifests/datasets.json`

Rebuild command on the job:

```bash
python3 /workspace/yans2@xiaopeng.com/agentic_rl/datasets/scripts/prepare_remote_datasets.py
```

Hugging Face is not directly reachable from this job as of 2026-06-09. The two
Parquet files were downloaded locally, SHA256 checked, and transferred over SSH.
