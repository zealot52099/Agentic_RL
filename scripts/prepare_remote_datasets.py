#!/usr/bin/env python3
import hashlib
import json
import subprocess
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path("/workspace/yans2@xiaopeng.com/agentic_rl/datasets")
SOURCES = ROOT / "sources"
PROCESSED = ROOT / "processed"
MANIFESTS = ROOT / "manifests"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def write_jsonl_from_parquet(source: Path, target: Path) -> dict:
    table = pq.read_table(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    message_count = 0
    assistant_count = 0
    tool_call_count = 0

    with target.open("w", encoding="utf-8") as output:
        for row in table.to_pylist():
            messages = row.get("messages") or []
            message_count += len(messages)
            assistant_count += sum(m.get("role") == "assistant" for m in messages)
            tool_call_count += sum(
                len(m.get("tool_calls") or []) for m in messages
            )
            output.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "rows": table.num_rows,
        "columns": table.column_names,
        "messages": message_count,
        "assistant_messages": assistant_count,
        "tool_calls": tool_call_count,
        "source_bytes": source.stat().st_size,
        "source_sha256": sha256(source),
        "jsonl_bytes": target.stat().st_size,
        "jsonl_sha256": sha256(target),
    }


def count_json_records(path: Path) -> int | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (UnicodeDecodeError, json.JSONDecodeError, OSError):
        return None
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        for key in ("tasks", "data", "examples"):
            if isinstance(value.get(key), list):
                return len(value[key])
    return None


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    MANIFESTS.mkdir(parents=True, exist_ok=True)

    datasets = []

    xlam_source = (
        SOURCES
        / "huggingface/xlam-function-calling-60k-parsed"
        / "xlam-function-calling-60k.parquet"
    )
    xlam_target = PROCESSED / "xlam-function-calling-60k/train.jsonl"
    datasets.append(
        {
            "name": "xlam-function-calling-60k-parsed",
            "kind": "sft_function_calling",
            "source": "https://huggingface.co/datasets/minpeter/xlam-function-calling-60k-parsed",
            "source_revision": "c6a0d61d66c5ac769c5c8c7de007322fa24f1825",
            "license": "CC-BY-4.0",
            "local_source": str(xlam_source),
            "processed": str(xlam_target),
            "stats": write_jsonl_from_parquet(xlam_source, xlam_target),
        }
    )

    swe_source = (
        SOURCES
        / "huggingface/SWE-Gym-OpenHands-SFT-Trajectories"
        / "swe-gym-openhands-sft.parquet"
    )
    swe_target = PROCESSED / "swe-gym-openhands-sft/train.jsonl"
    datasets.append(
        {
            "name": "SWE-Gym-OpenHands-SFT-Trajectories",
            "kind": "sft_terminal_agent_trajectory",
            "source": "https://huggingface.co/datasets/SWE-Gym/OpenHands-SFT-Trajectories",
            "source_revision": "4aaa5a4a4b5861f4799d2336908760c190ac3b17",
            "license": "MIT",
            "local_source": str(swe_source),
            "processed": str(swe_target),
            "stats": write_jsonl_from_parquet(swe_source, swe_target),
        }
    )

    tau_path = SOURCES / "tau2-bench"
    tau_task_files = sorted(
        (tau_path / "data/tau2/domains").glob("*/tasks.json")
    )
    datasets.append(
        {
            "name": "tau2-bench",
            "kind": "environment_tasks_and_evaluation",
            "source": "https://github.com/sierra-research/tau2-bench",
            "source_revision": git_revision(tau_path),
            "license": "See upstream repository",
            "local_source": str(tau_path),
            "task_files": [
                {"path": str(path), "records": count_json_records(path)}
                for path in tau_task_files
            ],
        }
    )

    agent_tuning_path = SOURCES / "AgentTuning"
    datasets.append(
        {
            "name": "AgentTuning-AgentBench-assets",
            "kind": "agent_tasks_prompts_and_evaluation_assets",
            "source": "https://github.com/THUDM/AgentTuning",
            "source_revision": git_revision(agent_tuning_path),
            "license": "See upstream repository",
            "local_source": str(agent_tuning_path),
        }
    )

    manifest = {
        "root": str(ROOT),
        "format_version": 1,
        "datasets": datasets,
    }
    manifest_path = MANIFESTS / "datasets.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
