#!/usr/bin/env python3
"""Download and inventory reproducible agent/model evaluation assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


HF_DATASETS = [
    {
        "id": "SWE-bench/SWE-bench_Verified",
        "group": "software_engineering",
        "purpose": "Primary 500-task human-validated SWE-bench evaluation set",
    },
    {
        "id": "SWE-bench/SWE-bench_Lite",
        "group": "software_engineering",
        "purpose": "Lower-cost SWE-bench development and regression set",
    },
    {
        "id": "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
        "group": "tool_calling",
        "purpose": "Official BFCL question and function data",
    },
    {
        "id": "google/IFEval",
        "group": "instruction_following",
        "purpose": "Official verifiable instruction-following prompts",
    },
    {
        "id": "livecodebench/code_generation_lite",
        "group": "code",
        "purpose": "Current lightweight LiveCodeBench code-generation data",
        "allow_patterns": [
            "README.md",
            "code_generation_lite.py",
            "test6.jsonl",
            "images/*",
        ],
    },
    {
        "id": "openai/openai_humaneval",
        "group": "code",
        "purpose": "HumanEval functional code-generation regression set",
    },
    {
        "id": "google-research-datasets/mbpp",
        "group": "code",
        "purpose": "MBPP Python code-generation regression set",
    },
    {
        "id": "openai/gsm8k",
        "group": "reasoning",
        "purpose": "Full GSM8K benchmark data",
    },
    {
        "id": "TIGER-Lab/MMLU-Pro",
        "group": "knowledge_reasoning",
        "purpose": "Full MMLU-Pro benchmark data",
    },
]

GIT_REPOSITORIES = [
    {
        "url": "https://github.com/swe-bench/SWE-bench.git",
        "directory": "SWE-bench",
        "group": "software_engineering",
        "purpose": "Official SWE-bench harness and Docker evaluation tooling",
    },
    {
        "url": "https://github.com/sierra-research/tau2-bench.git",
        "directory": "tau2-bench",
        "group": "conversational_agent",
        "purpose": "Official tau-bench tasks, environments, and evaluator",
    },
    {
        "url": "https://github.com/THUDM/AgentBench.git",
        "directory": "AgentBench",
        "group": "environment_agent",
        "purpose": "AgentBench tasks and environment definitions",
    },
    {
        "url": "https://github.com/LiveCodeBench/LiveCodeBench.git",
        "directory": "LiveCodeBench",
        "group": "code",
        "purpose": "Official LiveCodeBench evaluator",
    },
]

SPARSE_GIT_REPOSITORIES = [
    {
        "url": "https://github.com/ShishirPatil/gorilla.git",
        "directory": "gorilla-bfcl",
        "paths": ["berkeley-function-call-leaderboard"],
        "group": "tool_calling",
        "purpose": "Official BFCL V4 evaluator and model handlers",
    },
    {
        "url": "https://github.com/google-research/google-research.git",
        "directory": "google-ifeval",
        "paths": ["instruction_following_eval"],
        "group": "instruction_following",
        "purpose": "Official IFEval evaluator",
    },
]

GATED_DATASETS = [
    {
        "id": "gaia-benchmark/GAIA",
        "group": "general_agent",
        "status": "requires_manual_license_acceptance",
        "purpose": "Tool-augmented general assistant benchmark",
        "note": "Do not redistribute. Download only after accepting the dataset terms.",
    }
]


def run(command: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stdout.strip()}"
        )
    return result.stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clone_or_update(repo: dict, destination: Path, sparse: bool = False) -> str:
    if (destination / ".git").exists():
        run(["git", "fetch", "--depth", "1", "origin"], cwd=destination)
        run(["git", "reset", "--hard", "origin/HEAD"], cwd=destination)
    else:
        command = ["git", "clone", "--depth", "1"]
        if sparse:
            command.extend(["--filter=blob:none", "--sparse"])
        command.extend([repo["url"], str(destination)])
        run(command)
    if sparse:
        run(["git", "sparse-checkout", "set", *repo["paths"]], cwd=destination)
    return run(["git", "rev-parse", "HEAD"], cwd=destination)


def file_inventory(root: Path) -> tuple[list[dict], int]:
    files = []
    total_bytes = 0
    if not root.exists():
        return files, total_bytes
    for path in sorted(root.rglob("*")):
        if (
            not path.is_file()
            or ".git" in path.parts
            or ".huggingface" in path.parts
        ):
            continue
        size = path.stat().st_size
        total_bytes += size
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": size,
                "sha256": sha256(path),
            }
        )
    return files, total_bytes


def prepare_hf_datasets(root: Path, download: bool) -> list[dict]:
    api = HfApi()
    records = []
    for item in HF_DATASETS:
        repo_id = item["id"]
        destination = root / repo_id.replace("/", "__")
        record = dict(item)
        try:
            info = api.dataset_info(repo_id)
            record["revision"] = info.sha
            if download:
                snapshot_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    revision=info.sha,
                    local_dir=destination,
                    allow_patterns=item.get("allow_patterns"),
                )
            files, total_bytes = file_inventory(destination)
            record.update(
                {
                    "status": "downloaded" if files else "not_downloaded",
                    "local_path": str(destination.resolve()),
                    "bytes": total_bytes,
                    "files": files,
                }
            )
        except Exception as exc:  # Keep the manifest useful after partial failure.
            record.update({"status": "error", "error": str(exc)})
        records.append(record)
    return records


def prepare_git_repositories(root: Path, download: bool) -> list[dict]:
    records = []
    for item in GIT_REPOSITORIES + SPARSE_GIT_REPOSITORIES:
        destination = root / item["directory"]
        record = dict(item)
        try:
            if download:
                revision = clone_or_update(
                    item,
                    destination,
                    sparse=item in SPARSE_GIT_REPOSITORIES,
                )
            elif (destination / ".git").exists():
                revision = run(["git", "rev-parse", "HEAD"], cwd=destination)
            else:
                revision = None
            files, total_bytes = file_inventory(destination)
            record.update(
                {
                    "revision": revision,
                    "status": "downloaded" if files else "not_downloaded",
                    "local_path": str(destination.resolve()),
                    "bytes": total_bytes,
                    "files": files,
                }
            )
        except Exception as exc:
            record.update({"status": "error", "error": str(exc)})
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("datasets/eval_suite"),
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download/update assets before generating the manifest.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the output root first. Use only for a fresh reproducible snapshot.",
    )
    args = parser.parse_args()

    root = args.output_root.resolve()
    if args.clean and root.exists():
        shutil.rmtree(root)
    hf_root = root / "huggingface"
    git_root = root / "github"
    manifest_root = root / "manifests"
    for path in (hf_root, git_root, manifest_root):
        path.mkdir(parents=True, exist_ok=True)

    manifest = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "download_requested": args.download,
        "huggingface_datasets": prepare_hf_datasets(hf_root, args.download),
        "git_repositories": prepare_git_repositories(git_root, args.download),
        "gated_datasets": GATED_DATASETS,
        "execution_assets": {
            "swe_bench_docker_images": {
                "status": "not_downloaded",
                "reason": (
                    "Images are large and platform-specific; build/pull per instance "
                    "on the Linux GPU job."
                ),
            }
        },
    }
    manifest["total_bytes"] = sum(
        item.get("bytes", 0)
        for section in ("huggingface_datasets", "git_repositories")
        for item in manifest[section]
    )
    manifest_path = manifest_root / "eval_suite_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(
        {
            "manifest": str(manifest_path),
            "total_bytes": manifest["total_bytes"],
            "errors": [
                {"id": item.get("id", item.get("directory")), "error": item["error"]}
                for section in ("huggingface_datasets", "git_repositories")
                for item in manifest[section]
                if item["status"] == "error"
            ],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
