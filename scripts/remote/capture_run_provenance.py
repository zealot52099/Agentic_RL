#!/usr/bin/env python3
"""Capture reproducibility metadata without storing credentials."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


SENSITIVE_MARKERS = (
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "API_KEY",
    "ACCESS_KEY",
    "CREDENTIAL",
    "COOKIE",
)


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str]) -> dict:
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "error": repr(error)}


def safe_environment() -> dict[str, str]:
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if not any(marker in key.upper() for marker in SENSITIVE_MARKERS)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--scripts", type=Path, nargs="*", default=[])
    parser.add_argument("--launch-command", default="")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = [args.dataset, args.model / "config.json", *args.scripts]
    metadata = {
        "format_version": 1,
        "captured_at_unix": time.time(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "launch_command": args.launch_command,
        "model": str(args.model),
        "dataset": str(args.dataset),
        "artifact_sha256": {str(path): sha256(path) for path in artifacts},
        "environment": safe_environment(),
        "commands": {
            "pip_freeze": command_output([sys.executable, "-m", "pip", "freeze"]),
            "torch_env": command_output(
                [sys.executable, "-m", "torch.utils.collect_env"]
            ),
            "ppu_smi": command_output(
                ["/usr/local/PPU_SDK/ppu-smi/bin/nvidia-smi"]
            ),
        },
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
