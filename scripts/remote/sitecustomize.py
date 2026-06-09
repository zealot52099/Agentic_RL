"""Register local Agentic RL JSONL datasets without patching TorchTitan."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import load_dataset
from torchtitan.hf_datasets import DatasetConfig
from torchtitan.hf_datasets import text_datasets


def _resolve_jsonl(path_value: str) -> list[str]:
    path = Path(path_value)
    if path.is_file():
        return [str(path)]
    files = sorted(path.glob("*.jsonl"))
    if not files:
        files = sorted(path.rglob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No JSONL files found under {path}")
    return [str(file) for file in files]


def _load_agentic_jsonl(path: str):
    return load_dataset(
        "json",
        data_files={"train": _resolve_jsonl(path)},
        split="train",
    )


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, sort_keys=True)


def _process_agentic_sample(sample: dict[str, Any]) -> str:
    messages = sample.get("messages")
    if isinstance(messages, list):
        rendered: list[str] = []
        tools = sample.get("tools")
        if tools:
            rendered.append(f"<|tools|>\n{_content_to_text(tools)}")
        for message in messages:
            if not isinstance(message, dict):
                rendered.append(_content_to_text(message))
                continue
            role = str(message.get("role", "unknown"))
            rendered.append(f"<|{role}|>\n{_content_to_text(message.get('content', ''))}")
            tool_calls = message.get("tool_calls")
            if tool_calls:
                rendered.append(
                    "<|tool_calls|>\n"
                    + json.dumps(tool_calls, ensure_ascii=False, sort_keys=True)
                )
        return "\n".join(rendered)

    for key in ("text", "content", "prompt", "instruction"):
        value = sample.get(key)
        if value:
            return _content_to_text(value)
    return json.dumps(sample, ensure_ascii=False, sort_keys=True)


text_datasets.DATASETS["agentic_jsonl"] = DatasetConfig(
    path=".",
    loader=_load_agentic_jsonl,
    sample_processor=_process_agentic_sample,
)
