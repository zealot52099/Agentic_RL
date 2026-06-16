"""
Data Cleaning & Quality Control Pipeline for Tool-Calling Data
"""
import json
import hashlib
from typing import List, Dict, Any, Set, Tuple
from pathlib import Path
from collections import Counter


class ToolCallDataCleaner:
    """Cleans and filters tool-calling training data."""

    def __init__(self, min_tool_calls: int = 1, max_tool_calls: int = 10):
        self.min_tool_calls = min_tool_calls
        self.max_tool_calls = max_tool_calls
        self.stats = Counter()

    def _hash_sample(self, sample: Dict[str, Any]) -> str:
        """Compute a hash for deduplication."""
        content = json.dumps(sample.get("tool_calls", []), sort_keys=True)
        content += sample.get("user_prompt", "")
        return hashlib.md5(content.encode()).hexdigest()

    def filter_min_tool_calls(self, samples: List[Dict]) -> List[Dict]:
        """Remove samples with too few tool calls."""
        result = []
        for s in samples:
            n = len(s.get("tool_calls", []))
            if n >= self.min_tool_calls:
                result.append(s)
            else:
                self.stats["too_few_tool_calls"] += 1
        return result

    def filter_max_tool_calls(self, samples: List[Dict]) -> List[Dict]:
        """Remove samples with too many tool calls."""
        result = []
        for s in samples:
            n = len(s.get("tool_calls", []))
            if n <= self.max_tool_calls:
                result.append(s)
            else:
                self.stats["too_many_tool_calls"] += 1
        return result

    def filter_empty_prompts(self, samples: List[Dict]) -> List[Dict]:
        """Remove samples with empty or whitespace-only prompts."""
        result = []
        for s in samples:
            prompt = s.get("user_prompt", "").strip()
            if prompt:
                result.append(s)
            else:
                self.stats["empty_prompt"] += 1
        return result

    def filter_invalid_json_tool_calls(self, samples: List[Dict]) -> List[Dict]:
        """Remove samples where tool call arguments are not valid JSON objects."""
        result = []
        for s in samples:
            valid = True
            for tc in s.get("tool_calls", []):
                args = tc.get("arguments", {})
                if not isinstance(args, dict):
                    valid = False
                    break
                # Check for required fields
                if "name" not in tc and "tool_name" not in tc:
                    valid = False
                    break
            if valid:
                result.append(s)
            else:
                self.stats["invalid_tool_call"] += 1
        return result

    def deduplicate(self, samples: List[Dict], threshold: float = 0.85) -> List[Dict]:
        """Remove duplicate samples based on content hash."""
        seen: Set[str] = set()
        result = []
        for s in samples:
            h = self._hash_sample(s)
            if h not in seen:
                seen.add(h)
                result.append(s)
            else:
                self.stats["duplicate"] += 1
        return result

    def filter_by_difficulty_balance(self, samples: List[Dict],
                                     max_ratio: float = 0.6) -> List[Dict]:
        """Ensure no single difficulty level dominates."""
        by_diff = Counter(s.get("difficulty", "unknown") for s in samples)
        total = len(samples)
        max_allowed = int(total * max_ratio)

        counts = Counter()
        result = []
        for s in samples:
            diff = s.get("difficulty", "unknown")
            if counts[diff] < max_allowed:
                counts[diff] += 1
                result.append(s)
            else:
                self.stats[f"over_balance_{diff}"] += 1
        return result

    def filter_short_responses(self, samples: List[Dict], min_chars: int = 20) -> List[Dict]:
        """Remove samples with overly short final responses."""
        result = []
        for s in samples:
            resp = s.get("final_response", "")
            if len(resp.strip()) >= min_chars:
                result.append(s)
            else:
                self.stats["short_response"] += 1
        return result

    def clean(self, samples: List[Dict]) -> Tuple[List[Dict], Counter]:
        """Run the full cleaning pipeline."""
        self.stats = Counter()
        initial = len(samples)

        samples = self.filter_empty_prompts(samples)
        samples = self.filter_invalid_json_tool_calls(samples)
        samples = self.filter_min_tool_calls(samples)
        samples = self.filter_max_tool_calls(samples)
        samples = self.filter_short_responses(samples)
        samples = self.deduplicate(samples)
        samples = self.filter_by_difficulty_balance(samples)

        self.stats["initial"] = initial
        self.stats["final"] = len(samples)
        self.stats["removed"] = initial - len(samples)
        self.stats["retention"] = round(len(samples) / initial * 100, 1) if initial else 0

        return samples, self.stats

    def get_stats_report(self) -> str:
        """Generate a human-readable cleaning report."""
        lines = [
            "=" * 50,
            "Data Cleaning Report",
            "=" * 50,
            f"Initial samples:  {self.stats['initial']}",
            f"Final samples:    {self.stats['final']}",
            f"Removed:          {self.stats['removed']} ({self.stats['retention']}% retained)",
            "",
            "Removal reasons:",
        ]
        for reason, count in self.stats.most_common():
            if reason not in ("initial", "final", "removed", "retention"):
                lines.append(f"  - {reason}: {count}")
        return "\n".join(lines)
