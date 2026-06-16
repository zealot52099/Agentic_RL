"""Tests for tool schema and reward functions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.tool_schema import TOOL_LIBRARY, validate_tool_call, get_all_openai_schemas
from src.data.reward_funcs import extract_tool_calls, compute_tool_reward


def test_tool_library():
    """Verify all tools have valid schemas."""
    assert len(TOOL_LIBRARY) >= 17
    for name, tool in TOOL_LIBRARY.items():
        assert tool.name == name
        assert tool.description
        assert len(tool.parameters) > 0


def test_openai_format():
    """Verify OpenAI format conversion."""
    schemas = get_all_openai_schemas()
    assert len(schemas) == len(TOOL_LIBRARY)
    for s in schemas:
        assert s["type"] == "function"
        assert "function" in s
        assert "name" in s["function"]
        assert "parameters" in s["function"]


def test_validate_tool_call_valid():
    is_valid, err = validate_tool_call("web_search", {"query": "test"})
    assert is_valid, f"Expected valid, got: {err}"


def test_validate_tool_call_missing_param():
    is_valid, err = validate_tool_call("web_search", {})
    assert not is_valid


def test_validate_tool_call_unknown():
    is_valid, err = validate_tool_call("nonexistent_tool", {})
    assert not is_valid


def test_extract_tool_calls_standard():
    text = '{"tool_calls": [{"name": "web_search", "arguments": {"query": "AI"}}]}'
    calls = extract_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "web_search"


def test_extract_tool_calls_direct():
    text = '{"name": "calculate", "arguments": {"expression": "2+2"}}'
    calls = extract_tool_calls(text)
    assert len(calls) >= 1


def test_compute_tool_reward_perfect():
    completion = '{"tool_calls": [{"name": "web_search", "arguments": {"query": "AI news"}}]}'
    r = compute_tool_reward(
        completion=completion,
        expected_tools=["web_search"],
        expected_params=[{"query": "AI news"}],
        expected_outcomes=["success"],
    )
    assert r["tool_call_accuracy"] > 0, f"Expected accuracy > 0, got {r}"
    assert r["json_validity"] > 0
    assert r["total"] > 0


def test_compute_tool_reward_bad_json():
    completion = "I'll search for that... let me think..."
    r = compute_tool_reward(
        completion=completion,
        expected_tools=["web_search"],
    )
    assert r["json_validity"] == 0.0
    assert r["total"] < 0.5


if __name__ == "__main__":
    test_tool_library()
    test_openai_format()
    test_validate_tool_call_valid()
    test_validate_tool_call_missing_param()
    test_validate_tool_call_unknown()
    test_extract_tool_calls_standard()
    test_extract_tool_calls_direct()
    test_compute_tool_reward_perfect()
    test_compute_tool_reward_bad_json()
    print("All tests passed!")
