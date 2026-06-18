from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from agentic_rl.fingerprint import contamination_report, schema_fingerprint
from agentic_rl.pipeline import (
    build_dataset,
    evaluate_predictions,
    export_preferences,
    export_sft,
    synthesize_traces,
)
from agentic_rl.reward import score_tool_response
from agentic_rl.sandbox import build_office_sandbox
from agentic_rl.schema import AgentTrace, TraceValidationError


class SchemaTests(unittest.TestCase):
    def test_synthetic_traces_validate(self) -> None:
        for trace in synthesize_traces(16):
            AgentTrace.from_dict(trace)

    def test_unknown_tool_is_rejected(self) -> None:
        trace = synthesize_traces(1)[0]
        broken = copy.deepcopy(trace)
        broken["messages"][2]["tool_calls"][0]["tool_name"] = "does_not_exist"
        with self.assertRaises(TraceValidationError):
            AgentTrace.from_dict(broken)

    def test_schema_fingerprint_ignores_prose(self) -> None:
        left = {
            "type": "object",
            "description": "Old name",
            "properties": {"path": {"type": "string", "description": "A"}},
            "required": ["path"],
        }
        right = copy.deepcopy(left)
        right["description"] = "Renamed"
        right["properties"]["path"]["description"] = "B"
        self.assertEqual(schema_fingerprint(left), schema_fingerprint(right))


class SandboxTests(unittest.TestCase):
    def test_state_rolls_forward_on_success(self) -> None:
        env = build_office_sandbox()
        result = env.execute(
            "files", "write_file", {"path": "/tmp/result.txt", "content": "ok"}
        )
        self.assertTrue(result["ok"])
        self.assertEqual(env.state["files"]["/tmp/result.txt"], "ok")

    def test_permission_denial_does_not_mutate_state(self) -> None:
        env = build_office_sandbox()
        env.permissions.remove("send")
        before = copy.deepcopy(env.state)
        result = env.execute(
            "mail",
            "send_email",
            {"to": "a@example.com", "subject": "x", "body": "y"},
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "permission_denied")
        self.assertEqual(env.state, before)

    def test_planned_timeout_is_recorded(self) -> None:
        env = build_office_sandbox()
        env.failure_plan["files.read_file"] = ["timeout"]
        result = env.execute("files", "read_file", {"path": "/notes/todo.txt"})
        self.assertEqual(result["error"]["code"], "timeout")
        self.assertEqual(len(env.call_log), 1)


class RewardTests(unittest.TestCase):
    def test_format_is_a_reward_gate(self) -> None:
        verifier = {
            "expected_calls": [
                {
                    "server_id": "files",
                    "tool_name": "read_file",
                    "arguments": {"path": "/notes/todo.txt"},
                }
            ]
        }
        score = score_tool_response("not json", verifier)
        self.assertEqual(score["reward"], 0.0)
        self.assertFalse(score["format_valid"])

    def test_exact_call_gets_full_reward(self) -> None:
        calls = [
            {
                "server_id": "files",
                "tool_name": "read_file",
                "arguments": {"path": "/notes/todo.txt"},
            }
        ]
        score = score_tool_response(json.dumps(calls), {"expected_calls": calls})
        self.assertAlmostEqual(score["reward"], 1.0)
        self.assertTrue(score["success"])

    def test_no_tool_hallucination_is_penalized(self) -> None:
        response = json.dumps(
            [{"server_id": "files", "tool_name": "read_file", "arguments": {}}]
        )
        score = score_tool_response(
            response, {"decision": "no_tool", "expected_calls": []}
        )
        self.assertFalse(score["success"])
        self.assertLess(score["reward"], 0.5)


class PipelineTests(unittest.TestCase):
    def test_exports_have_expected_training_shapes(self) -> None:
        traces = synthesize_traces(8)
        sft = export_sft(traces)
        preferences = export_preferences(traces)
        self.assertTrue(all({"prompt", "completion", "verifier"} <= row.keys() for row in sft))
        self.assertTrue(all({"prompt", "chosen", "rejected"} <= row.keys() for row in preferences))

    def test_build_dataset_has_clean_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = build_dataset(Path(directory), 80, 7, 20)
            self.assertTrue(manifest["contamination"]["clean"])
            self.assertGreater(manifest["counts"]["sft"], 0)
            self.assertGreater(manifest["counts"]["ood_eval_traces"], 0)

    def test_internal_evaluation_labels_probe(self) -> None:
        traces = synthesize_traces(8)
        predictions = [
            {
                "task_id": trace["task_id"],
                "response": json.dumps(trace["verifier_results"]["expected_calls"]),
                "latency_ms": 10,
                "output_tokens": 5,
            }
            for trace in traces
        ]
        metrics = evaluate_predictions(traces, predictions)
        self.assertEqual(
            metrics["evaluation_type"], "internal_probe_not_official_benchmark"
        )
        self.assertEqual(metrics["metrics"]["task_success_rate"], 1.0)
        self.assertEqual(metrics["metrics"]["tool_hallucination_rate"], 0.0)

    def test_contamination_detects_shared_family(self) -> None:
        trace = synthesize_traces(1)[0]
        report = contamination_report([trace], [copy.deepcopy(trace)])
        self.assertFalse(report["clean"])


if __name__ == "__main__":
    unittest.main()
