import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.grading.normalization import slice_evidence_for_criterion
from app.grading.pipelines import _coerce_rubric_items, _run_staged_grading_pipeline


class _FakeStagedClient:
    """Returns deterministic JSON per staged prompt kind (no network)."""

    def chat_json(self, messages, *, temperature=0.3):
        content = messages[-1]["content"]
        if "You extract factual evidence only" in content:
            return {
                "claims": [{"text": "demo", "support": "x"}],
                "code_facts": [],
                "visualization_facts": [],
                "answers_by_question": [],
                "contradictions_spotted": [],
            }
        if "You score a single rubric criterion" in content:
            key = '{"criterion"'
            payload = json.loads(content[content.index(key) :])
            spec = payload["criterion"]
            mx = float(spec["max_points"])
            return {
                "name": spec["name"],
                "score": mx * 0.8,
                "max_points": mx,
                "min_points": float(spec["min_points"]),
                "confidence": 0.85,
                "rationale": "fake",
                "evidence": {"quotes": ["q"], "notes": "n"},
                "flags": [],
            }
        if "You review criterion scores together" in content:
            return {
                "adjustments": [],
                "new_flags": [],
                "contradictions": [],
            }
        raise AssertionError("unexpected prompt shape")


class CoerceRubricTests(unittest.TestCase):
    def test_criterion_and_max_score(self):
        rows = _coerce_rubric_items(
            [{"criterion": "Part A", "max_score": 15}, {"name": "B", "max_points": 5}]
        )
        self.assertEqual(rows[0]["name"], "Part A")
        self.assertEqual(rows[0]["max_points"], 15.0)
        self.assertEqual(rows[0]["weight"], 15.0)
        self.assertEqual(rows[1]["max_points"], 5.0)


class SliceEvidenceTests(unittest.TestCase):
    def test_truncates_large_bundle(self):
        big = {"claims": [{"text": "x" * 50000}]}
        out = slice_evidence_for_criterion(big, "n/a", max_chars=200)
        self.assertIn("truncated", out["evidence"])


class StagedPipelineTests(unittest.TestCase):
    @patch("app.grading.pipelines.build_grading_clients")
    def test_staged_runs_and_matches_schema(self, mock_clients):
        mock_clients.return_value = [(_FakeStagedClient(), "ollama:fake")]

        cfg = SimpleNamespace(
            STAGED_PROMPT_MAX_CHARS=50000,
            REVIEW_CONFIDENCE_THRESHOLD=0.72,
            REVIEW_NEAR_BOUNDARY_POINTS=2.0,
        )
        assignment = SimpleNamespace(
            title="T",
            description="Desc",
            rubric=[
                {"criterion": "A", "max_score": 10},
                {"criterion": "B", "max_score": 10},
            ],
            modality="notebook",
        )
        artifacts = {"txt": b"submission text"}
        ctx = {"modality": "notebook", "artifacts": {}, "tool_results": {}}

        result = _run_staged_grading_pipeline(
            cfg,
            assignment,
            artifacts,
            rubric_text=None,
            answer_key_text=None,
            assignment_prompt=None,
            ctx=ctx,
        )

        self.assertIn("evidence_bundle", ctx)
        self.assertEqual(len(result["criteria"]), 2)
        self.assertIn("overall", result)
        self.assertIn("score", result["overall"])
        self.assertEqual(result["_model_used"], "ollama:fake")
        self.assertEqual(result["_models_used"], ["ollama:fake"])
        self.assertEqual(result["_pipeline_meta"]["mode"], "staged")
        for c in result["criteria"]:
            self.assertIn("name", c)
            self.assertIn("score", c)
            self.assertIn("confidence", c)
            self.assertIn("evidence", c)


if __name__ == "__main__":
    unittest.main()
