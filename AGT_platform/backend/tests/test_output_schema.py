"""Tests for ``grading/output_schema.py`` validation."""

import unittest

from app.grading.output_schema import (
    GradingOutputValidationError,
    coerce_grading_output_shape,
    validate_grading_output,
)


class OutputSchemaTests(unittest.TestCase):
    def test_minimal_valid(self) -> None:
        d = {
            "overall": {"score": 80, "confidence": 0.8, "summary": "ok"},
            "criteria": [
                {
                    "name": "A",
                    "score": 40,
                    "confidence": 0.8,
                    "rationale": "r",
                    "evidence": {"quotes": ["x"], "notes": "n"},
                }
            ],
            "flags": [],
            "_model_used": "ollama:x",
        }
        out = validate_grading_output(d)
        self.assertEqual(out["overall"]["score"], 80.0)
        self.assertEqual(out["criteria"][0]["name"], "A")

    def test_criterion_alias_max_score(self) -> None:
        d = {
            "overall": {"score": 1, "confidence": 0.5, "summary": ""},
            "criteria": [
                {"criterion": "Q1", "max_score": 10, "score": 5, "confidence": 0.7}
            ],
        }
        out = validate_grading_output(d)
        self.assertEqual(out["criteria"][0]["name"], "Q1")
        self.assertEqual(out["criteria"][0]["max_points"], 10.0)

    def test_reject_missing_overall(self) -> None:
        with self.assertRaises(GradingOutputValidationError):
            validate_grading_output({"criteria": []})

    def test_coerce_scalar_overall_then_validates(self) -> None:
        d = {
            "overall": 85.0,
            "criteria": [
                {
                    "name": "A",
                    "score": 40,
                    "max_points": 50,
                    "confidence": 0.8,
                    "rationale": "r",
                    "evidence": {"quotes": [], "notes": ""},
                }
            ],
            "flags": [],
        }
        coerce_grading_output_shape(d)
        out = validate_grading_output(d)
        self.assertEqual(out["overall"]["score"], 85.0)

    def test_coerce_nested_grading(self) -> None:
        d = {
            "grading": {
                "overall": {"score": 10, "confidence": 0.9, "summary": "ok"},
                "criteria": [],
                "flags": [],
            }
        }
        coerce_grading_output_shape(d)
        validate_grading_output(d)

    def test_coerce_overall_from_criteria(self) -> None:
        d = {
            "criteria": [
                {
                    "name": "A",
                    "score": 8,
                    "max_points": 10,
                    "confidence": 0.7,
                    "rationale": "r",
                    "evidence": {"quotes": [], "notes": ""},
                },
                {
                    "name": "B",
                    "score": 5,
                    "max_points": 10,
                    "confidence": 0.7,
                    "rationale": "r",
                    "evidence": {"quotes": [], "notes": ""},
                },
            ],
            "flags": [],
        }
        coerce_grading_output_shape(d)
        out = validate_grading_output(d)
        self.assertIsInstance(out["overall"]["score"], float)


if __name__ == "__main__":
    unittest.main()
