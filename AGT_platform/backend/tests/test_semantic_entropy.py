"""Unit tests for semantic entropy helpers (no LLM)."""

import unittest

from app.grading.multimodal.sample_semantic_entropy import (
    aggregate_grading_json_samples,
    confidence_from_entropy_natural,
    grading_semantic_fingerprint,
    semantic_entropy_by_model,
    semantic_entropy_natural,
)


class SemanticEntropyTests(unittest.TestCase):
    def test_identical_samples_entropy_zero(self) -> None:
        s = {
            "overall": {"score": 80, "confidence": 0.8, "summary": "ok"},
            "criteria": [{"name": "A", "score": 40, "confidence": 0.8, "rationale": "x"}],
            "flags": [],
        }
        fps = [grading_semantic_fingerprint(s), grading_semantic_fingerprint(s)]
        h, clusters = semantic_entropy_natural(fps)
        self.assertAlmostEqual(h, 0.0, places=5)
        self.assertEqual(clusters, 1)
        self.assertAlmostEqual(confidence_from_entropy_natural(h), 1.0, places=5)

    def test_two_clusters_entropy_positive(self) -> None:
        a = {
            "overall": {"score": 80, "confidence": 0.8, "summary": "good"},
            "criteria": [{"name": "A", "score": 40, "confidence": 0.8, "rationale": "x"}],
            "flags": [],
        }
        b = {
            "overall": {"score": 20, "confidence": 0.5, "summary": "bad"},
            "criteria": [{"name": "A", "score": 10, "confidence": 0.5, "rationale": "y"}],
            "flags": [],
        }
        fps = [grading_semantic_fingerprint(a), grading_semantic_fingerprint(b)]
        h, clusters = semantic_entropy_natural(fps)
        self.assertGreater(h, 0.0)
        self.assertEqual(clusters, 2)
        self.assertLess(confidence_from_entropy_natural(h), 1.0)

    def test_aggregate_mean(self) -> None:
        samples = [
            {
                "overall": {"score": 60, "confidence": 0.6, "summary": "a"},
                "criteria": [{"name": "P", "score": 30, "confidence": 0.6, "rationale": "r"}],
                "flags": [],
            },
            {
                "overall": {"score": 80, "confidence": 0.8, "summary": "b"},
                "criteria": [{"name": "P", "score": 40, "confidence": 0.8, "rationale": "s"}],
                "flags": ["x"],
            },
        ]
        merged = aggregate_grading_json_samples(samples)
        self.assertEqual(merged["overall"]["score"], 70.0)
        self.assertEqual(merged["overall"]["confidence"], 0.7)
        self.assertEqual(merged["criteria"][0]["score"], 35.0)
        self.assertIn("x", merged["flags"])

    def test_semantic_entropy_by_model(self) -> None:
        s_same = {
            "overall": {"score": 80, "confidence": 0.8, "summary": "ok"},
            "criteria": [{"name": "A", "score": 40, "confidence": 0.8}],
            "flags": [],
        }
        s_diff = {
            "overall": {"score": 10, "confidence": 0.5, "summary": "no"},
            "criteria": [{"name": "A", "score": 5, "confidence": 0.5}],
            "flags": [],
        }
        labeled = [(s_same, "m1"), (s_same, "m1"), (s_diff, "m2"), (s_same, "m2")]
        by_m = semantic_entropy_by_model(labeled)
        self.assertAlmostEqual(float(by_m["m1"]["semantic_entropy"]), 0.0, places=4)
        self.assertGreater(float(by_m["m2"]["semantic_entropy"]), 0.01)


if __name__ == "__main__":
    unittest.main()
