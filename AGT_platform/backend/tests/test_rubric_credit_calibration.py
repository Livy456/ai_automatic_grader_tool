"""Tests for rubric-anchored partial credit helpers."""

import json
import unittest

from app.grading.multimodal.parser import parse_chunk_grade_json
from app.grading.multimodal.rubric_calibration import (
    DEFAULT_ANCHOR_HALFSTEPS_0_TO_4,
    anchor_map_monotone_increasing,
    ceiling_half_point_on_grid,
    compute_weighted_question_score,
    finalize_criterion_display_scores,
    get_anchor_map_for_criterion,
    interpolate_anchor_map_for_scale,
    map_raw_score_to_calibrated_credit,
    nearest_half_point_on_grid,
    validate_raw_score_increment,
)


class RubricCreditCalibrationTests(unittest.TestCase):
    def test_full_step_scores_valid(self) -> None:
        for r in (0.0, 1.0, 2.0, 3.0, 4.0):
            v = validate_raw_score_increment(r, 4.0)
            self.assertTrue(v.ok, msg=f"{r!r}")

    def test_half_point_scores_valid(self) -> None:
        for r in (0.5, 1.5, 2.5, 3.5):
            v = validate_raw_score_increment(r, 4.0)
            self.assertTrue(v.ok, msg=f"{r!r}")

    def test_invalid_decimal_rejected(self) -> None:
        v = validate_raw_score_increment(2.3, 4.0)
        self.assertFalse(v.ok)
        self.assertEqual(v.nearest_valid, 2.5)

    def test_ceiling_half_snap(self) -> None:
        self.assertEqual(ceiling_half_point_on_grid(2.28, 4.0), 2.5)
        self.assertEqual(ceiling_half_point_on_grid(2.2, 4.0), 2.5)
        self.assertEqual(ceiling_half_point_on_grid(2.0, 4.0), 2.0)
        self.assertEqual(nearest_half_point_on_grid(2.2, 4.0), 2.5)

    def test_default_anchor_table_endpoints(self) -> None:
        m = DEFAULT_ANCHOR_HALFSTEPS_0_TO_4
        self.assertEqual(m[0.0], 0.0)
        self.assertEqual(m[4.0], 1.0)
        self.assertLess(m[0.5], m[1.0])
        self.assertTrue(anchor_map_monotone_increasing(m))

    def test_interpolated_maps_monotone(self) -> None:
        for R in (1, 2, 3, 5):
            mp = interpolate_anchor_map_for_scale(float(R))
            self.assertTrue(
                anchor_map_monotone_increasing(mp),
                f"non-monotone R={R}: {mp}",
            )

    def test_map_raw_interpolation_mid(self) -> None:
        m = interpolate_anchor_map_for_scale(4.0)
        g = map_raw_score_to_calibrated_credit(1.25, m)
        self.assertGreater(g, m[1.0])
        self.assertLess(g, m[1.5])

    def test_finalize_blended_display_scores(self) -> None:
        raw_s, pts = finalize_criterion_display_scores(1.9375, 0.51875, 4.0)
        self.assertEqual(raw_s, 2.0)
        self.assertEqual(pts, 3.0)

    def test_weighted_question_score(self) -> None:
        pct, audit = compute_weighted_question_score(
            [(2.0, 1.0), (1.0, 0.5)],
            scale_to_100=True,
        )
        self.assertAlmostEqual(pct, 100.0 * (2.0 + 0.5) / 3.0, places=5)
        self.assertEqual(len(audit), 2)

    def test_custom_anchor_row(self) -> None:
        row = {
            "name": "X",
            "max_points": 2,
            "calibration_anchor_table": {0: 0.0, 1: 0.6, 2: 1.0},
        }
        m = get_anchor_map_for_criterion(row)
        self.assertTrue(anchor_map_monotone_increasing(m))


class ParseChunkGradeRawScorePolicyTests(unittest.TestCase):
    def test_regenerate_on_bad_increment(self) -> None:
        rubric_rows = [{"name": "A", "max_points": 4}]
        raw = json.dumps(
            {
                "rubric_type": "free_response",
                "criterion_scores": [
                    {
                        "name": "A",
                        "raw_score": 2.31,
                        "max_points": 4,
                        "evidence": "q",
                        "reasoning": "r",
                        "justification": "j",
                    }
                ],
                "criterion_justifications": ["j"],
                "confidence_note": "",
                "review_flag": False,
            }
        )
        parsed, warns = parse_chunk_grade_json(
            raw, rubric_rows=rubric_rows, invalid_raw_score_policy="regenerate"
        )
        self.assertIsNone(parsed)
        self.assertTrue(any("invalid_raw_score_regenerate" in w for w in warns))

    def test_nearest_half_snaps(self) -> None:
        rubric_rows = [{"name": "A", "max_points": 4}]
        raw = json.dumps(
            {
                "rubric_type": "free_response",
                "criterion_scores": [
                    {
                        "name": "A",
                        "score": 2.3,
                        "max_points": 4,
                        "evidence": "q",
                        "reasoning": "r",
                        "justification": "j",
                    }
                ],
                "criterion_justifications": ["j"],
                "confidence_note": "",
                "review_flag": False,
            }
        )
        parsed, warns = parse_chunk_grade_json(
            raw, rubric_rows=rubric_rows, invalid_raw_score_policy="nearest_half"
        )
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.criterion_scores[0].score, 2.5)
        self.assertTrue(any("raw_score_ceiled_to_half_grid" in w for w in warns))


if __name__ == "__main__":
    unittest.main()
