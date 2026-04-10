"""
Parse model JSON into ParsedChunkGrade with normalization.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .schemas import CriterionScore, ParsedChunkGrade, RubricType

_log = logging.getLogger(__name__)


def _coerce_rubric_type(val: Any) -> RubricType | str:
    if isinstance(val, RubricType):
        return val
    s = str(val or "").strip().lower()
    for rt in RubricType:
        if rt.value == s:
            return rt
    return s or "unknown"


def parse_chunk_grade_json(raw: str) -> tuple[ParsedChunkGrade | None, list[str]]:
    warnings: list[str] = []
    s = (raw or "").strip()
    if not s:
        return None, ["empty_model_output"]
    try:
        obj, _ = json.JSONDecoder().raw_decode(s[s.find("{") :] if "{" in s else s)
    except json.JSONDecodeError as e:
        _log.debug("chunk grade JSON parse failed: %s", e)
        return None, [f"json_decode:{e}"]

    if not isinstance(obj, dict):
        return None, ["top_level_not_object"]

    crit_raw = obj.get("criterion_scores") or obj.get("criteria") or []
    if not isinstance(crit_raw, list):
        crit_raw = []

    c_scores: list[CriterionScore] = []
    for i, row in enumerate(crit_raw):
        if not isinstance(row, dict):
            warnings.append(f"criterion_{i}_not_dict")
            continue
        name = str(row.get("name") or row.get("criterion") or f"criterion_{i}")
        try:
            score = float(row.get("score", 0))
            mx = float(row.get("max_points") or row.get("max_score") or 0)
        except (TypeError, ValueError):
            warnings.append(f"criterion_{i}_non_numeric")
            score, mx = 0.0, 0.0
        try:
            w = float(row.get("weight", 1.0))
        except (TypeError, ValueError):
            w = 1.0
        c_scores.append(CriterionScore(name=name, score=score, max_points=mx, weight=w))

    just = obj.get("criterion_justifications")
    if isinstance(just, list):
        just_list = [str(x) for x in just]
    elif isinstance(just, dict):
        just_list = [f"{k}: {v}" for k, v in just.items()]
    else:
        just_list = [str(just)] if just else []

    try:
        total = float(obj.get("total_score", 0))
    except (TypeError, ValueError):
        total = sum(c.score for c in c_scores)
        warnings.append("total_score_coerced_from_sum")

    try:
        norm = float(obj.get("normalized_score", 0))
    except (TypeError, ValueError):
        mx_total = sum(c.max_points for c in c_scores) or 1.0
        norm = max(0.0, min(1.0, total / mx_total))
        warnings.append("normalized_score_derived")

    norm = max(0.0, min(1.0, norm))
    if norm > 1.0:
        norm = norm / 100.0
        warnings.append("normalized_score_treated_as_percent")

    conf = str(obj.get("confidence_note") or "").strip()
    rf = bool(obj.get("review_flag", False))

    if not c_scores and total == 0 and not warnings:
        warnings.append("no_criteria_parsed")

    parsed = ParsedChunkGrade(
        rubric_type=_coerce_rubric_type(obj.get("rubric_type")),
        criterion_scores=c_scores,
        criterion_justifications=just_list,
        total_score=total,
        normalized_score=float(norm),
        confidence_note=conf,
        review_flag=rf,
        parse_warnings=list(warnings),
    )
    return parsed, warnings
