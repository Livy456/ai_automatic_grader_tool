"""Stage 5: weighted aggregation and human-review routing."""

from __future__ import annotations

from typing import Any

from .numpy_ops import criteria_rows_to_arrays, weighted_mean


def weighted_overall_confidence(criteria_results: list[dict[str, Any]]) -> float:
    if not criteria_results:
        return 0.5
    w, _s, conf = criteria_rows_to_arrays(criteria_results)
    mu = weighted_mean(conf, w, default=0.5)
    return round(mu, 2)


def weighted_overall_score(criteria_results: list[dict[str, Any]]) -> float:
    if not criteria_results:
        return 0.0
    w, scores, _c = criteria_rows_to_arrays(criteria_results)
    return round(weighted_mean(scores, w, default=0.0), 2)


def should_route_human_review(
    criteria_results: list[dict[str, Any]],
    *,
    confidence_threshold: float,
    near_boundary_points: float,
    cap_points: float = 100.0,
) -> tuple[bool, list[str]]:
    """
    Return (needs_review, reasons).
    """
    reasons: list[str] = []
    for c in criteria_results:
        conf = c.get("confidence")
        if conf is None:
            continue
        try:
            cf = float(conf)
        except (TypeError, ValueError):
            continue
        if cf < confidence_threshold:
            reasons.append(
                f"Low confidence on '{c.get('name','?')}': {cf}"
            )
        try:
            sc = float(c.get("score", 0))
        except (TypeError, ValueError):
            continue
        hi = float(c.get("max_points", cap_points))
        lo = float(c.get("min_points", 0))
        if hi > lo and (abs(sc - lo) <= near_boundary_points or abs(hi - sc) <= near_boundary_points):
            reasons.append(
                f"Near-boundary score on '{c.get('name','?')}': {sc}/{hi}"
            )
    flags_all: list[str] = []
    for c in criteria_results:
        fl = c.get("flags") or []
        if isinstance(fl, list):
            flags_all.extend(str(x) for x in fl)
    for f in flags_all:
        if f.upper() in ("UNCERTAIN_EVIDENCE", "CONFLICT", "MISSING_ARTIFACT"):
            if f not in reasons:
                reasons.append(f"Flag: {f}")
    return (len(reasons) > 0, reasons)

