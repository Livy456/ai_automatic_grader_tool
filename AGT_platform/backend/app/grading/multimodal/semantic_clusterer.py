"""
Semantic clustering of parsed grading outcomes (not raw LLM strings).

Minimum viable: cluster key from normalized total bin + rounded criterion scores.
"""

from __future__ import annotations

from .schemas import ParsedChunkGrade


def cluster_key_score_only(normalized: float, ndigits: int = 2) -> str:
    """Cluster by binned normalized total only."""
    return f"N:{round(float(normalized), ndigits):.2f}"


def cluster_key_pattern(parsed: ParsedChunkGrade, ndigits: int = 2) -> str:
    """
    Cluster by normalized total + discretized per-criterion scores (same order as parsed).

    Judgments that agree on totals and criterion pattern land in one cluster even if
    wording differs (wording is not hashed here—the **parsed** structure is).
    """
    base = cluster_key_score_only(parsed.normalized_score, ndigits=ndigits)
    parts = []
    for c in parsed.criterion_scores:
        g = float(getattr(c, "calibrated_credit", 0.0) or 0.0)
        if g <= 0.0 and c.max_points:
            g = c.score / c.max_points
        ratio = max(0.0, min(1.0, g))
        parts.append(f"{c.name}:{round(ratio, ndigits)}")
    pat = "|".join(parts) if parts else ""
    return f"{base}|{pat}"


def assign_cluster(parsed: ParsedChunkGrade, *, strong_pattern: bool = True) -> str:
    if strong_pattern:
        return cluster_key_pattern(parsed)
    return cluster_key_score_only(parsed.normalized_score)
