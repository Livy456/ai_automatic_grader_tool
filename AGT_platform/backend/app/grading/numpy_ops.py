"""
Vectorized numerics for grading (weighted averages, entropy, row-wise extraction).

Uses NumPy for batch operations on modest arrays (criteria rows, samples, hash embeddings).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def weighted_mean(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    default: float = 0.0,
) -> float:
    """``sum(w*v)/sum(w)`` for entries with ``w > 0``; ``default`` if no positive weights."""
    if values.size == 0:
        return default
    mask = weights > 0
    if not np.any(mask):
        return default
    w = weights[mask]
    v = values[mask]
    return float(np.dot(w, v) / np.sum(w))


def mean_round(values: Sequence[float], *, ndigits: int = 2) -> float:
    """Mean of a sequence, rounded (empty → 0)."""
    if not values:
        return 0.0
    a = np.asarray(values, dtype=np.float64)
    return round(float(np.mean(a)), ndigits)


def entropy_natural_from_multiset_counts(counts: Sequence[int], n_total: int) -> float:
    """
    Shannon entropy (natural log) for multiset fingerprint counts.
    ``counts`` = cluster sizes; ``n_total`` = number of draws.
    """
    if n_total <= 0 or not counts:
        return 0.0
    c = np.asarray(counts, dtype=np.float64)
    p = c / float(n_total)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log(p), 0.0)
    return float(-np.sum(terms))


def confidence_from_entropy(exp_neg_h: float) -> float:
    """Clip exp(-H) to [0, 1]."""
    return float(np.clip(exp_neg_h, 0.0, 1.0))


def criteria_rows_to_arrays(
    rows: list[dict[str, Any]],
    *,
    default_confidence: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack weight, score, confidence for each criterion dict."""
    n = len(rows)
    w = np.zeros(n, dtype=np.float64)
    s = np.zeros(n, dtype=np.float64)
    c = np.full(n, default_confidence, dtype=np.float64)
    for i, row in enumerate(rows):
        w[i] = float(row.get("weight") or 0.0)
        s[i] = float(row.get("score", 0.0))
        try:
            c[i] = float(row.get("confidence", default_confidence))
        except (TypeError, ValueError):
            c[i] = default_confidence
    return w, s, c
