"""
Semantic entropy from empirical cluster distribution over model samples.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def semantic_entropy_from_cluster_counts(counts: dict[str, int]) -> float:
    """
    SE_hat = - sum_c p_hat(c) log p_hat(c), natural log, p_hat = count / total.
    """
    if not counts:
        return 0.0
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    p = np.array([counts[k] for k in sorted(counts.keys())], dtype=np.float64)
    p = p / float(total)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log(p), 0.0)
    return float(-np.sum(terms))


def score_variance(normalized_scores: Sequence[float]) -> float:
    a = np.asarray(list(normalized_scores), dtype=np.float64)
    if a.size == 0:
        return 0.0
    return float(np.var(a))


def criterion_disagreement_max(
    samples: list[dict[str, float]],
) -> float:
    """
    ``samples`` = list of maps criterion_name -> score ratio [0,1].
    Return max over criteria of (max - min) ratio.
    """
    if not samples:
        return 0.0
    from collections import defaultdict

    acc: dict[str, list[float]] = defaultdict(list)
    for s in samples:
        for k, v in s.items():
            try:
                acc[k].append(float(v))
            except (TypeError, ValueError):
                continue
    spread = 0.0
    for vals in acc.values():
        if len(vals) < 2:
            continue
        a = np.asarray(vals, dtype=np.float64)
        spread = max(spread, float(np.max(a) - np.min(a)))
    return spread
