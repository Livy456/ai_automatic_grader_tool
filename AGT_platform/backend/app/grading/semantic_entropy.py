"""
Semantic entropy over stochastic grading samples (fingerprint clustering MVP).

Fingerprint: rounded per-criterion scores + digest of overall summary text.
Entropy: H = -sum_i p_i log(p_i) (natural log). confidence_from_entropy = exp(-H), clipped to [0, 1].
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from .numpy_ops import confidence_from_entropy, entropy_natural_from_multiset_counts


def grading_semantic_fingerprint(sample: dict[str, Any]) -> str:
    """
    Stable string key for clustering semantically similar grade JSON blobs.
    Uses criterion scores (1 decimal) plus a short hash of overall.summary.
    """
    criteria = sample.get("criteria") or []
    name_scores: dict[str, float] = {}
    for c in sorted(criteria, key=lambda x: str(x.get("name") or "")):
        name = str(c.get("name") or "")
        try:
            name_scores[name] = round(float(c.get("score", 0)), 1)
        except (TypeError, ValueError):
            name_scores[name] = 0.0
    overall = sample.get("overall") or {}
    try:
        overall_score = round(float(overall.get("score", 0)), 1)
    except (TypeError, ValueError):
        overall_score = 0.0
    summary = str(overall.get("summary") or "")[:2000]
    summary_hash = hashlib.sha256(
        summary.encode("utf-8", errors="replace")
    ).hexdigest()[:16]
    payload = {
        "criteria": name_scores,
        "overall_score": overall_score,
        "summary_hash": summary_hash,
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=True)


def semantic_entropy_natural(fingerprints: list[str]) -> tuple[float, int]:
    """
    Return (H, cluster_count) for natural-log entropy. Empty input -> (0.0, 0).
    """
    if not fingerprints:
        return 0.0, 0
    counts = Counter(fingerprints)
    n = len(fingerprints)
    h = 0.0
    for cnt in counts.values():
        p = cnt / n
        h -= p * math.log(p)
    return h, len(counts)


def confidence_from_entropy_natural(entropy: float) -> float:
    """Map entropy to [0, 1]; higher entropy -> lower confidence."""
    if entropy <= 0:
        return 1.0

    return max(0.0, min(1.0, math.exp(-entropy)))


def semantic_entropy_by_model(
    samples: list[tuple[dict[str, Any], str]],
) -> dict[str, dict[str, float | int]]:
    """
    For each model label, compute semantic entropy over that model's grading fingerprints only.
    """
    by_label: dict[str, list[str]] = defaultdict(list)
    for sample, label in samples:
        by_label[label].append(grading_semantic_fingerprint(sample))
    out: dict[str, dict[str, float | int]] = {}
    for label, fps in by_label.items():
        if not fps:
            continue
        h, clusters = semantic_entropy_natural(fps)
        out[label] = {
            "semantic_entropy": round(h, 4),
            "cluster_count": clusters,
            "sample_count": len(fps),
        }
    return out

def aggregate_grading_json_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Mean of overall.score / overall.confidence and per-criterion scores across samples.
    Same aggregation policy as multi-model averaging, but without per-model labels.
    """
    if not samples:
        raise ValueError("aggregate_grading_json_samples requires at least one sample")
    n = len(samples)
    overall_scores = np.zeros(n, dtype=np.float64)
    overall_confs = np.zeros(n, dtype=np.float64)
    summaries: list[str] = []
    for i, s in enumerate(samples):
        ov = s.get("overall") or {}
        overall_scores[i] = float(ov.get("score", 0))
        overall_confs[i] = float(ov.get("confidence", 0))
        summaries.append(str(ov.get("summary") or ""))

    criteria_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in samples:
        for c in s.get("criteria") or []:
            criteria_by_name[str(c.get("name", "unknown"))].append(c)

    merged_criteria: list[dict[str, Any]] = []
    for name in sorted(criteria_by_name.keys()):
        entries = criteria_by_name[name]
        scores = np.array(
            [float(e.get("score", 0)) for e in entries], dtype=np.float64
        )
        confs = np.array(
            [float(e.get("confidence", 0)) for e in entries], dtype=np.float64
        )
        avg_score = round(float(np.mean(scores)), 2)
        avg_conf = round(float(np.mean(confs)), 2)
        e0 = entries[0]
        max_pts = e0.get("max_points")
        if max_pts is None:
            max_pts = e0.get("max_score")
        rationales = [e.get("rationale", "") for e in entries if e.get("rationale")]
        all_quotes: list[Any] = []
        all_notes: list[str] = []
        for e in entries:
            ev = e.get("evidence", {})
            if isinstance(ev, dict):
                all_quotes.extend(ev.get("quotes", []) or [])
                if ev.get("notes"):
                    all_notes.append(str(ev["notes"]))
        merged_criteria.append(
            {
                "name": name,
                "score": avg_score,
                "max_points": max_pts,
                "confidence": avg_conf,
                "rationale": " | ".join(rationales),
                "evidence": {
                    "quotes": all_quotes,
                    "notes": " | ".join(all_notes) if all_notes else "",
                },
            }
        )

    all_flags: set[str] = set()
    for s in samples:
        all_flags.update(s.get("flags") or [])

    summary_joined = " | ".join(s for s in summaries if s)[:8000]

    return {
        "overall": {
            "score": round(float(np.mean(overall_scores)), 2),
            "confidence": round(float(np.mean(overall_confs)), 2),
            "summary": summary_joined,
        },
        "criteria": merged_criteria,
        "flags": list(all_flags),
    }
