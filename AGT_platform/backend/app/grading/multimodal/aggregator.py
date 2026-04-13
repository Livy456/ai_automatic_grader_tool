"""
Aggregate samples → chunk; chunks → assignment.
"""

from __future__ import annotations

import statistics
from typing import Sequence

from .entropy import criterion_disagreement_max
from .semantic_confidence import aggregate_assignment_confidence, summarize_chunk_confidence_from_counts
from .schemas import (
    AssignmentGradeResult,
    ChunkGradeOutcome,
    MultimodalGradingConfig,
    ParsedChunkGrade,
    ReviewStatus,
    SampledChunkGrade,
)


def _pick_representative_justifications(
    valid_parsed: list[SampledChunkGrade],
    consensus_score: float,
) -> tuple[dict[str, str], str]:
    """Select justifications from the sample closest to the consensus score.

    Returns (criterion_name -> justification, confidence_note).
    """
    if not valid_parsed:
        return {}, ""
    best = min(
        valid_parsed,
        key=lambda s: abs((s.parsed.normalized_score if s.parsed else 0.0) - consensus_score),
    )
    p = best.parsed
    if p is None:
        return {}, ""
    just_map: dict[str, str] = {}
    for i, cs in enumerate(p.criterion_scores):
        if i < len(p.criterion_justifications):
            just_map[cs.name] = p.criterion_justifications[i]
        else:
            just_map[cs.name] = ""
    return just_map, p.confidence_note


def consensus_normalized_score(
    scores: Sequence[float],
    *,
    mode: str = "mean",
) -> float:
    s = [float(x) for x in scores if x is not None]
    if not s:
        return 0.0
    if mode == "median":
        return float(statistics.median(s))
    return float(sum(s) / len(s))


def criterion_ratios(parsed: ParsedChunkGrade) -> dict[str, float]:
    out: dict[str, float] = {}
    for c in parsed.criterion_scores:
        denom = c.max_points if c.max_points else 1.0
        out[c.name] = max(0.0, min(1.0, float(c.score) / float(denom)))
    return out


def aggregate_chunk_samples(
    chunk_id: str,
    samples: list[SampledChunkGrade],
    *,
    cluster_counts: dict[str, int],
    cfg: MultimodalGradingConfig,
    question_point_weight: float = 1.0,
) -> ChunkGradeOutcome:
    valid_norms = [
        s.parsed.normalized_score
        for s in samples
        if s.parse_ok and s.parsed is not None
    ]
    estimate = consensus_normalized_score(
        valid_norms,
        mode=cfg.chunk_score_aggregator,
    )

    valid_parsed = [
        s for s in samples if s.parse_ok and s.parsed is not None
    ]
    crit_maps = [criterion_ratios(s.parsed) for s in valid_parsed]  # type: ignore[arg-type]
    # mean ratio per criterion name
    from collections import defaultdict

    acc: dict[str, list[float]] = defaultdict(list)
    for m in crit_maps:
        for k, v in m.items():
            acc[k].append(v)
    consensus_crit = {k: sum(vs) / len(vs) for k, vs in acc.items()}

    justifications, confidence_note = _pick_representative_justifications(
        valid_parsed, estimate,
    )

    n = max(1, len(samples))
    parse_fail_rate = sum(1 for s in samples if not s.parse_ok) / n
    review_flag_rate = sum(
        1 for s in samples if s.parse_ok and s.parsed and s.parsed.review_flag
    ) / max(1, len([s for s in samples if s.parse_ok]))

    total_draws = sum(cluster_counts.values()) or 1
    dist = {k: v / total_draws for k, v in cluster_counts.items()}

    conf_state = summarize_chunk_confidence_from_counts(dict(cluster_counts))

    aux = {
        "score_std_across_samples": float(
            statistics.pstdev(valid_norms) if len(valid_norms) > 1 else 0.0
        ),
        "criterion_disagreement_max": criterion_disagreement_max(crit_maps),
        "parse_fail_rate": parse_fail_rate,
        "review_flag_rate": review_flag_rate,
        "question_point_weight": float(question_point_weight),
        "criterion_justifications": justifications,
        "confidence_note": confidence_note,
    }

    return ChunkGradeOutcome(
        chunk_id=chunk_id,
        normalized_score_estimate=estimate,
        semantic_entropy_nats=float(conf_state["semantic_entropy_nats"]),
        ai_confidence=float(conf_state["ai_confidence"]),
        entropy_max_reference_nats=float(conf_state["entropy_max_reference_nats"]),
        cluster_counts=dict(cluster_counts),
        cluster_distribution=dist,
        samples=samples,
        criterion_consensus=consensus_crit,
        auxiliary=aux,
        review_status=ReviewStatus.AUTO_ACCEPTED,
        review_reasons=[],
        stage_artifacts={},
    )


def aggregate_assignment(
    assignment_id: str,
    student_id: str,
    chunks: list[ChunkGradeOutcome],
    *,
    weights: dict[str, float] | None = None,
) -> AssignmentGradeResult:
    wmap = dict(weights or {})
    if not chunks:
        return AssignmentGradeResult(
            assignment_id=assignment_id,
            student_id=student_id,
            assignment_normalized_score=0.0,
            assignment_ai_confidence=0.0,
            chunk_results=[],
            review_status=ReviewStatus.FLAGGED,
            review_reasons=["no_chunks"],
        )

    def _chunk_weight(c: ChunkGradeOutcome) -> float:
        if c.chunk_id in wmap:
            return float(wmap[c.chunk_id])
        aux = c.auxiliary or {}
        return float(aux.get("question_point_weight", 1.0) or 1.0)

    total_w = 0.0
    acc = 0.0
    resolved_weights: dict[str, float] = {}
    for c in chunks:
        w = _chunk_weight(c)
        resolved_weights[c.chunk_id] = w
        total_w += w
        acc += w * c.normalized_score_estimate
    assign_score = acc / total_w if total_w else 0.0

    assign_conf, conf_trace = aggregate_assignment_confidence(chunks, weights=wmap)

    reasons: list[str] = []
    statuses = [c.review_status for c in chunks]
    if any(s == ReviewStatus.ESCALATION for s in statuses):
        assign_status = ReviewStatus.ESCALATION
        reasons.append("chunk_escalation")
    elif any(s == ReviewStatus.FLAGGED for s in statuses):
        assign_status = ReviewStatus.FLAGGED
        reasons.append("chunk_flagged")
    elif any(s == ReviewStatus.CAUTION for s in statuses):
        assign_status = ReviewStatus.CAUTION
        reasons.append("chunk_caution")
    else:
        assign_status = ReviewStatus.AUTO_ACCEPTED

    stage = {
        "assignment_confidence_trace": conf_trace,
        "chunk_weights_resolved": resolved_weights,
    }

    return AssignmentGradeResult(
        assignment_id=assignment_id,
        student_id=student_id,
        assignment_normalized_score=assign_score,
        assignment_ai_confidence=assign_conf,
        chunk_results=chunks,
        chunk_weights=resolved_weights,
        review_status=assign_status,
        review_reasons=reasons,
        stage_artifacts=stage,
    )
