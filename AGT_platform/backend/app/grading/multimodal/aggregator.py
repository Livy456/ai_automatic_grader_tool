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


def _triplet_for_criterion(
    parsed: ParsedChunkGrade, name: str
) -> tuple[str, str, str]:
    """Return (justification, evidence, reasoning) for ``name`` from one parsed grade."""
    for i, cs in enumerate(parsed.criterion_scores):
        if cs.name != name:
            continue
        jt = (
            str(parsed.criterion_justifications[i])
            if i < len(parsed.criterion_justifications)
            else ""
        )
        ev = (
            str(parsed.criterion_evidence[i])
            if i < len(parsed.criterion_evidence)
            else ""
        )
        rs = (
            str(parsed.criterion_reasoning[i])
            if i < len(parsed.criterion_reasoning)
            else ""
        )
        return jt, ev, rs
    return "", "", ""


def align_criterion_text_maps_to_consensus(
    valid_parsed: list[SampledChunkGrade],
    consensus_crit: dict[str, float],
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """
    For each criterion, pick the sample whose **calibrated_credit** is closest to the
    consensus value (prefer non-empty evidence on ties). Aligns text with numeric
    consensus so strong justifications are not paired with empty ``evidence`` from an
    unrelated sample.
    """
    just_map: dict[str, str] = {}
    ev_map: dict[str, str] = {}
    reason_map: dict[str, str] = {}
    for name, target in consensus_crit.items():
        best_trip: tuple[str, str, str] = ("", "", "")
        best_key: tuple[float, int, int] | None = None
        for s in valid_parsed:
            p = s.parsed
            if p is None:
                continue
            g_here: float | None = None
            for cs in p.criterion_scores:
                if cs.name == name:
                    g_here = float(cs.calibrated_credit)
                    break
            if g_here is None:
                continue
            jt, ev, rs = _triplet_for_criterion(p, name)
            ev_n = str(ev).strip()
            key = (abs(g_here - float(target)), 0 if ev_n else 1, -len(ev_n))
            if best_key is None or key < best_key:
                best_key = key
                best_trip = (jt, ev, rs)
        just_map[name], ev_map[name], reason_map[name] = best_trip
    return just_map, ev_map, reason_map


def _backfill_empty_evidence_from_any_sample(
    valid_parsed: list[SampledChunkGrade],
    consensus_crit: dict[str, float],
    ev_map: dict[str, str],
) -> None:
    """If a criterion still has no evidence string, take the first non-empty from any sample."""
    for name in consensus_crit:
        if str(ev_map.get(name) or "").strip():
            continue
        for s in valid_parsed:
            p = s.parsed
            if p is None:
                continue
            _jt, ev, _rs = _triplet_for_criterion(p, name)
            if str(ev).strip():
                ev_map[name] = ev
                break


def _confidence_note_from_nearest_sample(
    valid_parsed: list[SampledChunkGrade],
    consensus_score: float,
) -> str:
    if not valid_parsed:
        return ""
    best = min(
        valid_parsed,
        key=lambda s: abs(
            (s.parsed.normalized_score if s.parsed else 0.0) - consensus_score
        ),
    )
    p = best.parsed
    if p is None:
        return ""
    return str(p.confidence_note or "")


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
    """Per-criterion calibrated credit in ``[0, 1]`` (rubric-anchored, not score/max)."""
    out: dict[str, float] = {}
    for c in parsed.criterion_scores:
        g = float(getattr(c, "calibrated_credit", 0.0) or 0.0)
        if g <= 0.0 and c.max_points:
            g = max(0.0, min(1.0, float(c.score) / float(c.max_points)))
        out[c.name] = max(0.0, min(1.0, g))
    return out


def aggregate_chunk_samples(
    chunk_id: str,
    samples: list[SampledChunkGrade],
    *,
    cluster_counts: dict[str, int],
    cfg: MultimodalGradingConfig,
    rubric_fallback_names: list[str] | None = None,
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
    fb = [str(n).strip() for n in (rubric_fallback_names or []) if str(n).strip()]
    if not consensus_crit and fb:
        consensus_crit = {n: 0.0 for n in fb}

    acc_raw: dict[str, list[float]] = defaultdict(list)
    for s in valid_parsed:
        p = s.parsed
        if p is None:
            continue
        for cs in p.criterion_scores:
            acc_raw[cs.name].append(float(cs.score))
    consensus_raw = {k: sum(vs) / len(vs) for k, vs in acc_raw.items()}
    if not consensus_raw and fb:
        consensus_raw = {n: 0.0 for n in fb}

    justifications, evidence, reasoning = align_criterion_text_maps_to_consensus(
        valid_parsed, consensus_crit
    )
    _backfill_empty_evidence_from_any_sample(
        valid_parsed, consensus_crit, evidence
    )
    confidence_note = _confidence_note_from_nearest_sample(valid_parsed, estimate)
    if not valid_parsed:
        confidence_note = (
            f"[{chunk_id}] No valid model JSON (all samples failed: timeouts, HTTP errors, "
            "or parse errors). Scores default to 0."
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
        "criterion_justifications": justifications,
        "criterion_evidence": evidence,
        "criterion_reasoning": reasoning,
        "confidence_note": confidence_note,
        "criterion_raw_scores": consensus_raw,
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
) -> AssignmentGradeResult:
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

    acc = sum(float(c.normalized_score_estimate) for c in chunks)
    assign_score = acc / len(chunks) if chunks else 0.0

    assign_conf, conf_trace = aggregate_assignment_confidence(chunks)

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

    stage = {"assignment_confidence_trace": conf_trace}

    return AssignmentGradeResult(
        assignment_id=assignment_id,
        student_id=student_id,
        assignment_normalized_score=assign_score,
        assignment_ai_confidence=assign_conf,
        chunk_results=chunks,
        review_status=assign_status,
        review_reasons=reasons,
        stage_artifacts=stage,
    )
