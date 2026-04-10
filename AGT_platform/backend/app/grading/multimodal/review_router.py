"""
Human-review routing from entropy, disagreement, parses, and flags.
"""

from __future__ import annotations

from .schemas import ChunkGradeOutcome, MultimodalGradingConfig, ReviewStatus, SampledChunkGrade


def evaluate_chunk_review(
    outcome: ChunkGradeOutcome,
    samples: list[SampledChunkGrade],
    cfg: MultimodalGradingConfig,
) -> ChunkGradeOutcome:
    reasons: list[str] = []
    status = ReviewStatus.AUTO_ACCEPTED

    if outcome.semantic_entropy_nats > cfg.semantic_entropy_high:
        reasons.append("semantic_entropy_above_threshold")
        status = ReviewStatus.FLAGGED

    if outcome.auxiliary.get("score_std_across_samples", 0) > cfg.score_spread_high:
        reasons.append("score_spread_above_threshold")
        status = ReviewStatus.FLAGGED

    if (
        outcome.auxiliary.get("criterion_disagreement_max", 0)
        > cfg.criterion_disagreement_high
    ):
        reasons.append("criterion_disagreement_above_threshold")
        status = ReviewStatus.FLAGGED

    if outcome.auxiliary.get("parse_fail_rate", 0) > cfg.parse_fail_rate_high:
        reasons.append("parse_failure_rate_high")
        status = ReviewStatus.FLAGGED

    if cfg.review_if_any_sample_flag and outcome.auxiliary.get("review_flag_rate", 0) > 0:
        reasons.append("sample_review_flag")
        status = ReviewStatus.FLAGGED

    # Escalation: both very high entropy and very high spread
    if (
        outcome.semantic_entropy_nats > cfg.semantic_entropy_high * 1.35
        and outcome.auxiliary.get("score_std_across_samples", 0) > cfg.score_spread_high * 1.5
    ):
        status = ReviewStatus.ESCALATION
        reasons.append("entropy_and_spread_escalation")

    outcome.review_status = status
    outcome.review_reasons = list(dict.fromkeys(reasons))
    return outcome
