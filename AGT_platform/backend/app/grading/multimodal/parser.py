"""
Parse model JSON into ParsedChunkGrade with normalization.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from typing import Any

from app.grading.rubric_allowlist import (
    match_criterion_name_to_allowlist,
    rubric_rows_to_allowlist,
)
from app.grading.multimodal.rubric_calibration import (
    anchor_map_monotone_increasing,
    ceiling_half_point_on_grid,
    format_anchor_map_for_log,
    get_anchor_map_for_criterion,
    validate_raw_score_increment,
)

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


def _align_parsed_to_rubric_rows(
    parsed: ParsedChunkGrade,
    rubric_rows: list[dict[str, Any]],
) -> ParsedChunkGrade:
    """Restrict scores to rubric rows for this chunk; remap names; recompute totals."""
    allowed = rubric_rows_to_allowlist(rubric_rows)
    if not allowed:
        return parsed
    ordered: list[tuple[str, float]] = []
    lims: dict[str, float] = {}
    for r in rubric_rows:
        if not isinstance(r, dict):
            continue
        nm = str(r.get("name") or r.get("criterion") or "").strip()
        if not nm:
            continue
        try:
            mp = float(r.get("max_points") or r.get("max_score") or 0)
        except (TypeError, ValueError):
            mp = 0.0
        ordered.append((nm, mp))
        lims[nm] = mp

    buckets: dict[str, list[tuple[int, CriterionScore]]] = {nm: [] for nm, _ in ordered}
    warns = list(parsed.parse_warnings)

    for i, cs in enumerate(parsed.criterion_scores):
        canon = match_criterion_name_to_allowlist(cs.name, allowed)
        if canon is None:
            warns.append(f"dropped_unknown_criterion:{cs.name!r}")
            continue
        if canon not in buckets:
            warns.append(f"dropped_unrouted_criterion:{cs.name!r}")
            continue
        buckets[canon].append((i, cs))

    new_scores: list[CriterionScore] = []
    new_just: list[str] = []
    new_ev: list[str] = []
    new_re: list[str] = []

    for nm, mp in ordered:
        pool = buckets.get(nm) or []
        if not pool:
            new_scores.append(
                CriterionScore(
                    name=nm,
                    score=0.0,
                    max_points=float(mp or 0),
                    weight=1.0,
                )
            )
            new_just.append("")
            new_ev.append("")
            new_re.append("")
            warns.append(f"missing_model_output_for:{nm}")
            continue
        _best_idx, best_cs = max(
            pool,
            key=lambda t: t[1].score / max(t[1].max_points or lims.get(nm) or 1e-9, 1e-9),
        )
        mx = float(lims.get(nm) or best_cs.max_points or 0)
        sc = max(0.0, min(float(best_cs.score), mx if mx else float(best_cs.score)))
        new_scores.append(
            CriterionScore(
                name=nm,
                score=sc,
                max_points=mx,
                weight=1.0,
            )
        )
        idx = int(_best_idx)
        if idx < len(parsed.criterion_justifications):
            new_just.append(str(parsed.criterion_justifications[idx] or ""))
        else:
            new_just.append("")
        if idx < len(parsed.criterion_evidence):
            new_ev.append(str(parsed.criterion_evidence[idx] or ""))
        else:
            new_ev.append("")
        if idx < len(parsed.criterion_reasoning):
            new_re.append(str(parsed.criterion_reasoning[idx] or ""))
        else:
            new_re.append("")

    total = sum(c.score for c in new_scores)
    mx_total = sum(c.max_points for c in new_scores) or 1.0
    norm = max(0.0, min(1.0, float(total) / float(mx_total)))

    return ParsedChunkGrade(
        rubric_type=parsed.rubric_type,
        criterion_scores=new_scores,
        criterion_justifications=new_just,
        total_score=float(total),
        normalized_score=float(norm),
        confidence_note=parsed.confidence_note,
        review_flag=parsed.review_flag,
        criterion_evidence=new_ev,
        criterion_reasoning=new_re,
        parse_warnings=warns,
        calibrated_question_score_0_100=float(norm) * 100.0,
    )


def _ordered_rubric_rows(rubric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rubric_rows:
        if not isinstance(r, dict):
            continue
        nm = str(r.get("name") or r.get("criterion") or "").strip()
        if nm:
            out.append(r)
    return out


def _finalize_rubric_half_steps(
    parsed: ParsedChunkGrade,
    rubric_rows: list[dict[str, Any]],
    *,
    invalid_raw_score_policy: str = "regenerate",
) -> tuple[ParsedChunkGrade | None, list[str]]:
    """
    Validate half-step raw scores per rubric row; recompute totals from raw ``score`` only
    (no calibrated credit blending).
    """
    extra: list[str] = []
    ordered_rr = _ordered_rubric_rows(rubric_rows)
    if len(ordered_rr) != len(parsed.criterion_scores):
        extra.append(
            f"calibration_rubric_row_count_mismatch:{len(ordered_rr)}"
            f"!={len(parsed.criterion_scores)}"
        )
    by_name: dict[str, dict[str, Any]] = {}
    for r in rubric_rows:
        if not isinstance(r, dict):
            continue
        nm = str(r.get("name") or r.get("criterion") or "").strip()
        if nm:
            by_name[nm] = r

    rebuilt: list[CriterionScore] = []
    for cs in parsed.criterion_scores:
        rr = by_name.get(cs.name) or {}
        try:
            R = float(
                rr.get("max_points")
                or rr.get("max_score")
                or cs.max_points
                or 0
            )
        except (TypeError, ValueError):
            R = float(cs.max_points or 0)
        raw = float(cs.score)
        v = validate_raw_score_increment(raw, R)
        if not v.ok:
            if invalid_raw_score_policy in ("nearest_half", "ceil_half"):
                raw = ceiling_half_point_on_grid(v.raw, v.max_raw)
                extra.append(
                    f"raw_score_ceiled_to_half_grid:{cs.name!r}:{v.raw!r}->{raw!r}"
                )
                v = validate_raw_score_increment(raw, R)
            else:
                return None, [
                    f"invalid_raw_score_regenerate:{cs.name!r}:"
                    f"raw={v.raw!r} R={v.max_raw!r} reason={v.message}"
                ]
        if not v.ok:
            return None, [
                f"invalid_raw_score_after_snap:{cs.name!r}:raw={v.raw!r}"
            ]
        raw = v.raw
        anchor = get_anchor_map_for_criterion(rr if rr else None)
        if not anchor_map_monotone_increasing(anchor):
            extra.append(f"anchor_map_non_monotone:{cs.name!r}")
        rebuilt.append(
            CriterionScore(
                name=cs.name,
                score=raw,
                max_points=float(R or cs.max_points or 0),
                weight=1.0,
            )
        )
        _log.info(
            "rubric_half_step name=%s R=%.3f raw=%.2f anchor=%s",
            cs.name,
            R,
            raw,
            format_anchor_map_for_log(anchor),
        )

    raw_sum = sum(c.score for c in rebuilt)
    mx_total = sum(c.max_points for c in rebuilt) or 1.0
    norm = max(0.0, min(1.0, float(raw_sum) / float(mx_total)))
    warns = list(parsed.parse_warnings) + extra
    return (
        ParsedChunkGrade(
            rubric_type=parsed.rubric_type,
            criterion_scores=rebuilt,
            criterion_justifications=list(parsed.criterion_justifications),
            total_score=float(raw_sum),
            normalized_score=float(norm),
            confidence_note=parsed.confidence_note,
            review_flag=parsed.review_flag,
            criterion_evidence=list(parsed.criterion_evidence),
            criterion_reasoning=list(parsed.criterion_reasoning),
            parse_warnings=warns,
            calibrated_question_score_0_100=float(norm) * 100.0,
        ),
        [],
    )


def parse_chunk_grade_json(
    raw: str,
    *,
    rubric_max_points: dict[str, float] | None = None,
    rubric_rows: list[dict[str, Any]] | None = None,
    invalid_raw_score_policy: str = "regenerate",
) -> tuple[ParsedChunkGrade | None, list[str]]:
    """Parse LLM JSON into ``ParsedChunkGrade``.

    ``rubric_max_points`` maps criterion name → max_points from the rubric
    so that dict-format ``criterion_scores`` (where the LLM omits max_points)
    can still resolve correct score ratios.

    When ``rubric_rows`` is non-empty (the chunk's routed rubric), scores are
    **aligned** to those rows only: unknown criterion names are dropped and
    missing rows get score 0 with a warning (prevents ``criterion_1`` style keys
    from propagating to the assignment JSON).

    With rubric rows, raw rubric scores must be on the half-point grid; invalid
    scores fail the parse when ``invalid_raw_score_policy`` is ``regenerate``,
    or are **ceiled** to the next valid half-step on ``[0, R]`` (capped at ``R``)
    when ``nearest_half`` or ``ceil_half``.
    """
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

    rubric_max: dict[str, float] = dict(rubric_max_points or {})

    crit_raw = obj.get("criterion_scores") or obj.get("criteria") or []

    if isinstance(crit_raw, dict):
        expanded: list[dict[str, Any]] = []
        for k, v in crit_raw.items():
            if isinstance(v, dict):
                row = dict(v)
                row.setdefault("name", str(k))
            else:
                row = {"name": str(k), "score": v}
            expanded.append(row)
        crit_raw = expanded
        warnings.append("criterion_scores_was_dict")

    if not isinstance(crit_raw, list):
        crit_raw = []

    c_scores: list[CriterionScore] = []
    inline_justifications: list[str] = []
    inline_evidence: list[str] = []
    inline_reasoning: list[str] = []
    rubric_list = list(rubric_rows or [])
    for i, row in enumerate(crit_raw):
        if not isinstance(row, dict):
            warnings.append(f"criterion_{i}_not_dict")
            continue
        default_nm = ""
        if i < len(rubric_list) and isinstance(rubric_list[i], dict):
            default_nm = str(rubric_list[i].get("name") or rubric_list[i].get("criterion") or "").strip()
        name = str(row.get("name") or row.get("criterion") or default_nm or "").strip()
        if not name:
            name = f"__unmapped_slot_{i}"
            warnings.append(f"criterion_{i}_missing_name_used_slot")
        raw_src = row.get("raw_score")
        if raw_src is None or raw_src == "":
            raw_src = row.get("score", 0)
        try:
            score = float(raw_src)
            mx = float(
                row.get("max_points")
                or row.get("max_score")
                or rubric_max.get(name)
                or 0
            )
        except (TypeError, ValueError):
            warnings.append(f"criterion_{i}_non_numeric")
            score, mx = 0.0, 0.0
        c_scores.append(
            CriterionScore(
                name=name,
                score=score,
                max_points=mx,
                weight=1.0,
            )
        )
        ij = str(row.get("justification") or row.get("reason") or row.get("explanation") or "")
        inline_justifications.append(ij)
        ev_raw = row.get("evidence") or ""
        if isinstance(ev_raw, dict):
            parts = []
            for q in ev_raw.get("quotes") or []:
                parts.append(str(q))
            notes = str(ev_raw.get("notes") or "")
            if notes:
                parts.append(notes)
            ev_str = " | ".join(parts)
        else:
            ev_str = str(ev_raw)
        inline_evidence.append(ev_str)
        reasoning_raw = str(
            row.get("reasoning")
            or row.get("chain_of_thought")
            or row.get("partial_credit_reasoning")
            or ""
        )
        inline_reasoning.append(reasoning_raw)

    just = obj.get("criterion_justifications")
    if isinstance(just, list):
        just_list = [str(x) for x in just]
    elif isinstance(just, dict):
        just_list = [str(just.get(cs.name, "")) for cs in c_scores] if c_scores else [
            f"{k}: {v}" for k, v in just.items()
        ]
    else:
        just_list = [str(just)] if just else []

    if not just_list and any(inline_justifications):
        just_list = inline_justifications
        warnings.append("justifications_from_inline_criterion_fields")

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
        criterion_evidence=inline_evidence,
        criterion_reasoning=inline_reasoning,
        parse_warnings=list(warnings),
        calibrated_question_score_0_100=float(norm) * 100.0,
    )
    if rubric_list:
        parsed = _align_parsed_to_rubric_rows(parsed, rubric_list)
        warnings = list(parsed.parse_warnings)
        fin, audit_msgs = _finalize_rubric_half_steps(
            parsed,
            rubric_list,
            invalid_raw_score_policy=invalid_raw_score_policy,
        )
        if fin is None:
            return None, warnings + audit_msgs
        parsed = fin
        warnings = list(parsed.parse_warnings) + audit_msgs
    elif c_scores:
        tot = sum(cs.score for cs in parsed.criterion_scores)
        mxt = sum(cs.max_points for cs in parsed.criterion_scores) or 1.0
        n_lin = max(0.0, min(1.0, float(tot) / float(mxt)))
        parsed = replace(
            parsed,
            total_score=float(tot),
            normalized_score=float(n_lin),
            calibrated_question_score_0_100=float(n_lin) * 100.0,
        )
    return parsed, warnings
