"""
Map :class:`AssignmentGradeResult` to the grading JSON contract used by
:func:`app.grading.output_schema.validate_grading_output` (``overall`` + ``criteria``).

Each ``question_grades`` entry includes one criterion row per routed rubric dimension
(with ``name``, ``score``, ``max_points``, ``confidence``, ``justification``, ``evidence``,
``reasoning``); ``raw_rubric_score`` / ``calibrated_credit`` are never emitted.
"""

from __future__ import annotations

from typing import Any

from app.grading.consistency_rules import run_rule_checks
from app.grading.output_schema import finalize_criterion_grading_fields
from app.grading.rubric_allowlist import filter_criteria_dicts_to_allowlist
from app.grading.multimodal.rubric_calibration import snap_half_nearest_display

from .schemas import AssignmentGradeResult, ChunkGradeOutcome


def _mean_criterion_score_fractions(rows: list[dict[str, Any]]) -> float:
    """Mean of ``score / max_points`` over criteria with positive cap (each criterion equal)."""
    parts: list[float] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        try:
            mp = float(r.get("max_points") or 0)
            if mp <= 0:
                continue
            parts.append(float(r.get("score", 0)) / mp)
        except (TypeError, ValueError):
            continue
    if not parts:
        return 0.0
    return max(0.0, min(1.0, sum(parts) / len(parts)))


def _mean_criterion_confidence(rows: list[dict[str, Any]]) -> float:
    vals: list[float] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        try:
            vals.append(float(r.get("confidence", 0.5)))
        except (TypeError, ValueError):
            continue
    if not vals:
        return 0.5
    return round(sum(vals) / len(vals), 4)


def _rubric_point_fraction(rows: list[dict[str, Any]]) -> float:
    """``sum(score) / sum(max_points)`` clipped to ``[0, 1]`` (matches blended criterion rows)."""
    earned = 0.0
    cap = 0.0
    for r in rows:
        if not isinstance(r, dict):
            continue
        try:
            earned += float(r.get("score", 0))
            cap += float(r.get("max_points", 0))
        except (TypeError, ValueError):
            continue
    if cap <= 0:
        return 0.0
    return max(0.0, min(1.0, earned / cap))


def _merge_criteria_max_by_name(
    question_grades: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pick the highest-scoring row per criterion name, preserving justification, evidence, and reasoning.

    If the highest-scoring row lacks evidence or justification text,
    backfill from other rows that do have it.
    """
    best: dict[str, dict[str, Any]] = {}
    backfill: dict[str, dict[str, str]] = {}
    for qg in question_grades:
        for c in qg.get("criteria") or []:
            if not isinstance(c, dict):
                continue
            name = str(c.get("name") or "")
            if not name:
                continue
            try:
                mp = float(c.get("max_points") or 0)
                sc = float(c.get("score", 0))
                frac = (sc / mp) if mp > 0 else 0.0
            except (TypeError, ValueError):
                frac, sc = 0.0, 0.0
            cur_frac = -1.0
            if name in best:
                try:
                    bmp = float(best[name].get("max_points") or 0)
                    bsc = float(best[name].get("score", 0))
                    cur_frac = (bsc / bmp) if bmp > 0 else 0.0
                except (TypeError, ValueError):
                    cur_frac = 0.0
            if name not in best or frac > cur_frac + 1e-12 or (
                abs(frac - cur_frac) < 1e-9 and sc > float(best[name].get("score", 0))
            ):
                best[name] = dict(c)
            bf = backfill.setdefault(name, {})
            for field in ("evidence", "justification", "reasoning"):
                val = str(c.get(field) or "")
                if val and not bf.get(field):
                    bf[field] = val
    for name, row in best.items():
        bf = backfill.get(name) or {}
        for field in ("evidence", "justification", "reasoning"):
            if not row.get(field):
                row[field] = bf.get(field, "")
        _snap_criterion_score_row(row)
    return [best[k] for k in sorted(best.keys())]


def _refresh_qg_overall_rubric_totals(qg: dict[str, Any]) -> None:
    """After per-chunk allowlist + blend sync, set ``overall.score`` = mean criterion fractions."""
    crits = [c for c in (qg.get("criteria") or []) if isinstance(c, dict)]
    earned = sum(float(c.get("score", 0)) for c in crits)
    cap = sum(float(c.get("max_points", 0)) for c in crits)
    frac = _mean_criterion_score_fractions(crits)
    o = qg.setdefault("overall", {})
    o["score"] = round(frac, 6)
    o["max_score"] = 1.0
    o["max_points"] = round(cap, 4)
    o["rubric_points_earned"] = round(earned, 4)


def _snap_criterion_score_row(row: dict[str, Any]) -> None:
    """Snap ``score`` to the nearest valid half-step on ``[0, max_points]``; drop legacy keys."""
    try:
        mp = float(row.get("max_points") or 0)
        raw = float(row.get("score", 0))
    except (TypeError, ValueError):
        return
    row["score"] = float(snap_half_nearest_display(raw, mp))
    row.pop("raw_rubric_score", None)
    row.pop("calibrated_credit", None)


def _sync_rubric_caps_on_criteria_rows(
    rows: list[dict[str, Any]], rubric_items: list[dict[str, Any]]
) -> None:
    """Fill missing ``max_points`` / ``min_points`` from the rubric; no weights."""
    by = {r["name"]: r for r in rubric_items}
    for c in rows:
        n = c.get("name")
        if n not in by:
            continue
        r = by[n]
        if c.get("max_points") is None:
            c["max_points"] = r["max_points"]
        if c.get("min_points") is None:
            c["min_points"] = r["min_points"]


def _coerce_rubric_items(rubric: list) -> list[dict[str, Any]]:
    """Normalize rubric rows: ``name``, ``max_points``, ``weight``, ``min_points``."""
    out: list[dict[str, Any]] = []
    for x in rubric or []:
        if not isinstance(x, dict):
            continue
        name = str(x.get("name") or x.get("criterion") or "").strip()
        if not name:
            continue
        try:
            mp = float(
                x.get("max_points")
                if x.get("max_points") is not None
                else (x.get("max_score") if x.get("max_score") is not None else 100.0)
            )
        except (TypeError, ValueError):
            mp = 100.0
        try:
            lo = float(x.get("min_points", 0.0))
        except (TypeError, ValueError):
            lo = 0.0
        out.append({"name": name, "max_points": mp, "min_points": lo})
    return out


def _max_by_name_for_chunk(
    chunk: ChunkGradeOutcome, full_max_by_name: dict[str, float]
) -> dict[str, float]:
    """Use routed rubric row names for this chunk when present; else full assignment rubric."""
    aux = chunk.auxiliary or {}
    names = aux.get("rubric_criterion_names") or []
    if not isinstance(names, list) or not names:
        return dict(full_max_by_name)
    out: dict[str, float] = {}
    for n in names:
        s = str(n).strip()
        if not s or s not in full_max_by_name:
            continue
        out[s] = float(full_max_by_name[s])
    return out or dict(full_max_by_name)


def _chunk_to_question_grade(
    chunk: ChunkGradeOutcome, max_by_name: dict[str, float]
) -> dict[str, Any]:
    aux = chunk.auxiliary or {}
    just_map: dict[str, str] = aux.get("criterion_justifications") or {}
    ev_map: dict[str, str] = aux.get("criterion_evidence") or {}
    reason_map: dict[str, str] = aux.get("criterion_reasoning") or {}
    conf_note: str = aux.get("confidence_note") or ""
    raw_map: dict[str, float] = {}
    raw_aux = aux.get("criterion_raw_scores") or {}
    if isinstance(raw_aux, dict):
        for k, v in raw_aux.items():
            try:
                raw_map[str(k)] = float(v)
            except (TypeError, ValueError):
                continue

    se_confidence = round(float(chunk.ai_confidence), 4)
    consensus_frac = dict(chunk.criterion_consensus or {})

    # One output row per rubric dimension for this chunk (never an empty ``criteria`` list).
    names_ordered = sorted(max_by_name.keys())

    criteria_rows: list[dict[str, Any]] = []
    for name in names_ordered:
        mp = float(max_by_name.get(name) or 0.0)
        if mp <= 0:
            continue
        if name in raw_map:
            try:
                raw_consensus = float(raw_map[name])
            except (TypeError, ValueError):
                raw_consensus = 0.0
        elif name in consensus_frac:
            raw_consensus = float(consensus_frac[name]) * mp
        else:
            raw_consensus = 0.0
        raw_s = float(snap_half_nearest_display(raw_consensus, mp))
        row: dict[str, Any] = {
            "name": name,
            "score": raw_s,
            "max_points": mp,
            "confidence": se_confidence,
            "justification": str(just_map.get(name) or "").strip(),
            "evidence": str(ev_map.get(name) or "").strip(),
            "reasoning": str(reason_map.get(name) or "").strip(),
        }
        criteria_rows.append(row)

    earned = sum(float(r["score"]) for r in criteria_rows)
    cap = sum(float(r["max_points"]) for r in criteria_rows)
    frac = _mean_criterion_score_fractions(criteria_rows)

    evidence_parts: list[str] = []
    if conf_note:
        evidence_parts.append(conf_note)
    for j_text in just_map.values():
        if j_text and j_text not in evidence_parts:
            evidence_parts.append(j_text)
    evidence_summary = " | ".join(evidence_parts)[:4000] if evidence_parts else ""

    return {
        "chunk_id": chunk.chunk_id,
        "criteria": criteria_rows,
        "overall": {
            "score": round(frac, 6),
            "max_score": 1.0,
            "max_points": round(cap, 4),
            "rubric_points_earned": round(earned, 4),
            "confidence": se_confidence,
            "summary": evidence_summary,
            "semantic_entropy": round(float(chunk.semantic_entropy_nats), 4),
            "entropy_max_reference_nats": round(
                float(chunk.entropy_max_reference_nats), 6
            ),
        },
        "semantic_entropy": float(chunk.semantic_entropy_nats),
        "review_status": chunk.review_status.value,
        "review_reasons": list(chunk.review_reasons),
        "flags": [],
    }


def multimodal_assignment_to_grading_dict(
    result: AssignmentGradeResult,
    *,
    rubric: list[dict],
    modality_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a dict suitable for :func:`~app.grading.output_schema.validate_grading_output`.

    Criteria are merged across chunks with **max** score per criterion name (same idea
    as chunk-entropy assignment merge).  Top-level ``overall.score`` is the **mean** of
    each question’s ``overall.score``, where each question score is the **mean** of
    ``criterion.score / criterion.max_points`` (equal weight per criterion). Assignment
    ``criteria_confidence_mean`` is the unweighted mean of merged criterion confidences.
    """
    rubric_items = _coerce_rubric_items(rubric)
    max_by_name = {r["name"]: float(r["max_points"]) for r in rubric_items}

    question_grades = [
        _chunk_to_question_grade(c, _max_by_name_for_chunk(c, max_by_name))
        for c in result.chunk_results
    ]
    sid = str(getattr(result, "student_id", "") or "")
    aid = str(getattr(result, "assignment_id", "") or "")
    for i, (qg, ch) in enumerate(
        zip(question_grades, result.chunk_results, strict=True), start=1
    ):
        qg["chunk_id"] = f"{sid}:{aid}:pair_{i}"
        qg["_source_chunk_id"] = ch.chunk_id

    allowed = frozenset(r["name"] for r in rubric_items)

    # --- Per-chunk allowlist, blend sync, rubric-aligned overall, then consistency ---
    all_consistency_issues: list[str] = []
    for qg in question_grades:
        cid = str(qg.get("chunk_id") or "chunk")
        crits = qg.get("criteria") or []
        filtered, q_issues = filter_criteria_dicts_to_allowlist(
            [c for c in crits if isinstance(c, dict)],
            allowed,
            context=cid,
        )
        qg["criteria"] = filtered
        all_consistency_issues.extend(q_issues)
        for c in qg.get("criteria") or []:
            if isinstance(c, dict):
                c.pop("weight", None)
                _snap_criterion_score_row(c)
                finalize_criterion_grading_fields(c)
        _refresh_qg_overall_rubric_totals(qg)
        issues = run_rule_checks(qg.get("criteria") or [])
        if issues:
            qg["flags"] = list(dict.fromkeys(qg.get("flags", []) + issues))
            all_consistency_issues.extend(issues)

    merged = _merge_criteria_max_by_name(question_grades)
    _sync_rubric_caps_on_criteria_rows(merged, rubric_items)
    merged, allow_issues = filter_criteria_dicts_to_allowlist(
        merged, allowed, context="assignment_merged"
    )
    all_consistency_issues.extend(allow_issues)
    for c in merged:
        if isinstance(c, dict):
            _snap_criterion_score_row(c)

    # --- Consistency checks on merged assignment-level criteria ---
    assignment_issues = run_rule_checks(merged)
    all_consistency_issues.extend(assignment_issues)

    cap_assign = sum(
        float(r.get("max_points", 0)) for r in merged if isinstance(r, dict)
    )
    per_q_scores: list[float] = []
    for qg in question_grades:
        ov = qg.get("overall")
        if isinstance(ov, dict):
            try:
                per_q_scores.append(float(ov.get("score", 0.0)))
            except (TypeError, ValueError):
                continue
    if per_q_scores:
        overall_score = round(sum(per_q_scores) / len(per_q_scores), 6)
    else:
        overall_score = round(_rubric_point_fraction(merged), 6)
    earned_from_chunks = sum(
        float((qg.get("overall") or {}).get("rubric_points_earned") or 0)
        for qg in question_grades
        if isinstance(qg, dict)
    )
    if question_grades:
        earned_assign = round(earned_from_chunks, 4)
    else:
        earned_assign = (
            round(overall_score * cap_assign, 4) if cap_assign > 0 else 0.0
        )
    conf_mean = _mean_criterion_confidence(merged)

    entropies = [float(c.semantic_entropy_nats) for c in result.chunk_results]
    avg_h = sum(entropies) / len(entropies) if entropies else 0.0
    assignment_conf = float(getattr(result, "assignment_ai_confidence", 0.0))

    model_ids: list[str] = []
    audit = (result.stage_artifacts or {}).get("pipeline_audit") or {}
    for row in audit.get("grading", []) or []:
        for mid in row.get("model_ids") or []:
            s = str(mid)
            if s and s not in model_ids:
                model_ids.append(s)

    crit_out: list[dict[str, Any]] = []
    for c in merged:
        row = dict(c)
        row.pop("weight", None)
        row.setdefault("evidence", "")
        row.setdefault("justification", "")
        row.setdefault("reasoning", "")
        _snap_criterion_score_row(row)
        finalize_criterion_grading_fields(row)
        crit_out.append(row)

    summary = "; ".join(
        f"{c.chunk_id}: norm={c.normalized_score_estimate:.3f}, H={c.semantic_entropy_nats:.3f}"
        for c in result.chunk_results
    )[:4000]

    flags = ["multimodal_pipeline", f"review_{result.review_status.value}"]
    if all_consistency_issues:
        flags.append("consistency_issues_detected")

    out: dict[str, Any] = {
        "overall": {
            "score": overall_score,
            "max_score": 1.0,
            "max_points": round(cap_assign, 4),
            "rubric_points_earned": round(earned_assign, 4),
            "confidence": round(assignment_conf, 4),
            "summary": summary,
            "semantic_entropy": round(avg_h, 4),
            "mean_chunk_semantic_entropy_nats": round(avg_h, 4),
            "criteria_confidence_mean": conf_mean,
        },
        "criteria": crit_out,
        "flags": flags,
        "question_grades": question_grades,
        "_multimodal_pipeline_audit": result.stage_artifacts,
        "_assignment_review_status": result.review_status.value,
    }
    if all_consistency_issues:
        out["_consistency_issues"] = list(dict.fromkeys(all_consistency_issues))
    if model_ids:
        out["_models_used"] = model_ids
        out["_model_used"] = ", ".join(model_ids)
    if modality_profile is not None:
        out["_modality"] = modality_profile
    aw = (result.stage_artifacts or {}).get("agentic_workflow")
    if isinstance(aw, list) and aw:
        out["_agentic_workflow"] = aw
    return out
