"""
Map :class:`AssignmentGradeResult` to the grading JSON contract used by
:func:`app.grading.output_schema.validate_grading_output` (``overall`` + ``criteria``).
"""

from __future__ import annotations

from typing import Any

from app.grading.aggregation import weighted_overall_confidence, weighted_overall_score
from app.grading.consistency_rules import run_rule_checks

from .schemas import AssignmentGradeResult, ChunkGradeOutcome


def _merge_criteria_max_by_name(
    question_grades: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pick the highest-scoring row per criterion name, preserving justification and evidence."""
    best: dict[str, dict[str, Any]] = {}
    for qg in question_grades:
        for c in qg.get("criteria") or []:
            if not isinstance(c, dict):
                continue
            name = str(c.get("name") or "")
            if not name:
                continue
            try:
                sc = float(c.get("score", 0))
            except (TypeError, ValueError):
                sc = 0.0
            if name not in best or sc > float(best[name].get("score", 0)):
                best[name] = dict(c)
    return [best[k] for k in sorted(best.keys())]


def _apply_rubric_weights_to_criteria(
    rows: list[dict[str, Any]], rubric_items: list[dict[str, Any]]
) -> None:
    by = {r["name"]: r for r in rubric_items}
    for c in rows:
        n = c.get("name")
        if n not in by:
            continue
        r = by[n]
        c["weight"] = r.get("weight", r["max_points"])
        if c.get("max_points") is None:
            c["max_points"] = r["max_points"]
        if c.get("min_points") is None:
            c["min_points"] = r["min_points"]


def _coerce_rubric_items(rubric: list) -> list[dict[str, Any]]:
    """Align with :func:`app.grading.pipelines._coerce_rubric_items` (subset copy)."""
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
        wt = x.get("weight")
        if wt is None:
            wt = mp
        try:
            wtf = float(wt)
        except (TypeError, ValueError):
            wtf = mp
        out.append(
            {"name": name, "max_points": mp, "min_points": lo, "weight": wtf}
        )
    return out


def _chunk_to_question_grade(
    chunk: ChunkGradeOutcome, max_by_name: dict[str, float]
) -> dict[str, Any]:
    aux = chunk.auxiliary or {}
    just_map: dict[str, str] = aux.get("criterion_justifications") or {}
    ev_map: dict[str, str] = aux.get("criterion_evidence") or {}
    conf_note: str = aux.get("confidence_note") or ""

    se_confidence = round(float(chunk.ai_confidence), 4)

    criteria_rows: list[dict[str, Any]] = []
    for name, ratio in (chunk.criterion_consensus or {}).items():
        mp = float(max_by_name.get(name) or 100.0)
        ratio_f = max(0.0, min(1.0, float(ratio)))
        row: dict[str, Any] = {
            "name": name,
            "score": round(ratio_f * mp, 4),
            "max_points": mp,
            "confidence": se_confidence,
        }
        j = just_map.get(name) or ""
        if j:
            row["justification"] = j
        ev = ev_map.get(name) or ""
        if ev:
            row["evidence"] = ev
        criteria_rows.append(row)

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
            "score": float(chunk.normalized_score_estimate),
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
    as chunk-entropy assignment merge).  ``overall.confidence`` is the weighted mean of
    chunk-level semantic-entropy-derived confidence (inverse normalized entropy over
    grading clusters).  ``overall.mean_chunk_semantic_entropy_nats`` is the mean of raw
    chunk entropies for diagnostics.
    """
    rubric_items = _coerce_rubric_items(rubric)
    max_by_name = {r["name"]: float(r["max_points"]) for r in rubric_items}

    question_grades = [
        _chunk_to_question_grade(c, max_by_name) for c in result.chunk_results
    ]

    # --- Consistency checks per chunk ---
    all_consistency_issues: list[str] = []
    for qg in question_grades:
        issues = run_rule_checks(qg.get("criteria") or [])
        if issues:
            qg["flags"] = list(dict.fromkeys(qg.get("flags", []) + issues))
            all_consistency_issues.extend(issues)

    merged = _merge_criteria_max_by_name(question_grades)
    _apply_rubric_weights_to_criteria(merged, rubric_items)

    # --- Consistency checks on merged assignment-level criteria ---
    assignment_issues = run_rule_checks(merged)
    all_consistency_issues.extend(assignment_issues)

    overall_score = weighted_overall_score(merged)
    wconf = weighted_overall_confidence(merged)

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
            "confidence": round(assignment_conf, 4),
            "summary": summary,
            "semantic_entropy": round(avg_h, 4),
            "mean_chunk_semantic_entropy_nats": round(avg_h, 4),
            "criteria_confidence_weighted_mean": round(wconf, 4),
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
    return out
