"""
Map :class:`AssignmentGradeResult` to the grading JSON contract used by
:func:`app.grading.output_schema.validate_grading_output` (``overall`` + ``criteria``).
"""

from __future__ import annotations

from typing import Any

from app.grading.aggregation import weighted_overall_confidence, weighted_overall_score
from app.grading.semantic_entropy import confidence_from_entropy_natural

from .schemas import AssignmentGradeResult, ChunkGradeOutcome


def _merge_criteria_max_by_name(
    question_grades: list[dict[str, Any]],
) -> list[dict[str, Any]]:
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
    criteria_rows: list[dict[str, Any]] = []
    for name, ratio in (chunk.criterion_consensus or {}).items():
        mp = float(max_by_name.get(name) or 100.0)
        ratio_f = max(0.0, min(1.0, float(ratio)))
        criteria_rows.append(
            {
                "name": name,
                "score": round(ratio_f * mp, 4),
                "max_points": mp,
                "confidence": 0.5,
            }
        )
    return {
        "chunk_id": chunk.chunk_id,
        "criteria": criteria_rows,
        "overall": {
            "score": float(chunk.normalized_score_estimate),
            "confidence": 0.5,
            "summary": "",
            "semantic_entropy": round(float(chunk.semantic_entropy_nats), 4),
        },
        "semantic_entropy": float(chunk.semantic_entropy_nats),
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
    as chunk-entropy assignment merge). ``overall.semantic_entropy`` is the mean of
    chunk-level semantic entropies (nats).
    """
    rubric_items = _coerce_rubric_items(rubric)
    max_by_name = {r["name"]: float(r["max_points"]) for r in rubric_items}

    question_grades = [
        _chunk_to_question_grade(c, max_by_name) for c in result.chunk_results
    ]
    merged = _merge_criteria_max_by_name(question_grades)
    _apply_rubric_weights_to_criteria(merged, rubric_items)
    overall_score = weighted_overall_score(merged)
    wconf = weighted_overall_confidence(merged)

    entropies = [float(c.semantic_entropy_nats) for c in result.chunk_results]
    avg_h = sum(entropies) / len(entropies) if entropies else 0.0
    final_conf_ent = confidence_from_entropy_natural(avg_h)

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

    out: dict[str, Any] = {
        "overall": {
            "score": overall_score,
            "confidence": round(final_conf_ent, 4),
            "summary": summary,
            "semantic_entropy": round(avg_h, 4),
            "confidence_from_entropy": round(final_conf_ent, 4),
            "classical_confidence": round(wconf, 4),
        },
        "criteria": crit_out,
        "flags": ["multimodal_pipeline"],
        "question_grades": question_grades,
        "_multimodal_pipeline_audit": result.stage_artifacts,
    }
    if model_ids:
        out["_models_used"] = model_ids
        out["_model_used"] = ", ".join(model_ids)
    if modality_profile is not None:
        out["_modality"] = modality_profile
    return out
