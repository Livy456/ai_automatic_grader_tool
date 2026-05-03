"""
Validation helpers for grading pipeline JSON output (stdlib only).

Checks ``overall``, ``criteria``, ``flags``, optional ``_entropy_meta`` /
``_model_used``, and optional ``_modality`` (pipeline-inferred submission type)
so producers and local tests share one contract.

Raises :class:`GradingOutputValidationError` with a human-readable message on failure.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.grading.rubric_allowlist import filter_criteria_dicts_to_allowlist

_log = logging.getLogger(__name__)


def _criteria_rows_to_arrays_for_weighted(
    rows: list[dict[str, Any]],
    *,
    default_confidence: float = 0.5,
) -> tuple[list[float], list[float], list[float]]:
    """Stack weight, score, confidence for each criterion dict (stdlib lists)."""
    w: list[float] = []
    s: list[float] = []
    c: list[float] = []
    for row in rows:
        wf: float | None = None
        wt = row.get("weight")
        if wt is not None:
            try:
                t = float(wt)
                if t > 0.0:
                    wf = t
            except (TypeError, ValueError):
                pass
        if wf is None:
            mp = row.get("max_points")
            if mp is None:
                mp = row.get("max_score")
            try:
                wf = float(mp) if mp is not None else 0.0
            except (TypeError, ValueError):
                wf = 0.0
        w.append(wf)
        s.append(float(row.get("score", 0.0)))
        try:
            c.append(float(row.get("confidence", default_confidence)))
        except (TypeError, ValueError):
            c.append(default_confidence)
    return w, s, c


def _weighted_mean_list(
    values: list[float],
    weights: list[float],
    *,
    default: float = 0.0,
) -> float:
    num = 0.0
    den = 0.0
    for vi, wi in zip(values, weights, strict=False):
        if wi > 0:
            num += wi * vi
            den += wi
    return default if den <= 0 else num / den


def weighted_overall_confidence(criteria_results: list[dict[str, Any]]) -> float:
    """Mean criterion confidence weighted by rubric weights / max_points."""
    if not criteria_results:
        return 0.5
    w, _s, conf = _criteria_rows_to_arrays_for_weighted(criteria_results)
    mu = _weighted_mean_list(conf, w, default=0.5)
    return round(mu, 2)

class GradingOutputValidationError(ValueError):
    """Pipeline result does not match the expected grading JSON contract."""


def _coerce_float(v: Any, default: float | None = None) -> float:
    if v is None and default is not None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError) as e:
        raise GradingOutputValidationError(f"expected numeric value, got {v!r}") from e


def _coerce_confidence(v: Any) -> float:
    x = _coerce_float(v, 0.5)
    return max(0.0, min(1.0, x))


def _rubric_point_fraction(rows: list[dict[str, Any]]) -> float:
    """``sum(score) / sum(max_points)`` in ``[0, 1]`` (assignment-style rubric rows)."""
    earned = 0.0
    cap = 0.0
    for r in rows:
        if not isinstance(r, dict):
            continue
        try:
            earned += float(r.get("score", 0))
            mp = r.get("max_points")
            if mp is None:
                mp = r.get("max_score")
            cap += float(mp if mp is not None else 0.0)
        except (TypeError, ValueError):
            continue
    if cap <= 0:
        return 0.0
    return max(0.0, min(1.0, earned / cap))


def _mean_criterion_fraction_from_crits(crits: list[dict[str, Any]]) -> float:
    parts: list[float] = []
    for c in crits:
        if not isinstance(c, dict):
            continue
        try:
            mp = float(c.get("max_points") or 0)
            if mp <= 0:
                continue
            parts.append(float(c.get("score", 0)) / mp)
        except (TypeError, ValueError):
            continue
    if not parts:
        return 0.0
    return max(0.0, min(1.0, sum(parts) / len(parts)))


def _mean_criterion_confidence_from_rows(rows: list[dict[str, Any]]) -> float:
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
    return float(sum(vals) / len(vals))


def _normalize_overall_score_fraction(overall: dict[str, Any]) -> None:
    """Mutates ``overall``: ``score`` is a fraction in ``[0, 1]``; ``max_score`` is ``1``."""
    sc = _coerce_float(overall.get("score"))
    if sc > 1.0:
        sc /= 100.0
    overall["score"] = max(0.0, min(1.0, sc))
    overall["max_score"] = 1.0


def _sync_question_grade_overall_from_criteria(qg_row: dict[str, Any]) -> None:
    """After criteria are filtered, recompute ``overall.score`` from remaining rows."""
    crits = [c for c in (qg_row.get("criteria") or []) if isinstance(c, dict)]
    ov = qg_row.setdefault("overall", {})
    if not crits:
        ov["score"] = 0.0
        ov["max_score"] = 1.0
        ov["max_points"] = 0.0
        ov["rubric_points_earned"] = 0.0
        return
    earned = 0.0
    cap = 0.0
    for c in crits:
        try:
            earned += float(c.get("score", 0))
            mp = c.get("max_points")
            if mp is not None:
                cap += float(mp)
        except (TypeError, ValueError):
            continue
    frac = _mean_criterion_fraction_from_crits(crits)
    ov = qg_row.setdefault("overall", {})
    ov["score"] = round(frac, 6)
    ov["max_score"] = 1.0
    if cap > 0:
        ov["max_points"] = round(cap, 4)
        ov["rubric_points_earned"] = round(earned, 4)
    _normalize_overall_score_fraction(ov)


def _criteria_rows_for_overall_synthesis(criteria: Any) -> list[dict[str, Any]]:
    if not isinstance(criteria, list):
        return []
    rows: list[dict[str, Any]] = []
    for c in criteria:
        if not isinstance(c, dict):
            continue
        r = dict(c)
        w = r.get("weight")
        if w is None:
            w = r.get("max_points") if r.get("max_points") is not None else r.get("max_score")
        if w is None:
            w = 1.0
        try:
            r["weight"] = float(w)
        except (TypeError, ValueError):
            r["weight"] = 1.0
        rows.append(r)
    return rows


def coerce_grading_output_shape(data: Any) -> dict[str, Any]:
    """
    Best-effort normalization so grading producers match :func:`validate_grading_output`.
    Mutates and returns the same dict.

    Handles missing or mistyped ``overall`` (scalar, wrong key capitalization, nested
    objects under ``grading`` / ``result``), and synthesizes ``overall`` from
    ``criteria`` when possible.
    """
    if not isinstance(data, dict):
        _log.warning("grader returned non-dict; substituting empty grading envelope")
        return {
            "overall": {
                "score": 0.0,
                "confidence": 0.25,
                "summary": "Grader returned a non-object response.",
            },
            "criteria": [],
            "flags": ["INVALID_GRADER_TOP_LEVEL"],
        }

    coerced = False
    alias_keys = ("Overall", "OVERALL", "grade_summary")
    for alt in alias_keys:
        if alt not in data or isinstance(data.get("overall"), dict):
            continue
        val = data.pop(alt)
        if isinstance(val, dict):
            data["overall"] = val
        elif isinstance(val, (int, float)):
            data["overall"] = {
                "score": float(val),
                "confidence": 0.5,
                "summary": "",
            }
        else:
            data[alt] = val
            continue
        coerced = True
        break

    for container_key in ("grading", "result", "output", "response"):
        inner = data.get(container_key)
        if (
            isinstance(inner, dict)
            and isinstance(inner.get("overall"), dict)
            and not isinstance(data.get("overall"), dict)
        ):
            if "overall" in inner:
                data["overall"] = inner["overall"]
            if data.get("criteria") is None and isinstance(inner.get("criteria"), list):
                data["criteria"] = inner["criteria"]
            if data.get("flags") is None and isinstance(inner.get("flags"), list):
                data["flags"] = inner["flags"]
            coerced = True
            break

    if not isinstance(data.get("overall"), dict):
        for nk in ("grading", "evaluation", "grade_output", "assessment"):
            inner = data.get(nk)
            if (
                isinstance(inner, dict)
                and isinstance(inner.get("overall"), dict)
            ):
                data["overall"] = inner["overall"]
                if data.get("criteria") is None and isinstance(
                    inner.get("criteria"), list
                ):
                    data["criteria"] = inner["criteria"]
                if data.get("flags") is None and isinstance(inner.get("flags"), list):
                    data["flags"] = inner["flags"]
                coerced = True
                break

    ov = data.get("overall")
    if isinstance(ov, (int, float)):
        data["overall"] = {
            "score": float(ov),
            "confidence": 0.5,
            "summary": "",
        }
        coerced = True
        ov = data["overall"]
    elif isinstance(ov, str):
        s = ov.strip()
        if s.startswith("{"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, dict) and "score" in parsed:
                    data["overall"] = parsed
                    coerced = True
                    ov = data["overall"]
            except json.JSONDecodeError:
                pass
        if not isinstance(data.get("overall"), dict):
            try:
                data["overall"] = {
                    "score": float(s),
                    "confidence": 0.5,
                    "summary": "",
                }
            except ValueError:
                data["overall"] = {
                    "score": 0.0,
                    "confidence": 0.4,
                    "summary": s[:4000],
                }
            coerced = True
            ov = data["overall"]

    if isinstance(ov, dict):
        if "score" not in ov:
            for bk in ("total", "total_score", "points", "Grade"):
                if bk in ov:
                    try:
                        ov["score"] = float(ov[bk])
                        coerced = True
                    except (TypeError, ValueError):
                        pass
                    break

    if isinstance(data.get("overall"), dict) and "score" not in data["overall"]:
        rows = _criteria_rows_for_overall_synthesis(data.get("criteria"))
        if rows:
            data["overall"]["score"] = _rubric_point_fraction(rows)
            data["overall"].setdefault(
                "confidence", weighted_overall_confidence(rows)
            )
            data["overall"].setdefault("summary", "")
            coerced = True
        else:
            data["overall"]["score"] = 0.0
            data["overall"].setdefault("confidence", 0.25)
            data["overall"].setdefault("summary", "")
            coerced = True

    if not isinstance(data.get("overall"), dict):
        rows = _criteria_rows_for_overall_synthesis(data.get("criteria"))
        if rows:
            data["overall"] = {
                "score": _rubric_point_fraction(rows),
                "confidence": weighted_overall_confidence(rows),
                "summary": "",
            }
            coerced = True
        else:
            data["overall"] = {
                "score": 0.0,
                "confidence": 0.25,
                "summary": "Grader output had no usable overall or criteria.",
            }
            coerced = True

    if data.get("criteria") is None:
        data["criteria"] = []
        coerced = True

    if coerced:
        fl = data.get("flags")
        if not isinstance(fl, list):
            data["flags"] = []
        tag = "grader_output_coerced"
        if tag not in (data.get("flags") or []):
            data["flags"] = list(dict.fromkeys([*(data.get("flags") or []), tag]))
        _log.info("Adjusted grader JSON to match expected overall/criteria shape")

    o_end = data.get("overall")
    if isinstance(o_end, dict) and "score" in o_end:
        try:
            o_end["score"] = float(o_end["score"])
        except (TypeError, ValueError):
            o_end["score"] = 0.0
        _normalize_overall_score_fraction(o_end)

    return data


def finalize_criterion_grading_fields(c: dict[str, Any]) -> None:
    """
    Ensure ``justification``, ``evidence``, and ``reasoning`` are present as strings.

    When the model omitted free-text fields but assigned points, insert explicit placeholders
    so downstream JSON matches the grading-output contract (and matches human-readable exports).
    """
    try:
        sc = float(c.get("score", 0) or 0)
    except (TypeError, ValueError):
        sc = 0.0
    try:
        mp = float(c.get("max_points") if c.get("max_points") is not None else c.get("max_score") or 0)
    except (TypeError, ValueError):
        mp = 0.0
    j = str(c.get("justification") or "").strip()
    ev = str(c.get("evidence") or "").strip()
    rs = str(c.get("reasoning") or "").strip()
    if j and ev and rs:
        return
    if mp <= 0.0:
        c["justification"] = j or "Not applicable (zero max points for this rubric row)."
        c["evidence"] = ev or ""
        c["reasoning"] = rs or ""
        return
    if sc <= 0.0:
        c["justification"] = j or (
            "No points awarded for this rubric dimension on this question segment."
        )
        c["evidence"] = ev or (
            "No credited student evidence for this criterion in this chunk."
        )
        c["reasoning"] = rs or (
            "A score of 0 reflects missing, off-topic, or non-defensible student work "
            "for this criterion relative to the rubric."
        )
        return
    c["justification"] = j or (
        "The grader did not return a separate justification string; the numeric score "
        "reflects rubric-aligned model output for this row."
    )
    c["evidence"] = ev or (
        "The grader did not return a verbatim student quote for this criterion; "
        "see other fields or the chunk summary for context."
    )
    c["reasoning"] = rs or (
        "The grader did not return explicit chain-of-thought text; the score follows "
        "the rubric row consensus for this chunk."
    )


def _normalize_one_criterion_dict(
    row: dict[str, Any], *, index: str | int
) -> dict[str, Any]:
    """Normalize a single criterion row (top-level or under ``question_grades``)."""
    if not isinstance(row, dict):
        raise GradingOutputValidationError(f"criterion[{index}] must be a dict")
    c = dict(row)
    name = (c.get("name") or c.get("criterion") or "").strip()
    if not name:
        name = "unknown"
    c["name"] = name
    c["score"] = _coerce_float(c.get("score", 0))
    c["confidence"] = _coerce_confidence(c.get("confidence", 0.5))
    if c.get("max_points") is None and c.get("max_score") is not None:
        c["max_points"] = _coerce_float(c["max_score"])
    elif c.get("max_points") is not None:
        c["max_points"] = _coerce_float(c["max_points"])
    ev = c.get("evidence")
    if ev is None:
        c["evidence"] = ""
    elif isinstance(ev, dict):
        parts = [str(q) for q in (ev.get("quotes") or [])]
        notes = str(ev.get("notes") or "")
        if notes:
            parts.append(notes)
        c["evidence"] = " | ".join(parts) if parts else ""
    elif not isinstance(ev, str):
        c["evidence"] = str(ev)

    reasoning = c.get("reasoning")
    if reasoning is None:
        c["reasoning"] = ""
    elif not isinstance(reasoning, str):
        c["reasoning"] = str(reasoning)

    if c.get("justification") is None:
        c["justification"] = ""
    elif not isinstance(c.get("justification"), str):
        c["justification"] = str(c.get("justification"))

    c.pop("calibrated_credit", None)
    c.pop("raw_rubric_score", None)
    c.pop("weight", None)
    finalize_criterion_grading_fields(c)
    return c


def _resync_assignment_overall_from_question_grades(data: dict[str, Any]) -> None:
    """
    Set assignment ``overall.score`` to the mean of ``question_grades[].overall.score``.

    When present, also align ``rubric_points_earned`` with the sum of per-question earned
    points so the headline fraction is not numerically inconsistent with chunk rows.
    """
    qg = data.get("question_grades")
    if not isinstance(qg, list) or not qg:
        return
    scores: list[float] = []
    earned_sum = 0.0
    for row in qg:
        if not isinstance(row, dict):
            continue
        ov = row.get("overall")
        if isinstance(ov, dict) and "score" in ov:
            try:
                scores.append(float(ov["score"]))
            except (TypeError, ValueError):
                continue
        if isinstance(ov, dict) and ov.get("rubric_points_earned") is not None:
            try:
                earned_sum += float(ov["rubric_points_earned"])
            except (TypeError, ValueError):
                pass
    if not scores:
        return
    mean_q = round(sum(scores) / len(scores), 6)
    top = data.get("overall")
    if not isinstance(top, dict):
        return
    prev = float(top.get("score", 0.0))
    if abs(prev - mean_q) > 1e-5:
        fl = data.setdefault("flags", [])
        if isinstance(fl, list):
            tag = "assignment_overall_resynced_from_question_grades_mean"
            if tag not in fl:
                fl.append(tag)
    top["score"] = mean_q
    top["max_score"] = 1.0
    _normalize_overall_score_fraction(top)
    top["rubric_points_earned"] = round(earned_sum, 4)


def validate_grading_output(
    data: dict[str, Any],
    *,
    allowed_criterion_names: frozenset[str] | None = None,
) -> dict[str, Any]:
    """
    Validate ``run_db_submission_multimodal_pipeline`` / Celery task result shape in-place (normalizes some fields).

    Required top-level keys: ``overall`` (dict with ``score``), ``criteria`` (list).

    When ``allowed_criterion_names`` is set (canonical LMS rubric names for this
    assignment), criteria rows are **filtered** to that set and remapped to canonical
    spelling; unknown names are removed and ``flags`` gain ``rubric_allowlist:*`` entries.
    ``overall.score`` is a **fraction in ``[0, 1]``**; legacy **percent in ``[0, 100]``**
    is divided by ``100``. ``overall.max_score`` is always ``1``. When allowlist filtering
    runs and ``question_grades`` are present, ``overall.score`` is the **mean** of each
    question’s ``overall.score`` (each of those is the mean of ``score/max_points`` per
    criterion). Otherwise it uses the mean criterion fraction on filtered top-level
    ``criteria``.

    Each criterion dict (top-level and under ``question_grades``) is normalized to include
    string ``justification``, ``evidence``, and ``reasoning`` (placeholders are inserted
    when the producer omitted them). ``raw_rubric_score`` / ``calibrated_credit`` are stripped.
    When ``question_grades`` is present, assignment-level ``overall.score`` and
    ``rubric_points_earned`` are re-synchronized from per-question aggregates so headline
    scores do not contradict chunk-level rubric rows.

    Returns the same ``data`` dict after light normalization (e.g. ``criterion`` → ``name``).
    """
    if not isinstance(data, dict):
        raise GradingOutputValidationError("grading result must be a dict")

    overall = data.get("overall")
    if not isinstance(overall, dict):
        raise GradingOutputValidationError("overall must be a dict")
    if "score" not in overall:
        raise GradingOutputValidationError("overall.score is required")
    overall["score"] = _coerce_float(overall.get("score"))
    _normalize_overall_score_fraction(overall)
    overall["confidence"] = _coerce_confidence(overall.get("confidence", 0.5))
    if overall.get("semantic_entropy") is not None:
        overall["semantic_entropy"] = float(overall["semantic_entropy"])
    if overall.get("confidence_from_entropy") is not None:
        overall["confidence_from_entropy"] = _coerce_confidence(
            overall["confidence_from_entropy"]
        )
    if overall.get("classical_confidence") is not None:
        overall["classical_confidence"] = _coerce_confidence(
            overall["classical_confidence"]
        )
    if "criteria_confidence_weighted_mean" in overall:
        overall["criteria_confidence_mean"] = _coerce_float(
            overall.pop("criteria_confidence_weighted_mean"), 0.5
        )
    elif overall.get("criteria_confidence_mean") is not None:
        overall["criteria_confidence_mean"] = _coerce_float(
            overall["criteria_confidence_mean"], 0.5
        )
    data["overall"] = overall

    crit_raw = data.get("criteria")
    if crit_raw is None:
        data["criteria"] = []
        crit_raw = []
    if not isinstance(crit_raw, list):
        raise GradingOutputValidationError("criteria must be a list")

    criteria_out: list[dict[str, Any]] = []
    for i, row in enumerate(crit_raw):
        criteria_out.append(_normalize_one_criterion_dict(row, index=i))
    data["criteria"] = criteria_out

    fl = data.get("flags")
    if fl is None:
        data["flags"] = []
    elif not isinstance(fl, list):
        raise GradingOutputValidationError("flags must be a list")
    else:
        data["flags"] = [str(x) for x in fl]

    ent = data.get("_entropy_meta")
    if ent is not None and not isinstance(ent, dict):
        raise GradingOutputValidationError("_entropy_meta must be a dict or omitted")

    qg = data.get("question_grades")
    if qg is not None:
        if not isinstance(qg, list):
            raise GradingOutputValidationError("question_grades must be a list or omitted")
        for i, row in enumerate(qg):
            if not isinstance(row, dict):
                raise GradingOutputValidationError(f"question_grades[{i}] must be a dict")
            qc_norm: list[dict[str, Any]] = []
            for j, crit in enumerate(row.get("criteria") or []):
                if isinstance(crit, dict):
                    qc_norm.append(_normalize_one_criterion_dict(crit, index=f"{i}.{j}"))
            row["criteria"] = qc_norm
            _sync_question_grade_overall_from_criteria(row)
            qov = row.get("overall")
            if isinstance(qov, dict):
                if "score" in qov:
                    qov["score"] = _coerce_float(qov.get("score"))
                    _normalize_overall_score_fraction(qov)
                if qov.get("max_points") is not None:
                    qov["max_points"] = _coerce_float(qov["max_points"])
                if qov.get("rubric_points_earned") is not None:
                    qov["rubric_points_earned"] = _coerce_float(
                        qov["rubric_points_earned"]
                    )
                if qov.get("semantic_entropy") is not None:
                    qov["semantic_entropy"] = float(qov["semantic_entropy"])
                if qov.get("entropy_max_reference_nats") is not None:
                    qov["entropy_max_reference_nats"] = float(
                        qov["entropy_max_reference_nats"]
                    )
        data["question_grades"] = qg

    mod = data.get("_modality")
    if mod is not None:
        if not isinstance(mod, dict):
            raise GradingOutputValidationError("_modality must be a dict or omitted")
        for key in ("modality", "modality_subtype"):
            if key in mod and mod[key] is not None:
                mod[key] = str(mod[key])
        ak = mod.get("artifact_keys")
        if ak is not None:
            if not isinstance(ak, list) or any(not isinstance(x, str) for x in ak):
                raise GradingOutputValidationError(
                    "_modality.artifact_keys must be a list of strings or omitted"
                )
        etc = mod.get("extracted_text_chars")
        if etc is not None:
            try:
                mod["extracted_text_chars"] = int(etc)
            except (TypeError, ValueError) as e:
                raise GradingOutputValidationError(
                    "_modality.extracted_text_chars must be an int or omitted"
                ) from e
        sig = mod.get("signals")
        if sig is not None and not isinstance(sig, dict):
            raise GradingOutputValidationError(
                "_modality.signals must be a dict or omitted"
            )
        data["_modality"] = mod

    if allowed_criterion_names:
        crit_top, al_top = filter_criteria_dicts_to_allowlist(
            data.get("criteria") or [],
            allowed_criterion_names,
            context="top_level",
        )
        data["criteria"] = [
            _normalize_one_criterion_dict(c, index=f"top:{i}")
            for i, c in enumerate(crit_top)
        ]
        fl = data.get("flags")
        if not isinstance(fl, list):
            data["flags"] = []
            fl = data["flags"]
        for msg in al_top[:80]:
            tag = f"rubric_allowlist:{msg}"
            if tag not in fl:
                fl.append(tag)
        qg = data.get("question_grades")
        if isinstance(qg, list):
            for row in qg:
                if not isinstance(row, dict):
                    continue
                cid = str(row.get("chunk_id") or "question_grade")
                qc = row.get("criteria") or []
                qf, al_q = filter_criteria_dicts_to_allowlist(
                    [c for c in qc if isinstance(c, dict)],
                    allowed_criterion_names,
                    context=cid,
                )
                row["criteria"] = [
                    _normalize_one_criterion_dict(c, index=f"{cid}:{j}")
                    for j, c in enumerate(qf)
                ]
                _sync_question_grade_overall_from_criteria(row)
                for msg in al_q[:40]:
                    t2 = f"rubric_allowlist:{msg}"
                    if t2 not in fl:
                        fl.append(t2)
        if crit_top:
            qg_scores: list[float] = []
            if isinstance(data.get("question_grades"), list):
                for row in data["question_grades"]:
                    if not isinstance(row, dict):
                        continue
                    qov = row.get("overall")
                    if isinstance(qov, dict) and "score" in qov:
                        try:
                            qg_scores.append(float(qov["score"]))
                        except (TypeError, ValueError):
                            continue
            if qg_scores:
                data["overall"]["score"] = round(
                    sum(qg_scores) / len(qg_scores), 6
                )
            else:
                data["overall"]["score"] = _mean_criterion_fraction_from_crits(crit_top)
            _normalize_overall_score_fraction(data["overall"])
            cap_top = sum(
                float(r.get("max_points", 0))
                for r in crit_top
                if isinstance(r, dict)
            )
            if cap_top > 0:
                data["overall"]["max_points"] = round(cap_top, 4)
                data["overall"]["rubric_points_earned"] = round(
                    float(data["overall"]["score"]) * cap_top, 4
                )
            data["overall"]["confidence"] = _coerce_confidence(
                _mean_criterion_confidence_from_rows(crit_top)
            )

    _resync_assignment_overall_from_question_grades(data)

    return data


def validate_grading_output_lenient(
    data: dict[str, Any],
    *,
    allowed_criterion_names: frozenset[str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Like :func:`validate_grading_output` but returns ``(data, issues)`` for soft checks."""
    issues: list[str] = []
    if not isinstance(data.get("criteria"), list):
        issues.append("criteria is not a list")
    elif len(data.get("criteria") or []) == 0:
        issues.append("criteria is empty")
    return validate_grading_output(
        data, allowed_criterion_names=allowed_criterion_names
    ), issues
