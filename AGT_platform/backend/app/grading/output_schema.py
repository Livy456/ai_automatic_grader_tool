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

from app.grading.aggregation import weighted_overall_confidence, weighted_overall_score
from app.grading.rubric_allowlist import filter_criteria_dicts_to_allowlist

_log = logging.getLogger(__name__)

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
    Best-effort normalization so locally hosted LLMs (legacy ``grade()`` path) match
    :func:`validate_grading_output`. Mutates and returns the same dict.

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
            data["overall"]["score"] = weighted_overall_score(rows)
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
                "score": weighted_overall_score(rows),
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

    return data


def validate_grading_output(
    data: dict[str, Any],
    *,
    allowed_criterion_names: frozenset[str] | None = None,
) -> dict[str, Any]:
    """
    Validate ``run_grading_pipeline`` / task result shape in-place (normalizes some fields).

    Required top-level keys: ``overall`` (dict with ``score``), ``criteria`` (list).

    When ``allowed_criterion_names`` is set (canonical LMS rubric names for this
    assignment), criteria rows are **filtered** to that set and remapped to canonical
    spelling; unknown names are removed and ``flags`` gain ``rubric_allowlist:*`` entries.
    ``overall.score`` is recomputed from the filtered assignment-level criteria when any
    filtering occurs.

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
    data["overall"] = overall

    crit_raw = data.get("criteria")
    if crit_raw is None:
        data["criteria"] = []
        crit_raw = []
    if not isinstance(crit_raw, list):
        raise GradingOutputValidationError("criteria must be a list")

    criteria_out: list[dict[str, Any]] = []
    for i, row in enumerate(crit_raw):
        if not isinstance(row, dict):
            raise GradingOutputValidationError(f"criteria[{i}] must be a dict")
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

        if c.get("calibrated_credit") is not None:
            c["calibrated_credit"] = _coerce_float(c["calibrated_credit"])
        if c.get("raw_rubric_score") is not None:
            c["raw_rubric_score"] = _coerce_float(c["raw_rubric_score"])

        criteria_out.append(c)
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
            for crit in row.get("criteria") or []:
                if isinstance(crit, dict):
                    crit.setdefault("evidence", "")
                    crit.setdefault("reasoning", "")
                    crit.setdefault("justification", "")
                    if crit.get("calibrated_credit") is not None:
                        crit["calibrated_credit"] = _coerce_float(
                            crit["calibrated_credit"]
                        )
                    if crit.get("raw_rubric_score") is not None:
                        crit["raw_rubric_score"] = _coerce_float(
                            crit["raw_rubric_score"]
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
        data["criteria"] = crit_top
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
                row["criteria"] = qf
                for msg in al_q[:40]:
                    t2 = f"rubric_allowlist:{msg}"
                    if t2 not in fl:
                        fl.append(t2)
        if crit_top:
            data["overall"]["score"] = weighted_overall_score(crit_top)
            data["overall"]["confidence"] = _coerce_confidence(
                weighted_overall_confidence(crit_top)
            )

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
