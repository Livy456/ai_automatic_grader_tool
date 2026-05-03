"""
Load the four assignment-level generic rubrics from ``rubric/`` as separate files.

Filenames match the course templates (``[Generic] …``). Callers pass the resulting
``dict[RubricType, list[dict]]`` into :class:`MultimodalGradingPipeline` so
:func:`~app.grading.multimodal.rubric_router.route_rubric` and
:func:`~app.grading.multimodal.custom_rubric_export.apply_custom_rubric_plan_to_chunks`
can pick one template for the assignment and filter criteria per chunk.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .schemas import RubricType

_log = logging.getLogger(__name__)

# Canonical on-disk names (repo ``rubric/`` directory).
GENERIC_RUBRIC_FILENAME: dict[RubricType, str] = {
    RubricType.PROGRAMMING_SCAFFOLDED: "[Generic] Scaffolded Coding Rubric.json",
    RubricType.FREE_RESPONSE: "[Generic] Free Response Rubric.json",
    RubricType.EDA_VISUALIZATION: "[Generic] Open-Ended EDA Rubric.json",
    RubricType.ORAL_INTERVIEW: "[Generic] Mock Interview.json",
}

_SECTION_NAME_FOR_TYPE: dict[RubricType, str] = {
    RubricType.PROGRAMMING_SCAFFOLDED: "Scaffolded Coding",
    RubricType.FREE_RESPONSE: "Free Response",
    RubricType.EDA_VISUALIZATION: "Open-Ended EDA",
    RubricType.ORAL_INTERVIEW: "Mock Interview / Oral Assessment",
}


def _max_points_from_range(points_range: object) -> float:
    if points_range is None:
        return 10.0
    s = str(points_range).strip().replace(" ", "")
    if "-" in s:
        parts = s.split("-", 1)
        try:
            return float(parts[1])
        except (IndexError, ValueError):
            pass
    try:
        return float(s)
    except ValueError:
        return 10.0


def _row_from_criterion(c: dict[str, Any]) -> dict[str, Any]:
    name = str(c.get("name") or "Criterion").strip()
    max_pts = _max_points_from_range(c.get("points_range"))
    levels = c.get("levels")
    desc = json.dumps(levels, ensure_ascii=False) if isinstance(levels, dict) else ""
    return {
        "name": name,
        "max_points": max_pts,
        "criterion": name,
        "max_score": max_pts,
        "description": desc,
    }


def _rows_from_criteria_list(criteria: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for c in criteria:
        if isinstance(c, dict):
            out.append(_row_from_criterion(c))
    return out


def _section_by_name(raw: dict[str, Any], want: str) -> dict[str, Any] | None:
    for sec in raw.get("sections") or []:
        if not isinstance(sec, dict):
            continue
        if str(sec.get("name") or "").strip() == want:
            return sec
    return None


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _log.warning("generic_rubric: could not read %s (%s)", path, exc)
        return None
    return obj if isinstance(obj, dict) else None


def rows_for_generic_file(rt: RubricType, path: Path) -> list[dict[str, Any]]:
    """Parse one generic JSON file into flat grader rows for ``rt`` only."""
    raw = _load_json(path)
    if not raw:
        return []

    if rt == RubricType.PROGRAMMING_SCAFFOLDED:
        want = _SECTION_NAME_FOR_TYPE[rt]
        sec = _section_by_name(raw, want)
        if sec is None and isinstance(raw.get("sections"), list) and raw["sections"]:
            sec0 = raw["sections"][0]
            sec = sec0 if isinstance(sec0, dict) else None
        if not isinstance(sec, dict):
            return []
        crit = sec.get("criteria")
        return _rows_from_criteria_list(crit) if isinstance(crit, list) else []

    if rt in (RubricType.FREE_RESPONSE, RubricType.EDA_VISUALIZATION):
        if isinstance(raw.get("criteria"), list):
            return _rows_from_criteria_list(raw["criteria"])
        want = _SECTION_NAME_FOR_TYPE[rt]
        sec = _section_by_name(raw, want)
        if isinstance(sec, dict) and isinstance(sec.get("criteria"), list):
            return _rows_from_criteria_list(sec["criteria"])
        return []

    if rt == RubricType.ORAL_INTERVIEW:
        want = _SECTION_NAME_FOR_TYPE[rt]
        sec = _section_by_name(raw, want)
        if isinstance(sec, dict) and isinstance(sec.get("criteria"), list):
            return _rows_from_criteria_list(sec["criteria"])
        return []

    return []


def load_four_generic_rubric_rows_by_type(
    rubric_dir: Path,
) -> dict[RubricType, list[dict[str, Any]]]:
    """Load all four generic files from ``rubric_dir`` (missing → empty list for that type)."""
    out: dict[RubricType, list[dict[str, Any]]] = {}
    for rt, fname in GENERIC_RUBRIC_FILENAME.items():
        path = rubric_dir / fname
        rows = rows_for_generic_file(rt, path)
        out[rt] = rows
        if not rows and path.is_file():
            _log.warning("generic_rubric: no rows parsed for %s from %s", rt.value, fname)
    return out


def four_generic_rubric_files_present(rubric_dir: Path) -> bool:
    """True when all four canonical ``[Generic] …`` JSON files exist."""
    return all((rubric_dir / fname).is_file() for fname in GENERIC_RUBRIC_FILENAME.values())


def flat_rubric_rows_from_by_type(
    by_type: dict[RubricType, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Stable concat of rows (type order: scaffolded, EDA, free response, oral)."""
    order = (
        RubricType.PROGRAMMING_SCAFFOLDED,
        RubricType.EDA_VISUALIZATION,
        RubricType.FREE_RESPONSE,
        RubricType.ORAL_INTERVIEW,
    )
    out: list[dict[str, Any]] = []
    for rt in order:
        out.extend(list(by_type.get(rt) or []))
    return out


def merge_four_generics_to_sections_document(
    rubric_dir: Path,
) -> dict[str, Any] | None:
    """
    Build one ``{"sections": [...]}`` object suitable for legacy ``_build_rubric_rows_by_type``.

    Used when ``rubric/default.json`` is absent but the four ``[Generic]`` files exist.
    """
    if not four_generic_rubric_files_present(rubric_dir):
        return None
    sections: list[dict[str, Any]] = []
    for rt in (
        RubricType.PROGRAMMING_SCAFFOLDED,
        RubricType.FREE_RESPONSE,
        RubricType.EDA_VISUALIZATION,
        RubricType.ORAL_INTERVIEW,
    ):
        path = rubric_dir / GENERIC_RUBRIC_FILENAME[rt]
        raw = _load_json(path)
        if not raw:
            return None
        want = _SECTION_NAME_FOR_TYPE[rt]
        if isinstance(raw.get("criteria"), list) and rt in (
            RubricType.FREE_RESPONSE,
            RubricType.EDA_VISUALIZATION,
        ):
            sections.append(
                {
                    "name": want,
                    "criteria": raw["criteria"],
                }
            )
            continue
        sec = _section_by_name(raw, want)
        if isinstance(sec, dict) and isinstance(sec.get("criteria"), list):
            sections.append(
                {
                    "name": want,
                    "criteria": sec["criteria"],
                }
            )
            continue
        if rt == RubricType.PROGRAMMING_SCAFFOLDED and isinstance(
            raw.get("sections"), list
        ):
            s0 = raw["sections"][0]
            if isinstance(s0, dict) and isinstance(s0.get("criteria"), list):
                sections.append({"name": want, "criteria": s0["criteria"]})
                continue
        return None
    prose_parts: list[str] = []
    for rt, fname in GENERIC_RUBRIC_FILENAME.items():
        raw = _load_json(rubric_dir / fname) or {}
        instr = raw.get("llm_grading_instructions")
        if isinstance(instr, str) and instr.strip():
            prose_parts.append(f"[{fname}]\n{instr.strip()}")
    out: dict[str, Any] = {"sections": sections}
    if prose_parts:
        out["llm_grading_instructions"] = "\n\n".join(prose_parts)
    return out
