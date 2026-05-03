"""
DB / Celery entry: map assignment + raw artifact bytes → multimodal pipeline → grading dict.

Produces the same top-level shape as the removed legacy ``run_grading_pipeline`` after
:func:`~app.grading.output_schema.coerce_grading_output_shape`.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.grading.modality_resolution import (
    augment_prompt_for_modality_profile,
    infer_modality_from_artifacts,
    resolve_modality_profile,
)
from app.grading.output_schema import coerce_grading_output_shape
from app.grading.submission_text import submission_text_from_artifacts

from .generic_rubric_loader import (
    _row_from_criterion,
    flat_rubric_rows_from_by_type,
)
from .grading_output import multimodal_assignment_to_grading_dict
from .ingestion import ingest_raw_submission
from .pipeline import create_multimodal_pipeline_from_app_config
from .rubric_fallback import DEFAULT_STANDALONE_RUBRIC
from .schemas import MultimodalGradingConfig, RubricType

_log = logging.getLogger(__name__)

_SECTION_NAME_TO_RUBRIC_TYPE: dict[str, RubricType] = {
    "Scaffolded Coding": RubricType.PROGRAMMING_SCAFFOLDED,
    "Free Response": RubricType.FREE_RESPONSE,
    "Open-Ended EDA": RubricType.EDA_VISUALIZATION,
    "Mock Interview / Oral Assessment": RubricType.ORAL_INTERVIEW,
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


def _flatten_sections_rubric(raw: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for sec in raw.get("sections") or []:
        if not isinstance(sec, dict):
            continue
        sec_name = str(sec.get("name") or "Section").strip()
        for c in sec.get("criteria") or []:
            if not isinstance(c, dict):
                continue
            cname = str(c.get("name") or "Criterion").strip()
            max_pts = _max_points_from_range(c.get("points_range"))
            levels = c.get("levels")
            desc = (
                json.dumps(levels, ensure_ascii=False) if isinstance(levels, dict) else ""
            )[:8000]
            label = f"{sec_name} — {cname}" if sec_name else cname
            out.append(
                {
                    "name": label,
                    "max_points": max_pts,
                    "criterion": label,
                    "max_score": max_pts,
                    "description": desc,
                }
            )
    return out


def _build_rubric_rows_by_type_from_sections_doc(
    rubric_json: dict[str, Any],
) -> dict[RubricType, list[dict[str, Any]]]:
    by_type: dict[RubricType, list[dict[str, Any]]] = {}
    for sec in rubric_json.get("sections") or []:
        if not isinstance(sec, dict):
            continue
        sec_name = str(sec.get("name") or "").strip()
        rt = _SECTION_NAME_TO_RUBRIC_TYPE.get(sec_name)
        if rt is None:
            continue
        rows: list[dict[str, Any]] = []
        for c in sec.get("criteria") or []:
            if not isinstance(c, dict):
                continue
            rows.append(_row_from_criterion(c))
        by_type[rt] = rows
    return by_type


def _coerce_flat_rubric_rows(raw_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for c in raw_list:
        if not isinstance(c, dict):
            continue
        if c.get("points_range") is not None or isinstance(c.get("levels"), dict):
            out.append(_row_from_criterion(c))
            continue
        name = str(c.get("name") or c.get("criterion") or "").strip()
        if not name:
            continue
        try:
            mp = float(
                c.get("max_points")
                if c.get("max_points") is not None
                else (c.get("max_score") if c.get("max_score") is not None else 10.0)
            )
        except (TypeError, ValueError):
            mp = 10.0
        desc = str(c.get("description") or "")[:8000]
        out.append(
            {
                "name": name,
                "max_points": mp,
                "criterion": name,
                "max_score": mp,
                "description": desc,
            }
        )
    return out


def rubric_column_to_by_type_and_flat(
    rubric_column: Any,
) -> tuple[dict[RubricType, list[dict[str, Any]]], list[dict[str, Any]]]:
    """
    From an LMS ``Assignment.rubric`` JSON column, build ``rubric_rows_by_type`` and a flat
    list for :func:`multimodal_assignment_to_grading_dict` allowlisting.
    """
    default_flat = [dict(x) for x in DEFAULT_STANDALONE_RUBRIC]
    if rubric_column is None:
        return {RubricType.FREE_RESPONSE: _coerce_flat_rubric_rows(default_flat)}, default_flat

    if isinstance(rubric_column, dict) and isinstance(rubric_column.get("sections"), list):
        by_typed = _build_rubric_rows_by_type_from_sections_doc(rubric_column)
        if by_typed:
            flat = flat_rubric_rows_from_by_type(by_typed)
            return by_typed, flat if flat else default_flat
        flat_sec = _flatten_sections_rubric(rubric_column)
        if flat_sec:
            rows = _coerce_flat_rubric_rows(flat_sec)
            if rows:
                return {RubricType.FREE_RESPONSE: rows}, rows

    raw_list: list[dict[str, Any]] = []
    if isinstance(rubric_column, list):
        raw_list = [x for x in rubric_column if isinstance(x, dict)]
    elif isinstance(rubric_column, dict):
        for key in ("rubric", "criteria", "items"):
            chunk = rubric_column.get(key)
            if isinstance(chunk, list):
                raw_list = [x for x in chunk if isinstance(x, dict)]
                break

    if not raw_list:
        return {RubricType.FREE_RESPONSE: _coerce_flat_rubric_rows(default_flat)}, default_flat

    rows = _coerce_flat_rubric_rows(raw_list)
    if not rows:
        return {RubricType.FREE_RESPONSE: _coerce_flat_rubric_rows(default_flat)}, default_flat
    return {RubricType.FREE_RESPONSE: rows}, rows


def _compose_assignment_task_description(
    assignment: Any,
    rubric_text: str | None,
    answer_key_text: str | None,
) -> str:
    parts: list[str] = []
    base = (
        getattr(assignment, "description", None) or getattr(assignment, "title", None) or ""
    ).strip()
    if base:
        parts.append(base)
    if answer_key_text and str(answer_key_text).strip():
        parts.append(
            "Answer key / reference (instructor context):\n" + str(answer_key_text).strip()
        )
    if rubric_text and str(rubric_text).strip():
        parts.append("Additional rubric notes:\n" + str(rubric_text).strip())
    return "\n\n".join(parts) if parts else "Grade this submission."


def _run_multimodal_once(
    cfg: Any,
    assignment: Any,
    artifacts_bytes: dict[str, bytes],
    *,
    assignment_id: int,
    envelope_student_id: str,
    rubric_text: str | None,
    answer_key_text: str | None,
    rubric_column: Any,
) -> dict[str, Any]:
    plaintext = submission_text_from_artifacts(artifacts_bytes).strip()
    modality = getattr(assignment, "modality", None) or infer_modality_from_artifacts(
        artifacts_bytes
    )
    profile = resolve_modality_profile(assignment, artifacts_bytes, plaintext)
    if profile.get("signals", {}).get("text_too_short_for_grading"):
        _log.warning(
            "multimodal course: submission text very short (%s chars); scores may be unreliable",
            profile.get("extracted_text_chars"),
        )

    base_desc = _compose_assignment_task_description(
        assignment, rubric_text, answer_key_text
    )
    augmented = augment_prompt_for_modality_profile(base_desc, profile)

    rubric_rows_by_type, flat_rubric = rubric_column_to_by_type_and_flat(rubric_column)

    hints: dict[str, Any] = {
        "answer_key_plaintext": (answer_key_text or "").strip(),
    }
    envelope = ingest_raw_submission(
        assignment_id=str(assignment_id),
        student_id=envelope_student_id,
        artifacts=dict(artifacts_bytes),
        extracted_plaintext=plaintext,
        modality_hints=hints,
    )

    mm_cfg = MultimodalGradingConfig(require_answer_key=False)
    pipeline = create_multimodal_pipeline_from_app_config(
        cfg,
        multimodal_cfg=mm_cfg,
        rubric_rows_by_type=rubric_rows_by_type,
        classifier=None,
        task_description=augmented,
    )
    mm_result = pipeline.run(envelope)
    out = multimodal_assignment_to_grading_dict(
        mm_result,
        rubric=flat_rubric,
        modality_profile={
            "modality": profile.get("modality"),
            "modality_subtype": profile.get("modality_subtype"),
            "artifact_keys": list(profile.get("artifact_keys") or []),
            "extracted_text_chars": profile.get("extracted_text_chars"),
            "signals": profile.get("signals")
            if isinstance(profile.get("signals"), dict)
            else {},
        },
    )
    return coerce_grading_output_shape(out)


def run_db_submission_multimodal_pipeline(
    cfg: Any,
    assignment: Any,
    artifacts_bytes: dict[str, bytes],
    *,
    submission_id: int,
    assignment_id: int,
    student_id: int | None,
    rubric_text: str | None,
    answer_key_text: str | None,
) -> dict[str, Any]:
    """Course or public autograder row: grade using :class:`MultimodalGradingPipeline`."""
    envelope_sid = (
        str(student_id) if student_id is not None else f"anon_sub_{submission_id}"
    )
    return _run_multimodal_once(
        cfg,
        assignment,
        artifacts_bytes,
        assignment_id=assignment_id,
        envelope_student_id=envelope_sid,
        rubric_text=rubric_text,
        answer_key_text=answer_key_text,
        rubric_column=getattr(assignment, "rubric", None),
    )


def run_standalone_multimodal_pipeline(
    cfg: Any,
    artifacts_bytes: dict[str, bytes],
    submission_id: int,
    title: str,
    rubric_text: str | None,
    answer_key_text: str | None,
    rubric_file_excerpt: str | None,
    answer_key_file_excerpt: str | None,
    grading_instructions: str | None = None,
) -> dict[str, Any]:
    """Standalone autograder: default structured rubric; prose rubric/AK in the prompt."""
    from types import SimpleNamespace

    merged_rubric_note_parts: list[str] = []
    if rubric_text and rubric_text.strip():
        merged_rubric_note_parts.append(rubric_text.strip())
    if rubric_file_excerpt and rubric_file_excerpt.strip():
        merged_rubric_note_parts.append(
            "Rubric (from uploaded file):\n" + rubric_file_excerpt.strip()
        )
    merged_rubric = "\n\n".join(merged_rubric_note_parts) if merged_rubric_note_parts else None

    merged_ak_parts: list[str] = []
    if answer_key_text and answer_key_text.strip():
        merged_ak_parts.append(answer_key_text.strip())
    if answer_key_file_excerpt and answer_key_file_excerpt.strip():
        merged_ak_parts.append(
            "Answer key (from uploaded file):\n" + answer_key_file_excerpt.strip()
        )
    merged_ak = "\n\n".join(merged_ak_parts) if merged_ak_parts else None

    desc_parts: list[str] = []
    base_title = (title or "Standalone submission").strip()
    if base_title:
        desc_parts.append(base_title)
    if grading_instructions and str(grading_instructions).strip():
        desc_parts.append(
            "Instructor grading instructions:\n" + str(grading_instructions).strip()
        )
    description = (
        "\n\n".join(desc_parts) if desc_parts else "Standalone autograder submission"
    )

    pseudo = SimpleNamespace(
        modality=infer_modality_from_artifacts(artifacts_bytes),
        rubric=list(DEFAULT_STANDALONE_RUBRIC),
        title=title or "Standalone submission",
        description=description,
    )
    return _run_multimodal_once(
        cfg,
        pseudo,
        artifacts_bytes,
        assignment_id=0,
        envelope_student_id=f"standalone_{submission_id}",
        rubric_text=merged_rubric,
        answer_key_text=merged_ak,
        rubric_column=list(DEFAULT_STANDALONE_RUBRIC),
    )
