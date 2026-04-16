"""
Chunking: build grading chunks (units) from ingestion + optional LMS question map.

Default path (non-ipynb): **structured submission chunking** — reflows each
``=== PDF TEXT ===`` region via :func:`app.grading.submission_chunks.reflow_pdf_sections_in_plaintext`,
then :func:`app.grading.submission_chunks.build_submission_chunks` (PDF vertical reflow again
per section, journal-style prompt boundaries when modality hints match) and
:func:`app.grading.grading_units.build_grading_units_from_chunks` to form Q/A units.
"""

from __future__ import annotations

from typing import Any, Protocol

from app.grading.grading_units import build_grading_units_from_chunks
from app.grading.submission_chunks import (
    build_submission_chunks,
    reflow_pdf_sections_in_plaintext,
)
from .ingestion import IngestionEnvelope
from .schemas import GradingChunk, Modality, TaskType


class GradingChunker(Protocol):
    def build_chunks(self, envelope: IngestionEnvelope) -> list[GradingChunk]: ...


def modality_from_hints(hints: dict[str, Any], default: Modality = Modality.UNKNOWN) -> Modality:
    raw = str(hints.get("modality") or "").strip().lower()
    for m in Modality:
        if m.value == raw:
            return m
    return default


def task_type_from_hints(hints: dict[str, Any], default: TaskType = TaskType.UNKNOWN) -> TaskType:
    raw = str(hints.get("task_type") or "").strip().lower()
    for t in TaskType:
        if t.value == raw:
            return t
    return default


def default_chunker_build_units(
    envelope: IngestionEnvelope,
    *,
    max_chunk_chars: int | None = None,
    modality_subtype: str = "",
) -> list[GradingChunk]:
    """
    Split ``extracted_plaintext`` into units (question + response bundles).

    Each unit becomes one ``GradingChunk`` with ``question_id`` = ``pair_X`` if
    instructor IDs are absent.

    ``envelope.modality_hints`` may override sizing with ``max_chunk_chars`` (int or
    ``null``/omitted for no cap) and ``modality_subtype`` (str) when callers omit kwargs.
    Optional ``max_grading_units`` (positive int) keeps only the first N units (for
    tests or cost limits).
    """
    plain = reflow_pdf_sections_in_plaintext((envelope.extracted_plaintext or "").strip())
    if not plain:
        return []

    hints = envelope.modality_hints or {}
    mc_hint = hints.get("max_chunk_chars")
    if mc_hint is not None:
        try:
            max_chunk_chars = int(mc_hint)
        except (TypeError, ValueError):
            pass
    st_hint = hints.get("modality_subtype")
    if st_hint is not None and str(st_hint).strip():
        modality_subtype = str(st_hint).strip()
    elif not modality_subtype:
        arts = envelope.artifacts or {}
        has_pdf = bool(isinstance(arts, dict) and arts.get("pdf"))
        u = plain.upper()
        if has_pdf or "=== PDF TEXT ===" in u:
            modality_subtype = "free_response"
        elif "=== NOTEBOOK" in u:
            modality_subtype = "notebook"
        else:
            modality_subtype = "notebook"

    chunks = build_submission_chunks(
        plain,
        assignment_title=envelope.assignment_id,
        modality_subtype=modality_subtype,
        max_chunk_chars=max_chunk_chars,
    )
    units = build_grading_units_from_chunks(chunks)
    modality = modality_from_hints(envelope.modality_hints)
    task = task_type_from_hints(envelope.modality_hints)
    out: list[GradingChunk] = []
    for u in units:
        pid = u.get("pair_id")
        qid = f"pair_{pid}" if pid is not None else "orphan"
        text_parts = [
            u.get("question_text") or "",
            u.get("response_text") or "",
        ]
        extracted = "\n\n".join(p for p in text_parts if str(p).strip()).strip()
        out.append(
            GradingChunk(
                chunk_id=f"{envelope.student_id}:{envelope.assignment_id}:{qid}",
                assignment_id=envelope.assignment_id,
                student_id=envelope.student_id,
                question_id=qid,
                modality=modality,
                task_type=task,
                extracted_text=extracted,
                evidence={
                    "chunk_ids": u.get("chunk_ids"),
                    "unit": u,
                },
            )
        )
    cap = hints.get("max_grading_units")
    if cap is not None:
        try:
            n = int(cap)
            if n >= 1:
                out = out[:n]
        except (TypeError, ValueError):
            pass
    return out
