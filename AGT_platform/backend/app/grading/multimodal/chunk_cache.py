"""
Serialize / deserialize :class:`GradingChunk` lists so chunking + unit embeddings run once.

The multimodal pipeline can read ``modality_hints["multimodal_chunk_cache_path"]`` and skip
``build_multimodal_grading_chunks`` + ``enrich_chunks_with_rag_embeddings`` when vectors are
present in the cached ``evidence["rag_embedding_bundle"]`` (and optional ``trio_segment_rag``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .schemas import GradingChunk, Modality, RubricType, TaskType

_log = logging.getLogger(__name__)


def _modality_from_value(raw: str) -> Modality:
    for m in Modality:
        if m.value == raw:
            return m
    return Modality.UNKNOWN


def _task_from_value(raw: str) -> TaskType:
    for t in TaskType:
        if t.value == raw:
            return t
    return TaskType.UNKNOWN


def _rubric_from_value(raw: str | None) -> RubricType | None:
    if not raw:
        return None
    for r in RubricType:
        if r.value == raw:
            return r
    return None


def grading_chunk_to_record(ch: GradingChunk) -> dict[str, Any]:
    return {
        "chunk_id": ch.chunk_id,
        "assignment_id": ch.assignment_id,
        "student_id": ch.student_id,
        "question_id": ch.question_id,
        "modality": ch.modality.value,
        "task_type": ch.task_type.value,
        "extracted_text": ch.extracted_text,
        "rubric_version": ch.rubric_version,
        "parent_chunk_id": ch.parent_chunk_id,
        "raw_content_ref": ch.raw_content_ref,
        "evidence": dict(ch.evidence or {}),
        "source_refs": list(ch.source_refs or []),
        "rubric_type": ch.rubric_type.value if ch.rubric_type else None,
        "rubric_rows": list(ch.rubric_rows or []),
        "routing_reason": ch.routing_reason,
        "classifier_fallback_used": ch.classifier_fallback_used,
    }


def grading_chunk_from_record(d: dict[str, Any]) -> GradingChunk:
    return GradingChunk(
        chunk_id=str(d["chunk_id"]),
        assignment_id=str(d["assignment_id"]),
        student_id=str(d["student_id"]),
        question_id=str(d["question_id"]),
        modality=_modality_from_value(str(d.get("modality") or "unknown")),
        task_type=_task_from_value(str(d.get("task_type") or "unknown")),
        extracted_text=str(d.get("extracted_text") or ""),
        rubric_version=str(d.get("rubric_version") or ""),
        parent_chunk_id=d.get("parent_chunk_id"),
        raw_content_ref=d.get("raw_content_ref"),
        evidence=dict(d.get("evidence") or {}),
        source_refs=list(d.get("source_refs") or []),
        rubric_type=_rubric_from_value(d.get("rubric_type")),
        rubric_rows=list(d.get("rubric_rows") or []),
        routing_reason=str(d.get("routing_reason") or ""),
        classifier_fallback_used=bool(d.get("classifier_fallback_used")),
    )


def save_grading_chunks_cache(path: Path, chunks: list[GradingChunk]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [grading_chunk_to_record(c) for c in chunks]
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def load_grading_chunks_cache(path: Path) -> list[GradingChunk] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        _log.warning("chunk_cache: could not read %s (%s)", path, exc)
        return None
    if not isinstance(data, list):
        return None
    out: list[GradingChunk] = []
    for row in data:
        if isinstance(row, dict):
            try:
                out.append(grading_chunk_from_record(row))
            except (KeyError, TypeError, ValueError) as exc:
                _log.warning("chunk_cache: bad row in %s (%s)", path, exc)
                return None
    return out if out else None


def chunks_have_unit_embeddings(chunks: list[GradingChunk]) -> bool:
    for ch in chunks:
        bundle = (ch.evidence or {}).get("rag_embedding_bundle")
        if not isinstance(bundle, dict):
            return False
        emb = bundle.get("embedding")
        if not isinstance(emb, list) or len(emb) < 8:
            return False
    return bool(chunks)
