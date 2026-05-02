"""
LLM-driven **question inventory** from a blank instructor ``.ipynb`` (``blank_assignments/``).

When a structure LLM is available (same stack as trio relabeling), we ask it to list
distinct gradable questions, then attach the best-matching student work (by embedding
cosine similarity) from :func:`notebook_chunker.build_notebook_qa_chunks`.

Falls back to heuristic template alignment in :mod:`template_aligned_notebook_chunks` when
this path returns nothing.

Uses **lazy** imports of :mod:`rag_embeddings` to avoid import cycles with that module.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from app.config import Config

from .ingestion import IngestionEnvelope
from .notebook_chunker import (
    build_notebook_qa_chunks,
    ipynb_to_plaintext_for_structure_llm,
)
from .schemas import GradingChunk, Modality, TaskType
from .chunker import modality_from_hints, task_type_from_hints
from .answer_key_chunk_enrich import _chunk_query_text, _cosine

_log = logging.getLogger(__name__)

_BLANK_Q_SYSTEM = """You are parsing an **instructor assignment template** (questions and
instructions only — no student answers). List every **separately gradable** item: each
distinct exercise, coding task, or written prompt students must complete.

Return **only** valid JSON (no markdown fences):
{"questions":[{"question_id":"short stable id","prompt":"verbatim or condensed prompt text"}]}

Rules:
- ``question_id``: use numbering from the document when visible (e.g. 1.1, 2.3a); else q1, q2.
- ``prompt``: the full instructor-facing text for that item (may include markdown).
- Do **not** invent items; skip purely informational cells with no student deliverable.
- Order questions as they appear in the template."""


def _blank_llm_questions_mode(cfg: Config | None) -> str:
    if cfg is not None:
        raw = str(getattr(cfg, "MULTIMODAL_BLANK_LLM_QUESTIONS", "") or "").strip().lower()
        if raw:
            return raw
    return (os.getenv("MULTIMODAL_BLANK_LLM_QUESTIONS", "auto") or "auto").strip().lower()


def blank_llm_questions_requested(*, blank_bytes: bytes, cfg: Config | None) -> bool:
    mode = _blank_llm_questions_mode(cfg)
    if mode in ("0", "false", "no", "off"):
        return False
    if mode in ("1", "true", "yes", "on"):
        return bool(blank_bytes.strip()) if blank_bytes else False
    return bool(blank_bytes and len(blank_bytes) > 32 and cfg is not None)


def _safe_qid(raw: str, idx: int) -> str:
    s = re.sub(r"\s+", " ", (raw or "").strip())
    return s[:120] if s else f"unit_{idx + 1}"


def _questions_from_blank_llm(plain: str, cfg: Config) -> list[dict[str, str]]:
    from . import rag_embeddings as _rag

    client, model = _rag._multimodal_structure_chat_client(cfg, purpose="trio")
    if client is None:
        return []
    cap = 28_000
    user = (plain or "").strip()[:cap]
    if len(user) < 20:
        return []
    try:
        obj = client.chat_json(
            [
                {"role": "system", "content": _BLANK_Q_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
        )
    except Exception:
        _log.warning("blank LLM question list failed model=%s", model, exc_info=True)
        return []
    if not isinstance(obj, dict):
        return []
    units = obj.get("questions")
    if not isinstance(units, list):
        return []
    out: list[dict[str, str]] = []
    for u in units:
        if not isinstance(u, dict):
            continue
        qid = str(u.get("question_id") or u.get("id") or "").strip()
        pr = str(u.get("prompt") or u.get("question") or "").strip()
        if not pr:
            continue
        if not qid:
            qid = f"q{len(out) + 1}"
        out.append({"question_id": qid, "prompt": pr})
    return out


def _best_student_chunk_for_prompt(
    prompt: str,
    student_chunks: list[GradingChunk],
    cfg: Config,
) -> GradingChunk | None:
    from app.grading.rag_embeddings import compute_submission_embedding

    if not student_chunks:
        return None
    pq = (prompt or "").strip()[:12_000]
    if not pq.strip():
        return student_chunks[0]
    try:
        q_vec, _ = compute_submission_embedding(pq, cfg)
    except Exception:
        _log.debug("blank LLM match: question embed failed", exc_info=True)
        return student_chunks[0]

    best: tuple[float, GradingChunk] = (-1.0, student_chunks[0])
    for ch in student_chunks:
        blob = _chunk_query_text(ch).strip()[:12_000] or (ch.extracted_text or "").strip()[:12_000]
        if not blob:
            continue
        try:
            b_vec, _ = compute_submission_embedding(blob, cfg)
            sim = _cosine(q_vec, b_vec)
        except Exception:
            continue
        if sim > best[0]:
            best = (sim, ch)
    return best[1]


def try_build_llm_blank_aligned_notebook_chunks(
    envelope: IngestionEnvelope,
    *,
    blank_ipynb_bytes: bytes,
    cfg: Config,
) -> tuple[list[GradingChunk], str] | None:
    """
    Return chunks when the LLM returns at least one question and the student notebook parses.
    """
    if not blank_llm_questions_requested(blank_bytes=blank_ipynb_bytes, cfg=cfg):
        return None
    raw = (envelope.artifacts or {}).get("ipynb")
    if not isinstance(raw, (bytes, bytearray)):
        return None
    student_bytes = bytes(raw)
    if not student_bytes.strip():
        return None

    plain = ipynb_to_plaintext_for_structure_llm(blank_ipynb_bytes)
    llm_qs = _questions_from_blank_llm(plain, cfg)
    if not llm_qs:
        return None

    hints = envelope.modality_hints or {}
    aid = envelope.assignment_id
    sid = envelope.student_id
    nb_mod = modality_from_hints(hints)
    if nb_mod == Modality.UNKNOWN:
        nb_mod = Modality.NOTEBOOK
    task = task_type_from_hints(hints)

    cap = hints.get("max_grading_units")
    max_units: int | None = None
    if cap is not None:
        try:
            max_units = int(cap)
        except (TypeError, ValueError):
            max_units = None

    student_chunks = build_notebook_qa_chunks(
        student_bytes,
        assignment_id=aid,
        student_id=sid,
        modality=nb_mod,
        task_type=task,
        max_grading_units=None,
    )
    if not student_chunks:
        return None

    out: list[GradingChunk] = []
    for i, unit in enumerate(llm_qs):
        if max_units is not None and max_units >= 1 and len(out) >= max_units:
            break
        raw_qid = str(unit.get("question_id") or f"q{i + 1}").strip() or f"q{i + 1}"
        qid_safe = _safe_qid(raw_qid, i)
        qtext = str(unit.get("prompt") or "").strip()
        sch = _best_student_chunk_for_prompt(qtext, student_chunks, cfg)
        sr, ic = "", ""
        if sch is not None:
            trio_s = (sch.evidence or {}).get("trio")
            if isinstance(trio_s, dict):
                sr = str(trio_s.get("student_response") or "").strip()
                ic = str(trio_s.get("instructor_context") or "").strip()
            if not sr:
                sr = str((sch.evidence or {}).get("response_preview") or "").strip()
            if not sr:
                sr = (sch.extracted_text or "").strip()
        ext = "\n\n".join(p for p in (qtext, sr) if p).strip() or qtext
        cid = f"{aid}:{sid}:template_trio:{i}:{qid_safe}"[:500]
        out.append(
            GradingChunk(
                chunk_id=cid,
                assignment_id=aid,
                student_id=sid,
                question_id=raw_qid[:200],
                modality=nb_mod,
                task_type=task,
                extracted_text=ext,
                evidence={
                    "chunker": "blank_llm_question_aligned_notebook",
                    "question_id": raw_qid[:200],
                    "question_text": qtext[:2000],
                    "response_preview": (sr or "")[:500],
                    "trio": {
                        "question": qtext,
                        "student_response": sr,
                        "instructor_context": ic,
                        "answer_key_segment": "",
                    },
                    "_blank_template_trio": True,
                    "blank_template_question_source": "blank_llm_questions",
                    "student_notebook_chunker": str(
                        ((sch.evidence or {}) if sch else {}).get("chunker")
                        or "notebook_cell_order"
                    ),
                },
            )
        )

    if not out:
        return None
    _log.info("blank_llm_question_chunker: %d LLM questions → %d chunks", len(llm_qs), len(out))
    return out, "blank_llm_question_aligned_notebook"
