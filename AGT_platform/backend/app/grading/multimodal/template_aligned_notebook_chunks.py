"""
Align **blank** instructor notebooks (``blank_assignments/``) with student ``.ipynb`` submissions.

When ``blank_assignments/`` bytes are available, chunking prefers **scaffold code-cell
anchors** (``# TODO``, ``# add your code``, placeholders) matched by ordinal between blank
and student, then optional LLM question lists from the blank, then question-boundary
chunks on the blank paired with student :func:`notebook_chunker.build_notebook_qa_chunks`.
Resulting chunks use ``chunk_id`` shaped like OpenAI trio frontload
(``{assignment_id}:{student_id}:template_trio:{i}:{qid}``) and set ``evidence["_blank_template_trio"]``
so the pipeline skips relabeling that would overwrite the alignment.
"""

from __future__ import annotations

import logging
import os
import re

from app.config import Config

from .ingestion import IngestionEnvelope
from .notebook_chunker import (
    build_notebook_qa_chunks,
    build_notebook_question_boundary_chunks,
    try_build_notebook_scaffold_aligned_chunks,
)
from .schemas import GradingChunk, Modality, TaskType
from .chunker import modality_from_hints, task_type_from_hints

_log = logging.getLogger(__name__)


def _blank_template_mode(cfg: Config | None) -> str:
    if cfg is not None:
        raw = str(getattr(cfg, "MULTIMODAL_BLANK_TEMPLATE_CHUNKING", "") or "").strip().lower()
        if raw:
            return raw
    return (os.getenv("MULTIMODAL_BLANK_TEMPLATE_CHUNKING", "auto") or "auto").strip().lower()


def blank_template_chunking_requested(*, blank_bytes: bytes, cfg: Config | None) -> bool:
    """Whether blank-template alignment should run (``off`` / ``on`` / ``auto``)."""
    mode = _blank_template_mode(cfg)
    if mode in ("0", "false", "no", "off"):
        return False
    if mode in ("1", "true", "yes", "on"):
        return bool(blank_bytes.strip()) if blank_bytes else False
    # auto — use blank bytes when they look like a real ipynb JSON object
    return bool(blank_bytes and len(blank_bytes) > 32)


def _safe_qid(raw: str, idx: int) -> str:
    s = re.sub(r"\s+", " ", (raw or "").strip())
    return s[:120] if s else f"unit_{idx + 1}"


def _norm_qid(q: str) -> str:
    return re.sub(r"\s+", "", (q or "").strip().lower())


def _pop_matching_student(
    student_chunks: list[GradingChunk],
    consumed: set[int],
    t_qid: str,
) -> GradingChunk | None:
    """First unused student chunk with exact ``question_id``, else normalized id match."""
    want = str(t_qid or "").strip()
    want_n = _norm_qid(want)
    for sch in student_chunks:
        sid = id(sch)
        if sid in consumed:
            continue
        sq = str(sch.question_id or "").strip()
        if want and sq == want:
            consumed.add(sid)
            return sch
    if want_n:
        for sch in student_chunks:
            sid = id(sch)
            if sid in consumed:
                continue
            if _norm_qid(str(sch.question_id or "")) == want_n:
                consumed.add(sid)
                return sch
    return None


def _trio_question_student(
    template_ch: GradingChunk,
    student_ch: GradingChunk | None,
) -> tuple[str, str, str, str]:
    """Return (question, student_response, instructor_context, extracted_join)."""
    tev = template_ch.evidence or {}
    tt = tev.get("trio")
    q = ""
    t_ic = ""
    if isinstance(tt, dict):
        q = str(tt.get("question") or "").strip()
        t_ic = str(tt.get("instructor_context") or "").strip()
    if not q:
        q = str(tev.get("question_text") or "").strip()
    if not q:
        q = (template_ch.extracted_text or "").strip()

    sr = ""
    s_ic = ""
    if student_ch is not None:
        sev = student_ch.evidence or {}
        st = sev.get("trio")
        if isinstance(st, dict):
            sr = str(st.get("student_response") or "").strip()
            s_ic = str(st.get("instructor_context") or "").strip()
        if not sr:
            sr = str(sev.get("response_preview") or "").strip()
        if not sr:
            sr = (student_ch.extracted_text or "").strip()

    ic_parts = [p for p in (t_ic, s_ic) if p]
    ic = "\n\n".join(ic_parts).strip()
    ext = "\n\n".join(p for p in (q, sr) if p).strip() or (template_ch.extracted_text or "").strip()
    return q, sr, ic, ext


def _max_grading_units_from_hints(hints: dict) -> int | None:
    cap = hints.get("max_grading_units")
    if cap is None:
        return None
    try:
        return int(cap)
    except (TypeError, ValueError):
        return None


def _try_scaffold_blank_student_alignment(
    envelope: IngestionEnvelope,
    blank_ipynb_bytes: bytes,
    *,
    max_grading_units: int | None = None,
) -> tuple[list[GradingChunk], str] | None:
    """
    When the blank and student notebooks share the same number of scaffold **code** cells
    (TODO / add code / placeholders), emit one template-style chunk per anchor.
    """
    hints = envelope.modality_hints or {}
    raw = (envelope.artifacts or {}).get("ipynb")
    if not isinstance(raw, (bytes, bytearray)):
        return None
    student_bytes = bytes(raw)
    if not student_bytes.strip():
        return None

    aid = envelope.assignment_id
    sid = envelope.student_id
    nb_mod = modality_from_hints(hints)
    if nb_mod == Modality.UNKNOWN:
        nb_mod = Modality.NOTEBOOK
    task = task_type_from_hints(hints)

    scaffold_try = try_build_notebook_scaffold_aligned_chunks(
        blank_ipynb_bytes,
        student_bytes,
        assignment_id=aid,
        student_id=sid,
        modality=nb_mod,
        task_type=task,
        max_grading_units=max_grading_units,
    )
    if not scaffold_try:
        return None

    out_sc: list[GradingChunk] = []
    for i, ch in enumerate(scaffold_try):
        ev0 = dict(ch.evidence or {})
        trio = ev0.get("trio")
        q, sr, ic = "", "", ""
        if isinstance(trio, dict):
            q = str(trio.get("question") or "").strip()
            sr = str(trio.get("student_response") or "").strip()
            ic = str(trio.get("instructor_context") or "").strip()
        qid = _safe_qid(str(ch.question_id or ""), i)
        cid = f"{aid}:{sid}:template_trio:{i}:{qid}"
        ext = "\n\n".join(p for p in (q, sr) if p).strip() or (ch.extracted_text or "").strip()
        ev0["chunker"] = "blank_scaffold_aligned_notebook"
        ev0["question_id"] = str(ch.question_id or qid)
        ev0["question_text"] = str(ev0.get("question_text") or q)
        ev0["response_preview"] = str(ev0.get("response_preview") or sr)
        ev0["trio"] = {
            "question": q,
            "student_response": sr,
            "instructor_context": ic,
            "answer_key_segment": "",
        }
        ev0["_blank_template_trio"] = True
        ev0["blank_template_question_source"] = str(
            hints.get("blank_assignment_matched_file") or "blank_assignments"
        )
        ev0["student_notebook_chunker"] = "notebook_scaffold_anchor_aligned"
        out_sc.append(
            GradingChunk(
                chunk_id=cid,
                assignment_id=aid,
                student_id=sid,
                question_id=str(ch.question_id or qid),
                modality=nb_mod,
                task_type=task,
                extracted_text=ext,
                evidence=ev0,
            )
        )
    if max_grading_units is not None and max_grading_units >= 1:
        out_sc = out_sc[:max_grading_units]
    if not out_sc:
        return None
    return out_sc, "blank_scaffold_aligned_notebook"


def build_blank_template_aligned_notebook_chunks(
    envelope: IngestionEnvelope,
    *,
    blank_ipynb_bytes: bytes,
    cfg: Config | None = None,
) -> tuple[list[GradingChunk], str] | None:
    """
    Return aligned chunks and chunker label, or ``None`` if inputs cannot produce units.

    Caller ensures ``blank_ipynb_bytes`` is non-empty when this is invoked.
    """
    hints = envelope.modality_hints or {}
    raw = (envelope.artifacts or {}).get("ipynb")
    if not isinstance(raw, (bytes, bytearray)):
        return None
    student_bytes = bytes(raw)
    if not student_bytes.strip():
        return None

    max_units = _max_grading_units_from_hints(hints)

    aid = envelope.assignment_id
    sid = envelope.student_id
    nb_mod = modality_from_hints(hints)
    if nb_mod == Modality.UNKNOWN:
        nb_mod = Modality.NOTEBOOK
    task = task_type_from_hints(hints)

    hit = _try_scaffold_blank_student_alignment(
        envelope, blank_ipynb_bytes, max_grading_units=max_units
    )
    if hit:
        return hit

    template_chunks = build_notebook_question_boundary_chunks(
        blank_ipynb_bytes,
        assignment_id=aid,
        student_id="__blank_template__",
        modality=nb_mod,
        task_type=task,
        max_grading_units=max_units,
    )
    if not template_chunks:
        _log.info("blank template: no question-boundary units from blank ipynb")
        return None

    student_chunks = build_notebook_qa_chunks(
        student_bytes,
        assignment_id=aid,
        student_id=sid,
        modality=nb_mod,
        task_type=task,
        max_grading_units=None,
    )
    if not student_chunks:
        _log.warning("blank template: student notebook produced no chunks; falling back")
        return None

    consumed_student: set[int] = set()
    out: list[GradingChunk] = []

    for i, tch in enumerate(template_chunks):
        sch = _pop_matching_student(student_chunks, consumed_student, str(tch.question_id or ""))
        q, sr, ic, ext = _trio_question_student(tch, sch)
        qid = _safe_qid(str(tch.question_id or ""), i)
        cid = f"{aid}:{sid}:template_trio:{i}:{qid}"
        qtxt = str((tch.evidence or {}).get("question_text") or q) or ""
        rprev = sr or ""
        out.append(
            GradingChunk(
                chunk_id=cid,
                assignment_id=aid,
                student_id=sid,
                question_id=str(tch.question_id or qid),
                modality=nb_mod,
                task_type=task,
                extracted_text=ext,
                evidence={
                    "chunker": "blank_template_aligned_notebook",
                    "question_id": str(tch.question_id or qid),
                    "question_text": qtxt,
                    "response_preview": rprev,
                    "trio": {
                        "question": q,
                        "student_response": sr,
                        "instructor_context": ic,
                        "answer_key_segment": "",
                    },
                    "_blank_template_trio": True,
                    "blank_template_question_source": str(
                        hints.get("blank_assignment_matched_file") or "blank_assignments"
                    ),
                    "student_notebook_chunker": str(
                        (sch.evidence or {}).get("chunker") or "notebook_cell_order"
                    ),
                },
            )
        )

    if max_units is not None and max_units >= 1:
        out = out[:max_units]

    if not out:
        return None
    return out, "blank_template_aligned_notebook"


def try_build_blank_template_aligned_chunks(
    envelope: IngestionEnvelope,
    cfg: Config | None,
) -> tuple[list[GradingChunk], str] | None:
    """If hints carry blank bytes and mode allows, return aligned chunks."""
    hints = envelope.modality_hints or {}
    raw_blank = hints.get("blank_assignment_template_bytes")
    if raw_blank is None:
        raw_blank = hints.get("blank_assignment_ipynb_bytes")
    blank_bytes = raw_blank if isinstance(raw_blank, (bytes, bytearray)) else b""
    blank_bytes = bytes(blank_bytes)
    if not blank_template_chunking_requested(blank_bytes=blank_bytes, cfg=cfg):
        return None
    sc = _try_scaffold_blank_student_alignment(
        envelope,
        blank_bytes,
        max_grading_units=_max_grading_units_from_hints(hints),
    )
    if sc:
        return sc
    if cfg is not None:
        from .blank_llm_question_chunker import try_build_llm_blank_aligned_notebook_chunks

        try:
            llm = try_build_llm_blank_aligned_notebook_chunks(
                envelope, blank_ipynb_bytes=blank_bytes, cfg=cfg
            )
        except (AttributeError, TypeError):
            llm = None
        if llm:
            return llm
    return build_blank_template_aligned_notebook_chunks(
        envelope, blank_ipynb_bytes=blank_bytes, cfg=cfg
    )
