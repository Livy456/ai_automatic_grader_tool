"""
Opt-in **LLM triplet chunking** using three full sources in one structured call:

1. Instructor **blank** template (``blank_assignments/``) — any supported type (``.ipynb``,
   ``.pdf``, ``.docx``, ``.py``, ``.txt``, ``.md``, ``.csv``, ``.xlsx``).
2. **Student** submission — all artifact keys the pipeline carries (``ipynb``, ``pdf``, ``docx``,
   ``py``, ``txt``, ``md``, ``csv``, ``xlsx``, ``mp4`` / audio with stub transcript, etc.).
3. **Answer key / reference** (resolved ``answer_key_plaintext``).

Plaintext conversion uses :mod:`app.grading.artifact_plaintext` so behavior stays consistent with
:func:`app.grading.submission_text.submission_text_from_artifacts`.

The model returns JSON ``units`` aligned with :mod:`openai_trio_rag_frontload` so downstream
answer-key enrichment, RAG embedding, trio JSON export, and grading stay unchanged.

**Enable:** ``MULTIMODAL_LLM_TRIPLET_THREE_SOURCE=on`` (requires ``cfg``, resolved blank template
bytes, non-empty ``answer_key_plaintext``, and at least one non-empty student artifact).

**Context size:** ``MULTIMODAL_LLM_TRIPLET_MAX_CHARS_PER_SOURCE`` (default ``1000000``) caps each
of the three bodies before the API call; provider context limits still apply.

**Client selection:** Uses **Claude** (:func:`~app.grading.llm_router.anthropic_multimodal_structure_client`)
when ``ANTHROPIC_API_KEY`` is set and ``MULTIMODAL_ANTHROPIC_ASSIGNMENT_PARSING`` is not ``off``,
unless ``MULTIMODAL_LLM_TRIPLET_THREE_SOURCE_PREFER_OPENAI=on`` forces
:class:`~app.grading.llm_router.OpenAIJsonClient` with ``OPENAI_TRIO_RAG_CHAT_MODEL``. If Claude is
unavailable, falls back to OpenAI when ``OPENAI_API_KEY`` is set. Ollama is not used.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from app.config import Config
from app.grading.artifact_plaintext import (
    artifacts_to_concatenated_plain,
    bytes_with_suffix_to_plain,
    infer_modality_from_artifact_keys,
)
from app.grading.llm_router import OpenAIJsonClient, anthropic_multimodal_structure_client

from .chunker import modality_from_hints, task_type_from_hints
from .ingestion import IngestionEnvelope
from .schemas import GradingChunk, Modality, TaskType

_log = logging.getLogger(__name__)

_TRIPLET_SYSTEM = """You are an expert instructional designer and grader preparing **atomic grading units**
for an automated pipeline.

You are given three labeled corpora in one message:

1. **BLANK_ASSIGNMENT** — the official instructor template (questions, instructions, starter code,
   tables, or prose depending on file type). This defines *what the student was asked to do*
   and the intended ordering / numbering of items.

2. **STUDENT_SUBMISSION** — the student's work in one or more files (notebook cells, PDF text,
   Python source, Word document text, CSV/tabular data, spreadsheet cells, or a video/audio
   placeholder with stub transcript). It may diverge from the template. Extract **only** what
   the student authored as their attempt for each gradable item.

3. **ANSWER_KEY_OR_REFERENCE** — sample solutions, rubric snippets, or reference code/text.
   For each unit, copy or excerpt the **minimal** passages that establish correctness for that item.
   If the key has numbered sections (e.g. "1.3.1", "## Problem 2"), align ``answer_key_segment`` to the same item.

**Task:** Emit a JSON object (no markdown fences, no commentary) with exactly this shape:
{"units":[
  {"question_id":"stable id from blank or numbering (e.g. 1.3.2, q4)",
   "question":"verbatim or faithful paraphrase of the instructor prompt for this item (from BLANK + context)",
   "student_response":"the student's answer / code / prose / table excerpt for this item only (from STUDENT)",
   "answer_key_segment":"matching excerpt from ANSWER_KEY_OR_REFERENCE (empty string if truly absent)",
   "extracted_text":"concatenation useful for grading: typically question + student_response + optional short hint from blank"}
]}

**Rules (strict):**
- **Coverage:** One unit per separately gradable item implied by the template (and visible in the student work when present). Do not merge unrelated problems.
- **Fidelity:** Prefer **verbatim** copying from the three sources; never invent requirements, grades, or work the student did not produce.
- **Alignment:** ``question_id`` should mirror numbering in the template when visible; otherwise use ``q1``, ``q2`` in document order.
- **Student-only work:** ``student_response`` must not include unreleased solution blocks from the template unless the student visibly reproduced or edited them.
- **Answer key:** If a unit has no applicable reference, set ``answer_key_segment`` to ``""`` (empty string), not null.
- **Granularity:** Short single-line answers are still complete units—do not drop them.
- **Ordering:** List ``units`` in the same pedagogical order as the instructor template.
- **No grades:** Do not output scores or pass/fail—only structural decomposition."""


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        raw = os.getenv(name, "").strip()
        return int(raw) if raw else default
    except (TypeError, ValueError):
        return default


def multimodal_llm_triplet_three_source_enabled(cfg: Config | None) -> bool:
    if cfg is None:
        return False
    raw = str(getattr(cfg, "MULTIMODAL_LLM_TRIPLET_THREE_SOURCE", "") or "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return _env_bool("MULTIMODAL_LLM_TRIPLET_THREE_SOURCE", default=False)


def _max_chars_per_source(cfg: Config | None) -> int:
    if cfg is not None:
        try:
            v = int(getattr(cfg, "MULTIMODAL_LLM_TRIPLET_MAX_CHARS_PER_SOURCE", 0) or 0)
            if v > 0:
                return max(8_000, min(v, 2_000_000))
        except (TypeError, ValueError):
            pass
    return max(8_000, min(_env_int("MULTIMODAL_LLM_TRIPLET_MAX_CHARS_PER_SOURCE", 1_000_000), 2_000_000))


def _safe_qid(raw: str, idx: int) -> str:
    s = re.sub(r"\s+", " ", (raw or "").strip())
    return s[:120] if s else f"unit_{idx + 1}"


def _chat_units(
    client: Any,
    user_body: str,
    *,
    use_openai_usage: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    messages = [
        {"role": "system", "content": _TRIPLET_SYSTEM},
        {"role": "user", "content": user_body},
    ]
    if use_openai_usage and hasattr(client, "chat_json_with_usage"):
        parsed, usage = client.chat_json_with_usage(
            messages, temperature=0.1, response_format={"type": "json_object"}
        )
    else:
        parsed = client.chat_json(messages, temperature=0.1)
        usage = {}
    raw = parsed.get("units") if isinstance(parsed, dict) else None
    raw_list = raw if isinstance(raw, list) else []
    clean = [u for u in raw_list if isinstance(u, dict)]
    return clean, usage


def _blank_template_plaintext(hints: dict[str, Any]) -> tuple[str, str]:
    """Return ``(plaintext, suffix)`` for the resolved blank template."""
    raw = hints.get("blank_assignment_template_bytes")
    if raw is None:
        raw = hints.get("blank_assignment_ipynb_bytes")
    data = bytes(raw) if isinstance(raw, (bytes, bytearray)) else b""
    suf = str(hints.get("blank_assignment_template_suffix") or "").strip().lower()
    if not suf and data:
        suf = ".ipynb"
    if not suf.startswith("."):
        suf = "." + suf if suf else ".ipynb"
    return bytes_with_suffix_to_plain(data, suf), suf


def _student_submission_plaintext(envelope: IngestionEnvelope) -> str:
    arts = envelope.artifacts or {}
    if isinstance(arts, dict):
        blob = artifacts_to_concatenated_plain(
            {k: v for k, v in arts.items() if isinstance(v, (bytes, bytearray))}
        )
        if blob.strip():
            return blob.strip()
    return (envelope.extracted_plaintext or "").strip()


def _modality_for_triplet_chunks(
    envelope: IngestionEnvelope, hints: dict[str, Any]
) -> Modality:
    raw = str(hints.get("modality") or "").strip().lower()
    if raw:
        for m in Modality:
            if m.value == raw:
                return m
    arts = envelope.artifacts or {}
    if isinstance(arts, dict):
        inferred = infer_modality_from_artifact_keys(
            {k: v for k, v in arts.items() if isinstance(v, (bytes, bytearray))}
        )
        try:
            return Modality(inferred)
        except ValueError:
            pass
    return modality_from_hints(hints)


def try_build_llm_triplet_three_source_chunks(
    envelope: IngestionEnvelope,
    cfg: Config,
    *,
    answer_key_plaintext: str,
) -> tuple[list[GradingChunk], str] | None:
    """
    Return grading chunks with ``evidence['trio']`` filled from one LLM call over blank + student
    + answer key, or ``None`` when disabled, inputs are missing, or the model returns no units.
    """
    if not multimodal_llm_triplet_three_source_enabled(cfg):
        return None
    hints = envelope.modality_hints or {}
    blank_plain, _blank_suf = _blank_template_plaintext(hints)
    stu_plain = _student_submission_plaintext(envelope)
    if len(blank_plain.strip()) < 10:
        _log.info("llm_triplet_three_source: blank template plaintext too short")
        return None
    if len(stu_plain.strip()) < 8:
        _log.info("llm_triplet_three_source: student submission plaintext too short")
        return None

    ak = (answer_key_plaintext or str(hints.get("answer_key_plaintext") or "")).strip()
    if len(ak) < 8:
        _log.info("llm_triplet_three_source: answer key too short or empty")
        return None

    cap = _max_chars_per_source(cfg)
    b_use = blank_plain if len(blank_plain) <= cap else blank_plain[:cap]
    s_use = stu_plain if len(stu_plain) <= cap else stu_plain[:cap]
    a_use = ak if len(ak) <= cap else ak[:cap]
    trunc_note = ""
    if len(blank_plain) > cap or len(stu_plain) > cap or len(ak) > cap:
        trunc_note = (
            f"\n\n[SYSTEM NOTE: One or more sources were truncated to {cap} characters for this API call. "
            "Prefer units that are fully visible inside the truncated regions; if a boundary is cut, "
            "still emit the best partial unit rather than omitting it.]\n"
        )

    user_body = (
        "### BLANK_ASSIGNMENT (instructor template)\n\n"
        + b_use
        + "\n\n### STUDENT_SUBMISSION\n\n"
        + s_use
        + "\n\n### ANSWER_KEY_OR_REFERENCE\n\n"
        + a_use
        + trunc_note
    )

    force_openai = os.getenv(
        "MULTIMODAL_LLM_TRIPLET_THREE_SOURCE_PREFER_OPENAI", "0"
    ).strip().lower() in ("1", "true", "yes", "on")
    anth = anthropic_multimodal_structure_client(cfg)
    oa_key = (cfg.OPENAI_API_KEY or "").strip()
    client: Any = None
    model_label = ""
    use_openai_usage = False
    if anth is not None and not force_openai:
        client, model_label = anth
        use_openai_usage = False
    elif oa_key:
        chat_model = (
            getattr(cfg, "OPENAI_TRIO_RAG_CHAT_MODEL", None) or ""
        ).strip() or "gpt-5.4-nano"
        client = OpenAIJsonClient(oa_key, chat_model)
        model_label = f"openai:{chat_model}"
        use_openai_usage = True
    elif anth is not None:
        client, model_label = anth
        use_openai_usage = False
    if client is None:
        _log.warning(
            "llm_triplet_three_source: no LLM client (set ANTHROPIC_API_KEY for Claude or OPENAI_API_KEY)"
        )
        return None

    try:
        units, _usage = _chat_units(client, user_body, use_openai_usage=use_openai_usage)
    except Exception:
        _log.warning("llm_triplet_three_source: chat failed model=%s", model_label, exc_info=True)
        return None

    if not units:
        return None

    nb_mod = _modality_for_triplet_chunks(envelope, hints)
    if nb_mod == Modality.UNKNOWN:
        nb_mod = Modality.MIXED
    task = task_type_from_hints(hints)
    aid = envelope.assignment_id
    sid = envelope.student_id
    max_units: int | None = None
    cap_u = hints.get("max_grading_units")
    if cap_u is not None:
        try:
            max_units = int(cap_u)
        except (TypeError, ValueError):
            max_units = None

    out: list[GradingChunk] = []
    for i, u in enumerate(units):
        if max_units is not None and max_units >= 1 and len(out) >= max_units:
            break
        q = str(u.get("question") or "").strip()
        sr = str(u.get("student_response") or "").strip()
        ak_seg = str(u.get("answer_key_segment") or "").strip()
        ext = str(u.get("extracted_text") or "").strip()
        if not ext:
            ext = "\n\n".join(p for p in (q, sr) if p).strip()
        if not q and not sr:
            continue
        qid = _safe_qid(str(u.get("question_id") or ""), i)
        cid = f"{aid}:{sid}:llm_triplet:{i}:{qid}"[:500]
        out.append(
            GradingChunk(
                chunk_id=cid,
                assignment_id=aid,
                student_id=sid,
                question_id=qid,
                modality=nb_mod,
                task_type=task,
                extracted_text=ext or (q + "\n\n" + sr).strip(),
                evidence={
                    "chunker": "llm_triplet_three_source",
                    "trio": {
                        "question": q,
                        "student_response": sr,
                        "answer_key_segment": ak_seg,
                        "instructor_context": "",
                    },
                    "_llm_triplet_three_source": True,
                    "llm_triplet_three_source_model": model_label,
                    "question_text": q[:8000],
                    "response_preview": (sr or "")[:2000],
                },
            )
        )

    if not out:
        return None
    _log.info(
        "llm_triplet_three_source: model=%s units_in=%d chunks_out=%d",
        model_label,
        len(units),
        len(out),
    )
    return out, "llm_triplet_three_source"
