"""
RAG-style chunk embeddings and optional LLM Q→A segmentation for multimodal grading.

Aligns with :mod:`app.grading.rag_embeddings` — per-chunk vectors from
:func:`app.grading.rag_embeddings.compute_submission_embedding` (default
``RAG_EMBEDDING_BACKEND=sentence_transformers`` / ``SENTENCE_TRANSFORMERS_MODEL``).

**Chunking stack** (see ``new_chunking_method.md``):

0. When ``MULTIMODAL_CLAUDE_STRUCTURED_CHUNKING`` is ``auto`` or ``on`` and
   ``ANTHROPIC_API_KEY`` is set, :mod:`claude_structured_assignment_chunker` may emit
   ``GradingChunk`` rows from a single Claude JSON ``units`` response (skipping steps 1–3
   on success). Default env is ``off``.
1. When ``MULTIMODAL_LLM_TRIPLET_THREE_SOURCE=on`` and blank template + answer key resolve,
   :mod:`llm_triplet_three_source` may emit trios in one LLM call (before notebook template
   heuristics). Blank may be ``.ipynb`` or other supported template types; student text comes
   from all submission artifacts. Else when ``modality_hints["blank_assignment_template_bytes"]``
   or ``blank_assignment_ipynb_bytes`` resolves and ``MULTIMODAL_BLANK_TEMPLATE_CHUNKING`` is not
   ``off``, try LLM question inventory
   (:mod:`blank_llm_question_chunker`) then heuristic blank/student alignment
   (:mod:`template_aligned_notebook_chunks`); otherwise
   :func:`notebook_chunker.build_notebook_qa_chunks` (cell-order Q/A).
2. Otherwise **PDF plaintext** is reflowed via
   :func:`app.grading.submission_chunks.reflow_pdf_sections_in_plaintext` before any LLM
   segmentation so verticalized extractors do not confuse the model.
3. If ``MULTIMODAL_ASSIGNMENT_PARSING`` or ``MULTIMODAL_OLLAMA_QA_SEGMENT`` (or
   ``MULTIMODAL_LLM_QA_SEGMENT``) is on, an **LLM** returns JSON Q/A units. **Claude** runs when
   ``ANTHROPIC_API_KEY`` is set and ``MULTIMODAL_ANTHROPIC_ASSIGNMENT_PARSING`` is not ``off``
   (default ``auto``); otherwise **OpenAI** is used for this step if ``OPENAI_API_KEY`` is set.
   Ollama is not used. On failure,
   :func:`chunker.default_chunker_build_units` runs structured chunking
   (:func:`app.grading.submission_chunks.build_submission_chunks`, which reflows each PDF
   section again and applies journal-style prompt boundaries when hints match).
4. If ``OPENAI_API_KEY`` is set and OpenAI trio frontload is enabled (default **auto**;
   disable with ``MULTIMODAL_OPENAI_TRIO_RAG_FRONTLOAD=off``),
   :func:`app.grading.multimodal.openai_trio_rag_frontload.run_openai_trio_rag_frontload`
   replaces steps 3–4 for
   that run (one or more OpenAI chats on overlapping windows when the submission is long,
   plus OpenAI Embeddings API). Otherwise, if
   ``MULTIMODAL_LLM_TRIO_CHUNKING`` is on, :func:`refine_chunks_trio_with_structure_llm` runs after
   units exist to label ``question`` / ``student_response`` / ``instructor_context`` via Claude
   or OpenAI (answer-key snippet alignment still uses answer-key enrich).

**Prompt (assignment parsing):** the system message sent to the LLM for step 3 is the string
``ASSIGNMENT_PARSING_SYSTEM_PROMPT`` (module constant below). The user message is the submission
plain text (artifact-derived when available, then reflowed).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from app.config import Config
from app.grading.artifact_plaintext import artifacts_to_concatenated_plain
from app.grading.llm_router import (
    OpenAIJsonClient,
    anthropic_multimodal_structure_client,
    openai_multimodal_grading_model,
)
from app.grading.rag_embeddings import compute_submission_embedding

from app.grading.submission_chunks import reflow_pdf_sections_in_plaintext

from .chunker import default_chunker_build_units, modality_from_hints, task_type_from_hints
from .ingestion import IngestionEnvelope
from .notebook_chunker import build_notebook_qa_chunks
from .schemas import GradingChunk, Modality, TaskType
from .template_aligned_notebook_chunks import try_build_blank_template_aligned_chunks

_log = logging.getLogger(__name__)


# System message for LLM assignment parsing (step 3). Sent as the chat ``system`` role; the user
# role is the full submission plaintext (see ``_qa_segment_plaintext``).
ASSIGNMENT_PARSING_SYSTEM_PROMPT = """You are an expert software engineer who specializes in parsing artifacts.
Your task is to parse this student assignment into distinct sections: each section corresponds to a different question, exercise, coding problem, or instruction the students were asked to complete.
For each section, capture the instructor-facing prompt (the question or directions) separately from the student's answer, code, data visualization, image,tables, or free-text response that fulfills that prompt.
Preserve ordering as it appears in the submission. Do not invent prompts or answers that are not present in the text.
Return **only** valid JSON with this shape (no markdown fences, no commentary outside the JSON object):
{"units":[{"key":"string_id","question":"prompt or problem text","response":"student answer or code"}]}
Use short unique keys like q1, q2, p1a. If the submission is one continuous block, emit a single unit with key q1.
The "question" field is the instructor prompt for that item; the "response" field is exactly what the student supplied for that prompt (use an empty string if there is no separable answer)."""


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def multimodal_llm_qa_segment_enabled() -> bool:
    """
    True to run LLM JSON Q/A segmentation on plain submissions.

    Enable with ``MULTIMODAL_ASSIGNMENT_PARSING=on`` (preferred), ``MULTIMODAL_LLM_QA_SEGMENT=on``,
    or the legacy env name ``MULTIMODAL_OLLAMA_QA_SEGMENT=on`` (structure LLM is Claude or OpenAI;
    see :data:`ASSIGNMENT_PARSING_SYSTEM_PROMPT`).
    """
    return (
        _env_bool("MULTIMODAL_ASSIGNMENT_PARSING", default=False)
        or _env_bool("MULTIMODAL_LLM_QA_SEGMENT", default=False)
        or _env_bool("MULTIMODAL_OLLAMA_QA_SEGMENT", default=False)
    )


def multimodal_rag_embed_units_enabled() -> bool:
    return _env_bool("MULTIMODAL_RAG_EMBED_UNITS", default=True)


def multimodal_llm_trio_chunking_enabled(cfg: Config | None = None) -> bool:
    """True when an LLM should re-label each chunk's ``evidence['trio']`` before AK enrich."""
    if cfg is not None and bool(getattr(cfg, "MULTIMODAL_LLM_TRIO_CHUNKING", False)):
        return True
    return _env_bool("MULTIMODAL_LLM_TRIO_CHUNKING", default=False)


def _multimodal_structure_chat_client(
    cfg: Config, *, purpose: str = "trio"
) -> tuple[Any, str]:
    """
    Client for multimodal **structure** only (assignment parsing, trio refinement, blank LLM).

    Uses **Claude** when :func:`app.grading.llm_router.anthropic_multimodal_structure_client`
    returns a client; otherwise **OpenAI** when ``OPENAI_API_KEY`` is set. Ollama is never used
    in this pipeline. ``purpose`` is kept for call-site compatibility and is ignored.
    """
    del purpose
    anth = anthropic_multimodal_structure_client(cfg)
    if anth is not None:
        return anth
    key = (getattr(cfg, "OPENAI_API_KEY", "") or "").strip()
    if not key:
        _log.warning(
            "_multimodal_structure_chat_client: no ANTHROPIC_API_KEY and no OPENAI_API_KEY; "
            "structure QA/trio calls are disabled."
        )
        return None, ""
    mid = openai_multimodal_grading_model(cfg)
    return OpenAIJsonClient(key, mid), f"openai:{mid}"


def _qa_segment_plaintext(envelope: IngestionEnvelope) -> str:
    """Prefer decoded artifact text, then ``extracted_plaintext``; reflow PDF-style blocks."""
    art_plain = ""
    arts = envelope.artifacts or {}
    if isinstance(arts, dict):
        blob = artifacts_to_concatenated_plain(
            {
                k: bytes(v)
                for k, v in arts.items()
                if isinstance(v, (bytes, bytearray)) and v
            }
        )
        if blob.strip():
            art_plain = blob.strip()
    base = art_plain or (envelope.extracted_plaintext or "").strip()
    if not base:
        return ""
    return reflow_pdf_sections_in_plaintext(base)


def _chunks_from_llm_qa_segmentation(
    envelope: IngestionEnvelope,
    cfg: Config,
) -> list[GradingChunk] | None:
    plain = _qa_segment_plaintext(envelope)
    if not plain:
        return None
    picked = _multimodal_structure_chat_client(cfg, purpose="qa")
    client, model_label = picked
    if client is None:
        return None
    user = plain
    try:
        obj = client.chat_json(
            [
                {"role": "system", "content": ASSIGNMENT_PARSING_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
        )
    except Exception:
        _log.warning(
            "LLM QA segmentation failed (%s); using heuristic chunker.",
            model_label,
            exc_info=True,
        )
        return None

    units = obj.get("units")
    if not isinstance(units, list) or not units:
        return None

    hints = envelope.modality_hints or {}
    modality = modality_from_hints(hints)
    task = task_type_from_hints(hints)
    if model_label.startswith("anthropic:"):
        chunker_tag = "anthropic_qa_segment"
    else:
        chunker_tag = "openai_qa_segment"
    out: list[GradingChunk] = []
    for i, u in enumerate(units, start=1):
        if not isinstance(u, dict):
            continue
        key = str(u.get("key") or f"q{i}").strip() or f"q{i}"
        q = str(u.get("question") or "").strip()
        r = str(u.get("response") or "").strip()
        extracted = "\n\n".join(p for p in (q, r) if p).strip()
        if not extracted:
            continue
        qid = f"pair_{i}"
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
                    "qa_pair_key": key,
                    "question_text": q,
                    "response_text": r,
                    "chunker": chunker_tag,
                    "qa_segment_model": model_label,
                    "assignment_parsing_system_prompt": "ASSIGNMENT_PARSING_SYSTEM_PROMPT",
                    "canonical_pair_index": i,
                    "trio": {
                        "question": q,
                        "student_response": r,
                        "instructor_context": "",
                        "answer_key_segment": "",
                    },
                },
            )
        )

    if not out:
        return None

    cap = hints.get("max_grading_units")
    if cap is not None:
        try:
            n = int(cap)
            if n >= 1:
                out = out[:n]
        except (TypeError, ValueError):
            pass
    return out


def _env_int(name: str, default: int) -> int:
    try:
        raw = os.getenv(name, "").strip()
        return int(raw) if raw else default
    except (TypeError, ValueError):
        return default


def _optional_positive_int_env(name: str) -> int | None:
    """
    Read an int env var: unset → ``None`` (caller treats as **no** character cap).

    ``0`` or negative values also mean no cap. Set a positive integer to enforce a limit.
    """
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        v = int(raw)
        return None if v <= 0 else v
    except (TypeError, ValueError):
        return None


def _chunks_use_openai_trio_rag_frontload(chunks: list[GradingChunk]) -> bool:
    """True when every chunk already has OpenAI-produced trio segment + bundle vectors."""
    if not chunks:
        return False
    for ch in chunks:
        ev = ch.evidence or {}
        if not ev.get("_openai_trio_rag_frontload"):
            return False
        tr = ev.get("trio_segment_rag")
        rb = ev.get("rag_embedding_bundle")
        if not isinstance(tr, dict) or not isinstance(rb, dict):
            return False
        src = str(rb.get("embedding_source") or "")
        if "openai:" not in src:
            return False
    return True


def enrich_chunks_with_rag_embeddings(chunks: list[GradingChunk], cfg: Config) -> None:
    """
    Attach per-unit vectors via :func:`compute_submission_embedding`.

    When ``evidence["trio"]`` is present (question / student_response / answer_key_segment),
    embed each non-empty segment into ``trio_segment_rag`` and set ``rag_embedding_bundle``
    from a canonical ``[QUESTION] / [STUDENT] / [REFERENCE]`` join (fallback: ``extracted_text``).
    """
    if not multimodal_rag_embed_units_enabled():
        return
    if _chunks_use_openai_trio_rag_frontload(chunks):
        return
    if _env_bool("MULTIMODAL_TRIO_EMBED_NO_CAPS", default=False):
        q_cap = r_cap = ak_cap = canon_cap = 10**9
    else:
        q_cap = _env_int("MULTIMODAL_TRIO_EMBED_QUESTION_MAX_CHARS", 12_000)
        r_cap = _env_int("MULTIMODAL_TRIO_EMBED_RESPONSE_MAX_CHARS", 16_000)
        ak_cap = _env_int("MULTIMODAL_TRIO_EMBED_ANSWER_KEY_MAX_CHARS", 16_000)
        canon_cap = _env_int("MULTIMODAL_TRIO_CANONICAL_EMBED_MAX_CHARS", 24_000)

    for ch in chunks:
        ev = dict(ch.evidence or {})
        trio = ev.get("trio")
        if isinstance(trio, dict):
            tq = str(trio.get("question") or "").strip()[:q_cap]
            tsr = str(trio.get("student_response") or "").strip()[:r_cap]
            tak = str(trio.get("answer_key_segment") or "").strip()[:ak_cap]
            seg_rag: dict[str, Any] = {}
            for key, blob in (
                ("question", tq),
                ("student_response", tsr),
                ("answer_key_segment", tak),
            ):
                b = (blob or "").strip()
                if not b:
                    seg_rag[key] = {
                        "embedding_dimension": 0,
                        "embedding_source": "empty_segment_skipped",
                        "embedding": [],
                    }
                    continue
                try:
                    vec, src = compute_submission_embedding(b, cfg)
                except Exception:
                    _log.warning(
                        "trio_segment_rag embed failed (%s / %s)",
                        ch.chunk_id,
                        key,
                        exc_info=True,
                    )
                    vec, src = [], "embedding_failed"
                seg_rag[key] = {
                    "embedding_dimension": len(vec),
                    "embedding_source": src,
                    "embedding": vec,
                }
            ev["trio_segment_rag"] = seg_rag
            canon_parts: list[str] = []
            if tq:
                canon_parts.append(f"[QUESTION]\n{tq}")
            if tsr:
                canon_parts.append(f"[STUDENT]\n{tsr}")
            if tak:
                canon_parts.append(f"[REFERENCE]\n{tak}")
            canon = "\n\n".join(canon_parts).strip()
            if not canon:
                canon = (ch.extracted_text or "").strip() or " "
            canon = canon[:canon_cap]
            try:
                vec2, src2 = compute_submission_embedding(canon, cfg)
            except Exception:
                _log.warning(
                    "trio canonical rag_embedding_bundle failed (%s)",
                    ch.chunk_id,
                    exc_info=True,
                )
                vec2, src2 = [], "embedding_failed"
            ev["rag_embedding_bundle"] = {
                "embedding_dimension": len(vec2),
                "embedding_source": f"trio_canonical:{src2}",
                "embedding": vec2,
            }
            ch.evidence = ev
            continue

        txt = (ch.extracted_text or "").strip() or " "
        vec, src = compute_submission_embedding(txt, cfg)
        ev["rag_embedding_bundle"] = {
            "embedding_dimension": len(vec),
            "embedding_source": src,
            "embedding": vec,
        }
        ch.evidence = ev


def _get_ipynb_bytes(envelope: IngestionEnvelope) -> bytes | None:
    """Return raw ipynb bytes from the envelope's artifacts, if present."""
    raw = (envelope.artifacts or {}).get("ipynb")
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw)
    return None


_TRIO_LLM_SYSTEM = """You split one assignment grading unit into three fields for downstream RAG and grading.
The input may mix instructor scaffolding, the assignment prompt, and the student's answer (markdown and/or code).
Return **only** valid JSON (no markdown fences, no prose outside the JSON object):
{"question":"the assigned prompt or problem statement only",
 "student_response":"everything the student authored as their answer for this unit",
 "instructor_context":"read-only setup, hidden tests, template or boilerplate not written by the student; use an empty string if none"}
Rules:
- Prefer copying phrases from the input; do not invent requirements or grades.
- Do not include official answer keys or full sample solutions in any field; if the input contains an instructor solution block, put at most one short neutral note in instructor_context or leave it empty.
- Use empty strings for fields that do not apply."""


def refine_chunks_trio_with_structure_llm(chunks: list[GradingChunk], cfg: Config) -> None:
    """
    For each chunk, call the **structure** LLM (Claude preferred, else OpenAI) once to populate
    ``evidence["trio"]`` (question / student_response / instructor_context). ``answer_key_segment``
    is left unchanged here and is filled by
    :func:`answer_key_chunk_enrich.enrich_chunks_with_per_question_answer_key`.

    Controlled by ``MULTIMODAL_LLM_TRIO_CHUNKING`` / ``cfg.MULTIMODAL_LLM_TRIO_CHUNKING`` and
    :func:`_multimodal_structure_chat_client` (Ollama is not used).

    ``MULTIMODAL_LLM_TRIO_INPUT_MAX_CHARS``: unset or ``0`` = send the full ``extracted_text``
    to the structure model; set a positive int to cap (clamped up to 2_000_000).
    """
    if not chunks or not multimodal_llm_trio_chunking_enabled(cfg):
        return
    picked = _multimodal_structure_chat_client(cfg, purpose="trio")
    client, model = picked
    if client is None:
        _log.warning(
            "refine_chunks_trio_with_structure_llm: no structure LLM client "
            "(set ANTHROPIC_API_KEY for Claude or OPENAI_API_KEY for OpenAI structure fallback)"
        )
        return
    for ch in chunks:
        ev0 = ch.evidence or {}
        if ev0.get("trio_llm_refined"):
            continue
        if ev0.get("_llm_triplet_three_source"):
            continue
        if ev0.get("_claude_structured_units"):
            continue
        raw_in = (ch.extracted_text or "").strip()
        if not raw_in:
            continue
        raw_cap = os.getenv("MULTIMODAL_LLM_TRIO_INPUT_MAX_CHARS", "").strip()
        if not raw_cap:
            user_payload = raw_in
        else:
            try:
                cap = int(raw_cap)
            except ValueError:
                cap = 28_000
            if cap <= 0:
                user_payload = raw_in
            else:
                cap = max(4000, min(cap, 2_000_000))
                user_payload = raw_in[:cap]
        try:
            obj = client.chat_json(
                [
                    {"role": "system", "content": _TRIO_LLM_SYSTEM},
                    {"role": "user", "content": user_payload},
                ],
                temperature=0.1,
            )
        except Exception:
            _log.warning(
                "trio LLM chunking failed chunk_id=%s model=%s",
                ch.chunk_id,
                model,
                exc_info=True,
            )
            continue
        if not isinstance(obj, dict):
            continue
        q = str(obj.get("question", "") or "").strip()
        s = str(obj.get("student_response", "") or "").strip()
        ic = str(obj.get("instructor_context", "") or "").strip()
        ev = dict(ch.evidence or {})
        prev = ev.get("trio") if isinstance(ev.get("trio"), dict) else {}
        ak_seg = str((prev or {}).get("answer_key_segment") or "")
        ev["trio"] = {
            "question": q or str((prev or {}).get("question") or ""),
            "student_response": s or str((prev or {}).get("student_response") or ""),
            "instructor_context": ic or str((prev or {}).get("instructor_context") or ""),
            "answer_key_segment": ak_seg,
        }
        unit = ev0.get("unit") if isinstance(ev0.get("unit"), dict) else {}
        unit_rt = str(unit.get("response_text") or "").strip()
        if not str(ev["trio"].get("student_response") or "").strip():
            if unit_rt:
                ev["trio"]["student_response"] = unit_rt
            elif raw_in.strip():
                pq = str(ev["trio"].get("question") or "").strip()
                if pq and raw_in.strip().startswith(pq):
                    tail = raw_in.strip()[len(pq) :].lstrip("\n").strip()
                    if tail:
                        ev["trio"]["student_response"] = tail
                    else:
                        ev["trio"]["student_response"] = raw_in.strip()
                else:
                    ev["trio"]["student_response"] = raw_in.strip()
        joined = "\n\n".join(
            p for p in (ev["trio"]["question"], ev["trio"]["student_response"]) if p
        ).strip()
        if joined:
            ch.extracted_text = joined
        ev["trio_llm_refined"] = True
        ev["trio_llm_chunking_model"] = model
        ch.evidence = ev


def refine_chunks_trio_with_ollama(chunks: list[GradingChunk], cfg: Config) -> None:
    """Backward-compatible alias for :func:`refine_chunks_trio_with_structure_llm`."""
    refine_chunks_trio_with_structure_llm(chunks, cfg)


def build_multimodal_grading_chunks(
    envelope: IngestionEnvelope,
    cfg: Config | None = None,
) -> tuple[list[GradingChunk], str]:
    """
    Build grading units using the best available chunker:

    0. **Claude structured assignment chunking** (optional) — when
       ``MULTIMODAL_CLAUDE_STRUCTURED_CHUNKING`` is ``auto`` or ``on`` and
       ``ANTHROPIC_API_KEY`` is set, a single Claude call returns JSON ``units`` mapped to
       :class:`GradingChunk` with ``evidence["trio"]`` (see ``claude_structured_assignment_chunker``).
       On success, notebook cell-order, triplet-three-source, and legacy QA-segment chunkers are skipped.
       When ``on`` and Claude fails, only :func:`chunker.default_chunker_build_units` is used next.
    1. **Notebook cell-order** — if the envelope carries raw ipynb bytes,
       parse cells directly and pair each question heading with the student's
       response cells (preserves the natural cell ordering).  Does not require
       ``cfg`` — works on the raw notebook JSON alone.
    2. **LLM QA segmentation** — JSON Q/A units (requires ``cfg``; skipped when ``cfg`` is ``None``).
       **Claude** when ``ANTHROPIC_API_KEY`` is set and parsing is not ``off``; otherwise **OpenAI**
       for this step if ``OPENAI_API_KEY`` is set. Uses ``ASSIGNMENT_PARSING_SYSTEM_PROMPT``.
       Plaintext is PDF-reflowed first; artifact bytes are preferred over ``extracted_plaintext``.
    3. **Structured heuristic** — :func:`chunker.default_chunker_build_units` on plaintext
       (PDF reflow + ``build_submission_chunks`` / journal-style boundaries per
       ``new_chunking_method.md``).

    ``cfg`` may be ``None`` when the pipeline runs without a full application
    config (e.g. lightweight tests).  The notebook and heuristic
    chunkers operate without it; only LLM QA segmentation is skipped.
    """
    hints = envelope.modality_hints or {}
    cap = hints.get("max_grading_units")
    max_units = None
    if cap is not None:
        try:
            max_units = int(cap)
        except (TypeError, ValueError):
            pass

    if cfg is not None:
        from .claude_structured_assignment_chunker import (
            claude_structured_chunking_forced_on,
            claude_structured_chunking_should_attempt,
            try_build_claude_structured_assignment_chunks,
        )

        if claude_structured_chunking_should_attempt(cfg):
            cl_chunks = try_build_claude_structured_assignment_chunks(
                envelope, cfg, max_units=max_units
            )
            if cl_chunks:
                return cl_chunks, "claude_structured_assignment"
            if claude_structured_chunking_forced_on(cfg):
                hc = default_chunker_build_units(envelope)
                return hc, "structured_heuristic_after_claude_fail"

    if cfg is not None:
        from .llm_triplet_three_source import (
            multimodal_llm_triplet_three_source_enabled,
            try_build_llm_triplet_three_source_chunks,
        )

        if multimodal_llm_triplet_three_source_enabled(cfg):
            hints0 = envelope.modality_hints or {}
            ak0 = str(hints0.get("answer_key_plaintext") or "").strip()
            raw_tpl = hints0.get("blank_assignment_template_bytes")
            raw_nb = hints0.get("blank_assignment_ipynb_bytes")
            blank_b = (
                bytes(raw_tpl) if isinstance(raw_tpl, (bytes, bytearray)) else b""
            )
            if not blank_b.strip():
                blank_b = (
                    bytes(raw_nb) if isinstance(raw_nb, (bytes, bytearray)) else b""
                )
            if ak0 and blank_b.strip():
                trip0 = try_build_llm_triplet_three_source_chunks(
                    envelope, cfg, answer_key_plaintext=ak0
                )
                if trip0:
                    return trip0

    ipynb_bytes = _get_ipynb_bytes(envelope)
    if ipynb_bytes is not None:
        tpl = None
        if cfg is not None:
            tpl = try_build_blank_template_aligned_chunks(envelope, cfg)
        if tpl:
            return tpl
        nb_mod = modality_from_hints(hints)
        if nb_mod == Modality.UNKNOWN:
            nb_mod = Modality.NOTEBOOK
        nb_chunks = build_notebook_qa_chunks(
            ipynb_bytes,
            assignment_id=envelope.assignment_id,
            student_id=envelope.student_id,
            modality=nb_mod,
            task_type=task_type_from_hints(hints),
            max_grading_units=max_units,
        )
        if nb_chunks:
            return nb_chunks, "notebook_cell_order"
        _log.info("Notebook chunker returned empty; trying other chunkers.")

    if cfg is not None and multimodal_llm_qa_segment_enabled():
        llm_units = _chunks_from_llm_qa_segmentation(envelope, cfg)
        if llm_units:
            tag = (llm_units[0].evidence or {}).get("chunker") or "llm_qa_segment"
            return llm_units, str(tag)
        _log.info("LLM QA segmentation disabled or empty; falling back to heuristic chunker.")

    hc = default_chunker_build_units(envelope)
    return hc, "structured_heuristic"


def sanitize_evidence_for_grading_prompt(evidence: dict[str, Any]) -> dict[str, Any]:
    """Drop raw embedding vectors from evidence for the grader JSON (keep dimension + source)."""
    out: dict[str, Any] = {}
    for k, v in (evidence or {}).items():
        if k == "rag_embedding_bundle" and isinstance(v, dict):
            rb = dict(v)
            rb.pop("embedding", None)
            rb["embedding_omitted_from_prompt"] = True
            out[k] = rb
        elif k == "answer_key_unit" and isinstance(v, dict):
            au = dict(v)
            rag = au.get("answer_key_rag")
            if isinstance(rag, dict):
                rag = dict(rag)
                rag.pop("embedding", None)
                rag["embedding_omitted_from_prompt"] = True
                au["answer_key_rag"] = rag
            out[k] = au
        elif k == "trio_segment_rag" and isinstance(v, dict):
            out[k] = {}
            for sk, sv in v.items():
                if isinstance(sv, dict):
                    d = dict(sv)
                    d.pop("embedding", None)
                    d["embedding_omitted_from_prompt"] = True
                    out[k][sk] = d
                else:
                    out[k][sk] = sv
        elif k in ("_openai_trio_rag_frontload", "_claude_structured_units"):
            continue
        elif k == "trio" and isinstance(v, dict):
            tq = _optional_positive_int_env("MULTIMODAL_TRIO_PROMPT_QUESTION_MAX_CHARS")
            tr = _optional_positive_int_env("MULTIMODAL_TRIO_PROMPT_RESPONSE_MAX_CHARS")
            ta = _optional_positive_int_env("MULTIMODAL_TRIO_PROMPT_ANSWER_KEY_MAX_CHARS")
            ti = _optional_positive_int_env("MULTIMODAL_TRIO_PROMPT_INSTRUCTOR_MAX_CHARS")
            trd = dict(v)
            q = str(trd.get("question") or "")
            s = str(trd.get("student_response") or "")
            a = str(trd.get("answer_key_segment") or "")
            ic = str(trd.get("instructor_context") or "")
            trd["question"] = (q[:tq] + ("…" if len(q) > tq else "")) if tq is not None else q
            trd["student_response"] = (s[:tr] + ("…" if len(s) > tr else "")) if tr is not None else s
            trd["answer_key_segment"] = (a[:ta] + ("…" if len(a) > ta else "")) if ta is not None else a
            trd["instructor_context"] = (ic[:ti] + ("…" if len(ic) > ti else "")) if ti is not None else ic
            out[k] = trd
        else:
            out[k] = v
    return out
