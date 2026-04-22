"""
RAG-style chunk embeddings and optional Ollama Q→A segmentation for multimodal grading.

Aligns with :mod:`app.grading.rag_embeddings` — per-chunk vectors from
:func:`app.grading.rag_embeddings.compute_submission_embedding` (default
``RAG_EMBEDDING_BACKEND=sentence_transformers`` / ``SENTENCE_TRANSFORMERS_MODEL``).

**Chunking stack** (see ``new_chunking_method.md``):

1. Raw **ipynb** bytes → :func:`notebook_chunker.build_notebook_qa_chunks` (cell-order Q/A).
2. Otherwise **PDF plaintext** is reflowed via
   :func:`app.grading.submission_chunks.reflow_pdf_sections_in_plaintext` before any LLM
   segmentation so verticalized extractors do not confuse the model.
3. If ``MULTIMODAL_OLLAMA_QA_SEGMENT`` is on, the **structure LLM** (Ollama or Hugging Face
   per ``MULTIMODAL_LLM_BACKEND``) returns JSON Q/A units; on failure,
   :func:`chunker.default_chunker_build_units` runs structured chunking
   (:func:`app.grading.submission_chunks.build_submission_chunks`, which reflows each PDF
   section again and applies journal-style prompt boundaries when hints match).
4. If ``OPENAI_API_KEY`` is set and OpenAI trio frontload is enabled (default **auto**;
   disable with ``MULTIMODAL_OPENAI_TRIO_RAG_FRONTLOAD=off``),
   :func:`app.grading.multimodal.openai_trio_rag_frontload.run_openai_trio_rag_frontload`
   replaces steps 3–4 for
   that run (one or more OpenAI chats on overlapping windows when the submission is long,
   plus OpenAI Embeddings API). Otherwise, if
   ``MULTIMODAL_LLM_TRIO_CHUNKING`` is on, :func:`refine_chunks_trio_with_ollama` runs after
   units exist to label ``question`` / ``student_response`` / ``instructor_context`` via the
   structure client (answer-key snippet alignment still uses answer-key enrich).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from app.config import Config
from app.grading.llm_router import (
    OllamaClient,
    OpenAIJsonClient,
    _ollama_keep_alive,
    huggingface_grading_model_id,
    huggingface_json_client_from_config,
    multimodal_llm_backend_uses_huggingface,
    multimodal_llm_backend_uses_openai,
    openai_multimodal_grading_model,
)
from app.grading.rag_embeddings import compute_submission_embedding

from app.grading.submission_chunks import reflow_pdf_sections_in_plaintext

from .chunker import default_chunker_build_units, modality_from_hints, task_type_from_hints
from .ingestion import IngestionEnvelope
from .notebook_chunker import build_notebook_qa_chunks
from .schemas import GradingChunk, Modality, TaskType

_log = logging.getLogger(__name__)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def multimodal_ollama_qa_segment_enabled() -> bool:
    return _env_bool("MULTIMODAL_OLLAMA_QA_SEGMENT", default=False)


def multimodal_rag_embed_units_enabled() -> bool:
    return _env_bool("MULTIMODAL_RAG_EMBED_UNITS", default=True)


def multimodal_llm_trio_chunking_enabled(cfg: Config | None = None) -> bool:
    """True when Ollama should re-label each chunk's ``evidence['trio']`` before AK enrich."""
    if cfg is not None and bool(getattr(cfg, "MULTIMODAL_LLM_TRIO_CHUNKING", False)):
        return True
    return _env_bool("MULTIMODAL_LLM_TRIO_CHUNKING", default=False)


def _multimodal_structure_llm_model(cfg: Config) -> str:
    """
    Ollama model for (a) optional PDF/plain Q→A JSON segmentation and (b) optional trio LLM chunking.

    Precedence: ``MULTIMODAL_TRIO_CHUNKING_MODEL`` → ``OLLAMA_MODEL`` →
    ``MULTIMODAL_QA_SEGMENT_MODEL`` (env) → ``llama3.2:1b``.
    """
    m = (getattr(cfg, "MULTIMODAL_TRIO_CHUNKING_MODEL", "") or "").strip()
    if m:
        return m
    om = (getattr(cfg, "OLLAMA_MODEL", "") or "").strip()
    if om:
        return om
    q = os.getenv("MULTIMODAL_QA_SEGMENT_MODEL", "").strip()
    if q:
        return q
    return "llama3.2:1b"


def _qa_segment_model(cfg: Config) -> str:
    return _multimodal_structure_llm_model(cfg)


def _ollama_base(cfg: Config) -> str:
    return (cfg.INTERNAL_OLLAMA_URL or cfg.OLLAMA_BASE_URL or "").strip().rstrip("/")


def _multimodal_structure_chat_client(cfg: Config, *, purpose: str = "trio"):
    """
    Return (client, model_label) for optional QA segmentation and trio chunking.

    When ``MULTIMODAL_LLM_BACKEND`` is ``huggingface`` / ``hf``, uses local transformers
    (:func:`~app.grading.llm_router.huggingface_json_client_from_config`). When it is
    ``openai``, uses :class:`~app.grading.llm_router.OpenAIJsonClient`` (requires
    ``OPENAI_API_KEY``). Otherwise uses Ollama (requires ``OLLAMA_BASE_URL``).

    ``purpose`` selects HTTP read timeout for Ollama only: ``"qa"`` (segmentation) vs
    ``"trio"`` (chunk relabeling).
    """
    if multimodal_llm_backend_uses_huggingface(cfg):
        try:
            mid = huggingface_grading_model_id(cfg)
            return huggingface_json_client_from_config(cfg, mid), mid
        except Exception:
            _log.warning(
                "Could not build Hugging Face structure LLM client; trio/QA segment skipped.",
                exc_info=True,
            )
            return None, ""
    if multimodal_llm_backend_uses_openai(cfg):
        key = (cfg.OPENAI_API_KEY or "").strip()
        if not key:
            _log.warning(
                "MULTIMODAL_LLM_BACKEND=openai but OPENAI_API_KEY is empty; "
                "structure QA/trio client unavailable."
            )
            return None, ""
        mid = openai_multimodal_grading_model(cfg)
        return OpenAIJsonClient(key, mid), f"openai:{mid}"
    base = _ollama_base(cfg)
    if not base:
        return None, ""
    model = _multimodal_structure_llm_model(cfg)
    if purpose == "qa":
        tout = _qa_segment_http_timeout_sec(cfg)
    else:
        try:
            tout = float(getattr(cfg, "OLLAMA_CHAT_TIMEOUT_SEC", 300))
            tout = min(1800.0, max(60.0, tout))
        except (TypeError, ValueError):
            tout = 300.0
    client = OllamaClient(
        base,
        model,
        request_json_format=getattr(cfg, "OLLAMA_CHAT_JSON_FORMAT", True),
        timeout_sec=tout,
        keep_alive=_ollama_keep_alive(cfg),
    )
    return client, model


def _chat_timeout(cfg: Config) -> float:
    return float(getattr(cfg, "OLLAMA_CHAT_TIMEOUT_SEC", 120))


def _qa_segment_http_timeout_sec(cfg: Config) -> float:
    """HTTP read timeout for Ollama QA segmentation (same order of magnitude as grading)."""
    raw = float(_chat_timeout(cfg))
    return min(1200.0, max(45.0, raw))


_QA_SYSTEM = """You are a teaching assistant. Split the student's assignment text into separate gradable units.
Each unit is one question, exercise, or coding problem and the student's answer/response/code that belongs to it.
Return **only** valid JSON with this shape (no markdown):
{"units":[{"key":"string_id","question":"prompt or problem text","response":"student answer or code"}]}
Use short unique keys like q1, q2, p1a. If there is only one blob of text, use one unit with key q1.
The "question" field is the instructor prompt; "response" is what the student wrote for that prompt."""


def _chunks_from_ollama_qa_segmentation(
    envelope: IngestionEnvelope,
    cfg: Config,
) -> list[GradingChunk] | None:
    plain = reflow_pdf_sections_in_plaintext((envelope.extracted_plaintext or "").strip())
    if not plain:
        return None
    client, model = _multimodal_structure_chat_client(cfg, purpose="qa")
    if client is None:
        return None
    # Full submission text so Q/A segmentation sees the entire document (model context limits may still apply).
    user = plain
    try:
        obj = client.chat_json(
            [
                {"role": "system", "content": _QA_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
        )
    except Exception:
        _log.warning("Ollama QA segmentation failed; using heuristic chunker.", exc_info=True)
        return None

    units = obj.get("units")
    if not isinstance(units, list) or not units:
        return None

    hints = envelope.modality_hints or {}
    modality = modality_from_hints(hints)
    task = task_type_from_hints(hints)
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
                    "chunker": "ollama_qa_segment",
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


def refine_chunks_trio_with_ollama(chunks: list[GradingChunk], cfg: Config) -> None:
    """
    For each chunk, call Ollama once to populate ``evidence["trio"]`` (question /
    student_response / instructor_context). ``answer_key_segment`` is left unchanged here
    and is filled by :func:`answer_key_chunk_enrich.enrich_chunks_with_per_question_answer_key`.

    Controlled by ``MULTIMODAL_LLM_TRIO_CHUNKING`` / ``cfg.MULTIMODAL_LLM_TRIO_CHUNKING`` and
    the structure LLM from :func:`_multimodal_structure_chat_client` (Ollama or Hugging Face).
    """
    if not chunks or not multimodal_llm_trio_chunking_enabled(cfg):
        return
    client, model = _multimodal_structure_chat_client(cfg, purpose="trio")
    if client is None:
        _log.warning(
            "refine_chunks_trio_with_ollama: no structure LLM client "
            "(set Ollama URL or MULTIMODAL_LLM_BACKEND=huggingface with HF token)"
        )
        return
    try:
        cap = int(os.getenv("MULTIMODAL_LLM_TRIO_INPUT_MAX_CHARS", "28000") or 28000)
    except ValueError:
        cap = 28_000
    cap = max(4000, min(cap, 120_000))
    for ch in chunks:
        ev0 = ch.evidence or {}
        if ev0.get("trio_llm_refined"):
            continue
        raw_in = (ch.extracted_text or "").strip()
        if not raw_in:
            continue
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
        joined = "\n\n".join(
            p for p in (ev["trio"]["question"], ev["trio"]["student_response"]) if p
        ).strip()
        if joined:
            ch.extracted_text = joined
        ev["trio_llm_refined"] = True
        ev["trio_llm_chunking_model"] = model
        ch.evidence = ev


def build_multimodal_grading_chunks(
    envelope: IngestionEnvelope,
    cfg: Config | None = None,
) -> tuple[list[GradingChunk], str]:
    """
    Build grading units using the best available chunker:

    1. **Notebook cell-order** — if the envelope carries raw ipynb bytes,
       parse cells directly and pair each question heading with the student's
       response cells (preserves the natural cell ordering).  Does not require
       ``cfg`` — works on the raw notebook JSON alone.
    2. **Ollama QA segmentation** — LLM-based JSON segmentation (requires
       ``cfg``; skipped when ``cfg`` is ``None``).  Plaintext is PDF-reflowed first.
    3. **Structured heuristic** — :func:`chunker.default_chunker_build_units` on plaintext
       (PDF reflow + ``build_submission_chunks`` / journal-style boundaries per
       ``new_chunking_method.md``).

    ``cfg`` may be ``None`` when the pipeline runs without a full application
    config (e.g. unit tests with a mock runner).  The notebook and heuristic
    chunkers operate without it; only Ollama QA segmentation is skipped.
    """
    hints = envelope.modality_hints or {}
    cap = hints.get("max_grading_units")
    max_units = None
    if cap is not None:
        try:
            max_units = int(cap)
        except (TypeError, ValueError):
            pass

    ipynb_bytes = _get_ipynb_bytes(envelope)
    if ipynb_bytes is not None:
        nb_chunks = build_notebook_qa_chunks(
            ipynb_bytes,
            assignment_id=envelope.assignment_id,
            student_id=envelope.student_id,
            modality=modality_from_hints(hints),
            task_type=task_type_from_hints(hints),
            max_grading_units=max_units,
        )
        if nb_chunks:
            return nb_chunks, "notebook_cell_order"
        _log.info("Notebook chunker returned empty; trying other chunkers.")

    if cfg is not None and multimodal_ollama_qa_segment_enabled():
        ollama_units = _chunks_from_ollama_qa_segmentation(envelope, cfg)
        if ollama_units:
            return ollama_units, "ollama_qa_segment"
        _log.info("Ollama QA segmentation disabled or empty; falling back to heuristic chunker.")

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
        elif k == "_openai_trio_rag_frontload":
            continue
        elif k == "trio" and isinstance(v, dict):
            tq = _env_int("MULTIMODAL_TRIO_PROMPT_QUESTION_MAX_CHARS", 8_000)
            tr = _env_int("MULTIMODAL_TRIO_PROMPT_RESPONSE_MAX_CHARS", 12_000)
            ta = _env_int("MULTIMODAL_TRIO_PROMPT_ANSWER_KEY_MAX_CHARS", 12_000)
            ti = _env_int("MULTIMODAL_TRIO_PROMPT_INSTRUCTOR_MAX_CHARS", 4_000)
            trd = dict(v)
            q = str(trd.get("question") or "")
            s = str(trd.get("student_response") or "")
            a = str(trd.get("answer_key_segment") or "")
            ic = str(trd.get("instructor_context") or "")
            trd["question"] = q[:tq] + ("…" if len(q) > tq else "")
            trd["student_response"] = s[:tr] + ("…" if len(s) > tr else "")
            trd["answer_key_segment"] = a[:ta] + ("…" if len(a) > ta else "")
            trd["instructor_context"] = ic[:ti] + ("…" if len(ic) > ti else "")
            out[k] = trd
        else:
            out[k] = v
    return out
