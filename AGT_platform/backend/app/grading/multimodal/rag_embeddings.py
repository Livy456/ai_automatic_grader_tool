"""
RAG-style chunk embeddings and optional Ollama Q→A segmentation for multimodal grading.

Aligns with :mod:`app.grading.rag_embeddings` (same ``compute_submission_embedding`` order)
and reuses structured chunking from :func:`app.grading.submission_chunks.build_submission_chunks`
when heuristic units are used.

When ``MULTIMODAL_OLLAMA_QA_SEGMENT`` is enabled, a **small** Ollama model (default
``llama3.2:1b`` via ``MULTIMODAL_QA_SEGMENT_MODEL``) returns JSON units with stable ``key``
fields mapping each question/prompt to its response; on failure the pipeline falls back
to :func:`default_chunker_build_units`.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from app.config import Config
from app.grading.llm_router import OllamaClient
from app.grading.rag_embeddings import compute_submission_embedding

from .chunker import default_chunker_build_units, modality_from_hints, task_type_from_hints
from .ingestion import IngestionEnvelope
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


def _qa_segment_model(cfg: Config) -> str:
    m = os.getenv("MULTIMODAL_QA_SEGMENT_MODEL", "").strip()
    if m:
        return m
    return "llama3.2:1b"


def _ollama_base(cfg: Config) -> str:
    return (cfg.INTERNAL_OLLAMA_URL or cfg.OLLAMA_BASE_URL or "").strip().rstrip("/")


def _chat_timeout(cfg: Config) -> float:
    return float(getattr(cfg, "OLLAMA_CHAT_TIMEOUT_SEC", 120))


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
    plain = (envelope.extracted_plaintext or "").strip()
    if not plain:
        return None
    base = _ollama_base(cfg)
    if not base:
        return None
    model = _qa_segment_model(cfg)
    client = OllamaClient(
        base,
        model,
        request_json_format=getattr(cfg, "OLLAMA_CHAT_JSON_FORMAT", True),
        timeout_sec=min(180.0, _chat_timeout(cfg)),
    )
    user = plain[:28000]
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
    for i, u in enumerate(units):
        if not isinstance(u, dict):
            continue
        key = str(u.get("key") or f"q{i + 1}").strip() or f"q{i + 1}"
        q = str(u.get("question") or "").strip()
        r = str(u.get("response") or "").strip()
        extracted = "\n\n".join(p for p in (q, r) if p).strip()
        if not extracted:
            continue
        qid = key.replace(" ", "_")[:120]
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


def enrich_chunks_with_rag_embeddings(chunks: list[GradingChunk], cfg: Config) -> None:
    """Attach per-unit vectors via :func:`compute_submission_embedding` (same stack as legacy RAG)."""
    if not multimodal_rag_embed_units_enabled():
        return
    for ch in chunks:
        txt = (ch.extracted_text or "").strip() or " "
        vec, src = compute_submission_embedding(txt, cfg)
        ev = dict(ch.evidence or {})
        ev["rag_embedding_bundle"] = {
            "embedding_dimension": len(vec),
            "embedding_source": src,
            "embedding": vec,
        }
        ch.evidence = ev


def build_multimodal_grading_chunks(
    envelope: IngestionEnvelope,
    cfg: Config,
) -> tuple[list[GradingChunk], str]:
    """
    Build grading units: optional Ollama JSON Q/A segmentation, else structured heuristic
    (``build_submission_chunks`` → ``build_grading_units_from_chunks``).
    """
    if multimodal_ollama_qa_segment_enabled():
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
        else:
            out[k] = v
    return out
