"""
Align the resolved answer key / sample response with each :class:`GradingChunk`.

After notebook (or heuristic) chunking, each unit has a ``question_id`` and optional
``evidence["question_text"]``.  This module:

1. Splits the full answer-key plaintext into coarse **sections** (markdown headings /
   numbered prompts).
2. Picks the section that best matches the chunk (heading contains ``question_id`` when
   possible; otherwise **cosine similarity** of embeddings vs. a chunk query string —
   same stack as :func:`app.grading.rag_embeddings.compute_submission_embedding`).
3. Writes ``chunk.evidence["answer_key_unit"]`` with parsed metadata, the matched
   **snippet**, and ``answer_key_rag`` (per-unit embedding) for downstream RAG / audit.

Raw vectors are stripped from grader prompts via
:func:`app.grading.multimodal.rag_embeddings.sanitize_evidence_for_grading_prompt`.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.grading.rag_embeddings import compute_submission_embedding

from .notebook_chunker import strip_assignment_placeholder_lines
from .schemas import GradingChunk

_log = logging.getLogger(__name__)

def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(a[i] * b[i] for i in range(len(a)))
    na = (sum(x * x for x in a)) ** 0.5
    nb = (sum(y * y for y in b)) ** 0.5
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return float(dot / (na * nb))


def split_answer_key_sections(answer_plain: str) -> list[tuple[str, str]]:
    """
    Split answer key text on markdown headings (``## ...`` at line start).

    Returns ``(heading_line, body)`` tuples; if no headings, a single ``("", full)``.
    """
    text = (answer_plain or "").strip()
    if not text:
        return []
    blocks = re.split(r"(?m)(?=^#{1,6}\s)", text)
    out: list[tuple[str, str]] = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n", 1)
        fl = lines[0].strip()
        body = lines[1].strip() if len(lines) > 1 else ""
        if fl.startswith("#"):
            out.append((fl, body))
        else:
            out.append(("", block))
    return out if out else [("", text)]


def _norm_code_line(line: str) -> str:
    return " ".join((line or "").strip().split())


def _executable_code_lines(text: str) -> list[str]:
    """Non-empty source lines, dropping full-line ``#`` comments only."""
    out: list[str] = []
    for raw in (text or "").splitlines():
        t = raw.strip()
        if not t or t.startswith("#"):
            continue
        out.append(_norm_code_line(t))
    return out


def code_reference_matches_student(*, student: str, reference: str) -> bool:
    """
    True when the student's code is the same **executable line sequence** as the reference,
    or (for a one-line student answer) that line appears as one executable line in the reference.

    Used to detect ``import csv`` == ``import csv`` even when ``reference`` is a longer section.
    """
    s_lines = _executable_code_lines(student)
    r_lines = _executable_code_lines(reference)
    if not s_lines:
        return False
    if s_lines == r_lines:
        return True
    if len(s_lines) == 1 and s_lines[0] in r_lines:
        return True
    if len(r_lines) == 1 and r_lines[0] in s_lines:
        return True
    joined_s, joined_r = "\n".join(s_lines), "\n".join(r_lines)
    if len(joined_s) <= 800 and joined_s and joined_s in joined_r:
        return True
    if len(joined_r) <= 800 and joined_r and joined_r in joined_s:
        return True
    return False


def grading_student_code_blob(chunk: GradingChunk) -> str:
    """Best-effort student code string for reference matching (trio → previews → chunk text)."""
    ev = chunk.evidence or {}
    trio = ev.get("trio")
    if isinstance(trio, dict):
        s = str(trio.get("student_response") or "").strip()
        if s:
            return s
    prev = str(ev.get("response_preview") or "").strip()
    if prev:
        return prev
    return (chunk.extracted_text or "").strip()


def narrow_answer_key_snippet_to_student_line(student: str, snippet: str) -> str | None:
    """
    If the student submission is effectively one executable line, return that same line
    from ``snippet`` when present (stabilizes prompts against huge matched sections).
    """
    s_lines = _executable_code_lines(student)
    if len(s_lines) != 1:
        return None
    want = s_lines[0]
    for raw in snippet.splitlines():
        t = raw.strip()
        if not t or t.startswith("#"):
            continue
        if _norm_code_line(t) == want:
            return t.strip()
    return None


def _normalize_qid(q: str) -> str:
    return re.sub(r"\s+", "", (q or "").strip().lower())

def _heading_matches_question_id(heading: str, qid: str) -> bool:
    if not heading or not qid:
        return False
    h = heading.lower()
    qn = _normalize_qid(qid)
    if qn and qn in re.sub(r"\s+", "", h.lower()):
        return True
    # Loose: "1.4.1" appears after "question" etc.
    if re.search(rf"\b{re.escape(qid)}\b", heading, re.I):
        return True
    return False


def _chunk_query_text(ch: GradingChunk) -> str:
    ev = ch.evidence or {}
    qtxt = str(ev.get("question_text") or "").strip()
    head = "\n".join((ch.extracted_text or "").splitlines()[:12]).strip()
    parts = [str(ch.question_id or "").strip(), qtxt, head]
    return "\n\n".join(p for p in parts if p).strip() or (ch.extracted_text or "").strip()


def _pick_section_for_chunk(
    ch: GradingChunk,
    sections: list[tuple[str, str]],
    cfg: Any,
) -> tuple[str, str, str, float | None]:
    """
    Return ``(snippet, match_method, matched_heading, cosine_or_none)``.

    ``snippet`` is capped for storage / embedding; may be shortened further in prompts.
    """
    qid = str(ch.question_id or "").strip()
    for hdr, body in sections:
        if _heading_matches_question_id(hdr, qid):
            snip = (hdr + "\n\n" + body).strip() if hdr else body.strip()
            return snip[:24_000], "question_id_heading", hdr, None

    if not sections:
        return "", "none", "", None

    query = _chunk_query_text(ch)
    if not query.strip():
        body0 = sections[0][1] if sections else ""
        hdr0 = sections[0][0] if sections else ""
        snip = (hdr0 + "\n\n" + body0).strip() if hdr0 else (body0 or "").strip()
        return snip[:24_000], "full_key_fallback", hdr0, None

    try:
        q_vec, _ = compute_submission_embedding(query, cfg)
    except Exception:
        _log.debug("answer_key_chunk_enrich: embedding query failed", exc_info=True)
        hdr, body = sections[0]
        snip = (hdr + "\n\n" + body).strip() if hdr else body.strip()
        return snip[:24_000], "fallback_first_section", hdr, None

    best_sim = -1.0
    best: tuple[str, str] = ("", "")
    for hdr, body in sections:
        blob = (hdr + "\n" + body).strip()[:20_000]
        if not blob.strip():
            continue
        try:
            b_vec, _ = compute_submission_embedding(blob, cfg)
            sim = _cosine(q_vec, b_vec)
        except Exception:
            continue
        if sim > best_sim:
            best_sim = sim
            best = (hdr, body)
    hdr, body = best
    snip = (hdr + "\n\n" + body).strip() if hdr else body.strip()
    return snip[:24_000], "embedding_cosine", hdr, float(best_sim)


def enrich_chunks_with_per_question_answer_key(
    chunks: list[GradingChunk],
    answer_key_plain: str,
    cfg: Any,
) -> None:
    """
    Mutate each chunk's ``evidence`` with ``answer_key_unit`` + ``answer_key_rag`` embedding.

    No-op when ``answer_key_plain`` is empty or ``cfg`` is ``None``.
    """
    if not chunks or not str(answer_key_plain or "").strip() or cfg is None:
        return
    answer_key_plain = strip_assignment_placeholder_lines(str(answer_key_plain))
    if not answer_key_plain.strip():
        return
    sections = split_answer_key_sections(answer_key_plain)
    if not sections:
        return

    for ch in chunks:
        snippet, method, hdr, cos = _pick_section_for_chunk(ch, sections, cfg)
        if not snippet.strip():
            continue
        student_blob = grading_student_code_blob(ch)
        ev0 = dict(ch.evidence or {})
        trio0 = ev0.get("trio")
        prior_seg = ""
        if isinstance(trio0, dict):
            prior_seg = str(trio0.get("answer_key_segment") or "").strip()
        if (
            prior_seg
            and student_blob.strip()
            and code_reference_matches_student(student=student_blob, reference=prior_seg)
        ):
            snippet = prior_seg
            method = f"{method};trio_answer_key_preferred"
        else:
            narrowed = narrow_answer_key_snippet_to_student_line(student_blob, snippet)
            if narrowed:
                snippet = narrowed
                method = f"{method};narrowed_to_student_matching_line"
        try:
            emb_vec, emb_src = compute_submission_embedding(snippet[:20_000], cfg)
        except Exception:
            _log.warning(
                "answer_key_chunk_enrich: per-unit embedding failed for %s",
                ch.chunk_id,
                exc_info=True,
            )
            emb_vec, emb_src = [], "embedding_failed"

        ev = dict(ch.evidence or {})
        unit: dict[str, Any] = {
            "question_id": str(ch.question_id or ""),
            "parsed": {
                "match_method": method,
                "matched_section_heading": hdr[:500],
                "snippet_char_count": len(snippet),
                "query_to_section_cosine": cos,
            },
            "snippet": snippet,
        }
        if isinstance(emb_vec, list) and len(emb_vec) >= 8:
            unit["answer_key_rag"] = {
                "embedding_dimension": len(emb_vec),
                "embedding_source": emb_src,
                "embedding": emb_vec,
            }
        ev["answer_key_unit"] = unit
        trio = ev.get("trio")
        if isinstance(trio, dict):
            trio = dict(trio)
            trio["answer_key_segment"] = snippet
            ev["trio"] = trio
        elif snippet.strip():
            rsp = str(ev.get("response_text") or ev.get("response_preview") or "").strip()
            ev["trio"] = {
                "question": str(ev.get("question_text") or "").strip(),
                "student_response": rsp,
                "instructor_context": "",
                "answer_key_segment": snippet,
            }
        ch.evidence = ev


def embed_full_answer_key_for_audit(answer_plain: str, cfg: Any) -> dict[str, Any] | None:
    """Single vector over the capped answer key (metadata suitable for ``pipeline_audit``)."""
    if not str(answer_plain or "").strip() or cfg is None:
        return None
    cap = 24_000
    blob = answer_plain.strip()[:cap]
    try:
        vec, src = compute_submission_embedding(blob, cfg)
    except Exception:
        return None
    return {
        "embedding_dimension": len(vec),
        "embedding_source": src,
        "plaintext_chars_capped": len(blob),
        "embedding_omitted_from_json": True,
    }
