"""
Split submission plain text into explicit **question** vs **answer** chunks for RAG and review.

Chunking strategy (high level)
-------------------------------

1. **Artifact sections** — If the text was produced by
   :func:`app.grading.submission_text.submission_text_from_artifacts`, it may contain
   banner lines such as ``=== NOTEBOOK CODE (ipynb) ===``. We split on those first so
   notebook *code* can be labeled ``role: \"code\"`` and prose/markdown as ``response``
   unless a line looks like source (see heuristics below).

2. **Question / answer pairs (prose sections)** — Within each prose block (PDF text,
   markdown export, plain txt, notebook markdown), we look for *question-like* single
   lines: ``Part 1.``, ``Question 2``, ``Q3:``, numbered items, ``## Heading``, etc.
   (same patterns as before, one line per header).

   * For each header line we emit one chunk with ``role: \"question\"`` whose ``text`` is
     exactly that prompt line (trimmed).
   * The material **after** that line until the next header is the student work; we emit
     one or more chunks with ``role: \"response\"`` or ``role: \"code\"`` depending on
     content (and section), packed to ``max_chunk_chars`` by paragraph.

3. **Leading body before the first header** — Treated as unattached submission text:
   ``role: \"response\"`` (or ``code`` if it clearly looks like code) with
   ``pair_id: null``.

4. **No headers found** — If there are no question-like lines, the whole section becomes
   ``response`` and/or ``code`` chunks only (typical for pasted essays or single-cell
   notebooks).

5. **Metadata** — Every chunk repeats ``assignment_title``, ``modality_subtype``,
   ``chunk_index``, and when applicable ``pair_id`` linking a question to its answer chunk(s).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence

# Banner lines from submission_text_from_artifacts: "=== LABEL (key) ===" or "=== LABEL ==="
_SECTION_BANNER = re.compile(
    r"(?m)^(===\s*.+?\s*===)\s*\n",
)

# Question / prompt lines (single line only; horizontal whitespace only after header key).
_CHUNK_HEADER = re.compile(
    r"(?m)^(?:"
    r"[ \t]*(?:Part|Question|Q)[ \t]*[\dA-Z]+[\.\):]?[ \t]*[^\n]*|"
    r"[ \t]*\d+[\.\)][ \t]+\S[^\n]*|"
    r"[ \t]*#{1,3}[ \t]+\S[^\n]*"
    r")\s*$",
    re.IGNORECASE,
)

# Lightweight "is this Python/code?" heuristic for classifying answer bodies.
_CODE_LINE_PATTERNS = (
    re.compile(r"^\s*(def |class |import |from .+ import |if __name__)"),
    re.compile(r"^\s*[{}\]\);]\s*$"),
    re.compile(r"^\s*#|^\s*//"),
    re.compile(r"=\s*\[|=\s*\{"),
)


def _infer_section_kind(banner: str, modality_subtype: str) -> str:
    u = banner.upper()
    if "NOTEBOOK CODE" in u or "PYTHON SOURCE" in u:
        return "code"
    if "NOTEBOOK MARKDOWN" in u or ".MD" in u or "MARKDOWN" in u:
        return "markdown"
    if "PDF TEXT" in u:
        return "pdf"
    if "TXT" in u and "NOTEBOOK" not in u:
        return "txt"
    if modality_subtype == "notebook":
        return "markdown"
    if modality_subtype in ("code", "mixed_notebook_pdf"):
        return "mixed"
    return "body"


def _split_artifact_sections(
    text: str, *, modality_subtype: str = ""
) -> list[tuple[str, str, str]]:
    """
    Split ``text`` into ``(section_banner, section_kind, body)`` tuples.
    If there are no ``===`` banners, returns one row ``("", inferred_kind, text)``.
    """
    raw = text or ""
    if not raw.strip():
        return []

    matches = list(_SECTION_BANNER.finditer(raw))
    if not matches:
        return [("", "body", raw.strip())]

    out: list[tuple[str, str, str]] = []
    for i, m in enumerate(matches):
        banner = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        body = raw[start:end].strip()
        kind = _infer_section_kind(banner, modality_subtype)
        out.append((banner, kind, body))
    return out


def _looks_like_code(text: str, *, section_kind: str) -> bool:
    if section_kind == "code":
        return True
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return False
    hits = 0
    for ln in lines[: min(40, len(lines))]:
        for pat in _CODE_LINE_PATTERNS:
            if pat.search(ln):
                hits += 1
                break
    # Many short lines with '=' assignments / brackets
    if len(lines) >= 3 and hits >= max(2, len(lines) // 5):
        return True
    return hits >= 3


def _answer_role_for_body(
    body: str,
    *,
    section_kind: str,
    modality_subtype: str,
) -> str:
    if section_kind == "code":
        return "code"
    if _looks_like_code(body, section_kind=section_kind):
        return "code"
    st = (modality_subtype or "").lower()
    if st in ("notebook", "mixed_notebook_pdf") and section_kind == "markdown":
        # Markdown cells can still contain fenced code blocks; long fenced blocks heuristic
        if body.count("```") >= 2 and len(body) > 200:
            return "code"
    return "response"


def _pack_text_by_paragraphs(
    text: str,
    *,
    role: str,
    pair_id: int | None,
    max_chunk_chars: int,
    assignment_title: str,
    modality_subtype: str,
    chunk_index_start: int,
    section_banner: str,
) -> tuple[list[dict[str, Any]], int]:
    text = text.strip()
    if not text:
        return [], chunk_index_start
    out: list[dict[str, Any]] = []
    idx = chunk_index_start

    def emit_piece(piece: str) -> None:
        nonlocal idx
        out.append(
            {
                "role": role,
                "pair_id": pair_id,
                "text": piece,
                "chunk_index": idx,
                "assignment_title": assignment_title,
                "modality_subtype": modality_subtype,
                "section_banner": section_banner or None,
            }
        )
        idx += 1

    if len(text) <= max_chunk_chars:
        emit_piece(text)
        return out, idx

    paras: list[str] = [p.strip() for p in text.split("\n\n") if p.strip()]
    buf: list[str] = []
    cur = 0
    for p in paras:
        add_len = len(p) + (2 if buf else 0)
        if cur + add_len > max_chunk_chars and buf:
            emit_piece("\n\n".join(buf))
            buf = []
            cur = 0
        buf.append(p)
        cur += add_len
    if buf:
        emit_piece("\n\n".join(buf))
    return out, idx


def _chunks_from_prose_section(
    body: str,
    *,
    section_kind: str,
    section_banner: str,
    assignment_title: str,
    modality_subtype: str,
    max_chunk_chars: int,
    chunk_index_start: int,
    pair_id_start: int,
) -> tuple[list[dict[str, Any]], int, int]:
    """
    Build question + answer chunks. Uses the same (question_line, answer_blob) segmentation
    as the previous implementation: each answer blob is paired with the **previous** header
    line as its question text.
    """
    chunks: list[dict[str, Any]] = []
    idx = chunk_index_start
    pair_id = pair_id_start

    boundaries = list(_CHUNK_HEADER.finditer(body))
    if not boundaries:
        ans_role = _answer_role_for_body(
            body, section_kind=section_kind, modality_subtype=modality_subtype
        )
        packed, idx = _pack_text_by_paragraphs(
            body,
            role=ans_role,
            pair_id=None,
            max_chunk_chars=max_chunk_chars,
            assignment_title=assignment_title,
            modality_subtype=modality_subtype,
            chunk_index_start=idx,
            section_banner=section_banner,
        )
        chunks.extend(packed)
        return chunks, idx, pair_id

    segments: list[tuple[str | None, str]] = []
    pos = 0
    current_q: str | None = None
    for m in boundaries:
        if m.start() > pos:
            blob = body[pos : m.start()].strip()
            if blob:
                segments.append((current_q, blob))
        current_q = m.group().strip()
        pos = m.end()
    tail = body[pos:].strip()
    if tail:
        segments.append((current_q, tail))

    for q_line, ans_text in segments:
        if q_line is None:
            ans_role = _answer_role_for_body(
                ans_text, section_kind=section_kind, modality_subtype=modality_subtype
            )
            packed, idx = _pack_text_by_paragraphs(
                ans_text,
                role=ans_role,
                pair_id=None,
                max_chunk_chars=max_chunk_chars,
                assignment_title=assignment_title,
                modality_subtype=modality_subtype,
                chunk_index_start=idx,
                section_banner=section_banner,
            )
            chunks.extend(packed)
            continue

        pid = pair_id
        pair_id += 1
        chunks.append(
            {
                "role": "question",
                "pair_id": pid,
                "text": q_line,
                "chunk_index": idx,
                "assignment_title": assignment_title,
                "modality_subtype": modality_subtype,
                "section_banner": section_banner or None,
            }
        )
        idx += 1
        ans_role = _answer_role_for_body(
            ans_text, section_kind=section_kind, modality_subtype=modality_subtype
        )
        packed, idx = _pack_text_by_paragraphs(
            ans_text,
            role=ans_role,
            pair_id=pid,
            max_chunk_chars=max_chunk_chars,
            assignment_title=assignment_title,
            modality_subtype=modality_subtype,
            chunk_index_start=idx,
            section_banner=section_banner,
        )
        chunks.extend(packed)

    return chunks, idx, pair_id


def build_submission_chunks(
    text: str,
    *,
    assignment_title: str = "",
    modality_subtype: str = "",
    max_chunk_chars: int = 4000,
) -> list[dict[str, Any]]:
    """
    Produce chunks with ``role`` of ``question``, ``response``, or ``code``.

    See module docstring for the full chunking explanation.
    """
    raw = (text or "").strip()
    if not raw:
        return []

    sections = _split_artifact_sections(raw)
    all_chunks: list[dict[str, Any]] = []
    idx = 0
    next_pair = 0

    for banner, sec_kind, body in sections:
        if not body.strip():
            continue

        if sec_kind == "code":
            packed, idx = _pack_text_by_paragraphs(
                body,
                role="code",
                pair_id=None,
                max_chunk_chars=max_chunk_chars,
                assignment_title=assignment_title,
                modality_subtype=modality_subtype,
                chunk_index_start=idx,
                section_banner=banner,
            )
            all_chunks.extend(packed)
            continue

        if sec_kind in ("markdown", "pdf", "txt", "body", "mixed"):
            effective_kind = sec_kind
            if sec_kind == "mixed" and _looks_like_code(body, section_kind="body"):
                effective_kind = "code"
            if effective_kind == "code":
                packed, idx = _pack_text_by_paragraphs(
                    body,
                    role="code",
                    pair_id=None,
                    max_chunk_chars=max_chunk_chars,
                    assignment_title=assignment_title,
                    modality_subtype=modality_subtype,
                    chunk_index_start=idx,
                    section_banner=banner,
                )
                all_chunks.extend(packed)
                continue

            part, idx, next_pair = _chunks_from_prose_section(
                body,
                section_kind="markdown" if sec_kind == "mixed" else sec_kind,
                section_banner=banner,
                assignment_title=assignment_title,
                modality_subtype=modality_subtype,
                max_chunk_chars=max_chunk_chars,
                chunk_index_start=idx,
                pair_id_start=next_pair,
            )
            all_chunks.extend(part)

    # Renumber chunk_index contiguously
    for i, c in enumerate(all_chunks):
        c["chunk_index"] = i

    return all_chunks


def write_chunks_json(
    out_path: Path,
    *,
    chunks: Sequence[dict[str, Any]],
    assignment_title: str = "",
    source_file: str = "",
    profile: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "assignment_title": assignment_title,
        "source_file": source_file,
        "chunk_count": len(chunks),
        "chunking_notes": (
            "question = rubric/part line; response/code = student work. "
            "pair_id links question to answer chunk(s); null = no detected prompt line."
        ),
        "chunks": list(chunks),
    }
    if profile:
        payload["modality"] = profile
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
