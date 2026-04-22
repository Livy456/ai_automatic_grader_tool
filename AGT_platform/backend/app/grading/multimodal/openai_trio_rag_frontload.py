"""
One-shot OpenAI chunking + OpenAI embeddings for multimodal RAG (optional).

When ``OPENAI_API_KEY`` is set and ``MULTIMODAL_OPENAI_TRIO_RAG_FRONTLOAD`` is not
``off``/``false`` (default **auto** = on), :func:`run_openai_trio_rag_frontload` calls one or
more chat completions with ``OPENAI_TRIO_RAG_CHAT_MODEL`` (default ``gpt-5.4-nano``) to emit JSON
``units`` with ``question``, ``student_response``, and ``answer_key_segment`` per
gradable item (overlapping **windows** when the submission exceeds
``MULTIMODAL_OPENAI_TRIO_WINDOW_CHARS``), then vectorizes each trio (and a canonical join) with the
OpenAI **Embeddings** API using ``OPENAI_TRIO_RAG_EMBEDDING_MODEL`` (default
``text-embedding-3-small``). Chat models do not produce embedding vectors; the
nano model is used only for structured trio extraction.

Downstream chunk grading uses ``MULTIMODAL_LLM_BACKEND``; this module does not grade chunks.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from app.config import Config
from app.grading.llm_router import OpenAIJsonClient
from app.grading.submission_chunks import reflow_pdf_sections_in_plaintext

from .chunker import modality_from_hints, task_type_from_hints
from .ingestion import IngestionEnvelope
from .schemas import GradingChunk, Modality, TaskType

_log = logging.getLogger(__name__)

_TRIO_RAG_SYSTEM = """You split a student assignment into ordered grading units.
Each unit must align with one question or sub-question when possible.

Return **only** a JSON object (no markdown fences) with this shape:
{"units":[
  {"question_id":"short id e.g. 1.4.1 or q1",
   "question":"the prompt or problem statement for this unit",
   "student_response":"the student's answer for this unit only",
   "answer_key_segment":"the matching excerpt from the provided answer key / sample solution for this unit (empty string if none applies)",
   "extracted_text":"verbatim or near-verbatim concatenation useful for grading (typically question + student response)"}
]}

Rules:
- Prefer copying text from the submission and answer key; do not invent grades.
- If the answer key has numbered sections, align ``answer_key_segment`` to the same ``question_id`` when headings match.
- Every unit needs non-empty ``question`` or ``student_response`` (at least one).
- Use as many units as the assignment visibly contains; merge tiny boilerplate into adjacent units.
- **Short coding cells count:** if the only student work for a problem is a single line (e.g. ``import csv``), that is still a **complete** unit—copy it verbatim and set ``answer_key_segment`` to the matching key line(s), even when the rest of the notebook is long.
- When the submission begins with a ``[STUDENT_SUBMISSION_PART …]`` header, you are seeing **one slice** of a longer file. Extract every gradable unit that appears **fully inside this slice**; still use the full ``ANSWER_KEY_OR_SAMPLE`` block below to choose ``answer_key_segment`` for each unit."""


def _submission_window_slices(text: str, window: int, overlap: int) -> list[tuple[int, int, str]]:
    """Return ``(start, end_exclusive, slice)`` covering ``text`` with optional overlap."""
    n = len(text)
    window = max(2_000, int(window))
    overlap = max(0, min(int(overlap), window // 2))
    if n <= window:
        return [(0, n, text)]
    step = window - overlap
    out: list[tuple[int, int, str]] = []
    start = 0
    while start < n:
        end = min(n, start + window)
        out.append((start, end, text[start:end]))
        if end >= n:
            break
        start += step
    return out


def _merge_two_raw_units(a: dict, b: dict) -> dict:
    out = dict(a)
    for k in ("question", "student_response", "answer_key_segment", "extracted_text"):
        va = str(out.get(k) or "").strip()
        vb = str(b.get(k) or "").strip()
        if len(vb) > len(va):
            out[k] = b.get(k, "")
        elif not va and vb:
            out[k] = b.get(k, "")
    ida = str(out.get("question_id") or "").strip()
    idb = str(b.get("question_id") or "").strip()
    if len(idb) > len(ida) and idb:
        out["question_id"] = idb
    elif not ida and idb:
        out["question_id"] = idb
    return out


def _units_should_merge(a: dict, b: dict) -> bool:
    """Heuristic dedupe for overlapping windows (same cell extracted twice)."""
    qa = str(a.get("question_id") or "").strip().lower()
    qb = str(b.get("question_id") or "").strip().lower()
    if qa and qb and qa != qb:
        return False
    sa = " ".join(str(a.get("student_response") or "").split())
    sb = " ".join(str(b.get("student_response") or "").split())
    if sa and sb:
        if sa == sb:
            return True
        if len(sa) >= len(sb) and sb and sb in sa:
            return True
        if len(sb) >= len(sa) and sa and sa in sb:
            return True
    if qa == qb and qa and (not sa and not sb):
        ta = " ".join(str(a.get("question") or "").split())
        tb = " ".join(str(b.get("question") or "").split())
        if ta and tb and (ta == tb or ta in tb or tb in ta):
            return True
    return False


def _dedupe_merge_trio_units(units: list[dict]) -> list[dict]:
    pool = [u for u in units if isinstance(u, dict)]
    out: list[dict] = []
    for u in pool:
        merged_into = False
        for i, existing in enumerate(out):
            if _units_should_merge(existing, u):
                out[i] = _merge_two_raw_units(existing, u)
                merged_into = True
                break
        if not merged_into:
            out.append(dict(u))
    return out


def _chat_trio_units(
    client: OpenAIJsonClient, user_body: str
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    messages = [
        {"role": "system", "content": _TRIO_RAG_SYSTEM},
        {"role": "user", "content": user_body},
    ]
    parsed, usage = client.chat_json_with_usage(
        messages, temperature=0.1, response_format={"type": "json_object"}
    )
    raw = parsed.get("units") if isinstance(parsed, dict) else None
    raw_list = raw if isinstance(raw, list) else []
    clean = [u for u in raw_list if isinstance(u, dict)]
    return clean, usage


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def multimodal_openai_trio_rag_frontload_enabled(cfg: Config) -> bool:
    """True unless explicitly disabled; requires ``OPENAI_API_KEY``."""
    if not (cfg.OPENAI_API_KEY or "").strip():
        return False
    raw = str(getattr(cfg, "MULTIMODAL_OPENAI_TRIO_RAG_FRONTLOAD", "") or "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    # ``auto`` or unset: use OpenAI for chunk+trio+RAG when a key is configured
    return raw in ("", "auto")


def _chars_to_token_estimate(chars: int) -> int:
    """Rough token count when API usage is unavailable (≈4 chars/token for English)."""
    return max(0, int(chars / 4))


def estimate_openai_trio_rag_cost_usd(
    cfg: Config,
    *,
    chat_prompt_tokens: int,
    chat_completion_tokens: int,
    embedding_tokens: int,
) -> dict[str, float]:
    pin = float(getattr(cfg, "OPENAI_TRIO_RAG_CHAT_INPUT_USD_PER_MTOK", 0.20))
    pout = float(getattr(cfg, "OPENAI_TRIO_RAG_CHAT_OUTPUT_USD_PER_MTOK", 1.25))
    pemb = float(getattr(cfg, "OPENAI_TRIO_RAG_EMBED_USD_PER_MTOK", 0.02))
    chat_usd = (chat_prompt_tokens / 1_000_000.0) * pin + (
        chat_completion_tokens / 1_000_000.0
    ) * pout
    emb_usd = (embedding_tokens / 1_000_000.0) * pemb
    return {
        "chat_input_usd": round((chat_prompt_tokens / 1_000_000.0) * pin, 8),
        "chat_output_usd": round((chat_completion_tokens / 1_000_000.0) * pout, 8),
        "embedding_usd": round(emb_usd, 8),
        "total_usd": round(chat_usd + emb_usd, 8),
    }


def _ipynb_bytes_to_plain(nb: bytes) -> str:
    try:
        obj = json.loads(nb.decode("utf-8", errors="replace"))
    except Exception:
        return ""
    parts: list[str] = []
    for cell in obj.get("cells") or []:
        src = cell.get("source")
        if isinstance(src, list):
            src = "".join(str(x) for x in src)
        elif not isinstance(src, str):
            src = str(src or "")
        parts.append(src)
    return "\n\n".join(parts).strip()


def _submission_plain_for_frontload(envelope: IngestionEnvelope, max_chars: int) -> str:
    plain = (envelope.extracted_plaintext or "").strip()
    if not plain:
        raw = (envelope.artifacts or {}).get("ipynb")
        if isinstance(raw, (bytes, bytearray)):
            plain = _ipynb_bytes_to_plain(bytes(raw))
    if not plain:
        return ""
    flowed = reflow_pdf_sections_in_plaintext(plain)
    if max_chars > 0 and len(flowed) > max_chars:
        return flowed[:max_chars]
    return flowed


def _safe_qid(raw: str, idx: int) -> str:
    s = re.sub(r"\s+", " ", (raw or "").strip())
    return s[:120] if s else f"unit_{idx + 1}"


def _openai_embed_batch(
    texts: list[str],
    *,
    api_key: str,
    model: str,
    max_chars_per_input: int = 8000,
) -> tuple[list[list[float]], int]:
    """Return one vector per input slot (empty list for empty strings) + total_tokens."""
    from openai import OpenAI

    n = len(texts)
    cleaned: list[str] = []
    for t in texts:
        s = (t or "").strip()
        cleaned.append(s[:max_chars_per_input] if s else "")

    need_idx = [i for i in range(n) if cleaned[i]]
    if not need_idx:
        return [[] for _ in range(n)], 0

    client = OpenAI(api_key=api_key)
    out: list[list[float]] = [[] for _ in range(n)]
    total_tok = 0
    batch_size = 256
    for start in range(0, len(need_idx), batch_size):
        batch_idx = need_idx[start : start + batch_size]
        batch_inputs = [cleaned[i] for i in batch_idx]
        resp = client.embeddings.create(model=model, input=batch_inputs)
        if getattr(resp, "usage", None) and getattr(resp.usage, "total_tokens", None):
            total_tok += int(resp.usage.total_tokens or 0)
        by_pos: dict[int, list[float]] = {}
        for d in resp.data:
            by_pos[int(d.index)] = list(d.embedding)
        for pos, i in enumerate(batch_idx):
            emb = by_pos.get(pos)
            if emb is None and pos < len(resp.data):
                emb = list(resp.data[pos].embedding)
            out[i] = list(emb or [])
    return out, total_tok


def run_openai_trio_rag_frontload(
    envelope: IngestionEnvelope,
    cfg: Config,
    answer_key_text: str,
) -> tuple[list[GradingChunk], dict[str, Any]]:
    """
    Build ``GradingChunk`` list with ``evidence['trio']``, ``trio_segment_rag``,
    and ``rag_embedding_bundle`` populated via OpenAI only.

    On failure returns ``([], {"ok": False, "error": ...})``.
    """
    audit: dict[str, Any] = {"ok": False}
    key = (cfg.OPENAI_API_KEY or "").strip()
    if not key:
        audit["error"] = "missing_OPENAI_API_KEY"
        return [], audit

    chat_model = (
        getattr(cfg, "OPENAI_TRIO_RAG_CHAT_MODEL", None) or ""
    ).strip() or "gpt-5.4-nano"
    embed_model = (
        getattr(cfg, "OPENAI_TRIO_RAG_EMBEDDING_MODEL", None) or ""
    ).strip() or "text-embedding-3-small"
    max_in = int(getattr(cfg, "MULTIMODAL_OPENAI_TRIO_INPUT_MAX_CHARS", 120_000) or 120_000)
    max_in = max(8_000, min(max_in, 500_000))

    submission = _submission_plain_for_frontload(envelope, max_in)
    if not submission.strip():
        audit["error"] = "empty_submission_plaintext"
        return [], audit

    ak = (answer_key_text or "").strip()
    ak_max = int(getattr(cfg, "MULTIMODAL_OPENAI_TRIO_ANSWER_KEY_MAX_CHARS", 32_000) or 32_000)
    ak_max = max(2_000, min(ak_max, max(8_000, max_in // 2)))
    ak_use = ak[:ak_max] if ak else ""

    win = int(getattr(cfg, "MULTIMODAL_OPENAI_TRIO_WINDOW_CHARS", 48_000) or 48_000)
    ovl = int(getattr(cfg, "MULTIMODAL_OPENAI_TRIO_WINDOW_OVERLAP_CHARS", 4_096) or 4_096)
    win = max(4_000, min(win, max_in))

    slices = _submission_window_slices(submission, win, ovl)
    audit["trio_submission_chars"] = len(submission)
    audit["trio_window_count"] = len(slices)
    audit["trio_answer_key_chars_in_prompt"] = len(ak_use)

    client = OpenAIJsonClient(key, chat_model)
    all_raw: list[dict[str, Any]] = []
    usage_chat: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    pre_chat_in = 0
    for wi, (start, end, slc) in enumerate(slices):
        prefix = ""
        if len(slices) > 1:
            prefix = (
                f"[STUDENT_SUBMISSION_PART {wi + 1}/{len(slices)}; "
                f"character_range={start}:{end}. "
                "Extract units that appear fully inside this slice; use ANSWER_KEY_OR_SAMPLE "
                "for answer_key_segment.]\n\n"
            )
        user_body = (
            "### STUDENT_SUBMISSION\n\n"
            + prefix
            + slc
            + "\n\n### ANSWER_KEY_OR_SAMPLE\n\n"
            + (ak_use or "(none provided; use empty answer_key_segment where unknown)")
        )
        pre_chat_in += _chars_to_token_estimate(len(_TRIO_RAG_SYSTEM) + len(user_body))
        try:
            part_units, u_chat = _chat_trio_units(client, user_body)
        except Exception as exc:
            _log.warning(
                "OpenAI trio frontload chat failed (window %s): %s",
                wi + 1,
                exc,
                exc_info=True,
            )
            audit["error"] = f"chat_failed:{exc!s}"
            audit["window_failed"] = wi + 1
            return [], audit
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            usage_chat[k] += int((u_chat or {}).get(k) or 0)
        all_raw.extend(part_units)

    units = _dedupe_merge_trio_units(all_raw)
    audit["trio_raw_units_before_dedupe"] = len(all_raw)
    audit["trio_units_after_dedupe"] = len(units)

    if not units:
        audit["error"] = "invalid_or_empty_units"
        return [], audit

    pre_chat_out = min(32_000, _chars_to_token_estimate(len(submission) // 2))

    hints = envelope.modality_hints or {}
    modality: Modality = modality_from_hints(hints)
    task_type: TaskType = task_type_from_hints(hints)

    chunks: list[GradingChunk] = []
    for i, u in enumerate(units):
        if not isinstance(u, dict):
            continue
        q = str(u.get("question") or "").strip()
        sr = str(u.get("student_response") or "").strip()
        ak_seg = str(u.get("answer_key_segment") or "").strip()
        ext = str(u.get("extracted_text") or "").strip()
        if not ext:
            parts = [p for p in (q, sr) if p]
            ext = "\n\n".join(parts).strip()
        if not q and not sr:
            continue
        qid = _safe_qid(str(u.get("question_id") or ""), i)
        cid = f"{envelope.assignment_id}:{envelope.student_id}:openai_trio:{i}:{qid}"
        ch = GradingChunk(
            chunk_id=cid,
            assignment_id=envelope.assignment_id,
            student_id=envelope.student_id,
            question_id=qid,
            modality=modality,
            task_type=task_type,
            extracted_text=ext or (q + "\n\n" + sr).strip(),
            evidence={
                "trio": {
                    "question": q,
                    "student_response": sr,
                    "answer_key_segment": ak_seg,
                    "instructor_context": "",
                },
                "question_text": q[:4000],
                "_openai_trio_rag_frontload": True,
            },
        )
        chunks.append(ch)

    if not chunks:
        audit["error"] = "no_valid_units_after_parse"
        return [], audit

    # --- Batch embeddings: per chunk, order [q, sr, ak, canonical] ---
    embed_inputs: list[str] = []

    canon_cap = int(os.getenv("MULTIMODAL_TRIO_CANONICAL_EMBED_MAX_CHARS", "24000") or 24000)

    for ch in chunks:
        trio = (ch.evidence or {}).get("trio") or {}
        if not isinstance(trio, dict):
            trio = {}
        for key in ("question", "student_response", "answer_key_segment"):
            embed_inputs.append(str(trio.get(key) or "").strip())
        tq = str(trio.get("question") or "").strip()
        tsr = str(trio.get("student_response") or "").strip()
        tak = str(trio.get("answer_key_segment") or "").strip()
        canon_parts: list[str] = []
        if tq:
            canon_parts.append(f"[QUESTION]\n{tq}")
        if tsr:
            canon_parts.append(f"[STUDENT]\n{tsr}")
        if tak:
            canon_parts.append(f"[REFERENCE]\n{tak}")
        canon = "\n\n".join(canon_parts).strip() or (ch.extracted_text or "").strip() or " "
        embed_inputs.append(canon[:canon_cap])

    nonempty_texts = [t for t in embed_inputs if t.strip()]
    pre_embed_tok = sum(_chars_to_token_estimate(len(t)) for t in nonempty_texts)

    try:
        vecs, embed_usage_tokens = _openai_embed_batch(
            embed_inputs,
            api_key=key,
            model=embed_model,
        )
    except Exception as exc:
        _log.warning("OpenAI trio frontload embeddings failed: %s", exc, exc_info=True)
        audit["error"] = f"embed_failed:{exc!s}"
        return [], audit

    if embed_usage_tokens <= 0:
        embed_usage_tokens = pre_embed_tok

    stride = 4
    for ci, ch in enumerate(chunks):
        trio = dict((ch.evidence or {}).get("trio") or {})
        base = ci * stride
        seg_rag: dict[str, Any] = {}
        for ki, key in enumerate(("question", "student_response", "answer_key_segment")):
            blob = str(trio.get(key) or "").strip()
            idx = base + ki
            if not blob:
                seg_rag[key] = {
                    "embedding_dimension": 0,
                    "embedding_source": "empty_segment_skipped",
                    "embedding": [],
                }
                continue
            vec = list(vecs[idx]) if idx < len(vecs) else []
            seg_rag[key] = {
                "embedding_dimension": len(vec),
                "embedding_source": f"openai:{embed_model}",
                "embedding": vec,
            }
        c_idx = base + 3
        vec2 = list(vecs[c_idx]) if c_idx < len(vecs) else []
        ev = dict(ch.evidence or {})
        ev["trio_segment_rag"] = seg_rag
        ev["rag_embedding_bundle"] = {
            "embedding_dimension": len(vec2),
            "embedding_source": f"trio_canonical:openai:{embed_model}",
            "embedding": vec2,
        }
        ch.evidence = ev

    pt = int(usage_chat.get("prompt_tokens") or 0)
    ct = int(usage_chat.get("completion_tokens") or 0)
    if pt + ct <= 0:
        pt = pre_chat_in
        try:
            ct = max(
                pre_chat_out,
                _chars_to_token_estimate(len(json.dumps({"units": units}))),
            )
        except Exception:
            ct = pre_chat_out

    cost = estimate_openai_trio_rag_cost_usd(
        cfg,
        chat_prompt_tokens=pt,
        chat_completion_tokens=ct,
        embedding_tokens=embed_usage_tokens,
    )
    pre_cost = estimate_openai_trio_rag_cost_usd(
        cfg,
        chat_prompt_tokens=pre_chat_in,
        chat_completion_tokens=pre_chat_out,
        embedding_tokens=pre_embed_tok,
    )

    n_ch = len(chunks)
    tot_usd = float(cost.get("total_usd") or 0.0)
    audit.update(
        {
            "ok": True,
            "chat_model": chat_model,
            "embedding_model": embed_model,
            "n_chunks": n_ch,
            "chat_usage_tokens": {"prompt": pt, "completion": ct, "total": pt + ct},
            "embedding_usage_tokens_est": embed_usage_tokens,
            "cost_usd": cost,
            "per_chunk_avg_cost_usd": round(tot_usd / n_ch, 10) if n_ch else 0.0,
            "per_chunk_avg_tokens_est": {
                "chat_prompt": round(pt / n_ch, 2) if n_ch else 0.0,
                "chat_completion": round(ct / n_ch, 2) if n_ch else 0.0,
                "embedding": round(embed_usage_tokens / n_ch, 2) if n_ch else 0.0,
            },
            "pre_call_cost_estimate_usd": pre_cost,
            "pre_call_token_estimates": {
                "chat_input": pre_chat_in,
                "chat_output_guess": pre_chat_out,
                "embedding": pre_embed_tok,
            },
            "pricing_note": "Chat USD/MTok from OPENAI_TRIO_RAG_CHAT_*; embedding from OPENAI_TRIO_RAG_EMBED_USD_PER_MTOK (see https://developers.openai.com/api/docs/models ).",
        }
    )
    _log.info(
        "OpenAI trio+RAG frontload: chunks=%s chat_tokens=%s+%s embed_tokens≈%s cost_usd≈%s",
        len(chunks),
        pt,
        ct,
        embed_usage_tokens,
        cost.get("total_usd"),
    )
    return chunks, audit
