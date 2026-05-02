"""
Claude-only **structured** assignment chunking: one Messages call returns JSON ``units``
that map directly to :class:`~app.grading.multimodal.schemas.GradingChunk` with
``evidence["trio"]`` in the same shape as OpenAI trio frontload / downstream graders.

This path does **not** use notebook cell-order chunking, QA-segment prompts, or heuristic
pairing helpers — only the raw submission view (notebook JSON or concatenated artifact text)
plus optional reference material from ``modality_hints``.

Enable with ``MULTIMODAL_CLAUDE_STRUCTURED_CHUNKING=auto`` (try when ``ANTHROPIC_API_KEY``
is set) or ``=on`` (same, but on failure fall back only to :func:`chunker.default_chunker_build_units`,
skipping triplet / notebook / legacy QA segmentation).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.config import Config
from app.grading.artifact_plaintext import (
    artifacts_to_concatenated_plain,
    infer_modality_from_artifact_keys,
)
from app.grading.llm_router import AnthropicJsonClient

from .chunker import modality_from_hints, task_type_from_hints
from .ingestion import IngestionEnvelope
from .schemas import GradingChunk, Modality, TaskType

_log = logging.getLogger(__name__)

# Prompt contract: model must return parseable JSON only (see AnthropicJsonClient.chat_json).
_CLAUDE_TRIO_CHUNK_SYSTEM = """You are splitting a student assignment into gradable units for an autograder.

Return **only** a single JSON object (no markdown fences, no commentary) with this exact shape:
{
  "units": [
    {
      "question_id": "string — stable id, e.g. 1.1 or pair_0",
      "question": "string — prompt text shown to the student for this part",
      "student_response": "string — only the student's answer/work for this part",
      "answer_key_segment": "string — matching excerpt from ANSWER_KEY_OR_REFERENCE, or empty if unknown",
      "instructor_context": "string — brief setup/readme/scaffolding that applies to this unit, or empty",
      "extracted_text": "optional string — if omitted, the pipeline joins question + student_response"
    }
  ]
}

Rules:
- Emit one unit per distinct gradable question or subpart; preserve numbering from the assignment when possible.
- Never put instructor/readme boilerplate in student_response; use instructor_context for that.
- If the submission is a Jupyter notebook JSON, parse cells and align questions with the following code/markdown.
- Use empty strings for unknown fields, not null.
"""


def claude_structured_chunking_should_attempt(cfg: Any) -> bool:
    """True when structured Claude chunking may run (requires an Anthropic API key)."""
    key = (getattr(cfg, "ANTHROPIC_API_KEY", "") or "").strip()
    if not key:
        return False
    mode = str(
        getattr(cfg, "MULTIMODAL_CLAUDE_STRUCTURED_CHUNKING", "off") or "off"
    ).strip().lower()
    if mode in ("off", "false", "0", "no", ""):
        return False
    return mode in ("on", "auto", "true", "1", "yes")


def claude_structured_chunking_forced_on(cfg: Any) -> bool:
    """When True, legacy notebook / triplet / QA-segment chunkers are skipped after Claude fails."""
    mode = str(
        getattr(cfg, "MULTIMODAL_CLAUDE_STRUCTURED_CHUNKING", "off") or "off"
    ).strip().lower()
    return mode in ("on", "true", "1", "yes")


def _anthropic_chunking_client(cfg: Config) -> tuple[AnthropicJsonClient, str] | None:
    """Anthropic client for assignment chunking only (independent of MULTIMODAL_ANTHROPIC_ASSIGNMENT_PARSING)."""
    key = (getattr(cfg, "ANTHROPIC_API_KEY", "") or "").strip()
    if not key:
        return None
    model = (
        (getattr(cfg, "MULTIMODAL_CLAUDE_CHUNKING_MODEL", "") or "").strip()
        or (getattr(cfg, "MULTIMODAL_ANTHROPIC_PARSING_MODEL", "") or "").strip()
        or "claude-opus-4-7"
    )
    try:
        mt = int(getattr(cfg, "MULTIMODAL_CLAUDE_CHUNKING_MAX_TOKENS", 0) or 0)
    except (TypeError, ValueError):
        mt = 0
    if mt <= 0:
        try:
            mt = int(
                getattr(cfg, "MULTIMODAL_ANTHROPIC_PARSING_MAX_TOKENS", 16384) or 16384
            )
        except (TypeError, ValueError):
            mt = 16384
    return AnthropicJsonClient(key, model, max_tokens=mt), f"anthropic:{model}"


def _max_student_chars(cfg: Any) -> int:
    try:
        n = int(getattr(cfg, "MULTIMODAL_CLAUDE_CHUNKING_MAX_STUDENT_CHARS", 120_000) or 120_000)
    except (TypeError, ValueError):
        n = 120_000
    return max(8_000, min(n, 500_000))


def _max_ref_chars(cfg: Any) -> int:
    try:
        n = int(getattr(cfg, "MULTIMODAL_CLAUDE_CHUNKING_MAX_REF_CHARS", 48_000) or 48_000)
    except (TypeError, ValueError):
        n = 48_000
    return max(1_000, min(n, 200_000))


def _submission_payload_for_claude(
    envelope: IngestionEnvelope, max_chars: int
) -> tuple[str, bool]:
    """
    Build the student submission string for the chunking prompt.

    Returns (text, is_raw_notebook_json). For a single ``ipynb`` artifact, the raw notebook
    JSON is sent so Claude performs notebook parsing (no local cell-order chunker).
    """
    arts = envelope.artifacts or {}
    bmap: dict[str, bytes] = {}
    for k, v in arts.items():
        if isinstance(v, (bytes, bytearray)) and v:
            bmap[str(k).lower()] = bytes(v)
    keys = set(bmap.keys())
    if keys == {"ipynb"}:
        raw = bmap["ipynb"].decode("utf-8", errors="replace")
        if len(raw) > max_chars:
            raw = raw[:max_chars] + "\n…[truncated]"
        return raw, True
    plain = artifacts_to_concatenated_plain(bmap).strip()
    if not plain:
        plain = (envelope.extracted_plaintext or "").strip()
    if len(plain) > max_chars:
        plain = plain[:max_chars] + "\n\n[TRUNCATED]"
    return plain, False


def _blank_template_snippet(hints: dict[str, Any], max_chars: int) -> str:
    raw_tpl = hints.get("blank_assignment_template_bytes")
    raw_nb = hints.get("blank_assignment_ipynb_bytes")
    blob = b""
    if isinstance(raw_tpl, (bytes, bytearray)) and raw_tpl:
        blob = bytes(raw_tpl)
    elif isinstance(raw_nb, (bytes, bytearray)) and raw_nb:
        blob = bytes(raw_nb)
    if not blob.strip():
        return ""
    try:
        s = blob.decode("utf-8", errors="replace").strip()
    except Exception:
        return ""
    if len(s) > max_chars:
        s = s[:max_chars] + "\n…[truncated]"
    return s


def _question_id_for_unit(raw: str, index: int) -> str:
    s = re.sub(r"\s+", " ", (raw or "").strip())
    return s[:120] if s else f"pair_{index}"


def _units_to_chunks(
    envelope: IngestionEnvelope,
    units: list[Any],
    *,
    modality: Modality,
    task_type: TaskType,
    model_label: str,
    max_units: int | None,
) -> list[GradingChunk]:
    out: list[GradingChunk] = []
    for i, u in enumerate(units):
        if not isinstance(u, dict):
            continue
        q = str(u.get("question") or "").strip()
        sr = str(u.get("student_response") or "").strip()
        ak_seg = str(u.get("answer_key_segment") or "").strip()
        ic = str(u.get("instructor_context") or "").strip()
        ext = str(u.get("extracted_text") or "").strip()
        if not ext:
            parts = [p for p in (q, sr) if p]
            ext = "\n\n".join(parts).strip()
        if not q and not sr:
            continue
        qid = _question_id_for_unit(str(u.get("question_id") or ""), i)
        cid = f"{envelope.assignment_id}:{envelope.student_id}:claude_structured:{i}:{qid}"
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
                    "instructor_context": ic,
                },
                "question_text": q[:4000],
                "chunker": "claude_structured_assignment",
                "_claude_structured_units": True,
                "claude_structured_chunking_model": model_label,
            },
        )
        out.append(ch)
        if max_units is not None and max_units > 0 and len(out) >= max_units:
            break
    return out


def try_build_claude_structured_assignment_chunks(
    envelope: IngestionEnvelope,
    cfg: Config,
    *,
    max_units: int | None = None,
) -> list[GradingChunk] | None:
    """
    Call Claude once; parse ``{"units": [...]}`` into :class:`GradingChunk` list.

    Returns ``None`` when disabled, misconfigured, the model errors, or ``units`` is empty.
    """
    if not claude_structured_chunking_should_attempt(cfg):
        return None
    pair = _anthropic_chunking_client(cfg)
    if pair is None:
        return None
    client, model_label = pair

    hints = envelope.modality_hints or {}
    cap_s = _max_student_chars(cfg)
    cap_r = _max_ref_chars(cfg)
    submission, _is_nb_json = _submission_payload_for_claude(envelope, cap_s)
    if not submission.strip():
        _log.info("claude_structured_chunking: empty submission view; skipping")
        return None

    ak = str(hints.get("answer_key_plaintext") or "").strip()
    if len(ak) > cap_r:
        ak = ak[:cap_r] + "\n…[truncated]"

    blank = _blank_template_snippet(hints, cap_r)
    parts = [
        "### STUDENT_SUBMISSION",
        submission,
        "### ANSWER_KEY_OR_REFERENCE",
        ak or "(none provided; use empty answer_key_segment where unknown)",
    ]
    if blank:
        parts.insert(0, "### BLANK_INSTRUCTOR_TEMPLATE (reference)")
        parts.insert(1, blank)
    user_body = "\n\n".join(parts)

    try:
        obj = client.chat_json(
            [
                {"role": "system", "content": _CLAUDE_TRIO_CHUNK_SYSTEM},
                {"role": "user", "content": user_body},
            ],
            temperature=0.1,
        )
    except Exception:
        _log.warning(
            "claude_structured_chunking: Claude request failed model=%s",
            model_label,
            exc_info=True,
        )
        return None

    if not isinstance(obj, dict):
        return None
    raw_units = obj.get("units")
    if not isinstance(raw_units, list) or not raw_units:
        _log.info("claude_structured_chunking: no units in model JSON")
        return None

    hints_mod = dict(hints)
    if not str(hints_mod.get("modality") or "").strip():
        arts = envelope.artifacts or {}
        bonly = {
            str(k).lower(): bytes(v)
            for k, v in arts.items()
            if isinstance(v, (bytes, bytearray)) and v
        }
        hints_mod["modality"] = infer_modality_from_artifact_keys(bonly)

    modality = modality_from_hints(hints_mod)
    task_type = task_type_from_hints(hints_mod)

    chunks = _units_to_chunks(
        envelope,
        raw_units,
        modality=modality,
        task_type=task_type,
        model_label=model_label,
        max_units=max_units,
    )
    if not chunks:
        return None
    return chunks
