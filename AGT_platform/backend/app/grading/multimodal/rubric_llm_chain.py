"""
Three-step rubric flow (separate LLM calls): (1) pick one generic type for the assignment,
(2) pick applicable criteria for each chunk, (3) scoring happens in the normal grader call.

Steps 1–2 use the same **structure** LLM stack as trio refinement (Claude if configured, else OpenAI).
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from app.config import Config

from .ingestion import IngestionEnvelope
from .rag_embeddings import _multimodal_structure_chat_client
from .schemas import GradingChunk, RubricType

_log = logging.getLogger(__name__)

_FOUR_TYPES: tuple[RubricType, ...] = (
    RubricType.PROGRAMMING_SCAFFOLDED,
    RubricType.EDA_VISUALIZATION,
    RubricType.FREE_RESPONSE,
    RubricType.ORAL_INTERVIEW,
)

_STEP1_SYSTEM = """You select **exactly ONE** grading-rubric template for an **entire** assignment.
Every question chunk will be graded with criteria drawn only from this template.

Chain-of-thought: think step-by-step in `cot_step1` (brief), then commit to one template.

Return **only** valid JSON (no markdown fences):
{
  "generic_rubric_type": "<one string from allowed list below>",
  "cot_step1": "<string — numbered reasoning is fine>",
  "anchor_scores": {
    "programming_scaffolded": <float 0.0-1.0>,
    "eda_visualization": <float 0.0-1.0>,
    "free_response": <float 0.0-1.0>,
    "oral_interview": <float 0.0-1.0>
  }
}

**Allowed** `generic_rubric_type` values (exact spelling):
- programming_scaffolded
- eda_visualization
- free_response
- oral_interview

**Hard rules**
1. `anchor_scores` MUST contain **all four** keys above (no extras, no omissions).
2. The key with the **largest** value in `anchor_scores` MUST equal `generic_rubric_type` (ties: prefer the type you name in `generic_rubric_type` and break ties by raising that key by +0.01 in your head — do not output ties).
3. Values are **confidence** that the template fits the assignment (not cosine similarities)."""


_STEP2_SYSTEM = """You select which **rubric criteria** from the instructor list apply to **this one** question chunk.
The assignment-level template is already fixed; only pick criteria relevant to what the student actually did in this chunk.

Chain-of-thought: brief reasoning in `cot_step2`, then output JSON only:
{
  "criterion_names": ["<exact string copied from the provided list>", ...],
  "cot_step2": "<string>"
}

Rules:
- `criterion_names` must be a **non-empty** subset of the provided list (copy names **exactly**).
- Prefer a tight set (omit criteria clearly not evidenced in this chunk)."""


def rubric_llm_chain_enabled(cfg: Config, hints: dict[str, Any]) -> bool:
    raw = str(os.getenv("MULTIMODAL_RUBRIC_LLM_CHAIN", "auto")).strip().lower()
    if hints.get("multimodal_rubric_llm_chain") is not None:
        v = str(hints.get("multimodal_rubric_llm_chain")).strip().lower()
        if v in ("0", "false", "off", "no"):
            return False
        if v in ("1", "true", "on", "yes"):
            return True
    if raw in ("0", "false", "off", "no"):
        return False
    if raw in ("1", "true", "on", "yes"):
        return True
    # auto: enable when a structure client exists
    client, _ = _multimodal_structure_chat_client(cfg, purpose="rubric_chain")
    return client is not None


def _structure_chat(cfg: Config) -> tuple[Any, str] | None:
    picked = _multimodal_structure_chat_client(cfg, purpose="rubric_chain")
    client, label = picked
    if client is None:
        return None
    return client, label


def _assignment_digest(envelope: IngestionEnvelope, chunks: list[GradingChunk], cap: int = 12000) -> str:
    arts = sorted((envelope.artifacts or {}).keys())
    parts = [
        f"assignment_id={envelope.assignment_id!r}",
        f"artifact_keys={arts}",
    ]
    blob = "\n\n---\n\n".join((c.extracted_text or "").strip() for c in chunks[:12])
    if len(blob) > cap:
        blob = blob[:cap] + "\n…[truncated]"
    parts.append("EXCERPT (first chunks):\n" + blob)
    return "\n".join(parts)


def llm_select_generic_rubric_type(
    envelope: IngestionEnvelope,
    chunks: list[GradingChunk],
    cfg: Config,
) -> tuple[RubricType | None, str, dict[str, float], str]:
    """Call #1: one generic type + anchor_scores (four keys only). Returns (type, reason, scores, cot)."""
    picked = _structure_chat(cfg)
    if picked is None:
        return None, "", {}, ""
    client, model_label = picked
    user = _assignment_digest(envelope, chunks)
    try:
        obj = client.chat_json(
            [
                {"role": "system", "content": _STEP1_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
    except Exception:
        _log.warning("rubric chain step1 failed (%s)", model_label, exc_info=True)
        return None, "", {}, ""
    rt_s = str(obj.get("generic_rubric_type") or "").strip()
    scores_raw = obj.get("anchor_scores") if isinstance(obj.get("anchor_scores"), dict) else {}
    scores: dict[str, float] = {}
    for t in _FOUR_TYPES:
        try:
            scores[t.value] = float(scores_raw.get(t.value, 0.0))
        except (TypeError, ValueError):
            scores[t.value] = 0.0
    cot = str(obj.get("cot_step1") or "").strip()
    rt: RubricType | None = None
    for r in _FOUR_TYPES:
        if r.value == rt_s:
            rt = r
            break
    if rt is None:
        return None, "", scores, cot
    # Enforce winner == selected type (model contract)
    win = max(_FOUR_TYPES, key=lambda k: scores.get(k.value, 0.0))
    if win != rt:
        _log.warning(
            "rubric chain step1: winner %s != declared %s; coercing to winner",
            win.value,
            rt.value,
        )
        rt = win
    # Ensure persisted scores are consistent with the chosen type (custom rubric validator).
    hi = max(scores.get(t.value, 0.0) for t in _FOUR_TYPES)
    scores[rt.value] = float(max(scores.get(rt.value, 0.0), hi + 1e-6))
    return rt, f"llm_chain_step1:{model_label}", scores, cot


def llm_select_criteria_for_chunk(
    chunk: GradingChunk,
    rt: RubricType,
    template_rows: list[dict[str, Any]],
    cfg: Config,
) -> tuple[list[str], str]:
    """Call #2: subset of criterion **names** (exact strings from template rows)."""
    picked = _structure_chat(cfg)
    if picked is None:
        return [], ""
    client, _model_label = picked
    allowed = [str(r.get("name") or "").strip() for r in template_rows if str(r.get("name") or "").strip()]
    if not allowed:
        return [], ""
    trio = (chunk.evidence or {}).get("trio")
    trio_s = json.dumps(trio, ensure_ascii=True)[:8000] if isinstance(trio, dict) else ""
    user = json.dumps(
        {
            "generic_rubric_type": rt.value,
            "allowed_criterion_names": allowed,
            "chunk_id": chunk.chunk_id,
            "question_id": chunk.question_id,
            "extracted_text": (chunk.extracted_text or "")[:14000],
            "trio": trio_s,
        },
        ensure_ascii=True,
        indent=2,
    )
    try:
        obj = client.chat_json(
            [
                {"role": "system", "content": _STEP2_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
    except Exception:
        _log.warning("rubric chain step2 failed chunk=%s", chunk.chunk_id, exc_info=True)
        return [], ""
    names = obj.get("criterion_names")
    cot = str(obj.get("cot_step2") or "").strip()
    if not isinstance(names, list):
        return [], cot
    allowed_set = set(allowed)
    picked_names = [str(x).strip() for x in names if str(x).strip() in allowed_set]
    if not picked_names:
        return [allowed[0]], cot
    return picked_names, cot


def build_plan_with_llm_chain(
    assignment_id: str,
    envelope: IngestionEnvelope,
    chunks: list[GradingChunk],
    cfg: Config,
    rubric_rows_by_type: dict[RubricType, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    """Build schema v2 plan using LLM steps 1–2 (step 3 is the normal grader)."""
    rt, reason, scores, cot1 = llm_select_generic_rubric_type(envelope, chunks, cfg)
    if rt is None:
        return None
    template = list(rubric_rows_by_type.get(rt) or [])
    if not template:
        return None
    per_q: list[dict[str, Any]] = []
    max_llm = int(os.getenv("MULTIMODAL_RUBRIC_LLM_CHAIN_MAX_CHUNKS", "48") or 48)
    max_llm = max(1, min(max_llm, 200))
    for ch in chunks[:max_llm]:
        names, cot2 = llm_select_criteria_for_chunk(ch, rt, template, cfg)
        per_q.append(
            {
                "assignment_id": assignment_id,
                "question_id": ch.question_id,
                "chunk_id": ch.chunk_id,
                "criterion_names": names,
                "cot_step2": cot2,
            }
        )
    # Any chunks beyond cap: deterministic full template (auditable)
    for ch in chunks[max_llm:]:
        per_q.append(
            {
                "assignment_id": assignment_id,
                "question_id": ch.question_id,
                "chunk_id": ch.chunk_id,
                "criterion_names": [str(r.get("name") or "").strip() for r in template if r.get("name")],
                "cot_step2": "skipped_llm_chain_cap",
            }
        )
    return {
        "schema_version": 2,
        "assignment_id": assignment_id,
        "generic_rubric_type": rt.value,
        "selection_reason": reason,
        "anchor_scores": scores,
        "cot_step1": cot1,
        "question_rubrics": per_q,
    }
