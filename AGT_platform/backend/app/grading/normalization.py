"""Stage 1: submission normalization — deterministic context for downstream stages."""

from __future__ import annotations

import json
from typing import Any

from app.grading import tools as grading_tools

NORMALIZATION_VERSION = "1.0"


def normalize_submission(
    ctx: dict[str, Any],
    *,
    assignment_instruction: str,
    rubric_items: list[dict[str, Any]],
    modalities: list[str],
    artifacts: dict[str, bytes],
    assignment_kind: str | None = None,
) -> None:
    """
    Populate ctx["normalized"] with modality-keyed blobs (JSON-serializable).
    """
    out: dict[str, Any] = {
        "normalization_version": NORMALIZATION_VERSION,
        "assignment_instruction": assignment_instruction or "",
        "assignment_kind": assignment_kind or "",
        "rubric_items": rubric_items,
        "modalities": list(modalities),
        "by_modality": {},
    }
    for mod in modalities:
        blob: dict[str, Any] = {"modality": mod}
        b = artifacts.get(mod) or b""
        if mod == "ipynb":
            blob["cells"] = grading_tools.extract_notebook_cells_structured(b)
            flat = grading_tools.extract_from_ipynb(b)
            blob["text_concat"] = {
                "code": flat.get("code", "")[:50000],
                "markdown": flat.get("markdown", "")[:50000],
            }
        elif mod == "py":
            try:
                blob["source"] = (b.decode("utf-8", errors="replace"))[:80000]
            except Exception:
                blob["source"] = ""
        elif mod == "md":
            try:
                blob["markdown"] = (b.decode("utf-8", errors="replace"))[:80000]
            except Exception:
                blob["markdown"] = ""
        elif mod == "txt":
            try:
                blob["text"] = (b.decode("utf-8", errors="replace"))[:80000]
            except Exception:
                blob["text"] = ""
        elif mod == "pdf":
            try:
                blob["extracted_text"] = grading_tools.extract_text_from_pdf(b)[
                    :80000
                ]
            except Exception:
                blob["extracted_text"] = ""
        elif mod == "url":
            blob["url"] = (b.decode("utf-8", errors="replace") if b else "")[
                :2000
            ]
        elif mod == "mp4":
            blob["note"] = (
                "Video modality; transcript may be provided in ctx or empty."
            )
        else:
            blob["raw_note"] = f"Unknown modality {mod}; bytes length {len(b)}"
        out["by_modality"][mod] = blob
    ctx["normalized"] = out


def normalized_to_json_str(ctx: dict[str, Any], *, max_chars: int) -> str:
    """Serialize normalized payload; truncate if needed."""
    raw = json.dumps(ctx.get("normalized", {}), ensure_ascii=False, default=str)
    if len(raw) <= max_chars:
        return raw
    return raw[: max_chars - 20] + "\n…[truncated]"


def criterion_dict_to_json(criterion: dict[str, Any], *, max_chars: int) -> str:
    raw = json.dumps(criterion, ensure_ascii=False, default=str)
    if len(raw) <= max_chars:
        return raw
    return raw[: max_chars - 20] + "\n…[truncated]"


def slice_evidence_for_criterion(
    evidence_bundle: dict[str, Any],
    criterion_name: str,
    *,
    max_chars: int,
) -> dict[str, Any]:
    """
    Prefer evidence snippets that mention the criterion; otherwise pass a truncated full bundle.
    """
    if not isinstance(evidence_bundle, dict):
        return {"criterion": criterion_name, "evidence": {}}
    needle = (criterion_name or "").lower().strip()
    if not needle:
        slim = json.dumps(evidence_bundle, ensure_ascii=False, default=str)
        return {
            "criterion": criterion_name,
            "evidence": evidence_bundle
            if len(slim) <= max_chars
            else {"truncated": slim[: max_chars - 40] + "…"},
        }

    def _pick_list(key: str) -> list[Any]:
        xs = evidence_bundle.get(key)
        if not isinstance(xs, list):
            return []
        hit = [x for x in xs if needle in json.dumps(x, default=str).lower()]
        return hit if hit else xs[:50]

    sliced = {
        "claims": _pick_list("claims"),
        "code_facts": _pick_list("code_facts"),
        "visualization_facts": _pick_list("visualization_facts"),
        "answers_by_question": _pick_list("answers_by_question"),
        "other": {
            k: v
            for k, v in evidence_bundle.items()
            if k
            not in (
                "claims",
                "code_facts",
                "visualization_facts",
                "answers_by_question",
                "contradictions_spotted",
            )
        },
    }
    contradictions = evidence_bundle.get("contradictions_spotted")
    if isinstance(contradictions, list):
        sliced["contradictions_spotted"] = [
            x for x in contradictions if needle in json.dumps(x, default=str).lower()
        ] or contradictions[:20]

    raw = json.dumps(sliced, ensure_ascii=False, default=str)
    if len(raw) <= max_chars:
        return {"criterion": criterion_name, "evidence": sliced}
    return {
        "criterion": criterion_name,
        "evidence": {"truncated": raw[: max_chars - 40] + "…"},
    }

