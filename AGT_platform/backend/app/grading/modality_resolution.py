"""
Modality detection for the grading pipeline: file types + assignment title heuristics + text signals.
"""

from __future__ import annotations

from typing import Any


def infer_modality_from_artifacts(artifacts_bytes: dict[str, bytes]) -> str:
    """Planner-facing modality label (legacy string)."""
    if "mp4" in artifacts_bytes:
        return "video"
    if "ipynb" in artifacts_bytes:
        return "notebook"
    if "py" in artifacts_bytes or "zip" in artifacts_bytes:
        return "code"
    if "pdf" in artifacts_bytes:
        return "written"
    if "txt" in artifacts_bytes:
        return "text"
    if "png" in artifacts_bytes or "jpg" in artifacts_bytes or "docx" in artifacts_bytes:
        return "written"
    return "written"


def resolve_modality_profile(
    assignment: Any,
    artifacts_bytes: dict[str, bytes],
    extracted_text_sample: str,
) -> dict[str, Any]:
    """
    Rich modality decision for logging, prompts, validation, and RAG.

    Returns dict with:
    - ``modality``: same bucket as :func:`infer_modality_from_artifacts` unless refined.
    - ``modality_subtype``: ``notebook`` | ``journal_entry`` | ``free_response_pdf`` | ``pdf_written`` | ``text`` | ``code`` | ``video`` | ``mixed``
    - ``artifact_keys``, ``extracted_text_chars``, ``signals``
    """
    keys = sorted(artifacts_bytes.keys())
    title = (
        str(getattr(assignment, "title", None) or "")
        + " "
        + str(getattr(assignment, "description", None) or "")
    )
    tl = title.lower()
    sample = extracted_text_sample or ""
    n_chars = len(sample.strip())

    base = infer_modality_from_artifacts(artifacts_bytes)
    subtype = "unknown"

    if "mp4" in artifacts_bytes:
        subtype = "video"
    elif "ipynb" in artifacts_bytes:
        subtype = "notebook"
    elif "py" in artifacts_bytes or "zip" in artifacts_bytes:
        subtype = "code"
    elif "pdf" in artifacts_bytes:
        if "journal" in tl or "journal entry" in tl:
            subtype = "journal_entry"
        elif "free response" in tl or "free-response" in tl or "short answer" in tl:
            subtype = "free_response_pdf"
        else:
            subtype = "pdf_written"
    elif "txt" in artifacts_bytes:
        subtype = "text"
    else:
        subtype = base if base != "written" else "written"

    if "ipynb" in artifacts_bytes and "pdf" in artifacts_bytes:
        subtype = "mixed_notebook_pdf"

    signals = {
        "title_mentions_journal": "journal" in tl,
        "title_mentions_free_response": "free response" in tl or "free-response" in tl,
        "has_substantive_text": n_chars >= 80,
        "text_too_short_for_grading": n_chars < 40,
    }

    return {
        "modality": base,
        "modality_subtype": subtype,
        "artifact_keys": keys,
        "extracted_text_chars": n_chars,
        "signals": signals,
    }


def augment_prompt_for_modality_profile(
    assignment_prompt: str,
    profile: dict[str, Any],
) -> str:
    """Append short instructions so small models don't treat prose PDFs as code tasks."""
    st = profile.get("modality_subtype") or ""
    parts = [assignment_prompt]
    if st in ("journal_entry", "free_response_pdf"):
        parts.append(
            "\n\n---\nGrading context: This submission is a **written / journal-style** response. "
            "Use the `submission` object's `artifacts.text` (extracted document text) and any "
            "`markdown` fields. **Do not** require executable code unless the prompt explicitly asks for it. "
            "If the extracted text is short, grade conservatively on what is present; **do not** assign "
            "all-zero scores unless there is no substantive student answer."
        )
    elif st == "pdf_written":
        parts.append(
            "\n\n---\nGrading context: Submission is primarily **PDF prose**. Rely on `artifacts.text` "
            "(full extracted text). Score each rubric criterion from the student's written evidence."
        )
    elif st in ("text",):
        parts.append(
            "\n\n---\nGrading context: Plain-text submission — use `artifacts.text` as the full answer."
        )
    return "\n".join(parts)
