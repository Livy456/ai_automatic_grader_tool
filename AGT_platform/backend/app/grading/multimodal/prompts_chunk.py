"""
Evidence-based chunk grading prompts (system + user skeleton).
"""

from __future__ import annotations

import json
from typing import Any

from .rag_embeddings import sanitize_evidence_for_grading_prompt
from .schemas import GradingChunk


SYSTEM_CHUNK_GRADER = """You are an evidence-based evaluator grading **one question chunk** from a student assignment.
Select scores only from the provided rubric definitions.
Use **only** the evidence present in the chunk content and any attached execution, test, chart, or transcript evidence.
Do **not** infer missing reasoning. Do **not** compare to a single sample answer.
Accept multiple valid approaches if supported by evidence.
Score **each criterion independently**.
If evidence for a criterion is weak or ambiguous, assign the **lower defensible score** and explain why.
Return **only** structured JSON (no markdown fences).
For **code** questions: verify correctness using outputs and tests, not assumptions.
For **visualization** questions: evaluate chart choice, labeling, analysis, and interpretation against what is shown.
For **oral/interview** questions: evaluate from transcript or summary only — do **not** invent delivery characteristics not evidenced in the text.
Ignore all other questions in the assignment; focus **only** on this chunk."""


OUTPUT_SCHEMA_HINT = {
    "rubric_type": "string (must match provided rubric_type)",
    "criterion_scores": [
        {"name": "string", "score": "number", "max_points": "number", "weight": "number"}
    ],
    "criterion_justifications": ["string aligned with criterion_scores order"],
    "total_score": "number",
    "normalized_score": "number in [0,1] relative to rubric max",
    "confidence_note": "string",
    "review_flag": "boolean",
}


def build_chunk_grading_prompt(
    chunk: GradingChunk,
    *,
    task_description: str = "",
) -> str:
    """Construct user message: task + rubric + chunk + strict instructions."""
    rubric = {
        "rubric_type": chunk.rubric_type.value if chunk.rubric_type else None,
        "rows": chunk.rubric_rows,
    }
    chunk_dict = chunk.to_prompt_dict()
    chunk_dict["evidence"] = sanitize_evidence_for_grading_prompt(
        chunk_dict.get("evidence") or {}
    )
    payload = {
        "instructions": (
            "Grade this single chunk. Output JSON with keys: "
            "rubric_type, criterion_scores, criterion_justifications, total_score, "
            "normalized_score, confidence_note, review_flag."
        ),
        "task_description": task_description or "(see assignment brief in LMS)",
        "chunk": chunk_dict,
        "rubric": rubric,
        "output_schema_hint": OUTPUT_SCHEMA_HINT,
    }
    return json.dumps(payload, ensure_ascii=True, indent=2)
