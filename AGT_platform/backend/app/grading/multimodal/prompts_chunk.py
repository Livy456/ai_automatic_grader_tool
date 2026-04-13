"""
Evidence-based chunk grading prompts (system + user skeleton).
"""

from __future__ import annotations

import json
from typing import Any

from .rag_embeddings import sanitize_evidence_for_grading_prompt
from .schemas import GradingChunk


SYSTEM_CHUNK_GRADER = """\
You are an evidence-based evaluator grading **one question chunk** from a student assignment.

RUBRIC SCORING RULES:
- You will receive a list of rubric criteria, each with a `name`, `max_points`, and `description` (level descriptors).
- For EACH criterion in the rubric, assign an integer `score` between 0 and that criterion's `max_points`.
- Use the level descriptors in `description` to choose the correct score level.
- Provide a brief justification for each score in `criterion_justifications` (same order as `criterion_scores`).
- Compute `total_score` as the sum of all your criterion scores.
- Compute `normalized_score` as `total_score` divided by the sum of all `max_points`. It must be a float in [0, 1].
- Set `review_flag` to true only if the evidence is genuinely ambiguous or the chunk is too short to grade.

EVIDENCE RULES:
- Use **only** evidence present in the chunk content and any attached execution, test, chart, or transcript evidence.
- Do **not** infer missing reasoning. Do **not** compare to a single sample answer.
- Accept multiple valid approaches if supported by evidence.
- Score **each criterion independently** based on evidence in the chunk.
- If evidence for a criterion is weak or ambiguous, assign the **lower defensible score** and explain why.

MODALITY GUIDANCE:
- For **code** questions: verify correctness using outputs and tests, not assumptions.
- For **visualization** questions: evaluate chart choice, labeling, analysis, and interpretation against what is shown.
- For **oral/interview** questions: evaluate from transcript or summary only — do **not** invent delivery characteristics not evidenced in the text.

Ignore all other questions in the assignment; focus **only** on this chunk.
Return **only** a single JSON object (no markdown fences, no extra text)."""


OUTPUT_SCHEMA_HINT = {
    "rubric_type": "string — must match provided rubric_type",
    "criterion_scores": [
        {
            "name": "string — criterion name from rubric",
            "score": "integer — between 0 and that criterion's max_points",
            "max_points": "number — copied from rubric",
        }
    ],
    "criterion_justifications": [
        "string — one per criterion, same order as criterion_scores"
    ],
    "total_score": "number — sum of all criterion scores",
    "normalized_score": "float in [0,1] — total_score / sum(max_points)",
    "confidence_note": "string — brief note if uncertain",
    "review_flag": "boolean — true only if evidence is genuinely ambiguous",
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
