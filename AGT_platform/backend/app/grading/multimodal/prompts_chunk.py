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

CHAIN-OF-THOUGHT — for **each** rubric criterion, in order:
  Step 1  EXTRACT: Quote or excerpt only what appears in the submission (chunk text / artifacts).
  Step 2  REASON:  Say what that evidence shows **for this criterion only**. Do not infer unstated student reasoning.
  Step 3  SCORE:   Assign the **raw rubric level** for this criterion (see RAW SCORE GRID below). It must follow from Steps 1–2.

RAW SCORE GRID — **mandatory; no exceptions**
- For each criterion, **R** = that row's `max_points` (top of the ordinal scale: levels **0 through R** inclusive).
- You **must** output `raw_score` or `score` as **exactly one** of these values only:
  **0, 0.5, 1, 1.5, 2, 2.5, … , R** — i.e. start at 0, add **0.5** each step, stop at **R**.
  Examples: R=1 → {0, 0.5, 1}. R=3 → {0, 0.5, 1, 1.5, 2, 2.5, 3}. R=4 → {0, 0.5, …, 4}.
- **Do not** output any other decimals (no 2.33, no 3.7, no integers not on this ladder unless they equal a valid step).
- **Server correction (if you slip):** values not on that ladder are adjusted **upward** to the **next** valid half-step on ``[0, R]``, and never above **R** (so the corrected score is always one of the allowed values above). You should still output a valid value yourself — do not rely on correction.
- Use a **half-point** when the submission clearly falls **between** two adjacent rubric descriptors; otherwise use a whole number on the same ladder.
- **0** = no defensible evidence for this criterion (blank/off-topic for that part). If there is a relevant attempt (even flawed), consider **≥ 0.5** rather than 0.

RUBRIC FIDELITY:
- **EXACT NAMES ONLY:** `criterion_scores[].name` MUST match `rubric.rows[].name` **exactly**. No `criterion_1`, no extra criteria.
- One object per rubric row; same order as `rubric.rows` when possible.
- Put the raw level in **`raw_score`** (preferred) or **`score`**. Copy `max_points` from the rubric row.
- Include `evidence`, `reasoning`, `justification` per criterion. `evidence` must be non-empty unless the submission is blank for that part.
- Also output `criterion_justifications` (one string per criterion, same order).
- Optional `total_score` / `normalized_score`: the **server recomputes** the official score from your raw levels; do not trust self-computed linear ratios.
- `review_flag`: true only if evidence is genuinely ambiguous or the chunk is too short to grade.

FAIRNESS — avoid **harsh** grading:
- Reward **partial mastery**: if the rubric text leaves room between “missing” and “complete,” prefer the level that **best matches quoted evidence**, not the lowest defensible level.
- **Ambiguity favors the student** only when the chunk genuinely supports the higher band; cite the lines that justify the higher score.
- Do **not** double-penalize: if one gap already lowers one criterion, do not use the same gap to justify min scores on unrelated criteria.
- If level descriptors are vague, interpret them **charitably** in favor of evident learning, while still requiring **some** quoted support for non-zero scores.

RUBRIC QUALITY (helps models grade less harshly and more consistently — apply when reading `rubric.rows`):
- Prefer **observable** level descriptors (“states X”, “shows Y”) over vague terms (“insightful”, “weak”) without anchors.
- Separate **dimensions** (e.g. correctness vs. communication) so one weakness does not collapse every score.
- Add **explicit partial-credit** bullets for mid levels (what “adequate” vs “strong” looks like).
- Keep **R** modest and aligned to real distinctions; very large R without rich descriptors invites score compression at the bottom.
- If the rubric has only high bars, treat “mostly meets” as **mid-scale**, not failure, when evidence supports it.

MODALITY GUIDANCE:
- **Code:** use outputs/tests shown in the chunk, not assumptions.
- **Visualization:** judge only what is visible in charts/tables provided.
- **Oral/interview:** transcript/summary only — do not invent delivery traits.

Ignore other questions; grade **only** this chunk.
Return **only** one JSON object (no markdown fences, no prose outside JSON)."""


OUTPUT_SCHEMA_HINT = {
    "rubric_type": "string — must match provided rubric_type",
    "criterion_scores": [
        {
            "name": "string — must equal rubric.rows[].name (same as criterion_name)",
            "raw_score": "number — raw rubric level on 0..max_points in steps of 0.5 only (alias: score)",
            "max_points": "number — max ordinal R, copied from rubric",
            "evidence": "REQUIRED string — quote/excerpt from submission only",
            "reasoning": "REQUIRED string — evidence-grounded; no invented student intent",
            "justification": "string — short summary tied to evidence",
        }
    ],
    "criterion_justifications": [
        "string — one per criterion, same order as criterion_scores",
    ],
    "total_score": "optional — sum of raw scores (server may ignore)",
    "normalized_score": "optional — server recomputes from calibrated mapping",
    "confidence_note": "string — brief note if uncertain",
    "review_flag": "boolean — true only if evidence is genuinely ambiguous",
    "note": (
        "Server adds calibrated_credit and weighted question score. "
        "If raw_score is not on 0,0.5,…,R, server ceils to the next half-step capped at R."
    ),
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
            "Grade this single chunk. Output one JSON object.\n"
            "Keys: rubric_type, criterion_scores, criterion_justifications, confidence_note, "
            "review_flag; optional total_score, normalized_score (server may override).\n"
            "Rubric fidelity: criterion_scores MUST list only rubric.rows names, exact spelling.\n"
            "Each criterion row MUST include: evidence, reasoning, raw_score (or score), "
            "name, max_points, justification.\n"
            "RAW SCORES — critical: use **only** the ladder 0, 0.5, 1, 1.5, …, max_points for "
            "each row. No other values. If you output an invalid decimal, the server will "
            "round **up** to the next valid half-step (capped at max_points); you should "
            "still emit a correct value yourself. Grade fairly: reward quoted partial work; "
            "do not default to the lowest band when evidence fits a mid level."
        ),
        "task_description": task_description or "(see assignment brief in LMS)",
        "chunk": chunk_dict,
        "rubric": rubric,
        "output_schema_hint": OUTPUT_SCHEMA_HINT,
    }
    return json.dumps(payload, ensure_ascii=True, indent=2)
