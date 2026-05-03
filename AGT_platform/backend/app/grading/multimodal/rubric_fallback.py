"""Fallback rubric rows when no course JSON rubric is attached (local tests, prose-only generics)."""

from __future__ import annotations

from typing import Any

# Shape matches course DB / API rows: ``criterion`` + ``max_score``.
DEFAULT_STANDALONE_RUBRIC: tuple[dict[str, Any], ...] = (
    {"criterion": "Correctness", "max_score": 10.0},
    {"criterion": "Completeness", "max_score": 10.0},
    {"criterion": "Clarity and presentation", "max_score": 5.0},
)
