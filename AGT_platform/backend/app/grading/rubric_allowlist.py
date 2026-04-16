"""
Map free-form LLM criterion labels to an allowed rubric name set.

Used by multimodal chunk parsing and grading-output validation so scores cannot
attach to hallucinated criterion keys (e.g. ``criterion_1``) that are not in
the LMS rubric for the routed modality.
"""

from __future__ import annotations

import difflib
from typing import Any


def normalize_rubric_criterion_key(s: str) -> str:
    """Lowercase, collapse internal whitespace (stable matching key)."""
    return " ".join((s or "").strip().lower().split())


def match_criterion_name_to_allowlist(name: str, allowed: frozenset[str]) -> str | None:
    """
    Return the **canonical** rubric string from ``allowed`` that best matches ``name``,
    or ``None`` if no acceptable match.

    Order: exact string match → normalized match → fuzzy (difflib, cutoff 0.78).
    """
    raw = (name or "").strip()
    if not raw or not allowed:
        return None
    if raw in allowed:
        return raw
    n = normalize_rubric_criterion_key(raw)
    by_norm: dict[str, str] = {}
    for a in allowed:
        by_norm.setdefault(normalize_rubric_criterion_key(a), a)
    if n in by_norm:
        return by_norm[n]
    norm_keys = list(by_norm.keys())
    hits = difflib.get_close_matches(n, norm_keys, n=1, cutoff=0.78)
    if hits:
        return by_norm[hits[0]]
    return None


def rubric_rows_to_allowlist(rubric_rows: list[dict[str, Any]] | None) -> frozenset[str]:
    """Criterion ``name`` values from routing rows (canonical spellings)."""
    out: list[str] = []
    for r in rubric_rows or []:
        if not isinstance(r, dict):
            continue
        nm = str(r.get("name") or r.get("criterion") or "").strip()
        if nm:
            out.append(nm)
    return frozenset(out)


def filter_criteria_dicts_to_allowlist(
    rows: list[dict[str, Any]],
    allowed: frozenset[str],
    *,
    context: str = "",
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Keep only rows whose ``name`` maps to an allowed rubric criterion; remap ``name``
    to the canonical spelling. Drops unknown names and records issues for flags/logs.
    """
    issues: list[str] = []
    if not allowed:
        return list(rows), issues
    prefix = f"{context}:" if context else ""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw = str(row.get("name") or row.get("criterion") or "").strip()
        canon = match_criterion_name_to_allowlist(raw, allowed)
        if canon is None:
            issues.append(f"{prefix}removed_extraneous_criterion:{raw!r}")
            continue
        if canon in seen:
            issues.append(f"{prefix}deduplicated_criterion:{canon!r}")
            continue
        seen.add(canon)
        nr = dict(row)
        nr["name"] = canon
        out.append(nr)
    return out, issues
