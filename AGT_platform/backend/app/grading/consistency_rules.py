"""Cheap deterministic checks before LLM consistency pass (stage 4)."""

from __future__ import annotations

from typing import Any


def run_rule_checks(
    criteria_results: list[dict[str, Any]],
    *,
    cap_points: float = 100.0,
) -> list[str]:
    """Return human-readable issues for prompts (may be empty)."""
    issues: list[str] = []
    seen: set[str] = set()
    for c in criteria_results:
        name = str(c.get("name") or "?")
        key = name.lower()
        if key in seen:
            issues.append(f"Duplicate criterion name: {name}")
        seen.add(key)
        try:
            sc = float(c.get("score", 0))
        except (TypeError, ValueError):
            issues.append(f"Non-numeric score for '{name}'")
            continue
        hi = float(c.get("max_points", cap_points))
        lo = float(c.get("min_points", 0))
        if sc < lo or sc > hi:
            issues.append(
                f"Score out of range for '{name}': {sc} not in [{lo}, {hi}]"
            )
        conf = c.get("confidence")
        if conf is not None:
            try:
                cf = float(conf)
                if cf < 0 or cf > 1:
                    issues.append(
                        f"Confidence out of [0,1] for '{name}': {cf}"
                    )
            except (TypeError, ValueError):
                issues.append(f"Non-numeric confidence for '{name}'")
    return issues
