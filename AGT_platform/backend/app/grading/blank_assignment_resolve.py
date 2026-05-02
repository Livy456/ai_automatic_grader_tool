"""
Resolve a **blank** instructor copy of an assignment (questions + instructions only).

Files live under ``blank_assignments/`` at the repository root (or a caller-provided
directory). Matching mirrors :func:`app.grading.answer_key_resolve.resolve_answer_key_plaintext`
stem logic and returns **raw bytes** plus the matched filename suffix for multimodal chunking.

Supported template suffixes: ``.ipynb``, ``.pdf``, ``.docx``, ``.py``, ``.txt``, ``.md``,
``.csv``, ``.xlsx``.
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Final

_BLANK_SUFFIXES: Final[tuple[str, ...]] = (
    ".ipynb",
    ".pdf",
    ".docx",
    ".py",
    ".txt",
    ".md",
    ".csv",
    ".xlsx",
)
_MIN_RATIO: Final[float] = 0.38


def _normalize_for_match(s: str) -> str:
    t = s.lower()
    t = re.sub(r"\[student\s*\d+\]\s*", "", t, flags=re.I)
    t = re.sub(r"\[[^\]]*\]", " ", t)
    t = re.sub(r"[^\w\s]+", " ", t)
    return " ".join(t.split())


def resolve_blank_assignment_template(
    assignment_stem: str,
    blank_dir: Path,
) -> tuple[bytes, str, str]:
    """
    Return ``(file_bytes, matched_relative_name, suffix_lower)``.

    ``suffix_lower`` includes the leading dot (e.g. ``".ipynb"``). Empty bytes when no
    suitable file is found under ``blank_dir``.
    """
    if not assignment_stem.strip() or not blank_dir.is_dir():
        return b"", "", ""

    for suf in _BLANK_SUFFIXES:
        exact = blank_dir / f"{assignment_stem}{suf}"
        if exact.is_file():
            try:
                return exact.read_bytes(), exact.name, suf.lower()
            except OSError:
                break

    stem_n = _normalize_for_match(assignment_stem)
    best_path: Path | None = None
    best_ratio = 0.0

    for path in sorted(blank_dir.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.name.lower() == "readme.md":
            continue
        suf = path.suffix.lower()
        if suf not in _BLANK_SUFFIXES:
            continue
        key_n = _normalize_for_match(path.stem)
        if not key_n:
            continue
        ratio = difflib.SequenceMatcher(None, stem_n, key_n).ratio()
        if stem_n and (stem_n in key_n or key_n in stem_n):
            ratio = max(ratio, 0.88)
        if ratio > best_ratio:
            best_ratio = ratio
            best_path = path

    if best_path is None or best_ratio < _MIN_RATIO:
        return b"", "", ""

    try:
        suf = best_path.suffix.lower()
        return best_path.read_bytes(), best_path.name, suf
    except OSError:
        return b"", "", ""


def resolve_blank_assignment_ipynb(
    assignment_stem: str,
    blank_dir: Path,
) -> tuple[bytes, str]:
    """
    Return ``(ipynb_bytes, matched_relative_name)`` — **notebook only** (backward compatible).

    When the matched blank is not ``.ipynb``, returns ``(b"", "")`` so legacy notebook-only
    chunkers skip; use :func:`resolve_blank_assignment_template` for all file types.
    """
    data, name, suf = resolve_blank_assignment_template(assignment_stem, blank_dir)
    if suf == ".ipynb" and data.strip():
        return data, name
    return b"", ""


__all__ = [
    "resolve_blank_assignment_ipynb",
    "resolve_blank_assignment_template",
    "_BLANK_SUFFIXES",
]
