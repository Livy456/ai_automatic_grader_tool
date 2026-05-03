"""
Resolve repository-side artifacts by assignment stem: **answer keys** and **blank
instructor templates**.

Uses normalized string similarity (``difflib``) so names like
``[Student 1] Week 1 PSet Part 1`` can match ``Week 1 PSet Part 1 [Answer_Key]``.
"""

from __future__ import annotations

import difflib
import json
import re
from pathlib import Path
from typing import Final

from app.grading.submission_text import submission_text_from_artifacts

_SUFFIXES: Final[tuple[str, ...]] = (".txt", ".md", ".json", ".ipynb")
_MIN_RATIO: Final[float] = 0.38

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


def _normalize_for_match(s: str) -> str:
    t = s.lower()
    t = re.sub(r"\[student\s*\d+\]\s*", "", t, flags=re.I)
    t = re.sub(r"\[[^\]]*\]", " ", t)
    t = re.sub(r"[^\w\s]+", " ", t)
    return " ".join(t.split())


def _read_file_plain(path: Path) -> str:
    raw = path.read_bytes()
    if path.suffix.lower() == ".ipynb":
        try:
            return submission_text_from_artifacts({"ipynb": raw}).strip()
        except (OSError, TypeError, ValueError, KeyError):
            try:
                nb = json.loads(raw.decode("utf-8", errors="replace"))
                parts: list[str] = []
                for cell in nb.get("cells") or []:
                    src = cell.get("source")
                    if isinstance(src, list):
                        parts.append("".join(src))
                    elif isinstance(src, str):
                        parts.append(src)
                return "\n".join(parts).strip()
            except (json.JSONDecodeError, TypeError, ValueError):
                return ""
    return path.read_text(encoding="utf-8", errors="replace").strip()


def resolve_answer_key_plaintext(
    assignment_stem: str,
    answer_key_dir: Path,
) -> tuple[str, str]:
    """
    Return ``(plaintext, matched_relative_name)``.

    Picks the best file in ``answer_key_dir`` by ``difflib.SequenceMatcher`` ratio
    between normalized stems. Empty string if nothing meets ``_MIN_RATIO``.
    """
    if not assignment_stem.strip() or not answer_key_dir.is_dir():
        return "", ""

    for suf in _SUFFIXES:
        exact = answer_key_dir / f"{assignment_stem}{suf}"
        if exact.is_file():
            try:
                return _read_file_plain(exact), exact.name
            except OSError:
                break

    stem_n = _normalize_for_match(assignment_stem)
    best_path: Path | None = None
    best_ratio = 0.0

    for path in sorted(answer_key_dir.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.name.lower() == "readme.md":
            continue
        if path.suffix.lower() not in _SUFFIXES:
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
        return "", ""

    try:
        text = _read_file_plain(best_path)
    except OSError:
        return "", ""
    return text, best_path.name


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
    "resolve_answer_key_plaintext",
    "resolve_blank_assignment_ipynb",
    "resolve_blank_assignment_template",
]
