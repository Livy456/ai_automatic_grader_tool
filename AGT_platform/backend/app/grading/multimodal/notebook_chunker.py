"""
Notebook-aware chunker: parse ipynb cell structure directly, preserving cell
order so that each question/prompt is paired with the student's actual response.

The text-based heuristic chunker receives pre-flattened plaintext where code
and markdown are split into separate sections (destroying Q/A pairing).  This
module parses the raw notebook JSON to produce accurate ``GradingChunk`` objects.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from .schemas import GradingChunk, Modality, TaskType

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cell classification patterns
# ---------------------------------------------------------------------------

_QUESTION_RE = [
    # ### Question 1.4.1, ## Problem 3, # Exercise 2a
    re.compile(
        r"^\s*#{1,4}\s*(?:Question|Problem|Exercise|Task)\s+[\w.\-]+",
        re.IGNORECASE | re.MULTILINE,
    ),
    # ##1.1 ..., # 1.4.2 ..., ### 2.3 ... (numbered sub-sections)
    re.compile(
        r"^\s*#{1,4}\s*\d+\.\d+[\.\d]*\b",
        re.MULTILINE,
    ),
]

_SECTION_RE = [
    # # Section 2: Data Preprocessing, ### Section 3: ...
    re.compile(
        r"^\s*#{1,4}\s*(?:Section|Part)\s+\d+",
        re.IGNORECASE | re.MULTILINE,
    ),
    # # Title-like single heading (all-caps or mixed-case without numbers)
    re.compile(
        r"^\s*#{1,2}\s+[A-Z][A-Za-z &,\-/]+\s*$",
        re.MULTILINE,
    ),
]

_TITLE_RE = re.compile(
    r"^\s*#{1,2}\s+.{10,}",
    re.MULTILINE,
)

_INSTRUCTOR_RE = [
    re.compile(r"DO\s+NOT\s+MODIFY", re.IGNORECASE),
    re.compile(r"INSTRUCTOR\s+CODE", re.IGNORECASE),
    re.compile(r"START\s+OF\s+INSTRUCTOR", re.IGNORECASE),
]

_STUDENT_WORK_RE = [
    re.compile(r"#\s*write\s+code\s+(?:for|here)", re.IGNORECASE),
    re.compile(r"#\s*your\s+(?:code|answer)", re.IGNORECASE),
    re.compile(r"END\s+OF\s+INSTRUCTOR\s+CODE", re.IGNORECASE),
]

_TEST_CODE_RE = [
    re.compile(r"^\s*#\s*[Tt]est\s+(?:code\s+)?for\s+(?:problem|question)", re.MULTILINE),
]

_ASSERT_HEAVY_THRESHOLD = 0.5

# Short plain-text markdown cells that act as section labels (no heading markers)
_LABEL_LIKE_RE = re.compile(r"^[\w\s/\-,&.():]+$")

_QID_RE = re.compile(
    r"(?:Question|Problem|Exercise|Task|Part|Section)?\s*"
    r"(\d+(?:\.\d+)*[a-zA-Z]?)",
    re.IGNORECASE,
)


def _cell_source(cell: dict) -> str:
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(src)
    return str(src or "")


def _extract_question_id(source: str) -> str:
    """Pull a short question identifier like ``1.4.1`` from a heading line."""
    for line in source.splitlines()[:3]:
        m = _QID_RE.search(line)
        if m:
            return m.group(1).strip()
    return ""


def _matches_any(patterns: list[re.Pattern], text: str) -> bool:
    return any(p.search(text) for p in patterns)


def _is_question(src: str) -> bool:
    return _matches_any(_QUESTION_RE, src)


def _is_section_header(src: str) -> bool:
    return _matches_any(_SECTION_RE, src)


def _is_instructor_code(src: str) -> bool:
    return _matches_any(_INSTRUCTOR_RE, src)


def _has_student_work_marker(src: str) -> bool:
    return _matches_any(_STUDENT_WORK_RE, src)


def _is_test_code(src: str) -> bool:
    if _matches_any(_TEST_CODE_RE, src):
        return True
    lines = [ln for ln in src.strip().splitlines() if ln.strip()]
    if not lines:
        return False
    assert_lines = sum(1 for ln in lines if ln.strip().startswith("assert "))
    return len(lines) <= 6 and assert_lines > 0 and assert_lines / len(lines) >= _ASSERT_HEAVY_THRESHOLD


def _is_section_label(src: str) -> bool:
    """Short plain-text markdown cell that acts as a section label
    (e.g. ``combining country data``, ``Data Preparation/...``)."""
    stripped = src.strip()
    if len(stripped) > 120 or not stripped:
        return False
    lines = [ln for ln in stripped.splitlines() if ln.strip()]
    if len(lines) > 2:
        return False
    return bool(_LABEL_LIKE_RE.match(stripped))


def _classify_markdown(src: str, cell_idx: int) -> str:
    """Return one of: ``question``, ``section_header``, ``preamble``, ``student_text``."""
    if _is_question(src):
        return "question"
    if _is_section_header(src):
        return "section_header"
    if cell_idx == 0 and _TITLE_RE.search(src):
        return "preamble"
    if _is_section_label(src):
        return "section_header"
    return "student_text"


def _split_instructor_prefix_and_student_suffix(src: str) -> tuple[str, str]:
    """
    Many assignments wrap instructor scaffolding in ``# DO NOT MODIFY`` blocks and put
    student work **after** an ``END OF INSTRUCTOR`` (or similar) marker. Without splitting,
    the whole cell is classified as instructor-only and student code never reaches
    ``response_parts`` — the grader then sees prompts but no code for evidence.
    """
    s = src.strip()
    if not s:
        return "", ""
    low = s.lower()
    delimiters = (
        "\n# end of instructor code",
        "\n# end of instructor",
        "# end of instructor code",
        "# end of instructor",
        "\n# your code below",
        "\n# your solution",
        "\n# write your code here",
        "\n# --- student",
        "\n# begin student",
    )
    best_tail = ""
    best_cut = -1
    for d in delimiters:
        idx = low.rfind(d)
        if idx < 0:
            continue
        cut = idx + len(d)
        while cut < len(s) and s[cut] in " \t":
            cut += 1
        if cut < len(s) and s[cut] == "\n":
            cut += 1
        tail = s[cut:].strip()
        if len(tail) >= 6 and len(tail) > len(best_tail):
            best_tail = tail
            best_cut = cut
    if best_cut < 0:
        return s, ""
    return s[:best_cut].strip(), best_tail


def _classify_code(src: str) -> str:
    """Return one of: ``student_code``, ``instructor_code``, ``test_code``, ``empty``."""
    stripped = src.strip()
    if not stripped:
        return "empty"
    _head, tail = _split_instructor_prefix_and_student_suffix(stripped)
    if tail and len(tail) >= 8:
        return "student_code"
    if _is_instructor_code(stripped) and _has_student_work_marker(stripped):
        return "student_code"
    if _is_instructor_code(stripped):
        return "instructor_code"
    if _is_test_code(stripped):
        return "test_code"
    return "student_code"


def _classify_cell(cell: dict, cell_idx: int) -> str:
    src = _cell_source(cell)
    ct = cell.get("cell_type", "")
    if ct == "markdown":
        return _classify_markdown(src, cell_idx)
    if ct == "code":
        return _classify_code(src)
    return "empty"


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

def _new_unit(question_id: str) -> dict[str, Any]:
    return {
        "question_id": question_id,
        "question_parts": [],
        "response_parts": [],
        "context_parts": [],
        "has_student_content": False,
    }


def _unit_trio_payload(unit: dict[str, Any]) -> dict[str, str]:
    """Structured question / student work / (answer key filled later) for RAG + prompts."""
    q = "\n\n".join(p for p in unit["question_parts"] if p.strip()).strip()
    r = "\n\n".join(p for p in unit["response_parts"] if p.strip()).strip()
    ctx = "\n\n".join(p for p in unit.get("context_parts", []) if p.strip()).strip()
    return {
        "question": q,
        "student_response": r,
        "instructor_context": ctx,
        "answer_key_segment": "",
    }


def _unit_to_extracted_text(unit: dict[str, Any]) -> str:
    """Flatten a unit into ``extracted_text`` for the ``GradingChunk``."""
    parts: list[str] = []

    q = "\n\n".join(p for p in unit["question_parts"] if p.strip())
    if q.strip():
        parts.append(q.strip())

    ctx = "\n\n".join(p for p in unit["context_parts"] if p.strip())
    if ctx.strip():
        parts.append(ctx.strip())

    r = "\n\n".join(p for p in unit["response_parts"] if p.strip())
    if r.strip():
        parts.append(r.strip())

    return "\n\n".join(parts).strip()


def build_notebook_qa_chunks(
    ipynb_bytes: bytes,
    *,
    assignment_id: str,
    student_id: str,
    modality: Modality = Modality.NOTEBOOK,
    task_type: TaskType = TaskType.UNKNOWN,
    max_grading_units: int | None = None,
) -> list[GradingChunk]:
    """
    Parse an ipynb file and return one :class:`GradingChunk` per detected
    question/answer pair, preserving cell order.

    Falls back to a single whole-notebook chunk when no question structure is
    detected (project-style notebooks with only code).
    """
    try:
        nb = json.loads(ipynb_bytes.decode("utf-8", errors="replace"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        _log.warning("notebook_chunker: could not parse ipynb JSON")
        return []

    cells = nb.get("cells", [])
    if not cells:
        return []

    # --- Pass 1: classify ---
    classified: list[tuple[dict, str, str, str]] = []
    for idx, cell in enumerate(cells):
        src = _cell_source(cell)
        role = _classify_cell(cell, idx)
        classified.append((cell, src, cell.get("cell_type", ""), role))

    # --- Pass 2: group into units ---
    units: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for _cell, src, _ctype, role in classified:
        if not src.strip() and role == "empty":
            continue

        if role in ("question", "section_header"):
            if current is not None:
                if current["has_student_content"]:
                    units.append(current)
                else:
                    question_prefix = current["question_parts"]
                    qid = _extract_question_id(src) or current["question_id"]
                    current = _new_unit(qid)
                    current["question_parts"] = question_prefix + [src]
                    continue
            qid = _extract_question_id(src) or f"q{len(units) + 1}"
            current = _new_unit(qid)
            current["question_parts"].append(src)

        elif role == "student_code":
            if current is None:
                current = _new_unit("preamble")
            current["response_parts"].append(src)
            current["has_student_content"] = True

        elif role == "student_text":
            if current is None:
                current = _new_unit("preamble")
            current["response_parts"].append(src)
            current["has_student_content"] = True

        elif role == "preamble":
            if current is None:
                current = _new_unit("preamble")
            current["question_parts"].append(src)

        elif role == "instructor_code":
            if current is None:
                current = _new_unit("preamble")
            inst, stud = _split_instructor_prefix_and_student_suffix(src)
            if stud.strip():
                if inst.strip():
                    current["context_parts"].append(inst.strip())
                current["response_parts"].append(stud.strip())
                current["has_student_content"] = True
            else:
                current["context_parts"].append(src)

        elif role == "test_code":
            if current is not None:
                current["context_parts"].append(src)

    if current is not None and current["has_student_content"]:
        units.append(current)

    # --- Fallback: if no structured questions found, one chunk per section ---
    if not units:
        full_text_parts: list[str] = []
        for _cell, src, ctype, _role in classified:
            s = src.strip()
            if s:
                full_text_parts.append(s)
        full = "\n\n".join(full_text_parts).strip()
        if full:
            units = [
                {
                    "question_id": "full",
                    "question_parts": ["(full notebook — no question structure detected)"],
                    "response_parts": [full],
                    "context_parts": [],
                    "has_student_content": True,
                }
            ]

    if not units:
        return []

    # --- Apply cap ---
    if max_grading_units is not None and max_grading_units >= 1:
        units = units[:max_grading_units]

    # --- Convert to GradingChunk ---
    out: list[GradingChunk] = []
    for unit in units:
        qid = unit["question_id"]
        extracted = _unit_to_extracted_text(unit)
        if not extracted:
            continue
        trio = _unit_trio_payload(unit)
        out.append(
            GradingChunk(
                chunk_id=f"{student_id}:{assignment_id}:{qid}",
                assignment_id=assignment_id,
                student_id=student_id,
                question_id=qid,
                modality=modality,
                task_type=task_type,
                extracted_text=extracted,
                evidence={
                    "chunker": "notebook_cell_order",
                    "question_id": qid,
                    "question_text": "\n\n".join(
                        unit["question_parts"]
                    ).strip()[:2000],
                    "response_preview": "\n\n".join(
                        unit["response_parts"]
                    ).strip()[:500],
                    "trio": trio,
                },
            )
        )

    _log.info(
        "notebook_chunker: %d cells → %d Q/A chunks for %s",
        len(cells),
        len(out),
        assignment_id,
    )
    return out
