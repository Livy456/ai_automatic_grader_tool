"""
Notebook-aware chunker: parse ipynb cell structure directly, preserving cell
order so that each question/prompt is paired with the student's actual response.

Question boundaries are **not** limited to ``### Question 1.1``-style headings: substantive
``##`` / ``###`` lines (e.g. ``## Step 4: …``), plain titles with time estimates
(``Loading CSV … (5 min)``), and instructional markdown that references ``code block below``
stay with the prompt until the next code cell. Readme-style prose (e.g. "In this section we will…",
homework setup notes) is classified as **question** text, not ``student_response``.
When a **blank** instructor copy is aligned
(:mod:`template_aligned_notebook_chunks`), **scaffold code-cell anchors** on the blank
(``# TODO`` / ``add code`` / placeholders) align the student's response tail by ordinal
anchor. The text-based heuristic chunker receives pre-flattened plaintext where code
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
    # ## Part 1, ### Part A — intro (Part without requiring a digit)
    re.compile(
        r"^\s*#{1,6}\s*Part\s+[:\.]?\s*[\w.\-/]+",
        re.IGNORECASE | re.MULTILINE,
    ),
    # ##1.1 ..., # 1.4.2 ..., ### 2.3 ... (numbered sub-sections)
    re.compile(
        r"^\s*#{1,4}\s*\d+\.\d+[\.\d]*\b",
        re.MULTILINE,
    ),
    # **Problem 1** / **Question 2** at line start (common in converted notebooks)
    re.compile(
        r"^\s*\*{2}\s*(?:Problem|Question|Exercise|Part|Task)\s+[\w.\-]+",
        re.IGNORECASE | re.MULTILINE,
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

# Whole-line assignment boilerplate (code cells) — strip so similarity / RAG / grading
# are not skewed by matching these to the answer key.
_ASSIGNMENT_PLACEHOLDER_LINE = re.compile(
    r"^\s*#\s*(?:"
    r"write\s+code\s+here\b"
    r"|write\s+code\s+for\b[^\n#]*"
    r"|write\s+your\s+code\s+here\b"
    r"|your\s+code\s+here\b"
    r"|your\s+answer\s+here\b"
    r"|insert\s+(?:your\s+)?code\b[^\n#]*"
    r"|put\s+(?:your\s+)?(?:code|answer)\s+here\b"
    r"|fill\s+in\s+(?:your\s+)?code\b[^\n#]*"
    r")\s*$",
    re.IGNORECASE,
)

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

# Instructional prose (markdown) that continues a prompt before student code — not a "response".
_PROMPT_EXTENSION_CUES = re.compile(
    r"\b("
    r"use\s+the\s+code|code\s+block\s+below|in\s+this\s+cell|first\s+code\s+cell|"
    r"second\s+code\s+cell|third\s+code\s+cell|fourth\s+code\s+cell|"
    r"your\s+task|you\s+should|let\'?s\s+|we\s+want\s+to|print\s+out|"
    r"follow\s+the|complete\s+the|fill\s+in\s+the|dataframe|column\s+names|"
    r"explore|familiar\s+with|loading\s+csv|into\s+pandas|now\s+that\s+we"
    r")\b",
    re.IGNORECASE,
)

_SUBSTANTIVE_HASH_HEADING = re.compile(
    r"^\s*(#{2,6})\s+(\S.{5,})\s*$",
    re.MULTILINE,
)

_PLAIN_TITLE_WITH_TIME_ESTIMATE = re.compile(
    r"^\s*.{10,}\(\s*\d+\s*min\s*\)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def ipynb_to_plaintext_for_structure_llm(ipynb_bytes: bytes) -> str:
    """
    Flatten an ``.ipynb`` to labeled plaintext for structure LLMs (blank template parsing).

    Preserves cell order and cell type so the model can infer distinct questions.
    """
    try:
        nb = json.loads(ipynb_bytes.decode("utf-8", errors="replace"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return ""
    parts: list[str] = []
    for i, cell in enumerate(nb.get("cells") or []):
        ct = str(cell.get("cell_type") or "")
        src = _cell_source(cell).strip()
        if not src:
            continue
        parts.append(f"\n--- cell {i} ({ct}) ---\n{src}")
    return "\n".join(parts).strip()


def _cell_source(cell: dict) -> str:
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(src)
    return str(src or "")


def _extract_question_id(source: str) -> str:
    """Pull a short question identifier like ``1.4.1`` from a heading line."""
    head = (source or "")[:800]
    # Prefer dotted numbering (4.1, 2.3.1) when present anywhere in the heading block.
    m = re.search(r"\b(\d+(?:\.\d+)+[a-zA-Z]?)\b", head)
    if m:
        return m.group(1).strip()
    for line in head.splitlines()[:6]:
        m = _QID_RE.search(line)
        if m:
            return m.group(1).strip()
    return ""


def _slug_from_instruction_text(source: str, ordinal: int) -> str:
    """Stable id from heading / instruction prose when no numbering exists (Week-4 style labs)."""
    lines = (source or "").strip().splitlines()
    head = lines[0] if lines else ""
    t = re.sub(r"^#{1,6}\s*", "", head).strip()
    t = re.sub(r"^\*+\s*|\s*\*+$", "", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"[^\w\s-]", " ", t, flags=re.UNICODE)
    t = re.sub(r"\s+", "_", t).strip("_").lower()[:72]
    if len(t) >= 6:
        return t
    return f"instr_{ordinal}"


def resolve_question_cell_id(source: str, *, ordinal: int) -> str:
    """Numeric / dotted id when present; else slug from first line; else ``instr_{ordinal}``."""
    nid = _extract_question_id(source)
    if nid:
        return nid
    slug = _slug_from_instruction_text(source, ordinal)
    if slug and not slug.startswith("instr_"):
        return slug
    return f"instr_{ordinal}"


def _markdown_leads_with_substantive_heading(src: str) -> bool:
    """True for ``## Step 4: …``, ``### Choose one column…``, etc. (no ``Question 1.1`` required)."""
    s = (src or "").strip()
    if not s:
        return False
    first = s.splitlines()[0].strip()
    m = _SUBSTANTIVE_HASH_HEADING.match(first)
    if not m:
        return False
    body = m.group(2).strip()
    if len(body) < 6:
        return False
    # Single-line mega heading is usually not a lab step boundary.
    if len(first) > 220 and "\n" not in s[:221]:
        return False
    return True


def _plain_instruction_title_cell(src: str, cell_idx: int) -> bool:
    """Titles like ``Loading CSV Into pandas (5 min)`` without ``#`` markdown."""
    if cell_idx == 0:
        return False
    s = (src or "").strip()
    if not s or len(s) > 600:
        return False
    if "#" in s.splitlines()[0]:
        return False
    first = s.splitlines()[0].strip()
    if _PLAIN_TITLE_WITH_TIME_ESTIMATE.match(first):
        return True
    if len(first) >= 12 and len(first) <= 140 and first.count("\n") == 0:
        if "**" in first or "__" in first:
            return True
    return False


def _markdown_looks_like_prompt_extension(src: str) -> bool:
    """
    Long instructional markdown before the next code cell (rubric text, not student prose).
    """
    s = (src or "").strip()
    if not s:
        return False
    if s.count("```") >= 2:
        return False
    if re.search(
        r"^\s*(import |from \w+ import |def |class |plt\.|sns\.|pd\.read_)",
        s,
        re.MULTILINE,
    ):
        return False
    low = s.lower()
    if _PROMPT_EXTENSION_CUES.search(low):
        return True
    if len(s) > 220 and s.count("\n") <= 14:
        return True
    return False


def _markdown_reads_as_instructor_readme_or_setup(src: str) -> bool:
    """
    Markdown that describes the lab, dataset, or homework setup—not student-authored answers.

    Without this, :func:`build_notebook_qa_chunks` classifies such cells as ``student_text`` and
    appends them to ``response_parts`` right after a ``section_header``, so ``trio.student_response``
    duplicates instructions that belong in ``question`` / ``instructor_context``.
    """
    low = (src or "").lower().strip()
    if not low:
        return False
    cues = (
        "in this section we will",
        "in this section, we will",
        "we will preprocess",
        "we have modified",
        "for the purpose of the homework",
        "for the purpose of this homework",
        "for the homework assignment",
        "practice cleaning your data",
        "help you practice",
        "we modified the data",
        "currently does not have any missing",
        "synthetic insurance data",
        "read through this entire",
        "read through all",
        "before moving on",
        "overview of this section",
    )
    if any(c in low for c in cues):
        return True
    if "**note:**" in low and (
        "homework" in low or "assignment" in low or "practice" in low or "synthetic" in low
    ):
        return True
    if "synthetic" in low and ("nan" in low or "missing values" in low):
        return True
    return False


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


def _line_is_assignment_placeholder(line: str) -> bool:
    t = (line or "").strip()
    if not t.startswith("#") or len(t) > 240:
        return False
    if _ASSIGNMENT_PLACEHOLDER_LINE.match(line):
        return True
    return _has_student_work_marker(t + "\n")


def strip_assignment_placeholder_lines(text: str) -> str:
    """
    Remove full-line scaffold comments such as ``# write code here`` or
    ``# write code for problem 1.1 here`` so chunk text aligns with real student code
    and does not spuriously match the answer key.
    """
    if not (text or "").strip():
        return (text or "").strip()
    kept: list[str] = []
    for line in text.splitlines():
        if _line_is_assignment_placeholder(line):
            continue
        kept.append(line)
    out = "\n".join(kept)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def sanitize_grading_chunks_placeholders(chunks: list[GradingChunk]) -> None:
    """In-place: strip boilerplate ``# write code …`` lines from chunk text and trio fields."""
    for ch in chunks:
        ch.extracted_text = strip_assignment_placeholder_lines(ch.extracted_text or "")
        ev = dict(ch.evidence or {})
        qt = ev.get("question_text")
        if isinstance(qt, str) and qt.strip():
            ev["question_text"] = strip_assignment_placeholder_lines(qt)
        rp = ev.get("response_preview")
        if isinstance(rp, str) and rp.strip():
            ev["response_preview"] = strip_assignment_placeholder_lines(rp)
        trio = ev.get("trio")
        if isinstance(trio, dict):
            trio = dict(trio)
            for k in ("question", "student_response", "instructor_context", "answer_key_segment"):
                v = trio.get(k)
                if isinstance(v, str) and v.strip():
                    trio[k] = strip_assignment_placeholder_lines(v)
            ev["trio"] = trio
        ch.evidence = ev


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
    if _markdown_leads_with_substantive_heading(src):
        return "question"
    if _plain_instruction_title_cell(src, cell_idx):
        return "question"
    if _is_section_label(src):
        return "section_header"
    if _markdown_reads_as_instructor_readme_or_setup(src):
        return "question"
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


# Code-cell scaffolds: instructor prompts the student where to write (blank + student alignment).
_SCAFFOLD_LINE_HINT = re.compile(
    r"(?:"
    r"to\s*do\b|todo\b|fixme\b|tbd\b|"
    r"add\s+(your\s+)?code|insert\s+(your\s+)?code|"
    r"your\s+code\s+(here|below)|write\s+your\s+code|"
    r"fill\s+in|complete\s+(this|the)|replace\s+(this|the)|"
    r"student\s+(code|answer)\s+below|\+{2,}\s*your|\.{3}\s*add"
    r")",
    re.IGNORECASE,
)


def code_cell_has_scaffold_instruction(src: str) -> bool:
    """
    True when a **code** cell signals a student deliverable (TODO / add code / placeholders).

    Used with the **blank** notebook to locate anchor cells; the same indices (ordinal)
    align the **student** notebook for ``trio.student_response``.
    """
    if not (src or "").strip():
        return False
    low = src.lower()
    for line in src.splitlines():
        t = line.strip()
        if not t:
            continue
        if t.startswith("#") or t.startswith("//"):
            if _SCAFFOLD_LINE_HINT.search(line):
                return True
        if _line_is_assignment_placeholder(line):
            return True
    if any(
        k in low
        for k in (
            "add your code",
            "insert your code",
            "add code here",
            "your code below",
        )
    ):
        return True
    return False


def split_code_cell_at_scaffold_for_student_tail(src: str) -> tuple[str, str]:
    """
    Split a scaffold **code** cell into ``(prefix_for_prompt, student_tail)``.

    Uses the **last** matching placeholder / scaffold comment line as the cut so
    ``# setup`` then ``# TODO`` keeps setup in the prompt and code after TODO as the response.
    """
    lines = src.splitlines()
    split_at = -1
    for i, line in enumerate(lines):
        if _line_is_assignment_placeholder(line):
            split_at = i
            continue
        if line.strip().startswith("#") and _SCAFFOLD_LINE_HINT.search(line):
            split_at = i
    if split_at < 0:
        return src.strip(), ""
    rest = "\n".join(lines[split_at + 1 :]).strip()
    pref = "\n".join(lines[: split_at + 1]).strip()
    return pref, rest


def _notebook_cells_list(ipynb_bytes: bytes) -> list[dict[str, Any]]:
    try:
        nb = json.loads(ipynb_bytes.decode("utf-8", errors="replace"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []
    cells = nb.get("cells")
    return cells if isinstance(cells, list) else []


def scaffold_anchor_code_cell_indices(cells: list[dict[str, Any]]) -> list[int]:
    """Indices of **code** cells that carry student scaffold / TODO instructions."""
    out: list[int] = []
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        src = _cell_source(cell)
        if code_cell_has_scaffold_instruction(src):
            out.append(i)
    return out


def _collect_blank_question_for_segment(
    cells: list[dict[str, Any]],
    start_i: int,
    end_i: int,
) -> list[str]:
    """Cells ``start_i`` … ``end_i`` inclusive from the blank; scaffold cell uses prefix only."""
    parts: list[str] = []
    for j in range(start_i, end_i + 1):
        if j < 0 or j >= len(cells):
            continue
        cell = cells[j]
        src = _cell_source(cell).strip()
        if not src:
            continue
        ct = cell.get("cell_type", "")
        if ct == "code" and j == end_i:
            pref, _tail = split_code_cell_at_scaffold_for_student_tail(src)
            parts.append(pref if pref.strip() else src)
        else:
            parts.append(src)
    return parts


def _collect_student_response_for_segment(
    cells: list[dict[str, Any]],
    anchor_i: int,
    next_anchor_i: int,
) -> list[str]:
    """Student work: tail of scaffold cell at ``anchor_i`` plus following cells until next anchor."""
    parts: list[str] = []
    if anchor_i < 0 or anchor_i >= len(cells):
        return parts
    anchor_cell = cells[anchor_i]
    if anchor_cell.get("cell_type") == "code":
        src = _cell_source(anchor_cell)
        _pref, tail = split_code_cell_at_scaffold_for_student_tail(src)
        if tail.strip():
            parts.append(tail)
    end = min(next_anchor_i, len(cells))
    for j in range(anchor_i + 1, end):
        src = _cell_source(cells[j]).strip()
        if src:
            parts.append(src)
    return parts


def try_build_notebook_scaffold_aligned_chunks(
    blank_ipynb_bytes: bytes,
    student_ipynb_bytes: bytes,
    *,
    assignment_id: str,
    student_id: str,
    modality: Modality = Modality.NOTEBOOK,
    task_type: TaskType = TaskType.UNKNOWN,
    max_grading_units: int | None = None,
) -> list[GradingChunk] | None:
    """
    One chunk per **scaffold code cell** shared ordinal between blank and student notebooks.

    Question / instructions come from the blank segment up to and including the scaffold
    prefix; ``student_response`` is the scaffold tail plus following cells until the next
    scaffold anchor. Returns ``None`` when anchors are missing or counts disagree.
    """
    blank_cells = _notebook_cells_list(blank_ipynb_bytes)
    student_cells = _notebook_cells_list(student_ipynb_bytes)
    if not blank_cells or not student_cells:
        return None
    b_idx = scaffold_anchor_code_cell_indices(blank_cells)
    s_idx = scaffold_anchor_code_cell_indices(student_cells)
    if not b_idx or len(b_idx) != len(s_idx):
        _log.info(
            "scaffold align: anchor mismatch or empty (blank=%s student=%s)",
            len(b_idx),
            len(s_idx),
        )
        return None

    units: list[dict[str, Any]] = []
    for k, (ib, isc) in enumerate(zip(b_idx, s_idx, strict=True)):
        prev_b = b_idx[k - 1] if k > 0 else -1
        q_parts = _collect_blank_question_for_segment(blank_cells, prev_b + 1, ib)
        next_s = s_idx[k + 1] if k + 1 < len(s_idx) else len(student_cells)
        r_parts = _collect_student_response_for_segment(student_cells, isc, next_s)
        q_blob = "\n\n".join(q_parts).strip()
        r_blob = "\n\n".join(r_parts).strip()
        if not q_blob and not r_blob:
            continue
        qid = resolve_question_cell_id(q_blob[:1200] if q_blob else f"scaffold_{k}", ordinal=k + 1)
        u = _new_unit(qid)
        u["question_parts"] = q_parts if q_parts else [q_blob or f"(scaffold segment {k + 1})"]
        u["response_parts"] = r_parts if r_parts else ([] if not r_blob else [r_blob])
        u["has_student_content"] = bool(r_blob.strip())
        units.append(u)

    if not units:
        return None

    if max_grading_units is not None and max_grading_units >= 1:
        units = units[:max_grading_units]

    out: list[GradingChunk] = []
    for i, unit in enumerate(units):
        qid = str(unit["question_id"])
        extracted = _unit_to_extracted_text(unit)
        if not extracted:
            continue
        trio = _unit_trio_payload(unit)
        qtxt = strip_assignment_placeholder_lines(
            "\n\n".join(unit["question_parts"]).strip()
        )
        rprev = strip_assignment_placeholder_lines(
            "\n\n".join(unit["response_parts"]).strip()
        )
        out.append(
            GradingChunk(
                chunk_id=f"{student_id}:{assignment_id}:{qid}:scaffold_{i}",
                assignment_id=assignment_id,
                student_id=student_id,
                question_id=qid,
                modality=modality,
                task_type=task_type,
                extracted_text=extracted,
                evidence={
                    "chunker": "notebook_scaffold_anchor_aligned",
                    "question_id": qid,
                    "question_text": qtxt,
                    "response_preview": rprev,
                    "trio": trio,
                    "scaffold_anchor_index": i,
                    "n_scaffold_anchors": len(b_idx),
                    "blank_scaffold_cell_indices": list(b_idx),
                    "student_scaffold_cell_indices": list(s_idx),
                },
            )
        )

    _log.info(
        "notebook_chunker (scaffold anchors): %d anchors → %d chunks for %s",
        len(b_idx),
        len(out),
        assignment_id,
    )
    return out


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
    q = strip_assignment_placeholder_lines(q)
    r = strip_assignment_placeholder_lines(r)
    ctx = strip_assignment_placeholder_lines(ctx)
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

    return strip_assignment_placeholder_lines("\n\n".join(parts).strip())


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
                    qid = resolve_question_cell_id(src, ordinal=len(units) + 1)
                    current = _new_unit(qid)
                    current["question_parts"] = question_prefix + [src]
                    continue
            qid = resolve_question_cell_id(src, ordinal=len(units) + 1)
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
            if _markdown_looks_like_prompt_extension(src):
                current["question_parts"].append(src)
            else:
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
        full = strip_assignment_placeholder_lines("\n\n".join(full_text_parts).strip())
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
        qtxt = strip_assignment_placeholder_lines(
            "\n\n".join(unit["question_parts"]).strip()
        )
        rprev = strip_assignment_placeholder_lines(
            "\n\n".join(unit["response_parts"]).strip()
        )
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
                    "question_text": qtxt,
                    "response_preview": rprev,
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


def build_notebook_question_boundary_chunks(
    ipynb_bytes: bytes,
    *,
    assignment_id: str,
    student_id: str,
    modality: Modality = Modality.NOTEBOOK,
    task_type: TaskType = TaskType.UNKNOWN,
    max_grading_units: int | None = None,
) -> list[GradingChunk]:
    """
    One unit per **question** or **section** heading (flush on each), including
    question-only cells (no student answer required).

    Used to chunk a **blank** instructor template: boundaries follow headings so
    :mod:`template_aligned_notebook_chunks` can align student work by ``question_id``.
    """
    try:
        nb = json.loads(ipynb_bytes.decode("utf-8", errors="replace"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        _log.warning("notebook_chunker: could not parse ipynb JSON (boundary mode)")
        return []

    cells = nb.get("cells", [])
    if not cells:
        return []

    classified: list[tuple[dict, str, str, str]] = []
    for idx, cell in enumerate(cells):
        src = _cell_source(cell)
        role = _classify_cell(cell, idx)
        classified.append((cell, src, cell.get("cell_type", ""), role))

    units: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    preamble_parts: list[str] = []

    def flush_current() -> None:
        nonlocal current
        if current is None:
            return
        if (
            current["question_parts"]
            or current["response_parts"]
            or current.get("context_parts", [])
        ):
            units.append(current)
        current = None

    for _cell, src, _ctype, role in classified:
        if not src.strip() and role == "empty":
            continue

        if role in ("question", "section_header"):
            flush_current()
            qid = resolve_question_cell_id(src, ordinal=len(units) + 1)
            current = _new_unit(qid)
            if preamble_parts:
                current["question_parts"].extend(preamble_parts)
                preamble_parts.clear()
            current["question_parts"].append(src)
            continue

        if role == "preamble":
            preamble_parts.append(src)
            continue

        if current is None:
            current = _new_unit("preamble")
            if preamble_parts:
                current["question_parts"].extend(preamble_parts)
                preamble_parts.clear()

        if role == "student_text":
            if _markdown_looks_like_prompt_extension(src):
                current["question_parts"].append(src)
            else:
                current["response_parts"].append(src)
                current["has_student_content"] = True
        elif role == "student_code":
            current["response_parts"].append(src)
            current["has_student_content"] = True
        elif role == "instructor_code":
            inst, stud = _split_instructor_prefix_and_student_suffix(src)
            if stud.strip():
                if inst.strip():
                    current["context_parts"].append(inst.strip())
                current["response_parts"].append(stud.strip())
                current["has_student_content"] = True
            else:
                current["context_parts"].append(src)
        elif role == "test_code":
            current["context_parts"].append(src)

    flush_current()

    if preamble_parts and not units:
        u = _new_unit("preamble")
        u["question_parts"].extend(preamble_parts)
        units.append(u)
        preamble_parts.clear()

    if not units:
        full_text_parts: list[str] = []
        for _cell, src, _ctype, _role in classified:
            s = src.strip()
            if s:
                full_text_parts.append(s)
        full = strip_assignment_placeholder_lines("\n\n".join(full_text_parts).strip())
        if full:
            units = [
                {
                    "question_id": "full",
                    "question_parts": ["(full notebook — no question headings detected)"],
                    "response_parts": [full],
                    "context_parts": [],
                    "has_student_content": True,
                }
            ]

    if not units:
        return []

    if max_grading_units is not None and max_grading_units >= 1:
        units = units[:max_grading_units]

    out: list[GradingChunk] = []
    for unit in units:
        qid = unit["question_id"]
        extracted = _unit_to_extracted_text(unit)
        if not extracted:
            continue
        trio = _unit_trio_payload(unit)
        qtxt = strip_assignment_placeholder_lines(
            "\n\n".join(unit["question_parts"]).strip()
        )
        rprev = strip_assignment_placeholder_lines(
            "\n\n".join(unit["response_parts"]).strip()
        )
        out.append(
            GradingChunk(
                chunk_id=f"{student_id}:{assignment_id}:{qid}:question_boundary",
                assignment_id=assignment_id,
                student_id=student_id,
                question_id=qid,
                modality=modality,
                task_type=task_type,
                extracted_text=extracted,
                evidence={
                    "chunker": "notebook_question_boundaries",
                    "question_id": qid,
                    "question_text": qtxt,
                    "response_preview": rprev,
                    "trio": trio,
                },
            )
        )

    _log.info(
        "notebook_chunker (boundary): %d cells → %d template chunks for %s",
        len(cells),
        len(out),
        assignment_id,
    )
    return out
