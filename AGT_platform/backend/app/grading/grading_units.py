"""
Build grading units from :func:`submission_chunks.build_submission_chunks` output.

Units group a **question** line with the **student response** chunks that share its
``pair_id``. Orphan responses (no detected prompt) form a separate unit. Used by the
``chunk_entropy`` grading pipeline with optional per-unit embeddings.
"""

from __future__ import annotations

from typing import Any


def build_grading_units_from_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Return one dict per gradable unit:

    - ``pair_id``: int or ``None``
    - ``question_text``: prompt line(s) or empty / placeholder
    - ``response_text``: concatenated student ``response`` / ``code`` text
    - ``chunk_ids``: sorted ``chunk_index`` values included
    """
    if not chunks:
        return []

    sorted_chunks = sorted(
        chunks,
        key=lambda x: int(x.get("chunk_index", 0)),
    )

    by_pair: dict[Any, dict[str, Any]] = {}
    orphans: list[dict[str, Any]] = []

    def bucket(pid: Any) -> dict[str, Any]:
        if pid not in by_pair:
            by_pair[pid] = {
                "question_parts": [],
                "response_parts": [],
                "chunk_ids": [],
            }
        return by_pair[pid]

    for ch in sorted_chunks:
        role = ch.get("role")
        pid = ch.get("pair_id")
        cid = ch.get("chunk_index")
        text = str(ch.get("text") or "")

        if role == "question":
            if pid is not None:
                b = bucket(pid)
                b["question_parts"].append(text)
                if cid is not None:
                    b["chunk_ids"].append(cid)
        elif role in ("response", "code"):
            if pid is None:
                orphans.append(ch)
            else:
                b = bucket(pid)
                b["response_parts"].append(text)
                if cid is not None:
                    b["chunk_ids"].append(cid)

    units: list[dict[str, Any]] = []

    for pid in sorted((k for k in by_pair if k is not None), key=lambda x: int(x)):
        u = by_pair[pid]
        q = "\n".join(u["question_parts"]).strip()
        r = "\n\n".join(u["response_parts"]).strip()
        cids = sorted({x for x in u["chunk_ids"] if x is not None})
        if not q and not r:
            continue
        units.append(
            {
                "pair_id": pid,
                "question_text": q,
                "response_text": r,
                "chunk_ids": cids,
            }
        )

    if orphans:
        rtext = "\n\n".join(
            str(o.get("text") or "").strip()
            for o in orphans
            if str(o.get("text") or "").strip()
        )
        ocids = sorted(
            {int(o["chunk_index"]) for o in orphans if o.get("chunk_index") is not None}
        )
        if rtext.strip():
            units.insert(
                0,
                {
                    "pair_id": None,
                    "question_text": "(no detected prompt line; preamble or unstructured excerpt)",
                    "response_text": rtext.strip(),
                    "chunk_ids": ocids,
                },
            )

    if not units:
        blob = "\n\n".join(
            str(c.get("text") or "").strip()
            for c in sorted_chunks
            if str(c.get("text") or "").strip()
        )
        if blob.strip():
            units.append(
                {
                    "pair_id": None,
                    "question_text": "",
                    "response_text": blob.strip(),
                    "chunk_ids": sorted(
                        {
                            int(c["chunk_index"])
                            for c in sorted_chunks
                            if c.get("chunk_index") is not None
                        }
                    ),
                }
            )

    return units


def format_unit_for_grader_prompt(unit: dict[str, Any]) -> str:
    q = unit.get("question_text") or ""
    r = unit.get("response_text") or ""
    return (
        "\n\n---\n### Focus for this grading pass\n"
        f"**Question / prompt:**\n{q}\n\n"
        f"**Student work (response or code):**\n{r}\n\n"
        "Apply the rubric to this excerpt. Use score 0 for criteria clearly not "
        "evidenced in this excerpt. Return the standard grading JSON.\n"
    )
