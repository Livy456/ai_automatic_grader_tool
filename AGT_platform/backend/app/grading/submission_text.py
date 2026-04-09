"""
Extract a single plain-text representation from submission artifacts for RAG / embeddings.

Supports the same artifact keys as the grading pipeline (``ipynb``, ``pdf``, ``txt``, etc.).
"""

from __future__ import annotations

from app.grading.tools import extract_from_ipynb, extract_text_from_pdf


def submission_text_from_artifacts(artifacts: dict[str, bytes]) -> str:
    """
    Concatenate normalized text from all present artifact kinds (stable key order).

    Empty or missing parts are skipped. Large notebooks/PDFs are not truncated here;
    callers may cap length before embedding (see ``Config.RAG_EMBED_MAX_CHARS``).
    """
    chunks: list[str] = []
    order = ("ipynb", "pdf", "txt", "md", "py", "mp4")
    for key in order:
        raw = artifacts.get(key)
        if not raw:
            continue
        try:
            if key == "ipynb":
                nb = extract_from_ipynb(raw)
                code = (nb.get("code") or "").strip()
                md = (nb.get("markdown") or "").strip()
                if code:
                    chunks.append(f"=== NOTEBOOK CODE ({key}) ===\n{code}")
                if md:
                    chunks.append(f"=== NOTEBOOK MARKDOWN ({key}) ===\n{md}")
            elif key == "pdf":
                txt = extract_text_from_pdf(raw).strip()
                if txt:
                    chunks.append(f"=== PDF TEXT ===\n{txt}")
            elif key in ("txt", "md"):
                chunks.append(
                    f"=== {key.upper()} ===\n"
                    + raw.decode("utf-8", errors="replace").strip()
                )
            elif key == "py":
                chunks.append(
                    f"=== PYTHON SOURCE ===\n"
                    + raw.decode("utf-8", errors="replace").strip()
                )
            elif key == "mp4":
                chunks.append(
                    f"=== VIDEO (binary {len(raw)} bytes; no transcript in artifact) ==="
                )
        except Exception:
            chunks.append(f"=== ERROR parsing artifact {key} ===\n")
    if not chunks:
        return ""
    return "\n\n".join(chunks)
