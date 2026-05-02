"""
Extract a single plain-text representation from submission artifacts for RAG / embeddings.

Supports the same artifact keys as the grading pipeline (``ipynb``, ``pdf``, ``txt``, ``docx``,
``csv``, ``xlsx``, ``py``, ``mp4``, etc.).
"""

from __future__ import annotations

from app.grading.artifact_plaintext import artifacts_to_concatenated_plain


def submission_text_from_artifacts(artifacts: dict[str, bytes]) -> str:
    """
    Concatenate normalized text from all present artifact kinds (stable key order).

    Empty or missing parts are skipped. Large notebooks/PDFs are not truncated here;
    callers may cap length before embedding (see ``Config.RAG_EMBED_MAX_CHARS``).
    """
    return artifacts_to_concatenated_plain(artifacts)
