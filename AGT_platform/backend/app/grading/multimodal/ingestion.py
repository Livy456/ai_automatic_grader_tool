"""
Ingestion: normalize raw assignment artifacts into an envelope for chunking.

This stage is intentionally shallow: it defines the contract. LMS-specific
fetching belongs in adapters above this layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class IngestionEnvelope:
    """Normalized submission after ingestion."""

    assignment_id: str
    student_id: str
    artifacts: dict[str, Any]
    """e.g. ipynb_bytes, pdf_bytes, mp4_ref, transcripts, rubric_bundle refs"""
    extracted_plaintext: str = ""
    modality_hints: dict[str, Any] = field(default_factory=dict)


def ingest_raw_submission(
    *,
    assignment_id: str,
    student_id: str,
    artifacts: dict[str, Any],
    extracted_plaintext: str = "",
    modality_hints: dict[str, Any] | None = None,
) -> IngestionEnvelope:
    """Build an envelope; callers run PDF/notebook extractors before this if needed."""
    return IngestionEnvelope(
        assignment_id=assignment_id,
        student_id=student_id,
        artifacts=dict(artifacts),
        extracted_plaintext=extracted_plaintext,
        modality_hints=dict(modality_hints or {}),
    )
