"""
Multimodal grading pipeline: ingestion → chunking → rubric routing → per-chunk grading →
uncertainty → aggregation → output.

**Scope:** This package owns **unit-level** multimodal grading (notebooks, PDFs, mixed
artifacts). Shared helpers that are also used outside multimodal (e.g. ``submission_chunks``,
``grading_units``, ``app.grading.rag_embeddings.compute_submission_embedding``) stay under
``app.grading`` to avoid circular imports.

**Celery / DB grading:** :mod:`app.tasks` calls
:func:`~app.grading.multimodal.course_multimodal_runner.run_db_submission_multimodal_pipeline`
and :func:`~app.grading.multimodal.course_multimodal_runner.run_standalone_multimodal_pipeline`
(both wrap :class:`MultimodalGradingPipeline`). Local multimodal runs use the same factory;
tests live under ``tests/test_multimodal_pipeline.py``.

See ``AGT_platform/backend/docs/multimodal_grading_pipeline.md`` for architecture.
"""

from __future__ import annotations

from .grading_output import multimodal_assignment_to_grading_dict
from .model_runner import ChunkModelRunner, MultiModelChunkRunner
from .semantic_confidence import (
    aggregate_assignment_confidence,
    cluster_assignment,
    compute_semantic_entropy,
    estimate_cluster_distribution,
    normalize_entropy_to_confidence,
    summarize_chunk_confidence_from_counts,
)
from .pipeline import (
    MultimodalGradingPipeline,
    PipelineArtifactStore,
    build_envelope_from_plaintext,
    create_multimodal_pipeline_from_app_config,
)
from .schemas import (
    AssignmentGradeResult,
    ChunkGradeOutcome,
    GradingChunk,
    Modality,
    MultimodalGradingConfig,
    ParsedChunkGrade,
    ReviewStatus,
    RubricType,
    SampledChunkGrade,
    TaskType,
)

__all__ = [
    "aggregate_assignment_confidence",
    "AssignmentGradeResult",
    "ChunkGradeOutcome",
    "ChunkModelRunner",
    "cluster_assignment",
    "compute_semantic_entropy",
    "estimate_cluster_distribution",
    "GradingChunk",
    "MultiModelChunkRunner",
    "multimodal_assignment_to_grading_dict",
    "normalize_entropy_to_confidence",
    "summarize_chunk_confidence_from_counts",
    "Modality",
    "MultimodalGradingConfig",
    "MultimodalGradingPipeline",
    "ParsedChunkGrade",
    "PipelineArtifactStore",
    "ReviewStatus",
    "RubricType",
    "SampledChunkGrade",
    "TaskType",
    "build_envelope_from_plaintext",
    "create_multimodal_pipeline_from_app_config",
]
