"""
Multimodal grading pipeline: ingestion → chunking → rubric routing → grading →
uncertainty → aggregation → output.

See ``AGT_platform/backend/docs/multimodal_grading_pipeline.md`` for architecture.
"""

from __future__ import annotations

from .grading_output import multimodal_assignment_to_grading_dict
from .model_runner import ChunkModelRunner, MockChunkModelRunner, MultiModelChunkRunner
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
    "AssignmentGradeResult",
    "ChunkGradeOutcome",
    "ChunkModelRunner",
    "GradingChunk",
    "MockChunkModelRunner",
    "MultiModelChunkRunner",
    "multimodal_assignment_to_grading_dict",
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
