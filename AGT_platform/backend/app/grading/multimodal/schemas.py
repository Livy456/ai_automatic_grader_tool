"""
Typed data structures for the multimodal grading pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Modality(str, Enum):
    NOTEBOOK = "notebook"
    CODE = "code"
    WRITTEN = "written"
    VISUALIZATION = "visualization"
    PROGRAMMING_ANALYSIS = "programming_analysis"
    VIDEO_ORAL = "video_oral"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class TaskType(str, Enum):
    SCAFFOLDED_CODING = "scaffolded_coding"
    FREE_RESPONSE_SHORT = "free_response_short"
    FREE_RESPONSE_LONG = "free_response_long"
    EDA_VISUALIZATION = "eda_visualization"
    PROGRAMMING_ANALYSIS_OPEN = "programming_analysis_open"
    ORAL_INTERVIEW = "oral_interview"
    UNKNOWN = "unknown"


class RubricType(str, Enum):
    PROGRAMMING_SCAFFOLDED = "programming_scaffolded"
    FREE_RESPONSE = "free_response"
    EDA_VISUALIZATION = "eda_visualization"
    PROGRAMMING_ANALYSIS = "programming_analysis"
    ORAL_INTERVIEW = "oral_interview"


class ReviewStatus(str, Enum):
    AUTO_ACCEPTED = "auto_accepted"
    FLAGGED = "flagged"
    ESCALATION = "escalation"


@dataclass
class MultimodalGradingConfig:
    """Configurable thresholds for entropy, disagreement, and review routing."""

    models_count_cap: int = 4
    samples_per_model_cap: int = 8
    semantic_entropy_high: float = 1.0
    score_spread_high: float = 0.12
    criterion_disagreement_high: float = 0.25
    parse_fail_rate_high: float = 0.35
    review_if_any_sample_flag: bool = True
    chunk_score_aggregator: str = "mean"  # mean | median


@dataclass
class GradingChunk:
    """One gradable unit (question or rubric-relevant subpart)."""

    chunk_id: str
    assignment_id: str
    student_id: str
    question_id: str
    modality: Modality
    task_type: TaskType
    extracted_text: str
    rubric_version: str = ""
    parent_chunk_id: str | None = None
    raw_content_ref: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    source_refs: list[dict[str, Any]] = field(default_factory=list)
    rubric_type: RubricType | None = None
    rubric_rows: list[dict[str, Any]] = field(default_factory=list)
    routing_reason: str = ""
    classifier_fallback_used: bool = False

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "assignment_id": self.assignment_id,
            "student_id": self.student_id,
            "question_id": self.question_id,
            "modality": self.modality.value,
            "task_type": self.task_type.value,
            "extracted_text": self.extracted_text,
            "evidence": self.evidence,
            "source_refs": self.source_refs,
            "rubric_type": (self.rubric_type.value if self.rubric_type else None),
            "rubric_version": self.rubric_version,
        }


@dataclass
class CriterionScore:
    name: str
    score: float
    max_points: float
    weight: float = 1.0


@dataclass
class ParsedChunkGrade:
    rubric_type: RubricType | str
    criterion_scores: list[CriterionScore]
    criterion_justifications: list[str]
    total_score: float
    normalized_score: float
    confidence_note: str = ""
    review_flag: bool = False
    parse_warnings: list[str] = field(default_factory=list)


@dataclass
class SampledChunkGrade:
    model_id: str
    sample_index: int
    raw_text: str
    parsed: ParsedChunkGrade | None
    parse_ok: bool
    parse_warnings: list[str] = field(default_factory=list)
    cluster_key: str | None = None


@dataclass
class ChunkGradeOutcome:
    chunk_id: str
    normalized_score_estimate: float
    semantic_entropy_nats: float
    cluster_counts: dict[str, int]
    cluster_distribution: dict[str, float]
    samples: list[SampledChunkGrade]
    criterion_consensus: dict[str, float]
    auxiliary: dict[str, Any] = field(default_factory=dict)
    review_status: ReviewStatus = ReviewStatus.AUTO_ACCEPTED
    review_reasons: list[str] = field(default_factory=list)
    stage_artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass
class AssignmentGradeResult:
    assignment_id: str
    student_id: str
    assignment_normalized_score: float
    chunk_results: list[ChunkGradeOutcome]
    chunk_weights: dict[str, float] = field(default_factory=dict)
    review_status: ReviewStatus = ReviewStatus.AUTO_ACCEPTED
    review_reasons: list[str] = field(default_factory=list)
    stage_artifacts: dict[str, Any] = field(default_factory=dict)
