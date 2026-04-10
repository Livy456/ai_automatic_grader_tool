"""
Orchestrator: ingestion → chunking → rubric routing → grading → entropy → aggregation → output.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

from .aggregator import aggregate_assignment, aggregate_chunk_samples
from .chunker import default_chunker_build_units
from .rag_embeddings import build_multimodal_grading_chunks, enrich_chunks_with_rag_embeddings
from .entropy import semantic_entropy_from_cluster_counts
from .ingestion import IngestionEnvelope, ingest_raw_submission
from .model_runner import ChunkModelRunner, MultiModelChunkRunner
from .parser import parse_chunk_grade_json
from .prompts_chunk import SYSTEM_CHUNK_GRADER, build_chunk_grading_prompt
from .review_router import evaluate_chunk_review
from .rubric_router import route_rubric
from .semantic_clusterer import assign_cluster
from .schemas import (
    AssignmentGradeResult,
    ChunkGradeOutcome,
    GradingChunk,
    MultimodalGradingConfig,
    RubricType,
    SampledChunkGrade,
)


@dataclass
class PipelineArtifactStore:
    """Per-stage audit log (in-memory; persist to DB/S3 in production)."""

    stages: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def append(self, stage: str, payload: dict[str, Any]) -> None:
        self.stages.setdefault(stage, []).append(payload)


class MultimodalGradingPipeline:
    def __init__(
        self,
        config: MultimodalGradingConfig,
        runner: ChunkModelRunner,
        *,
        rubric_rows_by_type: dict[RubricType, list[dict]] | None = None,
        classifier: Callable[[GradingChunk], Any] | None = None,
        task_description: str = "",
    ):
        self.config = config
        self.runner = runner
        self.rubric_rows_by_type = rubric_rows_by_type or {}
        self.classifier = classifier
        self.task_description = task_description

    def run(
        self,
        envelope: IngestionEnvelope,
        *,
        artifacts: PipelineArtifactStore | None = None,
    ) -> AssignmentGradeResult:
        art = artifacts or PipelineArtifactStore()
        art.append("ingestion", {"envelope": envelope.artifacts.keys()})

        app_cfg = self._resolve_app_config()
        if app_cfg is not None:
            chunks, chunker_mode = build_multimodal_grading_chunks(envelope, app_cfg)
            enrich_chunks_with_rag_embeddings(chunks, app_cfg)
        else:
            chunks = default_chunker_build_units(envelope)
            chunker_mode = "heuristic_no_app_config"
        art.append(
            "chunking",
            {
                "chunk_ids": [c.chunk_id for c in chunks],
                "chunker_mode": chunker_mode,
                "rag_unit_embeddings": app_cfg is not None,
            },
        )

        chunk_outcomes: list[ChunkGradeOutcome] = []
        for chunk in chunks:
            route_rubric(
                chunk,
                classifier=self.classifier,
                rubric_rows_by_type=self.rubric_rows_by_type,
            )
            art.append(
                "rubric_routing",
                {
                    "chunk_id": chunk.chunk_id,
                    "rubric_type": chunk.rubric_type.value
                    if chunk.rubric_type
                    else None,
                    "reason": chunk.routing_reason,
                },
            )

            user_prompt = build_chunk_grading_prompt(chunk, task_description=self.task_description)
            raw_samples = self.runner.run_chunk_samples(
                chunk,
                system_prompt=SYSTEM_CHUNK_GRADER,
                user_prompt=user_prompt,
            )

            parsed_samples: list[SampledChunkGrade] = []
            cluster_counts: Counter[str] = Counter()

            for s in raw_samples:
                parsed, warns = parse_chunk_grade_json(s.raw_text)
                parse_ok = parsed is not None
                pw = list(warns)
                ck: str | None = None
                if parsed:
                    ck = assign_cluster(parsed)
                    cluster_counts[ck] += 1
                parsed_samples.append(
                    SampledChunkGrade(
                        model_id=s.model_id,
                        sample_index=s.sample_index,
                        raw_text=s.raw_text,
                        parsed=parsed,
                        parse_ok=parse_ok,
                        parse_warnings=pw,
                        cluster_key=ck,
                    )
                )

            se = semantic_entropy_from_cluster_counts(dict(cluster_counts))
            outcome = aggregate_chunk_samples(
                chunk.chunk_id,
                parsed_samples,
                cluster_counts=dict(cluster_counts),
                semantic_entropy=se,
                cfg=self.config,
            )
            outcome = evaluate_chunk_review(outcome, parsed_samples, self.config)
            model_ids = sorted({s.model_id for s in raw_samples})
            meta_spm: int | None = None
            if isinstance(self.runner, MultiModelChunkRunner):
                meta_spm = int(self.runner.app_config.GRADING_SAMPLES_PER_MODEL)
            outcome.stage_artifacts = {
                "system_prompt": SYSTEM_CHUNK_GRADER,
                "user_prompt": user_prompt,
                "raw_sample_count": len(raw_samples),
                "model_ids": model_ids,
                "samples_per_model": meta_spm,
            }
            art.append(
                "grading",
                {
                    "chunk_id": chunk.chunk_id,
                    "semantic_entropy": se,
                    "cluster_counts": dict(cluster_counts),
                    "review": outcome.review_status.value,
                    "model_ids": model_ids,
                    "total_samples": len(raw_samples),
                },
            )
            chunk_outcomes.append(outcome)

        assign = aggregate_assignment(
            envelope.assignment_id,
            envelope.student_id,
            chunk_outcomes,
        )
        assign.stage_artifacts["pipeline_audit"] = art.stages
        art.append("output", {"score": assign.assignment_normalized_score})
        return assign


# Public re-export helper
def build_envelope_from_plaintext(
    *,
    assignment_id: str,
    student_id: str,
    plaintext: str,
    artifact_refs: dict[str, Any] | None = None,
    modality_hints: dict[str, Any] | None = None,
) -> IngestionEnvelope:
    return ingest_raw_submission(
        assignment_id=assignment_id,
        student_id=student_id,
        artifacts=artifact_refs or {},
        extracted_plaintext=plaintext,
        modality_hints=modality_hints,
    )


def create_multimodal_pipeline_from_app_config(
    app_cfg: Any,
    *,
    multimodal_cfg: MultimodalGradingConfig | None = None,
    rubric_rows_by_type: dict[RubricType, list[dict]] | None = None,
    classifier: Callable[[GradingChunk], Any] | None = None,
    task_description: str = "",
) -> MultimodalGradingPipeline:
    """
    Build :class:`MultimodalGradingPipeline` with :class:`MultiModelChunkRunner`.

    Uses ``GRADING_MODEL_2`` / ``GRADING_MODEL_3`` plus primary Ollama from ``app_cfg``,
    and ``GRADING_SAMPLES_PER_MODEL`` / ``GRADING_SAMPLE_TEMPERATURE`` for stochastic
    multi-sample grading; semantic entropy is computed downstream from cluster counts.
    """
    mm = multimodal_cfg or MultimodalGradingConfig()
    runner = MultiModelChunkRunner(app_cfg)
    return MultimodalGradingPipeline(
        mm,
        runner,
        rubric_rows_by_type=rubric_rows_by_type,
        classifier=classifier,
        task_description=task_description,
        app_cfg=app_cfg,
    )
