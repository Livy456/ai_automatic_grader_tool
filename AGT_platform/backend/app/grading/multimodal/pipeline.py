"""
Orchestrator: ingestion → chunking (and optional trio LLM) → answer-key alignment → **per-chunk
RAG vectorization** (when ``MULTIMODAL_RAG_EMBED_UNITS`` is on) → rubric routing → LLM grading →
entropy → aggregation → output.

Chunking uses :func:`app.grading.multimodal.rag_embeddings.build_multimodal_grading_chunks`
(notebook cell-order, optional LLM QA on PDF-reflowed text, structured Q/A chunking), unless
``OPENAI_API_KEY`` is set and ``MULTIMODAL_OPENAI_TRIO_RAG_FRONTLOAD`` is not ``off`` (default
**auto** = on) — then
:func:`app.grading.multimodal.openai_trio_rag_frontload.run_openai_trio_rag_frontload`
runs one or more chat calls (overlapping windows when the submission is long) with
``OPENAI_TRIO_RAG_CHAT_MODEL`` (default ``gpt-5.4-nano``) to emit
trio JSON plus OpenAI **Embeddings** for each unit (``OPENAI_TRIO_RAG_EMBEDDING_MODEL``), and
trio relabeling via ``MULTIMODAL_LLM_TRIO_CHUNKING`` is skipped. Otherwise optional
:func:`app.grading.multimodal.rag_embeddings.refine_chunks_trio_with_ollama` uses the structure
model from ``MULTIMODAL_LLM_BACKEND`` (``openai``, Ollama, or Hugging Face / Maverick). Per-chunk
**grading** uses the same ``MULTIMODAL_LLM_BACKEND`` (default **openai** / ``gpt-5.4-nano`` when
``OPENAI_API_KEY`` is set, else **ollama**). RAG embeddings otherwise use
``RAG_EMBEDDING_BACKEND`` (``sentence_transformers`` default, or ``openai`` for the same OpenAI
embeddings API without frontload) / ``SENTENCE_TRANSFORMERS_MODEL``.

**Answer key:** if ``modality_hints["answer_key_plaintext"]`` is empty, the pipeline calls
:func:`app.grading.answer_key_resolve.resolve_answer_key_plaintext` against
``modality_hints["answer_key_dir"]`` or the repository ``answer_key/`` folder.

**Chunk cache:** set ``modality_hints["multimodal_chunk_cache_path"]`` to a JSON file produced
by :func:`app.grading.multimodal.chunk_cache.save_grading_chunks_cache` to skip rebuilding
chunks and (when embeddings are present in the file) skip per-unit embedding calls.

**Trio export:** after chunking, the pipeline writes ``{assignment_id}_trio_chunks.json`` under
``modality_hints["rag_embedding_output_dir"]`` or the repository ``RAG_embedding/`` folder
(slim trio text per chunk, no embedding vectors). Set ``skip_trio_chunks_json_export`` in hints
to disable.

**Agentic trace:** ordered phases are stored on the result as
``stage_artifacts["agentic_workflow"]`` and copied into grading JSON as ``_agentic_workflow``.

**Answer key size:** the string passed into chunk prompts is capped at
``MULTIMODAL_ANSWER_KEY_PROMPT_MAX_CHARS`` (default 18000) to avoid huge prompts that
often cause local Ollama timeouts.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from app.config import Config
from app.grading.answer_key_resolve import resolve_answer_key_plaintext
from app.grading.llm_router import multimodal_structure_llm_trace_label
from app.grading.dataset_resolve import attach_dataset_context_for_notebook

from .aggregator import aggregate_assignment, aggregate_chunk_samples
from .chunk_cache import (
    chunks_have_unit_embeddings,
    load_grading_chunks_cache,
    save_grading_chunks_cache,
)
from .ingestion import IngestionEnvelope, ingest_raw_submission
from .model_runner import ChunkModelRunner, MultiModelChunkRunner
from .parser import parse_chunk_grade_json
from .prompts_chunk import SYSTEM_CHUNK_GRADER, build_chunk_grading_prompt
from .review_router import evaluate_chunk_review
from .rubric_router import route_rubric
from .semantic_confidence import (
    cluster_assignment,
    summarize_chunk_confidence_from_counts,
)
from .openai_trio_rag_frontload import (
    multimodal_openai_trio_rag_frontload_enabled,
    run_openai_trio_rag_frontload,
)
from .rag_embeddings import (
    build_multimodal_grading_chunks,
    enrich_chunks_with_rag_embeddings,
    multimodal_llm_trio_chunking_enabled,
    multimodal_rag_embed_units_enabled,
    refine_chunks_trio_with_ollama,
)
from .schemas import (
    AssignmentGradeResult,
    ChunkGradeOutcome,
    GradingChunk,
    MultimodalGradingConfig,
    RubricType,
    SampledChunkGrade,
)

_log = logging.getLogger(__name__)


def default_answer_key_dir() -> Path:
    """``…/ai-automatic-grader-tool/answer_key`` (repo root)."""
    return Path(__file__).resolve().parents[5] / "answer_key"


def default_rag_embedding_dir() -> Path:
    """``…/ai-automatic-grader-tool/RAG_embedding`` (repo root)."""
    return Path(__file__).resolve().parents[5] / "RAG_embedding"


def _safe_trio_export_stem(assignment_id: str) -> str:
    s = re.sub(r"[^\w\-.]+", "_", (assignment_id or "assignment").strip())
    return s[:180] or "assignment"


def _trio_chunks_export_payload(chunks: list[GradingChunk]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ch in chunks:
        ev = dict(ch.evidence or {})
        trio = ev.get("trio")
        if not isinstance(trio, dict):
            trio = {}
        rows.append(
            {
                "chunk_id": ch.chunk_id,
                "assignment_id": ch.assignment_id,
                "student_id": ch.student_id,
                "question_id": ch.question_id,
                "extracted_text": ch.extracted_text,
                "modality": ch.modality.value,
                "task_type": ch.task_type.value,
                "trio": {
                    "question": str(trio.get("question") or ""),
                    "student_response": str(trio.get("student_response") or ""),
                    "answer_key_segment": str(trio.get("answer_key_segment") or ""),
                    "instructor_context": str(trio.get("instructor_context") or ""),
                },
            }
        )
    return rows


def _try_persist_trio_chunks_json(
    envelope: IngestionEnvelope,
    chunks: list[GradingChunk],
    hints: dict[str, Any],
    wf: Callable[..., None],
) -> None:
    if hints.get("skip_trio_chunks_json_export"):
        return
    raw_dir = hints.get("rag_embedding_output_dir")
    root = Path(str(raw_dir)).expanduser() if raw_dir else default_rag_embedding_dir()
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _log.warning("Could not create RAG_embedding directory %s: %s", root, exc)
        return
    path = root / f"{_safe_trio_export_stem(envelope.assignment_id)}_trio_chunks.json"
    try:
        payload = {
            "assignment_id": envelope.assignment_id,
            "student_id": envelope.student_id,
            "n_chunks": len(chunks),
            "chunks": _trio_chunks_export_payload(chunks),
        }
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        wf("persist_trio_chunks_json", path=str(path))
    except OSError as exc:
        _log.warning("Could not write trio chunks export %s: %s", path, exc)


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
        app_cfg: Any | None = None,
    ):
        self.config = config
        self.runner = runner
        self.rubric_rows_by_type = rubric_rows_by_type or {}
        self.classifier = classifier
        self.task_description = task_description
        self._app_cfg = app_cfg

    def _resolve_app_config(self) -> Any | None:
        if self._app_cfg is not None:
            return self._app_cfg
        if isinstance(self.runner, MultiModelChunkRunner):
            return self.runner.app_config
        return None

    def run(
        self,
        envelope: IngestionEnvelope,
        *,
        artifacts: PipelineArtifactStore | None = None,
    ) -> AssignmentGradeResult:
        art = artifacts or PipelineArtifactStore()
        hints = envelope.modality_hints
        workflow: list[dict[str, Any]] = []

        def wf(phase: str, **extra: Any) -> None:
            row: dict[str, Any] = {"phase": phase}
            row.update(extra)
            workflow.append(row)

        wf(
            "ingest",
            assignment_id=envelope.assignment_id,
            student_id=envelope.student_id,
            artifact_keys=sorted(envelope.artifacts.keys()),
        )

        answer_key_plain = str(hints.get("answer_key_plaintext") or "").strip()
        raw_akd = str(hints.get("answer_key_dir") or "").strip()
        ak_dir = Path(raw_akd).expanduser() if raw_akd else default_answer_key_dir()
        if not answer_key_plain:
            ak_text, ak_name = resolve_answer_key_plaintext(envelope.assignment_id, ak_dir)
            if ak_text.strip():
                answer_key_plain = ak_text.strip()
                hints["answer_key_plaintext"] = ak_text
                if ak_name:
                    hints["answer_key_matched_file"] = ak_name
        wf(
            "resolve_answer_key",
            chars=len(answer_key_plain),
            matched_file=str(hints.get("answer_key_matched_file") or ""),
            search_dir=str(ak_dir),
        )

        require_ak = bool(self.config.require_answer_key) or (
            os.getenv("MULTIMODAL_REQUIRE_ANSWER_KEY", "").strip().lower() in ("1", "true", "yes")
        )
        if require_ak and not answer_key_plain:
            raise ValueError(
                "Multimodal grading requires an answer key or sample response: set "
                "modality_hints['answer_key_plaintext'] or add a matching file under "
                f"{ak_dir} (see resolve_answer_key_plaintext)."
            )

        ingest_meta: dict[str, Any] = {
            "envelope": envelope.artifacts.keys(),
            "answer_key_chars": len(answer_key_plain),
        }
        ak_match = str(hints.get("answer_key_matched_file") or "").strip()
        if ak_match:
            ingest_meta["answer_key_matched_file"] = ak_match
        art.append("ingestion", ingest_meta)

        answer_key_for_prompt = answer_key_plain
        ak_cap = int(os.getenv("MULTIMODAL_ANSWER_KEY_PROMPT_MAX_CHARS", "18000") or 18000)
        ak_cap = max(4000, min(ak_cap, 100_000))
        if len(answer_key_for_prompt) > ak_cap:
            answer_key_for_prompt = answer_key_for_prompt[:ak_cap]

        app_cfg = self._resolve_app_config()

        cache_read = str(hints.get("multimodal_chunk_cache_path") or "").strip()
        cache_write = str(hints.get("multimodal_chunk_cache_write_path") or "").strip()
        cache_p = Path(cache_read).expanduser() if cache_read else None
        chunks: list[GradingChunk]
        chunker_mode: str
        reused_embeddings = False
        openai_trio_rag_frontload_audit: dict[str, Any] = {}
        openai_frontload_ok = False
        loaded: list[GradingChunk] | None = None
        if cache_p is not None and cache_p.is_file():
            loaded = load_grading_chunks_cache(cache_p)
        if loaded:
            chunks = loaded
            chunker_mode = "cached_units"
            reused_embeddings = app_cfg is not None and chunks_have_unit_embeddings(chunks)
            wf(
                "chunk_and_embed",
                chunker_mode=chunker_mode,
                cache_path=str(cache_p),
                n_chunks=len(chunks),
                reused_embeddings=reused_embeddings,
            )
        else:
            if cache_read:
                _log.warning(
                    "multimodal_chunk_cache_path missing or invalid; rebuilding chunks (%s)",
                    cache_read,
                )
            if app_cfg is not None and multimodal_openai_trio_rag_frontload_enabled(app_cfg):
                fl_chunks, fl_audit = run_openai_trio_rag_frontload(
                    envelope, app_cfg, answer_key_for_prompt
                )
                openai_trio_rag_frontload_audit = dict(fl_audit or {})
                if fl_chunks:
                    chunks = fl_chunks
                    chunker_mode = "openai_trio_rag_frontload"
                    openai_frontload_ok = bool(fl_audit.get("ok"))
                    art.append("openai_trio_rag_frontload", openai_trio_rag_frontload_audit)
                else:
                    chunks, chunker_mode = build_multimodal_grading_chunks(envelope, app_cfg)
                    if openai_trio_rag_frontload_audit.get("error"):
                        _log.warning(
                            "OpenAI trio+RAG frontload skipped: %s",
                            openai_trio_rag_frontload_audit.get("error"),
                        )
            else:
                chunks, chunker_mode = build_multimodal_grading_chunks(envelope, app_cfg)
            wf(
                "chunk_and_embed",
                chunker_mode=chunker_mode,
                cache_path=None,
                n_chunks=len(chunks),
                reused_embeddings=False,
                openai_trio_rag_frontload=openai_frontload_ok,
            )

        if app_cfg is not None and chunks and multimodal_llm_trio_chunking_enabled(app_cfg):
            if not openai_frontload_ok:
                refine_chunks_trio_with_ollama(chunks, app_cfg)
                wf(
                    "trio_llm_chunking",
                    n_chunks=len(chunks),
                    model=str(multimodal_structure_llm_trace_label(app_cfg)),
                )
            else:
                wf(
                    "trio_llm_chunking",
                    skipped=True,
                    reason="openai_trio_rag_frontload",
                    n_chunks=len(chunks),
                )

        if openai_frontload_ok and openai_trio_rag_frontload_audit:
            wf(
                "openai_trio_rag_cost",
                chat_model=openai_trio_rag_frontload_audit.get("chat_model"),
                embedding_model=openai_trio_rag_frontload_audit.get("embedding_model"),
                cost_usd=openai_trio_rag_frontload_audit.get("cost_usd"),
                per_chunk_avg_cost_usd=openai_trio_rag_frontload_audit.get(
                    "per_chunk_avg_cost_usd"
                ),
                chat_usage_tokens=openai_trio_rag_frontload_audit.get("chat_usage_tokens"),
                embedding_usage_tokens_est=openai_trio_rag_frontload_audit.get(
                    "embedding_usage_tokens_est"
                ),
                per_chunk_avg_tokens_est=openai_trio_rag_frontload_audit.get(
                    "per_chunk_avg_tokens_est"
                ),
            )

        if app_cfg is not None and answer_key_plain:
            from .answer_key_chunk_enrich import (
                embed_full_answer_key_for_audit,
                enrich_chunks_with_per_question_answer_key,
            )

            enrich_chunks_with_per_question_answer_key(
                chunks, answer_key_plain, app_cfg
            )
            ak_audit = embed_full_answer_key_for_audit(answer_key_plain, app_cfg)
            if ak_audit:
                hints["answer_key_embedding_audit"] = ak_audit
                art.append("answer_key", dict(ak_audit))
                wf("answer_key_vectorize", **dict(ak_audit))

        # Per-chunk RAG vectors (trio + bundle) must exist before LLM grading when enabled.
        embed_cfg = app_cfg if app_cfg is not None else Config()
        rag_embed_ran = False
        if app_cfg is not None and multimodal_rag_embed_units_enabled():
            if (not reused_embeddings) or (not chunks_have_unit_embeddings(chunks)):
                enrich_chunks_with_rag_embeddings(chunks, app_cfg)
                rag_embed_ran = True
            if not chunks_have_unit_embeddings(chunks):
                _log.warning(
                    "multimodal: RAG unit embeddings still missing before grading; forcing embed pass"
                )
                enrich_chunks_with_rag_embeddings(chunks, app_cfg)
                rag_embed_ran = True

        if envelope.artifacts.get("ipynb"):
            attach_dataset_context_for_notebook(envelope, embed_cfg, art)
        dataset_plain = str(hints.get("dataset_context_plaintext") or "").strip()

        if cache_write:
            out_p = Path(cache_write).expanduser()
            try:
                save_grading_chunks_cache(out_p, chunks)
                wf("persist_chunk_cache", path=str(out_p))
            except OSError as exc:
                _log.warning("Could not save multimodal chunk cache to %s: %s", out_p, exc)

        _try_persist_trio_chunks_json(envelope, chunks, hints, wf)

        art.append(
            "chunking",
            {
                "chunk_ids": [c.chunk_id for c in chunks],
                "chunker_mode": chunker_mode,
                "rag_unit_embeddings": rag_embed_ran,
                "reused_cached_embeddings": reused_embeddings and not rag_embed_ran,
            },
        )

        wf(
            "route_rubric_and_grade",
            description=(
                "Per chunk: modality-aware rubric routing, multi-model JSON sampling, "
                "parse + semantic entropy → confidence and normalized scores."
            ),
            n_chunks=len(chunks),
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

            user_prompt = build_chunk_grading_prompt(
                chunk,
                task_description=self.task_description,
                answer_key_text=answer_key_for_prompt,
                dataset_context_text=dataset_plain,
            )
            raw_samples = self.runner.run_chunk_samples(
                chunk,
                system_prompt=SYSTEM_CHUNK_GRADER,
                user_prompt=user_prompt,
            )

            parsed_samples: list[SampledChunkGrade] = []
            cluster_counts: Counter[str] = Counter()

            rubric_mx: dict[str, float] = {}
            for rr in chunk.rubric_rows or []:
                rn = str(rr.get("name") or "").strip()
                if rn:
                    try:
                        rubric_mx[rn] = float(rr.get("max_points") or rr.get("max_score") or 0)
                    except (TypeError, ValueError):
                        pass

            strong = bool(self.config.confidence_clustering_strong_pattern)
            for s in raw_samples:
                parsed, warns = parse_chunk_grade_json(
                    s.raw_text,
                    rubric_max_points=rubric_mx,
                    rubric_rows=list(chunk.rubric_rows or []),
                    invalid_raw_score_policy=str(
                        getattr(self.config, "raw_score_invalid_policy", "regenerate")
                        or "regenerate"
                    ),
                )
                parse_ok = parsed is not None
                pw = list(warns)
                ck: str | None = cluster_assignment(
                    parsed, strong_pattern=strong
                )
                if ck is not None:
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

            co = summarize_chunk_confidence_from_counts(dict(cluster_counts))
            rubric_fb = [
                str(rr.get("name") or "").strip()
                for rr in (chunk.rubric_rows or [])
                if str(rr.get("name") or "").strip()
            ]
            outcome = aggregate_chunk_samples(
                chunk.chunk_id,
                parsed_samples,
                cluster_counts=dict(cluster_counts),
                cfg=self.config,
                rubric_fallback_names=rubric_fb or None,
            )
            sample_details = [
                {
                    "model_id": s.model_id,
                    "sample_index": s.sample_index,
                    "parse_ok": s.parse_ok,
                    "normalized_score": s.parsed.normalized_score
                    if s.parsed
                    else None,
                    "cluster_key": s.cluster_key,
                }
                for s in parsed_samples
            ]
            outcome.stage_artifacts = {
                "system_prompt": SYSTEM_CHUNK_GRADER,
                "user_prompt": user_prompt,
                "raw_sample_count": len(raw_samples),
                "confidence_trace": {
                    "clustering_strong_pattern": strong,
                    "cluster_counts": dict(cluster_counts),
                    "p_hat": co["p_hat"],
                    "semantic_entropy_nats": co["semantic_entropy_nats"],
                    "entropy_max_reference_nats": co["entropy_max_reference_nats"],
                    "ai_confidence": co["ai_confidence"],
                    "n_observed_clusters": co["n_observed_clusters"],
                    "n_valid_samples": co["n_valid_samples"],
                    "samples": sample_details,
                },
            }
            outcome = evaluate_chunk_review(outcome, parsed_samples, self.config)
            trace = outcome.stage_artifacts.get("confidence_trace")
            if isinstance(trace, dict):
                trace["review_status"] = outcome.review_status.value
                trace["review_reasons"] = list(outcome.review_reasons)
            model_ids = sorted({s.model_id for s in raw_samples})
            meta_spm: int | None = None
            if isinstance(self.runner, MultiModelChunkRunner):
                meta_spm = int(
                    getattr(
                        self.runner.app_config,
                        "MULTIMODAL_SAMPLES_PER_MODEL",
                        5,
                    )
                )
            outcome.stage_artifacts["model_ids"] = model_ids
            outcome.stage_artifacts["samples_per_model"] = meta_spm
            art.append(
                "grading",
                {
                    "chunk_id": chunk.chunk_id,
                    "semantic_entropy": co["semantic_entropy_nats"],
                    "ai_confidence": co["ai_confidence"],
                    "entropy_max_reference_nats": co["entropy_max_reference_nats"],
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
        wf(
            "aggregate",
            assignment_normalized_score=assign.assignment_normalized_score,
            review_status=assign.review_status.value,
        )
        assign.stage_artifacts["pipeline_audit"] = art.stages
        assign.stage_artifacts["agentic_workflow"] = workflow
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

    Uses :func:`~app.grading.llm_router.build_multimodal_grading_clients` (primary from
    ``MULTIMODAL_LLM_BACKEND``: ``openai`` (``OPENAI_MULTIMODAL_GRADING_MODEL`` / API key),
    Ollama ``OLLAMA_MODEL``, or Hugging Face repo id, plus optional
    ``GRADING_MODEL_2`` / ``GRADING_MODEL_3``), ``MULTIMODAL_SAMPLES_PER_MODEL``
    and ``GRADING_SAMPLE_TEMPERATURE`` for stochastic multi-sample grading; semantic entropy
    is computed downstream from cluster counts over those samples.
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
