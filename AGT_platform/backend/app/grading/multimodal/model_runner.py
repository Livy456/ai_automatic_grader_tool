"""
Protocol for M models × k samples per chunk.

``MultiModelChunkRunner`` uses :func:`app.grading.llm_router.build_grading_clients`
(primary Ollama + ``GRADING_MODEL_2`` + ``GRADING_MODEL_3``) and draws
``GRADING_SAMPLES_PER_MODEL`` stochastic samples per model at
``GRADING_SAMPLE_TEMPERATURE`` — same knobs as the legacy entropy grading path.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Protocol

from app.config import Config
from app.grading.llm_router import ChatClient, build_grading_clients

from .schemas import GradingChunk, SampledChunkGrade

_log = logging.getLogger(__name__)

ClientBuilder = Callable[[Config], list[tuple[ChatClient, str]]]


class ChunkModelRunner(Protocol):
    """M models × k samples per chunk; returns raw model outputs for parsing + entropy."""

    def run_chunk_samples(
        self,
        chunk: GradingChunk,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> list[SampledChunkGrade]: ...


_DEFAULT_MOCK_RESPONSE = json.dumps({
    "rubric_type": "free_response",
    "criterion_scores": [
        {"name": "Conceptual Correctness", "score": 3, "max_points": 4},
        {"name": "Evidence & Justification", "score": 2, "max_points": 3},
    ],
    "criterion_justifications": [
        "Student demonstrates solid understanding of the core concept with minor gaps.",
        "Two concrete examples cited from the reading; could be more specific.",
    ],
    "total_score": 5,
    "normalized_score": 0.71,
    "confidence_note": "Evidence is clear for both criteria.",
    "review_flag": False,
})


class MockChunkModelRunner:
    """Deterministic stub for tests."""

    def __init__(self, responses: list[str] | None = None):
        self._responses = responses or [_DEFAULT_MOCK_RESPONSE]

    def run_chunk_samples(
        self,
        chunk: GradingChunk,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> list[SampledChunkGrade]:
        out: list[SampledChunkGrade] = []
        for i, raw in enumerate(self._responses):
            out.append(
                SampledChunkGrade(
                    model_id="mock",
                    sample_index=i,
                    raw_text=raw,
                    parsed=None,
                    parse_ok=False,
                    parse_warnings=[],
                )
            )
        return out


class MultiModelChunkRunner:
    """
    For each configured grading client (up to three when ``GRADING_MODEL_2`` /
    ``GRADING_MODEL_3`` are set), run ``GRADING_SAMPLES_PER_MODEL`` ``chat_json`` calls.

    Semantic entropy over parsed outcomes is computed in
    :class:`MultimodalGradingPipeline` from cluster assignments of these samples.
    """

    def __init__(
        self,
        cfg: Config,
        *,
        build_clients: ClientBuilder | None = None,
    ):
        self._cfg = cfg
        self._build_clients: ClientBuilder = build_clients or build_grading_clients

    @property
    def app_config(self) -> Config:
        return self._cfg

    def run_chunk_samples(
        self,
        chunk: GradingChunk,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> list[SampledChunkGrade]:
        clients = self._build_clients(self._cfg)
        k = max(1, int(getattr(self._cfg, "GRADING_SAMPLES_PER_MODEL", 1)))
        temp = float(getattr(self._cfg, "GRADING_SAMPLE_TEMPERATURE", 0.3))

        if len(clients) < 3:
            _log.info(
                "Multimodal grading has %s model(s); configure GRADING_MODEL_2 and "
                "GRADING_MODEL_3 for three models (primary Ollama is always included).",
                len(clients),
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        out: list[SampledChunkGrade] = []
        idx = 0
        for client, model_label in clients:
            for _rep in range(k):
                raw_text = ""
                try:
                    obj = client.chat_json(messages, temperature=temp)
                    raw_text = json.dumps(obj, ensure_ascii=True, default=str)
                except Exception:
                    _log.warning(
                        "chunk sample failed chunk_id=%s model=%s rep=%s/%s",
                        chunk.chunk_id,
                        model_label,
                        _rep + 1,
                        k,
                        exc_info=True,
                    )
                out.append(
                    SampledChunkGrade(
                        model_id=model_label,
                        sample_index=idx,
                        raw_text=raw_text,
                        parsed=None,
                        parse_ok=False,
                        parse_warnings=[],
                    )
                )
                idx += 1
        return out
