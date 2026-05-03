"""
Protocol for k samples per chunk.

``MultiModelChunkRunner`` uses :func:`app.grading.llm_router.build_multimodal_grading_clients`
(OpenAI-only for per-chunk grading) and draws
``MULTIMODAL_SAMPLES_PER_MODEL`` stochastic samples **per client** at
``GRADING_SAMPLE_TEMPERATURE``.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Protocol

from app.config import Config
from app.grading.llm_router import ChatClient, build_multimodal_grading_clients

from .schemas import GradingChunk, SampledChunkGrade

_log = logging.getLogger(__name__)

ClientBuilder = Callable[[Config], list[tuple[ChatClient, str]]]


class ChunkModelRunner(Protocol):
    """k samples per chunk; returns raw model outputs for parsing + entropy."""

    def run_chunk_samples(
        self,
        chunk: GradingChunk,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> list[SampledChunkGrade]: ...


class MultiModelChunkRunner:
    """
    For each configured grading client, run ``MULTIMODAL_SAMPLES_PER_MODEL``
    ``chat_json`` calls (default 3 for a single primary OpenAI model).

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
        self._build_clients: ClientBuilder = (
            build_clients or build_multimodal_grading_clients
        )

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
        k = max(1, int(getattr(self._cfg, "MULTIMODAL_SAMPLES_PER_MODEL", 5)))
        temp = float(getattr(self._cfg, "GRADING_SAMPLE_TEMPERATURE", 0.3))

        _log.debug(
            "Multimodal grading: %d model(s), %d sample(s) each → %d total calls/chunk",
            len(clients),
            k,
            len(clients) * k,
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
                except Exception as e:
                    _log.warning(
                        "grading_llm_sample_failed (not chunking): chunk_id=%s model=%s "
                        "rep=%s/%s: %s: %s",
                        chunk.chunk_id,
                        model_label,
                        _rep + 1,
                        k,
                        type(e).__name__,
                        e,
                        exc_info=_log.isEnabledFor(logging.DEBUG),
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
