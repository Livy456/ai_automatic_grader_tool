"""
Chunk-aware rubric routing: deterministic rules first, optional classifier second.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .schemas import GradingChunk, Modality, RubricType, TaskType


@dataclass
class RubricRouteResult:
    rubric_type: RubricType
    reason: str
    classifier_fallback: bool = False


# (modality, task_type) → rubric — partial; UNKNOWN tasks may need classifier
_DETERMINISTIC: dict[tuple[Modality, TaskType], RubricType] = {
    (Modality.CODE, TaskType.SCAFFOLDED_CODING): RubricType.PROGRAMMING_SCAFFOLDED,
    (Modality.NOTEBOOK, TaskType.SCAFFOLDED_CODING): RubricType.PROGRAMMING_SCAFFOLDED,
    (Modality.WRITTEN, TaskType.FREE_RESPONSE_SHORT): RubricType.FREE_RESPONSE,
    (Modality.WRITTEN, TaskType.FREE_RESPONSE_LONG): RubricType.FREE_RESPONSE,
    (Modality.WRITTEN, TaskType.ORAL_INTERVIEW): RubricType.ORAL_INTERVIEW,
    (Modality.VISUALIZATION, TaskType.EDA_VISUALIZATION): RubricType.EDA_VISUALIZATION,
    (Modality.NOTEBOOK, TaskType.EDA_VISUALIZATION): RubricType.EDA_VISUALIZATION,
    (Modality.CODE, TaskType.PROGRAMMING_ANALYSIS_OPEN): RubricType.PROGRAMMING_ANALYSIS,
    (Modality.NOTEBOOK, TaskType.PROGRAMMING_ANALYSIS_OPEN): RubricType.PROGRAMMING_ANALYSIS,
    (Modality.PROGRAMMING_ANALYSIS, TaskType.PROGRAMMING_ANALYSIS_OPEN): RubricType.PROGRAMMING_ANALYSIS,
    (Modality.VIDEO_ORAL, TaskType.ORAL_INTERVIEW): RubricType.ORAL_INTERVIEW,
    (Modality.VIDEO_ORAL, TaskType.UNKNOWN): RubricType.ORAL_INTERVIEW,
}

_HEURISTIC_TASK_FROM_MODALITY: dict[Modality, TaskType] = {
    Modality.VIDEO_ORAL: TaskType.ORAL_INTERVIEW,
    Modality.VISUALIZATION: TaskType.EDA_VISUALIZATION,
}


ClassifierFn = Callable[[GradingChunk], RubricRouteResult | None]


def route_rubric(
    chunk: GradingChunk,
    *,
    classifier: ClassifierFn | None = None,
    rubric_rows_by_type: dict[RubricType, list[dict]] | None = None,
) -> GradingChunk:
    """
    Mutates ``chunk`` with ``rubric_type``, ``rubric_rows``, ``routing_reason``,
    and optional ``classifier_fallback_used``.
    """
    rows_map = rubric_rows_by_type or {}
    if chunk.rubric_type is not None and chunk.rubric_rows:
        chunk.routing_reason = "instructor_override"
        return chunk

    key = (chunk.modality, chunk.task_type)
    if key in _DETERMINISTIC:
        rt = _DETERMINISTIC[key]
        chunk.rubric_type = rt
        chunk.rubric_rows = list(rows_map.get(rt, []))
        chunk.routing_reason = f"deterministic:{key[0].value}+{key[1].value}"
        return chunk

    if chunk.task_type == TaskType.UNKNOWN:
        hinted = _HEURISTIC_TASK_FROM_MODALITY.get(chunk.modality)
        if hinted is not None:
            key2 = (chunk.modality, hinted)
            if key2 in _DETERMINISTIC:
                chunk.task_type = hinted
                rt = _DETERMINISTIC[key2]
                chunk.rubric_type = rt
                chunk.rubric_rows = list(rows_map.get(rt, []))
                chunk.routing_reason = f"heuristic_modality:{chunk.modality.value}"
                return chunk

    if classifier:
        res = classifier(chunk)
        if res:
            chunk.rubric_type = res.rubric_type
            chunk.rubric_rows = list(rows_map.get(res.rubric_type, []))
            chunk.routing_reason = res.reason
            chunk.classifier_fallback_used = res.classifier_fallback
            return chunk

    # Safe default: free response
    chunk.rubric_type = RubricType.FREE_RESPONSE
    chunk.rubric_rows = list(rows_map.get(RubricType.FREE_RESPONSE, []))
    chunk.routing_reason = "default_free_response"
    return chunk
