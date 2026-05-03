"""
Chunk-aware rubric routing: deterministic rules first, optional classifier second.

Notebook / ``.ipynb``-sourced chunks (see :func:`_chunk_is_ipynb_submission`) are routed only
to **Scaffolded Coding** (``programming_scaffolded``) or **Open-Ended EDA**
(``eda_visualization``), never to the prose **free_response** rubric or
``programming_analysis``.
"""

from __future__ import annotations

import re
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
    # Notebooks use the "Open-Ended EDA" rubric section only (not programming_analysis).
    (Modality.NOTEBOOK, TaskType.PROGRAMMING_ANALYSIS_OPEN): RubricType.EDA_VISUALIZATION,
    (Modality.PROGRAMMING_ANALYSIS, TaskType.PROGRAMMING_ANALYSIS_OPEN): RubricType.PROGRAMMING_ANALYSIS,
    (Modality.VIDEO_ORAL, TaskType.ORAL_INTERVIEW): RubricType.ORAL_INTERVIEW,
    (Modality.VIDEO_ORAL, TaskType.UNKNOWN): RubricType.ORAL_INTERVIEW,
}

_STRONG_EDA_SIGNAL = re.compile(
    r"\b("
    r"plt\.|pyplot|matplotlib|seaborn|sns\.|plotly|plotly\.express|px\.|"
    r"\.plot\(|scatter\(|hist\(|histogram|heatmap|subplot|subplots\(|figure\(|"
    r"ggplot|bokeh|altair"
    r")\b",
    re.IGNORECASE,
)

_MEDIUM_EDA_SIGNAL = re.compile(
    r"\b("
    r"groupby\(|\.describe\(|\.corr\(|value_counts\(|crosstab\(|pivot_table|"
    r"exploratory|\beda\b|visuali[sz]ation|data\s+wrangling"
    r")\b",
    re.IGNORECASE,
)

_HEURISTIC_TASK_FROM_MODALITY: dict[Modality, TaskType] = {
    Modality.VIDEO_ORAL: TaskType.ORAL_INTERVIEW,
    Modality.VISUALIZATION: TaskType.EDA_VISUALIZATION,
}


ClassifierFn = Callable[[GradingChunk], RubricRouteResult | None]


def _chunk_is_ipynb_submission(chunk: GradingChunk) -> bool:
    """True for units produced from ``.ipynb`` (cell-order chunker, OpenAI trio frontload, etc.)."""
    ev = chunk.evidence or {}
    if ev.get("chunker") == "notebook_cell_order":
        return True
    if ev.get("chunker") == "blank_template_aligned_notebook":
        return True
    if ev.get("chunker") == "blank_llm_question_aligned_notebook":
        return True
    if ev.get("chunker") == "blank_scaffold_aligned_notebook":
        return True
    if ev.get("_blank_template_trio"):
        return True
    if ev.get("_openai_trio_rag_frontload"):
        return True
    if chunk.modality == Modality.NOTEBOOK:
        return True
    return False


def _notebook_ipynb_grading_blob(chunk: GradingChunk) -> str:
    parts: list[str] = [chunk.extracted_text or ""]
    ev = chunk.evidence or {}
    trio = ev.get("trio")
    if isinstance(trio, dict):
        for k in ("question", "student_response", "answer_key_segment"):
            parts.append(str(trio.get(k) or ""))
    return "\n".join(parts)


def _notebook_ipynb_pick_scaffolded_vs_eda(chunk: GradingChunk) -> RubricType:
    """Choose between programming_scaffolded and eda_visualization (Open-Ended EDA section)."""
    blob = _notebook_ipynb_grading_blob(chunk)
    if _STRONG_EDA_SIGNAL.search(blob):
        return RubricType.EDA_VISUALIZATION
    if len(_MEDIUM_EDA_SIGNAL.findall(blob)) >= 2:
        return RubricType.EDA_VISUALIZATION
    return RubricType.PROGRAMMING_SCAFFOLDED


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
    ipynb = _chunk_is_ipynb_submission(chunk)
    rr = str(chunk.routing_reason or "")
    if chunk.rubric_type is not None:
        if chunk.rubric_rows:
            return chunk
        # Custom plan: name resolution can yield [] while type is fixed — refill full template.
        if rr.startswith("custom_rubric"):
            full = list(rows_map.get(chunk.rubric_type, []))
            if full:
                chunk.rubric_rows = full
                return chunk
            return chunk

    key = (chunk.modality, chunk.task_type)
    if key in _DETERMINISTIC:
        rt = _DETERMINISTIC[key]
        if ipynb and rt not in (
            RubricType.PROGRAMMING_SCAFFOLDED,
            RubricType.EDA_VISUALIZATION,
        ):
            rt = _notebook_ipynb_pick_scaffolded_vs_eda(chunk)
            chunk.routing_reason = (
                f"deterministic:{key[0].value}+{key[1].value};notebook_ipynb_adjusted"
            )
        else:
            chunk.routing_reason = f"deterministic:{key[0].value}+{key[1].value}"
        chunk.rubric_type = rt
        chunk.rubric_rows = list(rows_map.get(rt, []))
        return chunk

    if chunk.task_type == TaskType.UNKNOWN:
        hinted = _HEURISTIC_TASK_FROM_MODALITY.get(chunk.modality)
        if hinted is not None:
            key2 = (chunk.modality, hinted)
            if key2 in _DETERMINISTIC:
                chunk.task_type = hinted
                rt = _DETERMINISTIC[key2]
                if ipynb and rt not in (
                    RubricType.PROGRAMMING_SCAFFOLDED,
                    RubricType.EDA_VISUALIZATION,
                ):
                    rt = _notebook_ipynb_pick_scaffolded_vs_eda(chunk)
                    chunk.routing_reason = (
                        f"heuristic_modality:{chunk.modality.value};notebook_ipynb_adjusted"
                    )
                else:
                    chunk.routing_reason = f"heuristic_modality:{chunk.modality.value}"
                chunk.rubric_type = rt
                chunk.rubric_rows = list(rows_map.get(rt, []))
                return chunk

    if classifier:
        res = classifier(chunk)
        if res:
            rt = res.rubric_type
            fb = res.classifier_fallback
            reason = res.reason
            if ipynb and rt not in (
                RubricType.PROGRAMMING_SCAFFOLDED,
                RubricType.EDA_VISUALIZATION,
            ):
                rt = _notebook_ipynb_pick_scaffolded_vs_eda(chunk)
                reason = f"{reason};notebook_ipynb_coerced"
            chunk.rubric_type = rt
            chunk.rubric_rows = list(rows_map.get(rt, []))
            chunk.routing_reason = reason
            chunk.classifier_fallback_used = fb
            return chunk

    if ipynb:
        rt = _notebook_ipynb_pick_scaffolded_vs_eda(chunk)
        chunk.rubric_type = rt
        chunk.rubric_rows = list(rows_map.get(rt, []))
        chunk.routing_reason = "default_notebook_ipynb_heuristic"
        return chunk

    chunk.rubric_type = RubricType.FREE_RESPONSE
    chunk.rubric_rows = list(rows_map.get(RubricType.FREE_RESPONSE, []))
    chunk.routing_reason = "default_free_response"
    return chunk
