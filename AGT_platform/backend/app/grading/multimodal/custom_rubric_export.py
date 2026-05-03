"""
Assignment-wide generic rubric selection, per-question criterion views, and ``custom_rubric/`` JSON.

After unit embeddings exist, we pick **one** :class:`RubricType` for the whole assignment
(via mean chunk embedding vs short type anchors, with deterministic fallbacks), then assign
each chunk a filtered subset of that type's rubric rows. Results are cached under
``custom_rubric/`` so re-runs skip re-embedding anchors.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from app.config import Config
from app.grading.rag_embeddings import compute_submission_embedding

from . import rubric_llm_chain as _rubric_llm_chain
from .ingestion import IngestionEnvelope
from .rubric_router import _MEDIUM_EDA_SIGNAL, _STRONG_EDA_SIGNAL
from .schemas import GradingChunk, Modality, RubricType, TaskType

_log = logging.getLogger(__name__)

_SCHEMA_VERSION = 2

# Short anchor texts — embedded with the same backend as submission units for comparability.
_RUBRIC_TYPE_ANCHORS: dict[RubricType, str] = {
    RubricType.PROGRAMMING_SCAFFOLDED: (
        "Jupyter notebook Python programming homework: implement functions, "
        "fill in cells, pass unit tests, structured scaffolded coding assignment."
    ),
    RubricType.EDA_VISUALIZATION: (
        "Exploratory data analysis matplotlib seaborn plots visualization "
        "histogram scatterplot dataframe summarize distributions."
    ),
    RubricType.FREE_RESPONSE: (
        "Written essay short answer reflection argument prose explanation "
        "conceptual reasoning without programming tasks."
    ),
    RubricType.ORAL_INTERVIEW: (
        "Video or audio recorded oral interview spoken response verbal assessment "
        "live presentation answers."
    ),
}

_FOUR_ASSIGNMENT_RUBRICS: frozenset[str] = frozenset(
    t.value for t in (
        RubricType.PROGRAMMING_SCAFFOLDED,
        RubricType.EDA_VISUALIZATION,
        RubricType.FREE_RESPONSE,
        RubricType.ORAL_INTERVIEW,
    )
)

# Criterion ``name`` (lowercased) → tags a chunk must intersect to include that row.
_SCAFFOLD_APPLICABILITY: dict[str, frozenset[str]] = {
    "functional correctness": frozenset({"code"}),
    "logical implementation": frozenset({"code", "logic"}),
    "code quality": frozenset({"code"}),
}

_EDA_APPLICABILITY: dict[str, frozenset[str]] = {
    "problem framing": frozenset({"written", "data"}),
    "data manipulation": frozenset({"code", "data"}),
    "method appropriateness": frozenset({"data", "written", "logic"}),
    "visualization": frozenset({"viz", "plot", "code"}),
    "interpretation": frozenset({"written", "data", "viz"}),
    "insight": frozenset({"written", "data", "viz"}),
    "limitations": frozenset({"written"}),
}

_ORAL_APPLICABILITY: dict[str, frozenset[str]] = {
    "star method": frozenset({"oral", "written"}),
    "technical explanation": frozenset({"oral", "written", "code"}),
    "analytical thinking": frozenset({"oral", "written"}),
    "bias & limitations awareness": frozenset({"oral", "written"}),
    "communication clarity": frozenset({"oral", "written"}),
}

_FREE_RESPONSE_APPLICABILITY: dict[str, frozenset[str]] = {
    "conceptual correctness": frozenset({"written"}),
    "evidence & justification": frozenset({"written"}),
    "depth of understanding": frozenset({"written"}),
    "clarity": frozenset({"written"}),
}


def _norm_suffix(full: str) -> str:
    """Lowercase criterion basename (strip section prefix like ``Scaffolded Coding —``)."""
    s = (full or "").strip()
    for sep in ("—", "–"):
        if sep in s:
            return s.split(sep, 1)[-1].strip().lower()
    if " - " in s:
        return s.rsplit(" - ", 1)[-1].strip().lower()
    return s.lower()


def default_custom_rubric_dir() -> Path:
    """``<repo_root>/custom_rubric`` (sibling of ``RAG_embedding``)."""
    return Path(__file__).resolve().parents[5] / "custom_rubric"


def _custom_rubric_dir(hints: dict[str, Any]) -> Path:
    raw = str(hints.get("custom_rubric_output_dir") or "").strip()
    if raw:
        return Path(raw).expanduser()
    envd = os.getenv("MULTIMODAL_CUSTOM_RUBRIC_OUTPUT_DIR", "").strip()
    if envd:
        return Path(envd).expanduser()
    return default_custom_rubric_dir()


def _safe_filename_stem(assignment_id: str) -> str:
    s = re.sub(r"[^\w\-.]+", "_", (assignment_id or "assignment").strip())
    return s.strip("._") or "assignment"


def _mean_unit_embedding(chunks: list[GradingChunk]) -> np.ndarray | None:
    vecs: list[np.ndarray] = []
    for ch in chunks:
        bundle = (ch.evidence or {}).get("rag_embedding_bundle")
        if not isinstance(bundle, dict):
            continue
        emb = bundle.get("embedding")
        if not isinstance(emb, list) or len(emb) < 8:
            continue
        vecs.append(np.asarray(emb, dtype=np.float64))
    if not vecs:
        return None
    return np.mean(np.stack(vecs, axis=0), axis=0)


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v
    return v / n


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
    if na < 1e-12:
        return 0.0
    return float(np.dot(a, b) / na)


def _infer_chunk_tags(chunk: GradingChunk) -> set[str]:
    """Mechanical tags for applicability (no LLM)."""
    blob = (chunk.extracted_text or "") + "\n" + json.dumps(
        chunk.evidence or {}, default=str
    )[:4000]
    low = blob.lower()
    tags: set[str] = set()
    if any(
        x in low
        for x in (
            "def ",
            "import ",
            "class ",
            "```",
            ".apply(",
            "pandas.",
            "np.",
            "plt.",
        )
    ):
        tags.add("code")
    if _STRONG_EDA_SIGNAL.search(blob) or _MEDIUM_EDA_SIGNAL.search(blob):
        tags.add("viz")
        tags.add("plot")
    if any(
        x in low
        for x in (
            "dataframe",
            "read_csv",
            "dropna",
            "fillna",
            "merge(",
            "groupby",
            "describe(",
        )
    ):
        tags.add("data")
    if any(x in low for x in ("theorem", "prove", "lemma", "algorithm", "complexity", "o(n)")):
        tags.add("logic")
    if len(low.split()) > 120 and not tags.intersection({"code", "viz"}):
        tags.add("written")
    if chunk.modality == Modality.VIDEO_ORAL or "transcript" in (chunk.evidence or {}):
        tags.add("oral")
    if not tags:
        tags.add("written")
    return tags


def _applicability_map(rt: RubricType) -> dict[str, frozenset[str]]:
    if rt == RubricType.PROGRAMMING_SCAFFOLDED:
        return _SCAFFOLD_APPLICABILITY
    if rt == RubricType.EDA_VISUALIZATION:
        return _EDA_APPLICABILITY
    if rt == RubricType.FREE_RESPONSE:
        return _FREE_RESPONSE_APPLICABILITY
    if rt == RubricType.ORAL_INTERVIEW:
        return _ORAL_APPLICABILITY
    return {}


def _filter_rows_for_chunk(
    rt: RubricType, all_rows: list[dict[str, Any]], chunk: GradingChunk
) -> list[dict[str, Any]]:
    """Drop criteria whose applicability tags do not intersect chunk tags (auditable)."""
    tags = _infer_chunk_tags(chunk)
    app = _applicability_map(rt)
    if not app or not all_rows:
        return [copy.deepcopy(r) for r in all_rows]
    kept: list[dict[str, Any]] = []
    for row in all_rows:
        nm = str(row.get("name") or row.get("criterion") or "").strip()
        base = _norm_suffix(nm)
        need = app.get(base)
        if need is None:
            # Unknown row name: include (same as legacy full rubric) but log once.
            kept.append(copy.deepcopy(row))
            continue
        if tags & need:
            kept.append(copy.deepcopy(row))
    return kept if kept else [copy.deepcopy(all_rows[0])]


def _match_template_row(template_rows: list[dict[str, Any]], want: str) -> dict[str, Any] | None:
    """Match by full ``name``, case-insensitive full name, or basename after ``—``/``–``."""
    w = str(want).strip()
    if not w:
        return None
    wl = w.lower()
    w_base = _norm_suffix(w)
    for r in template_rows:
        fn = str(r.get("name") or "").strip()
        if not fn:
            continue
        if fn == w or fn.lower() == wl:
            return r
    for r in template_rows:
        fn = str(r.get("name") or "").strip()
        if _norm_suffix(fn) == w_base:
            return r
    return None


def _rows_by_names(
    template_rows: list[dict[str, Any]], names: list[str]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for n in names:
        row = _match_template_row(template_rows, str(n))
        if row is not None:
            out.append(copy.deepcopy(row))
    return out


def validate_multimodal_custom_rubric(doc: dict[str, Any]) -> list[str]:
    """Return human-readable errors; empty means the plan is usable."""
    errs: list[str] = []
    if not isinstance(doc, dict):
        return ["document is not an object"]
    ver = int(doc.get("schema_version") or 0)
    gt = str(doc.get("generic_rubric_type") or "").strip()
    if not gt:
        errs.append("missing generic_rubric_type")
    elif gt not in _FOUR_ASSIGNMENT_RUBRICS:
        errs.append(
            f"generic_rubric_type {gt!r} must be one of: {sorted(_FOUR_ASSIGNMENT_RUBRICS)}"
        )
    scores = doc.get("anchor_scores")
    if not isinstance(scores, dict):
        scores = {}
    extra = set(scores) - set(_FOUR_ASSIGNMENT_RUBRICS)
    if extra:
        errs.append(f"anchor_scores has disallowed keys: {sorted(extra)}")
    if ver >= 2:
        missing = set(_FOUR_ASSIGNMENT_RUBRICS) - set(scores)
        if missing:
            errs.append(f"anchor_scores missing keys: {sorted(missing)}")
        if gt and not extra and not missing:
            try:
                vals = {k: float(scores[k]) for k in _FOUR_ASSIGNMENT_RUBRICS}
            except (TypeError, ValueError):
                errs.append("anchor_scores values must be numeric")
            else:
                mval = max(vals.values())
                winners = {k for k, v in vals.items() if abs(v - mval) < 1e-9}
                if gt not in winners:
                    errs.append(
                        f"anchor_scores top score(s) {sorted(winners)!r} "
                        f"do not include generic_rubric_type {gt!r}"
                    )
    qr = doc.get("question_rubrics")
    if not isinstance(qr, list) or not qr:
        errs.append("question_rubrics must be a non-empty list")
        return errs
    aid_top = str(doc.get("assignment_id") or "").strip()
    for i, row in enumerate(qr):
        if not isinstance(row, dict):
            errs.append(f"question_rubrics[{i}] must be an object")
            continue
        if not str(row.get("question_id") or "").strip():
            errs.append(f"question_rubrics[{i}] missing question_id")
        if ver >= 2:
            if not str(row.get("chunk_id") or "").strip():
                errs.append(f"question_rubrics[{i}] missing chunk_id")
            ra = str(row.get("assignment_id") or "").strip()
            if not ra and not aid_top:
                errs.append(f"question_rubrics[{i}] missing assignment_id")
        cn = row.get("criterion_names")
        if isinstance(cn, list) and not cn:
            errs.append(f"question_rubrics[{i}] criterion_names is empty")
        elif cn is not None and not isinstance(cn, list):
            errs.append(f"question_rubrics[{i}] criterion_names must be a list")
    return errs


def resolve_assignment_rubric_type(
    chunks: list[GradingChunk],
    envelope: IngestionEnvelope,
    cfg: Config,
) -> tuple[RubricType, str, dict[str, float]]:
    """
    Pick exactly one generic rubric type for ``chunks``.

    Prefer mean unit embedding vs type anchors; fall back to artifact + text heuristics.
    """
    scores: dict[str, float] = {}
    mean = _mean_unit_embedding(chunks)
    skip = str(os.getenv("MULTIMODAL_SKIP_RUBRIC_ANCHOR_EMBED", "")).strip().lower() in (
        "1",
        "true",
        "yes",
    ) or str((envelope.modality_hints or {}).get("skip_rubric_anchor_embed") or "").strip().lower() in (
        "1",
        "true",
        "yes",
    )

    if mean is not None and not skip:
        mean_n = _l2_normalize(mean)
        for rt, anchor in _RUBRIC_TYPE_ANCHORS.items():
            try:
                vec, _src = compute_submission_embedding(anchor, cfg)
            except Exception:
                _log.debug("anchor embed failed for %s", rt.value, exc_info=True)
                continue
            if not isinstance(vec, list) or len(vec) != len(mean):
                continue
            a = np.asarray(vec, dtype=np.float64)
            scores[rt.value] = _cosine(mean_n, _l2_normalize(a))

    if scores:
        best_rt = max(_RUBRIC_TYPE_ANCHORS, key=lambda r: scores.get(r.value, -1.0))
        return best_rt, "rag_mean_cosine_vs_type_anchors", scores

    # --- Deterministic fallbacks (no embedding agreement) ---
    arts = envelope.artifacts or {}
    if arts.get("ipynb"):
        return RubricType.PROGRAMMING_SCAFFOLDED, "fallback_artifact_ipynb", scores
    blob = "\n\n".join((c.extracted_text or "") for c in chunks).lower()
    if "mp4" in arts or "webm" in arts or "wav" in arts or "oral" in blob[:2000]:
        return RubricType.ORAL_INTERVIEW, "fallback_media_oral_heuristic", scores
    if _STRONG_EDA_SIGNAL.search(blob) or len(_MEDIUM_EDA_SIGNAL.findall(blob)) >= 3:
        return RubricType.EDA_VISUALIZATION, "fallback_text_eda_signals", scores
    if any(c.modality == Modality.NOTEBOOK for c in chunks):
        return RubricType.PROGRAMMING_SCAFFOLDED, "fallback_modality_notebook", scores
    return RubricType.FREE_RESPONSE, "fallback_default_free_response", scores


def _build_plan_payload(
    assignment_id: str,
    rt: RubricType,
    reason: str,
    scores: dict[str, float],
    chunks: list[GradingChunk],
    template_rows: list[dict[str, Any]],
    *,
    cot_step1: str = "",
) -> dict[str, Any]:
    """Persisted plan (schema v2): one generic type; per chunk criterion names + ids."""
    four_scores = {k: float(scores.get(k, 0.0)) for k in sorted(_FOUR_ASSIGNMENT_RUBRICS)}
    per_q: list[dict[str, Any]] = []
    for ch in chunks:
        rows = _filter_rows_for_chunk(rt, template_rows, ch)
        names = [str(r.get("name") or "").strip() for r in rows if str(r.get("name") or "").strip()]
        per_q.append(
            {
                "assignment_id": assignment_id,
                "question_id": ch.question_id,
                "chunk_id": ch.chunk_id,
                "criterion_names": names,
            }
        )
    return {
        "schema_version": _SCHEMA_VERSION,
        "assignment_id": assignment_id,
        "generic_rubric_type": rt.value,
        "selection_reason": reason,
        "anchor_scores": four_scores,
        "cot_step1": cot_step1,
        "question_rubrics": per_q,
    }


def _normalize_notebook_chunks_from_artifact(
    envelope: IngestionEnvelope, chunks: list[GradingChunk]
) -> None:
    """If submission includes ``ipynb`` but chunker left modality unknown, fix for routing."""
    arts = envelope.artifacts or {}
    if not arts.get("ipynb"):
        return
    for ch in chunks:
        if ch.modality == Modality.UNKNOWN:
            ch.modality = Modality.NOTEBOOK
        if ch.task_type == TaskType.UNKNOWN:
            ch.task_type = TaskType.SCAFFOLDED_CODING


def apply_custom_rubric_plan_to_chunks(
    chunks: list[GradingChunk],
    envelope: IngestionEnvelope,
    cfg: Config,
    rubric_rows_by_type: dict[RubricType, list[dict[str, Any]]],
    hints: dict[str, Any],
) -> dict[str, Any] | None:
    """
    Load or create ``custom_rubric`` JSON, then stamp each chunk's ``rubric_type``,
    ``rubric_rows``, and ``routing_reason`` so :func:`route_rubric` treats them as overrides.

    Returns the persisted (or loaded) plan dict, or ``None`` if ``rubric_rows_by_type`` is empty.
    """
    if not chunks or not rubric_rows_by_type:
        return None

    env_dir = os.getenv("MULTIMODAL_CUSTOM_RUBRIC_OUTPUT_DIR", "").strip()
    if env_dir and not hints.get("custom_rubric_output_dir"):
        hints["custom_rubric_output_dir"] = env_dir

    _normalize_notebook_chunks_from_artifact(envelope, chunks)

    out_dir = _custom_rubric_dir(hints)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _log.warning("custom_rubric: could not mkdir %s (%s)", out_dir, exc)
        return None

    stem = _safe_filename_stem(envelope.assignment_id)
    path = out_dir / f"{stem}_multimodal_custom_rubric.json"

    plan: dict[str, Any] | None = None
    if path.is_file():
        try:
            plan = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            _log.warning("custom_rubric: corrupt %s (%s); rebuilding", path, exc)
            plan = None

    def _plan_stale(p: dict[str, Any] | None) -> bool:
        if not p:
            return True
        return int(p.get("schema_version") or 0) < _SCHEMA_VERSION

    def _plan_errors(p: dict[str, Any] | None) -> list[str]:
        if not p:
            return ["missing plan"]
        return validate_multimodal_custom_rubric(p)

    errs_load = _plan_errors(plan)
    if plan is not None and (_plan_stale(plan) or errs_load):
        for e in errs_load:
            _log.warning("custom_rubric: invalid or stale plan (%s)", e)
        plan = None

    if plan is None:
        built: dict[str, Any] | None = None
        if _rubric_llm_chain.rubric_llm_chain_enabled(cfg, hints):
            built = _rubric_llm_chain.build_plan_with_llm_chain(
                envelope.assignment_id,
                envelope,
                chunks,
                cfg,
                rubric_rows_by_type,
            )
        if built is not None and not _plan_errors(built):
            plan = built
        if plan is None:
            rt, reason, scores = resolve_assignment_rubric_type(chunks, envelope, cfg)
            template = list(rubric_rows_by_type.get(rt) or [])
            if not template:
                _log.warning(
                    "custom_rubric: no template rows for resolved type %s; using FREE_RESPONSE",
                    rt.value,
                )
                rt = RubricType.FREE_RESPONSE
                template = list(rubric_rows_by_type.get(rt) or [])
            if not template:
                return None
            plan = _build_plan_payload(
                envelope.assignment_id,
                rt,
                reason,
                scores,
                chunks,
                template,
                cot_step1="",
            )
        try:
            path.write_text(json.dumps(plan, indent=2, ensure_ascii=True), encoding="utf-8")
            _log.info("custom_rubric: wrote %s", path)
        except OSError as exc:
            _log.warning("custom_rubric: could not write %s (%s)", path, exc)

    if plan is None:
        return None

    assert plan is not None
    rt_val = str(plan.get("generic_rubric_type") or "").strip()
    rt: RubricType | None = None
    for r in RubricType:
        if r.value == rt_val:
            rt = r
            break
    if rt is None:
        _log.warning("custom_rubric: unknown type %r in %s", rt_val, path)
        return plan

    template_rows = list(rubric_rows_by_type.get(rt) or [])
    if not template_rows:
        return plan

    qr_list = [e for e in (plan.get("question_rubrics") or []) if isinstance(e, dict)]
    by_cid = {str(e.get("chunk_id") or ""): e for e in qr_list}
    by_qid = {str(e.get("question_id") or ""): e for e in qr_list}
    for ch in chunks:
        entry = by_cid.get(str(ch.chunk_id or "")) or by_qid.get(str(ch.question_id or ""))
        if isinstance(entry, dict):
            names = entry.get("criterion_names")
            if isinstance(names, list) and names:
                rows = _rows_by_names(template_rows, [str(x) for x in names])
            else:
                rows = _filter_rows_for_chunk(rt, template_rows, ch)
        else:
            rows = _filter_rows_for_chunk(rt, template_rows, ch)
        if not rows:
            rows = [copy.deepcopy(r) for r in template_rows]
        ch.rubric_type = rt
        ch.rubric_rows = rows
        ch.routing_reason = (
            f"custom_rubric_file:{path.name}"
            if path.is_file()
            else "custom_rubric_built_in_memory"
        )
    hints["custom_rubric_path"] = str(path)
    hints["custom_rubric_plan"] = plan
    return plan
