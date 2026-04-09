import hashlib
import logging
import time
from collections import defaultdict
from types import SimpleNamespace
from typing import Any

import numpy as np

from .agent import check_consistency, extract_evidence, grade, plan, score_criterion
from .aggregation import (
    should_route_human_review,
    weighted_overall_confidence,
    weighted_overall_score,
)
from .consistency_rules import run_rule_checks
from .llm_router import (
    build_grading_clients,
    maybe_escalate_grade,
    openai_client_if_configured,
    primary_ollama_client,
)
from .normalization import (
    normalize_submission,
    normalized_to_json_str,
    slice_evidence_for_criterion,
)
from .output_schema import coerce_grading_output_shape
from .semantic_entropy import (
    aggregate_grading_json_samples,
    confidence_from_entropy_natural,
    grading_semantic_fingerprint,
    semantic_entropy_by_model,
    semantic_entropy_natural,
)
from .grading_units import build_grading_units_from_chunks, format_unit_for_grader_prompt
from .rag_embeddings import compute_submission_embedding
from .submission_chunks import build_submission_chunks
from .modality_resolution import (
    augment_prompt_for_modality_profile,
    infer_modality_from_artifacts,
    resolve_modality_profile,
)
from .submission_text import submission_text_from_artifacts
from .tools import extract_from_ipynb, extract_text_from_pdf, run_python_tests, transcribe_video_stub

_log = logging.getLogger(__name__)

# Staged pipeline (stages 2–4) uses only the primary grading client to avoid
# N_models × N_criteria explosive cost. Multi-LLM averaging remains on legacy path only.


def _ensure_submission_artifacts_in_ctx(
    ctx: dict[str, Any], artifacts_bytes: dict[str, bytes]
) -> None:
    """
    Materialize submission text/code into ctx['artifacts'] even when the planner
    omits extract_text / run_tests (common cause of empty context and all-zero grades on PDFs).
    """
    arts = ctx.setdefault("artifacts", {})
    tr = ctx.setdefault("tool_results", {})

    if "pdf" in artifacts_bytes and not (str(arts.get("text") or "").strip()):
        arts["text"] = extract_text_from_pdf(artifacts_bytes["pdf"])

    if "ipynb" in artifacts_bytes:
        if not (arts.get("code") or arts.get("markdown")):
            nb = extract_from_ipynb(artifacts_bytes["ipynb"])
            arts["code"] = nb.get("code") or ""
            arts["markdown"] = nb.get("markdown") or ""

    if not (str(arts.get("text") or "").strip()):
        if "txt" in artifacts_bytes:
            arts["text"] = artifacts_bytes["txt"].decode("utf-8", errors="ignore")
        elif "md" in artifacts_bytes:
            arts["text"] = artifacts_bytes["md"].decode("utf-8", errors="ignore")

    if "py" in artifacts_bytes and "tests" not in tr:
        try:
            tr["tests"] = run_python_tests(artifacts_bytes["py"], "submission.py")
        except Exception:
            _log.debug("run_python_tests skipped after error", exc_info=True)

    if "mp4" in artifacts_bytes and not tr.get("transcript"):
        tr["transcript"] = transcribe_video_stub(artifacts_bytes["mp4"])


def _plaintext_for_modality_profile(
    artifacts_bytes: dict[str, bytes], ctx: dict[str, Any]
) -> str:
    t = submission_text_from_artifacts(artifacts_bytes).strip()
    if t:
        return t
    inner = str((ctx.get("artifacts") or {}).get("text") or "").strip()
    return inner


def _attach_modality_to_result(result: dict[str, Any], ctx: dict[str, Any]) -> None:
    prof = ctx.get("_modality_resolution")
    if not isinstance(prof, dict):
        return
    result["_modality"] = {
        "modality": prof.get("modality"),
        "modality_subtype": prof.get("modality_subtype"),
        "artifact_keys": list(prof.get("artifact_keys") or []),
        "extracted_text_chars": prof.get("extracted_text_chars"),
        "signals": prof.get("signals") if isinstance(prof.get("signals"), dict) else {},
    }


def _submission_context_id(artifacts_bytes: dict) -> str:
    h = hashlib.sha256()
    for kind in sorted(artifacts_bytes.keys()):
        b = artifacts_bytes.get(kind) or b""
        h.update(kind.encode())
        h.update(len(b).to_bytes(8, "big", signed=False))
    return h.hexdigest()[:16]


def _coerce_rubric_items(rubric: list) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in rubric or []:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("criterion") or "Criterion"
        max_pts = item.get("max_points")
        if max_pts is None:
            max_pts = item.get("max_score")
        if max_pts is None:
            max_pts = 100.0
        try:
            max_pts_f = float(max_pts)
        except (TypeError, ValueError):
            max_pts_f = 100.0
        min_pts = item.get("min_points", 0.0)
        try:
            min_pts_f = float(min_pts)
        except (TypeError, ValueError):
            min_pts_f = 0.0
        weight = item.get("weight")
        if weight is None:
            weight = max_pts_f
        try:
            weight_f = float(weight)
        except (TypeError, ValueError):
            weight_f = max_pts_f
        desc = item.get("description") or item.get("text") or ""
        out.append(
            {
                "name": str(name),
                "max_points": max_pts_f,
                "min_points": min_pts_f,
                "weight": weight_f,
                "description": str(desc),
            }
        )
    return out


def _apply_consistency_adjustments(
    criteria_results: list[dict[str, Any]], consistency_out: dict[str, Any]
) -> None:
    by_name = {c["name"]: c for c in criteria_results}
    for adj in consistency_out.get("adjustments") or []:
        if not isinstance(adj, dict):
            continue
        cn = adj.get("criterion_name")
        if cn not in by_name:
            continue
        c = by_name[cn]
        override = adj.get("score_override")
        if override is not None:
            try:
                c["score"] = float(override)
            except (TypeError, ValueError):
                continue
        else:
            try:
                delta = float(adj.get("score_delta") or 0.0)
            except (TypeError, ValueError):
                delta = 0.0
            try:
                c["score"] = float(c["score"]) + delta
            except (TypeError, ValueError):
                c["score"] = delta
        lo = float(c.get("min_points", 0.0))
        hi = float(c.get("max_points", 100.0))
        try:
            sc = float(c["score"])
        except (TypeError, ValueError):
            sc = lo
        c["score"] = max(lo, min(hi, round(sc, 2)))


def _staged_criterion_error_row(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": spec["name"],
        "score": 0.0,
        "max_points": spec["max_points"],
        "min_points": spec["min_points"],
        "confidence": 0.25,
        "rationale": "Scoring failed; defaulted for human review.",
        "evidence": {"quotes": [], "notes": ""},
        "flags": ["SCORING_ERROR"],
    }


def _average_staged_criterion_samples(
    samples: list[dict[str, Any]], spec: dict[str, Any]
) -> dict[str, Any]:
    """Mean scores/confidence across models; merge evidence text (staged multi-LLM)."""
    valid = [s for s in samples if isinstance(s, dict)]
    if not valid:
        return _staged_criterion_error_row(spec)
    if len(valid) == 1:
        return dict(valid[0])

    def _usable(x: dict[str, Any]) -> bool:
        fl = x.get("flags")
        if not isinstance(fl, list):
            return True
        return "SCORING_ERROR" not in fl

    usable = [s for s in valid if _usable(s)]
    merged_from = usable if usable else valid
    n = len(merged_from)
    scores = []
    confs = []
    for s in merged_from:
        try:
            scores.append(float(s.get("score", 0.0)))
        except (TypeError, ValueError):
            scores.append(0.0)
        try:
            confs.append(float(s.get("confidence", 0.5)))
        except (TypeError, ValueError):
            confs.append(0.5)
    avg_score = round(sum(scores) / n, 2)
    avg_conf = round(sum(confs) / n, 2)
    rationales = [
        str(s.get("rationale") or "").strip()
        for s in merged_from
        if str(s.get("rationale") or "").strip()
    ]
    all_quotes: list[str] = []
    all_notes: list[str] = []
    for s in merged_from:
        ev = s.get("evidence") or {}
        if isinstance(ev, dict):
            for q in ev.get("quotes") or []:
                if q is not None and str(q).strip():
                    all_quotes.append(str(q))
            if ev.get("notes"):
                all_notes.append(str(ev["notes"]))
    u0 = merged_from[0]
    fl_flat: list[str] = []
    for s in merged_from:
        for f in s.get("flags") or []:
            fs = str(f)
            if fs and fs not in fl_flat:
                fl_flat.append(fs)
    if "STAGED_MULTI_LLM_AVERAGE" not in fl_flat:
        fl_flat.append("STAGED_MULTI_LLM_AVERAGE")
    return {
        "name": str(u0.get("name") or spec["name"]),
        "score": avg_score,
        "max_points": float(u0.get("max_points", spec["max_points"])),
        "min_points": float(u0.get("min_points", spec["min_points"])),
        "confidence": avg_conf,
        "rationale": " | ".join(rationales),
        "evidence": {
            "quotes": all_quotes,
            "notes": " | ".join(all_notes) if all_notes else "",
        },
        "flags": fl_flat,
    }


def _run_staged_grading_pipeline(
    cfg,
    assignment,
    artifacts_bytes: dict,
    *,
    rubric_text: str | None,
    answer_key_text: str | None,
    assignment_prompt: str | None,
    ctx: dict[str, Any],
) -> dict[str, Any]:
    t0 = time.perf_counter()
    context_id = _submission_context_id(artifacts_bytes)
    meta_stages: dict[str, float] = {}

    rubric = getattr(assignment, "rubric", None) or []
    rubric_items = _coerce_rubric_items(rubric)
    if not rubric_items:
        rubric_items = [
            {
                "name": "Overall",
                "max_points": 100.0,
                "min_points": 0.0,
                "weight": 100.0,
                "description": "Holistic quality when no rubric rows exist.",
            }
        ]
    if assignment_prompt is None:
        assignment_prompt = _build_assignment_prompt(
            assignment, rubric_text, answer_key_text
        )
    prof = ctx.get("_modality_resolution")
    if not isinstance(prof, dict):
        prof = {}
    modalities = list(artifacts_bytes.keys())
    assignment_kind = (
        getattr(assignment, "modality", None)
        or prof.get("modality")
        or infer_modality_from_artifacts(artifacts_bytes)
    )

    grading_clients = build_grading_clients(cfg)
    primary_client, primary_label = grading_clients[0]
    use_staged_multi = bool(getattr(cfg, "STAGED_MULTI_LLM", False)) and len(
        grading_clients
    ) > 1
    scoring_clients: list[tuple[Any, str]] = (
        grading_clients if use_staged_multi else [grading_clients[0]]
    )
    max_chars = cfg.STAGED_PROMPT_MAX_CHARS

    s = time.perf_counter()
    normalize_submission(
        ctx,
        assignment_instruction=assignment_prompt,
        rubric_items=rubric_items,
        modalities=modalities,
        artifacts=artifacts_bytes,
        assignment_kind=assignment_kind,
    )
    meta_stages["normalize"] = round(time.perf_counter() - s, 3)
    _log.info(
        "grading_stage submission=%s stage=normalize duration_s=%.3f",
        context_id,
        meta_stages["normalize"],
    )

    s = time.perf_counter()
    norm_json = normalized_to_json_str(ctx, max_chars=max_chars)
    try:
        evidence_bundle = extract_evidence(
            primary_client, norm_json, assignment_prompt, rubric_items
        )
    except Exception:
        _log.exception(
            "grading_stage submission=%s stage=extract_evidence failed",
            context_id,
        )
        evidence_bundle = {
            "claims": [],
            "code_facts": [],
            "visualization_facts": [],
            "answers_by_question": [],
            "contradictions_spotted": [],
            "_error": "extract_evidence_failed",
        }
    ctx["evidence_bundle"] = evidence_bundle
    meta_stages["extract_evidence"] = round(time.perf_counter() - s, 3)
    _log.info(
        "grading_stage submission=%s stage=extract_evidence duration_s=%.3f",
        context_id,
        meta_stages["extract_evidence"],
    )

    s = time.perf_counter()
    criteria_results: list[dict[str, Any]] = []
    slice_budget = max(4000, max_chars // 2)
    for spec in rubric_items:
        sl = slice_evidence_for_criterion(
            evidence_bundle, spec["name"], max_chars=slice_budget
        )
        per_model: list[dict[str, Any]] = []
        for _sc_client, sc_label in scoring_clients:
            try:
                per_model.append(
                    score_criterion(_sc_client, spec, sl, assignment_prompt)
                )
            except Exception:
                _log.exception(
                    "grading_stage submission=%s stage=score_criterion model=%s criterion=%s",
                    context_id,
                    sc_label,
                    spec["name"],
                )
                per_model.append(_staged_criterion_error_row(spec))
        if len(per_model) > 1:
            one = _average_staged_criterion_samples(per_model, spec)
        else:
            one = per_model[0]
        ev = one.get("evidence")
        if not isinstance(ev, dict):
            ev = {"quotes": [], "notes": str(ev or "")}
        crit_flags = one.get("flags")
        if not isinstance(crit_flags, list):
            crit_flags = []
        try:
            score_v = float(one.get("score", 0.0))
        except (TypeError, ValueError):
            score_v = 0.0
        try:
            conf_v = float(one.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf_v = 0.5
        row = {
            "name": str(one.get("name") or spec["name"]),
            "score": score_v,
            "max_points": float(
                one.get("max_points", spec["max_points"]),
            ),
            "min_points": float(
                one.get("min_points", spec["min_points"]),
            ),
            "confidence": conf_v,
            "rationale": str(one.get("rationale") or ""),
            "evidence": ev,
            "flags": crit_flags,
            "weight": spec["weight"],
        }
        row["score"] = max(
            row["min_points"],
            min(row["max_points"], round(row["score"], 2)),
        )
        criteria_results.append(row)
    meta_stages["score_criteria"] = round(time.perf_counter() - s, 3)
    _log.info(
        "grading_stage submission=%s stage=score_criteria duration_s=%.3f",
        context_id,
        meta_stages["score_criteria"],
    )

    s = time.perf_counter()
    rule_issues = run_rule_checks(criteria_results, cap_points=100.0)
    flags_seed: list[str] = []
    if rule_issues:
        flags_seed.append("RULE_ISSUES")
    consistency_out: dict[str, Any] = {
        "adjustments": [],
        "new_flags": [],
        "contradictions": [],
    }
    try:
        consistency_out = check_consistency(
            primary_client,
            criteria_results,
            evidence_bundle,
            rule_issues,
        )
    except Exception:
        _log.exception(
            "grading_stage submission=%s stage=consistency failed",
            context_id,
        )
    _apply_consistency_adjustments(criteria_results, consistency_out)
    flat_flags = list(dict.fromkeys(flags_seed + (consistency_out.get("new_flags") or [])))
    for ct in consistency_out.get("contradictions") or []:
        if isinstance(ct, dict) and (ct.get("detail") or ct.get("text")):
            flat_flags.append("CONTRADICTION")
            break
    meta_stages["consistency"] = round(time.perf_counter() - s, 3)
    _log.info(
        "grading_stage submission=%s stage=consistency duration_s=%.3f",
        context_id,
        meta_stages["consistency"],
    )

    s = time.perf_counter()
    overall_score = weighted_overall_score(criteria_results)
    overall_conf = weighted_overall_confidence(criteria_results)
    summary = "; ".join(
        f"{c['name']}: {c['score']}/{c['max_points']}" for c in criteria_results
    )[:4000]

    needs_review, reasons = should_route_human_review(
        criteria_results,
        confidence_threshold=cfg.REVIEW_CONFIDENCE_THRESHOLD,
        near_boundary_points=cfg.REVIEW_NEAR_BOUNDARY_POINTS,
    )
    if rule_issues:
        needs_review = True
        for ri in rule_issues[:8]:
            if ri not in reasons:
                reasons.append(ri)

    for c in criteria_results:
        for f in c.get("flags") or []:
            fs = str(f)
            if fs and fs not in flat_flags:
                flat_flags.append(fs)
    for f in flat_flags:
        if f.upper() in ("UNCERTAIN_EVIDENCE", "CONFLICT", "MISSING_ARTIFACT", "SCORING_ERROR"):
            needs_review = True
            break
    if needs_review and "needs_review" not in flat_flags:
        flat_flags.append("needs_review")

    meta_stages["aggregate"] = round(time.perf_counter() - s, 3)
    total_s = round(time.perf_counter() - t0, 3)
    _log.info(
        "grading_stage submission=%s stage=aggregate duration_s=%.3f total_s=%.3f",
        context_id,
        meta_stages["aggregate"],
        total_s,
    )

    criteria_out = []
    for c in criteria_results:
        row = dict(c)
        row.pop("weight", None)
        criteria_out.append(row)

    return {
        "overall": {
            "score": overall_score,
            "confidence": overall_conf,
            "summary": summary,
        },
        "criteria": criteria_out,
        "flags": flat_flags,
        "_model_used": ", ".join(lbl for _, lbl in scoring_clients)
        if use_staged_multi
        else primary_label,
        "_models_used": [lbl for _, lbl in scoring_clients],
        "_pipeline_meta": {
            "mode": "staged",
            "submission_context_id": context_id,
            "stage_timings_s": meta_stages,
            "review_reasons": reasons,
            "total_s": total_s,
            "staged_multi_llm": use_staged_multi,
            "staged_scoring_models": [lbl for _, lbl in scoring_clients],
        },
    }

DEFAULT_STANDALONE_RUBRIC = [
    {"criterion": "Clarity", "max_score": 25},
    {"criterion": "Correctness", "max_score": 25},
    {"criterion": "Completeness", "max_score": 25},
    {"criterion": "Organization", "max_score": 25},
]


def _build_assignment_prompt(assignment, rubric_text: str | None, answer_key_text: str | None) -> str:
    parts = []
    base = (getattr(assignment, "description", None) or getattr(assignment, "title", None) or "").strip()
    if base:
        parts.append(base)
    if answer_key_text and str(answer_key_text).strip():
        parts.append("Answer key / reference (instructor context):\n" + str(answer_key_text).strip())
    if rubric_text and str(rubric_text).strip():
        parts.append("Additional rubric notes:\n" + str(rubric_text).strip())
    return "\n\n".join(parts) if parts else "Grade this submission."


def _average_grading_results(results: list[tuple[dict, str]]) -> dict:
    """
    Average scores from multiple LLM grading results.

    Each entry is (grading_result_dict, model_label).
    """
    if len(results) == 1:
        merged = results[0][0]
        merged["_model_used"] = results[0][1]
        merged["_models_used"] = [results[0][1]]
        return merged

    n = len(results)

    overall_scores = np.zeros(n, dtype=np.float64)
    overall_confidences = np.zeros(n, dtype=np.float64)
    summaries: list[str] = []
    for i, (res, _label) in enumerate(results):
        ov = res.get("overall", {}) or {}
        overall_scores[i] = float(ov.get("score", 0))
        overall_confidences[i] = float(ov.get("confidence", 0))
        summaries.append(ov.get("summary", "") or "")

    avg_overall_score = round(float(np.mean(overall_scores)), 2)
    avg_overall_confidence = round(float(np.mean(overall_confidences)), 2)

    criteria_by_name: dict[str, list[dict]] = defaultdict(list)
    for res, _label in results:
        for c in res.get("criteria", []) or []:
            criteria_by_name[c.get("name", "unknown")].append(c)

    merged_criteria = []
    for name in sorted(criteria_by_name.keys()):
        entries = criteria_by_name[name]
        sc = np.array([float(e.get("score", 0)) for e in entries], dtype=np.float64)
        cf = np.array([float(e.get("confidence", 0)) for e in entries], dtype=np.float64)
        avg_score = round(float(np.mean(sc)), 2)
        avg_conf = round(float(np.mean(cf)), 2)
        e0 = entries[0]
        max_pts = e0.get("max_points")
        if max_pts is None:
            max_pts = e0.get("max_score")
        rationales = [e.get("rationale", "") for e in entries if e.get("rationale")]
        all_quotes = []
        all_notes = []
        for e in entries:
            ev = e.get("evidence", {})
            if isinstance(ev, dict):
                all_quotes.extend(ev.get("quotes", []) or [])
                if ev.get("notes"):
                    all_notes.append(str(ev["notes"]))

        merged_criteria.append(
            {
                "name": name,
                "score": avg_score,
                "max_points": max_pts,
                "confidence": avg_conf,
                "rationale": " | ".join(rationales),
                "evidence": {
                    "quotes": all_quotes,
                    "notes": " | ".join(all_notes) if all_notes else "",
                },
            }
        )

    all_flags: set[str] = set()
    for res, _label in results:
        all_flags.update(res.get("flags") or [])

    model_labels = [label for _, label in results]
    summary_parts = []
    for (res, label), s in zip(results, summaries):
        if s:
            summary_parts.append(f"[{label}] {s}")

    return {
        "overall": {
            "score": avg_overall_score,
            "confidence": avg_overall_confidence,
            "summary": " | ".join(summary_parts),
        },
        "criteria": merged_criteria,
        "flags": list(all_flags),
        "_model_used": ", ".join(model_labels),
        "_models_used": model_labels,
    }


def _merge_criteria_max_by_name(
    question_grades: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for qg in question_grades:
        for c in qg.get("criteria") or []:
            if not isinstance(c, dict):
                continue
            name = str(c.get("name") or "")
            if not name:
                continue
            try:
                sc = float(c.get("score", 0))
            except (TypeError, ValueError):
                sc = 0.0
            if name not in best or sc > float(best[name].get("score", 0)):
                best[name] = dict(c)
    return [best[k] for k in sorted(best.keys())]


def _apply_rubric_weights_to_criteria(
    rows: list[dict[str, Any]], rubric_items: list[dict[str, Any]]
) -> None:
    by = {r["name"]: r for r in rubric_items}
    for c in rows:
        n = c.get("name")
        if n not in by:
            continue
        r = by[n]
        c["weight"] = r.get("weight", r["max_points"])
        if c.get("max_points") is None:
            c["max_points"] = r["max_points"]
        if c.get("min_points") is None:
            c["min_points"] = r["min_points"]


def _run_chunk_entropy_grading_pipeline(
    cfg,
    assignment,
    artifacts_bytes: dict[str, bytes],
    rubric: list,
    assignment_prompt: str,
    ctx: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    """
    RAG-style units from chunking (+ per-unit embeddings), ``k`` stochastic ``grade()`` samples
    per configured model per unit, semantic entropy per unit (pooled + per-model), then merge
    criteria across units (max score per criterion name) for the assignment aggregate.
    """
    t0 = time.perf_counter()
    plain = submission_text_from_artifacts(artifacts_bytes).strip()
    if not plain:
        plain = _plaintext_for_modality_profile(artifacts_bytes, ctx)

    title = str(getattr(assignment, "title", "") or "")
    st = str(profile.get("modality_subtype") or "")
    chunk_cap = int(getattr(cfg, "RAG_EMBED_MAX_CHARS", 4000) or 4000)
    chunks = build_submission_chunks(
        plain,
        assignment_title=title,
        modality_subtype=st,
        max_chunk_chars=max(2000, min(chunk_cap, 12000)),
    )
    units = build_grading_units_from_chunks(chunks)
    if not units:
        units = [
            {
                "pair_id": None,
                "question_text": "",
                "response_text": plain,
                "chunk_ids": [],
            }
        ]

    k = max(1, int(getattr(cfg, "GRADING_SAMPLES_PER_MODEL", 1)))
    temp = float(getattr(cfg, "GRADING_SAMPLE_TEMPERATURE", 0.3))
    clients = build_grading_clients(cfg)
    rubric_items = _coerce_rubric_items(rubric)
    if not rubric_items:
        rubric_items = [
            {
                "name": "Overall",
                "max_points": 100.0,
                "min_points": 0.0,
                "weight": 100.0,
                "description": "Holistic quality when no rubric rows exist.",
            }
        ]

    question_grades: list[dict[str, Any]] = []
    max_embed = int(getattr(cfg, "RAG_EMBED_MAX_CHARS", 24000))

    for u in units:
        unit_block = format_unit_for_grader_prompt(u)
        prompt_u = assignment_prompt + unit_block
        utext = f"{u.get('question_text') or ''}\n\n{u.get('response_text') or ''}"
        vec_dim = 0
        vec_src = "none"
        try:
            vec, vec_src = compute_submission_embedding(utext[:max_embed], cfg)
            vec_dim = len(vec) if vec else 0
        except Exception:
            _log.debug("unit embedding skipped", exc_info=True)

        labeled: list[tuple[dict[str, Any], str]] = []
        attempted = 0
        for cl, label in clients:
            for _i in range(k):
                attempted += 1
                try:
                    res = grade(
                        client=cl,
                        rubric=rubric,
                        assignment_prompt=prompt_u,
                        submission_context=ctx,
                        temperature=temp,
                    )
                    labeled.append((res, label))
                except Exception:
                    _log.warning(
                        "chunk_entropy grade failed pair_id=%s model=%s sample=%s",
                        u.get("pair_id"),
                        label,
                        _i,
                        exc_info=True,
                    )

        if not labeled:
            merged_flat: dict[str, Any] = {
                "overall": {
                    "score": 0.0,
                    "confidence": 0.25,
                    "summary": "All grading samples failed for this unit.",
                },
                "criteria": [],
                "flags": ["SCORING_ERROR"],
            }
            H, cl_n = 0.0, 0
            cfe = confidence_from_entropy_natural(0.0)
            by_m: dict[str, dict[str, float | int]] = {}
        else:
            samples_only = [s for s, _ in labeled]
            try:
                merged_flat = aggregate_grading_json_samples(samples_only)
                merged_flat = coerce_grading_output_shape(merged_flat)
            except Exception:
                _log.exception("chunk_entropy aggregate failed for unit")
                merged_flat = {
                    "overall": {"score": 0.0, "confidence": 0.25, "summary": ""},
                    "criteria": [],
                    "flags": ["AGGREGATE_ERROR"],
                }
            fingerprints = [grading_semantic_fingerprint(s) for s in samples_only]
            H, cl_n = semantic_entropy_natural(fingerprints)
            cfe = confidence_from_entropy_natural(H)
            by_m = semantic_entropy_by_model(labeled)

        ov = merged_flat.get("overall") or {}
        question_grades.append(
            {
                "pair_id": u.get("pair_id"),
                "chunk_ids": u.get("chunk_ids") or [],
                "question_preview": (u.get("question_text") or "")[:1200],
                "response_preview": (u.get("response_text") or "")[:1200],
                "embedding_source": vec_src,
                "embedding_dim": vec_dim,
                "overall": {
                    "score": float(ov.get("score", 0.0)),
                    "confidence": float(ov.get("confidence", 0.5)),
                    "summary": str(ov.get("summary") or "")[:4000],
                    "semantic_entropy": round(H, 4),
                    "confidence_from_entropy": round(cfe, 4),
                },
                "criteria": list(merged_flat.get("criteria") or []),
                "flags": list(merged_flat.get("flags") or []),
                "semantic_entropy": round(H, 4),
                "confidence_from_entropy": round(cfe, 4),
                "cluster_count": cl_n,
                "per_model_semantic_entropy": by_m,
                "samples_valid": len(labeled),
                "samples_attempted": attempted,
            }
        )

    merged_criteria = _merge_criteria_max_by_name(question_grades)
    _apply_rubric_weights_to_criteria(merged_criteria, rubric_items)
    overall_score = weighted_overall_score(merged_criteria)
    wconf = weighted_overall_confidence(merged_criteria)

    entropies = np.asarray(
        [float(qg["semantic_entropy"]) for qg in question_grades],
        dtype=np.float64,
    )
    avg_h = float(np.mean(entropies)) if entropies.size else 0.0
    final_conf_ent = confidence_from_entropy_natural(avg_h)

    summary_parts = [
        f"unit[{qg.get('pair_id')}]: score={qg['overall']['score']}, H={qg['semantic_entropy']}"
        for qg in question_grades
    ]

    flags_acc: list[str] = []
    for qg in question_grades:
        for f in qg.get("flags") or []:
            fs = str(f)
            if fs and fs not in flags_acc:
                flags_acc.append(fs)
    if len(units) > 1:
        flags_acc.append("chunk_entropy_multi_unit")

    crit_out = []
    for c in merged_criteria:
        row = dict(c)
        row.pop("weight", None)
        crit_out.append(row)

    model_labels = [lbl for _, lbl in clients]
    elapsed = round(time.perf_counter() - t0, 3)

    return {
        "overall": {
            "score": overall_score,
            "confidence": round(final_conf_ent, 4),
            "summary": "; ".join(summary_parts)[:4000],
            "semantic_entropy": round(avg_h, 4),
            "confidence_from_entropy": round(final_conf_ent, 4),
            "classical_confidence": round(wconf, 4),
        },
        "criteria": crit_out,
        "question_grades": question_grades,
        "flags": flags_acc,
        "_model_used": ", ".join(model_labels),
        "_models_used": model_labels,
        "_entropy_meta": {
            "mode": "chunk_entropy",
            "samples_per_model": k,
            "temperature": temp,
            "aggregate_semantic_entropy": round(avg_h, 4),
            "confidence_from_aggregate_entropy": round(final_conf_ent, 4),
            "per_question": [
                {
                    "pair_id": qg["pair_id"],
                    "chunk_ids": qg.get("chunk_ids"),
                    "semantic_entropy": qg["semantic_entropy"],
                    "confidence_from_entropy": qg["confidence_from_entropy"],
                    "per_model_semantic_entropy": qg["per_model_semantic_entropy"],
                    "grade": qg["overall"]["score"],
                }
                for qg in question_grades
            ],
            "chunk_count": len(chunks),
            "unit_count": len(units),
            "duration_s": elapsed,
        },
    }


def _use_entropy_sampling(cfg) -> bool:
    """Legacy pipeline only; requires explicit flag and k>1."""
    mode = (getattr(cfg, "GRADING_PIPELINE_MODE", None) or "legacy").strip().lower()
    if mode in ("staged", "chunk_entropy"):
        return False
    if not getattr(cfg, "GRADING_ENTROPY_MODE", False):
        return False
    return int(getattr(cfg, "GRADING_SAMPLES_PER_MODEL", 1)) > 1


def _run_entropy_sampling_grading(
    cfg,
    assignment,
    ctx: dict[str, Any],
    rubric: list,
    assignment_prompt: str,
    *,
    fallback_client,
) -> dict[str, Any]:
    """
    k stochastic grade() calls per model in build_grading_clients(); aggregate scores by
    arithmetic mean over *all* valid samples (all models × k). Semantic entropy uses
    fingerprint clusters across those samples. OpenAI escalation (maybe_escalate_grade) is
    skipped here to avoid extra cost after many samples.
    """
    k = int(cfg.GRADING_SAMPLES_PER_MODEL)
    temp = float(cfg.GRADING_SAMPLE_TEMPERATURE)
    embed_mode = getattr(cfg, "GRADING_ENTROPY_EMBEDDINGS", "fingerprint") or "fingerprint"
    if embed_mode not in ("off", "fingerprint", "openai"):
        _log.warning(
            "Unknown GRADING_ENTROPY_EMBEDDINGS=%r; using fingerprint",
            embed_mode,
        )
    if embed_mode == "openai":
        _log.warning(
            "GRADING_ENTROPY_EMBEDDINGS=openai not implemented; using fingerprint"
        )

    t0 = time.perf_counter()
    grading_clients = build_grading_clients(cfg)
    labeled: list[tuple[dict[str, Any], str]] = []
    attempted = 0

    for grading_client, model_label in grading_clients:
        for _ in range(k):
            attempted += 1
            try:
                res = grade(
                    client=grading_client,
                    rubric=rubric,
                    assignment_prompt=assignment_prompt,
                    submission_context=ctx,
                    temperature=temp,
                )
                labeled.append((res, model_label))
            except Exception:
                _log.warning(
                    "Entropy sample failed model=%s (attempt %s/%s)",
                    model_label,
                    attempted,
                    len(grading_clients) * k,
                    exc_info=True,
                )

    if not labeled:
        _log.error("All entropy samples failed; single deterministic fallback grade")
        result = grade(
            fallback_client,
            rubric,
            assignment_prompt,
            ctx,
            temperature=None,
        )
        om = (cfg.OLLAMA_MODEL or "llama3.2:3b").strip()
        result["_model_used"] = f"ollama:{om}"
        result["_models_used"] = [f"ollama:{om}"]
        return result

    samples = [s for s, _ in labeled]
    merged = aggregate_grading_json_samples(samples)
    fingerprints = [grading_semantic_fingerprint(s) for s in samples]
    h, clusters = semantic_entropy_natural(fingerprints)
    conf_ent = confidence_from_entropy_natural(h)

    model_labels = list(dict.fromkeys(label for _, label in labeled))
    valid_n = len(samples)
    min_rate = float(getattr(cfg, "GRADING_ENTROPY_MIN_SUCCESS_RATE", 0.5))
    high_h = float(getattr(cfg, "GRADING_ENTROPY_REVIEW_NATURAL_H", 1.0))

    flags = list(merged.get("flags") or [])
    if attempted > 0 and (valid_n / attempted) < min_rate:
        flags.append("entropy_low_sample_success_rate")
    if h > high_h:
        flags.append("high_semantic_entropy")
    if "entropy_low_sample_success_rate" in flags or "high_semantic_entropy" in flags:
        if "needs_review" not in flags:
            flags.append("needs_review")

    merged["flags"] = flags
    merged["_model_used"] = ", ".join(model_labels)
    merged["_models_used"] = model_labels
    merged["_entropy_meta"] = {
        "semantic_entropy": round(h, 4),
        "entropy_base": "e",
        "confidence_from_entropy": round(conf_ent, 4),
        "samples_per_model": k,
        "valid_samples": valid_n,
        "attempted_samples": attempted,
        "cluster_count": clusters,
        "sample_temperature": temp,
        "aggregation": "mean_over_all_valid_samples_all_models",
    }
    overall = dict(merged["overall"])
    overall["semantic_entropy"] = merged["_entropy_meta"]["semantic_entropy"]
    overall["confidence_from_entropy"] = merged["_entropy_meta"]["confidence_from_entropy"]
    merged["overall"] = overall

    elapsed = round(time.perf_counter() - t0, 3)
    _log.info(
        "grading_entropy valid=%s attempted=%s H=%.4f clusters=%s duration_s=%.3f temp=%s",
        valid_n,
        attempted,
        h,
        clusters,
        elapsed,
        temp,
    )
    return merged


def run_grading_pipeline(
    cfg,
    assignment,
    artifacts_bytes: dict,
    *,
    rubric_text: str | None = None,
    answer_key_text: str | None = None,
):
    """
    artifacts_bytes: {kind: bytes}

    Heavy inference runs on GPU workers (Ollama). OpenAI is optional server-side escalation.
    Optional rubric_text / answer_key_text add instructor context to the grader prompt.

    When GRADING_MODEL_2 / GRADING_MODEL_3 are configured, multiple LLMs grade independently
    and the final score is the arithmetic average of their grades.

    When ``GRADING_ENTROPY_MODE=on`` and ``GRADING_SAMPLES_PER_MODEL`` > 1 (legacy pipeline only;
    not when ``GRADING_PIPELINE_MODE`` is ``staged`` or ``chunk_entropy``), each model runs k
    stochastic ``grade()`` calls at ``GRADING_SAMPLE_TEMPERATURE`` (default 0.3); scores are
    averaged over all valid samples and ``_entropy_meta`` / ``overall.semantic_entropy`` record
    fingerprint-cluster entropy.

    When ``GRADING_PIPELINE_MODE=chunk_entropy``, submission text is chunked into units (question +
    response via :func:`build_grading_units_from_chunks`); each unit is embedded for metadata,
    graded with k samples per configured model, and per-unit semantic entropy is pooled (and
    split per model). Assignment-level scores merge criterion rows with **max** score per
    criterion across units; ``overall.semantic_entropy`` is the mean of unit entropies and
    ``overall.confidence`` / ``overall.confidence_from_entropy`` use that aggregate.

    Results may include ``_modality`` (``modality``, ``modality_subtype``, artifact keys,
    text length signals) from :func:`resolve_modality_profile`.
    """
    client = primary_ollama_client(cfg)
    secondary = openai_client_if_configured(cfg)
    modality = getattr(assignment, "modality", None) or infer_modality_from_artifacts(artifacts_bytes)
    p = plan(client, modality)

    ctx = {"modality": modality, "artifacts": {}, "tool_results": {}}

    for step in p.get("plan", []):
        tool = step.get("tool", "none")

        if tool == "extract_text":
            if "pdf" in artifacts_bytes:
                txt = extract_text_from_pdf(artifacts_bytes["pdf"])
                ctx["artifacts"]["text"] = txt
        elif tool == "run_tests":
            if "py" in artifacts_bytes:
                ctx["tool_results"]["tests"] = run_python_tests(artifacts_bytes["py"], "submission.py")
            if "ipynb" in artifacts_bytes:
                nb = extract_from_ipynb(artifacts_bytes["ipynb"])
                ctx["artifacts"]["code"] = nb["code"]
                ctx["artifacts"]["markdown"] = nb["markdown"]
        elif tool == "transcribe_video":
            if "mp4" in artifacts_bytes:
                ctx["tool_results"]["transcript"] = transcribe_video_stub(artifacts_bytes["mp4"])

    _ensure_submission_artifacts_in_ctx(ctx, artifacts_bytes)

    plain_profile = _plaintext_for_modality_profile(artifacts_bytes, ctx)
    profile = resolve_modality_profile(
        assignment, artifacts_bytes, plain_profile[:12000]
    )
    ctx["_modality_resolution"] = profile
    if profile.get("signals", {}).get("text_too_short_for_grading"):
        _log.warning(
            "grading submission text very short (%s chars); scores may be unreliable",
            profile.get("extracted_text_chars"),
        )

    rubric = getattr(assignment, "rubric", None) or []
    base_prompt = _build_assignment_prompt(assignment, rubric_text, answer_key_text)
    assignment_prompt = augment_prompt_for_modality_profile(base_prompt, profile)

    mode = (getattr(cfg, "GRADING_PIPELINE_MODE", None) or "legacy").strip().lower()
    if mode == "chunk_entropy":
        result = _run_chunk_entropy_grading_pipeline(
            cfg,
            assignment,
            artifacts_bytes,
            rubric,
            assignment_prompt,
            ctx,
            profile,
        )
    elif mode == "staged":
        result = _run_staged_grading_pipeline(
            cfg,
            assignment,
            artifacts_bytes,
            rubric_text=rubric_text,
            answer_key_text=answer_key_text,
            assignment_prompt=assignment_prompt,
            ctx=ctx,
        )
    elif _use_entropy_sampling(cfg):
        result = _run_entropy_sampling_grading(
            cfg,
            assignment,
            ctx,
            rubric,
            assignment_prompt,
            fallback_client=client,
        )
    else:
        grading_clients = build_grading_clients(cfg)
        grading_results: list[tuple[dict, str]] = []

        for grading_client, model_label in grading_clients:
            try:
                res = grade(
                    client=grading_client,
                    rubric=rubric,
                    assignment_prompt=assignment_prompt,
                    submission_context=ctx,
                )
                grading_results.append((res, model_label))
            except Exception:
                _log.warning(
                    "Grading failed for model %s, skipping", model_label, exc_info=True
                )

        if not grading_results:
            result = grade(
                client=client,
                rubric=rubric,
                assignment_prompt=assignment_prompt,
                submission_context=ctx,
            )
            om = (cfg.OLLAMA_MODEL or "llama3.2:3b").strip()
            result["_model_used"] = f"ollama:{om}"
            result["_models_used"] = [f"ollama:{om}"]
        elif len(grading_results) == 1:
            result = grading_results[0][0]
            single_label = grading_results[0][1]
            result = maybe_escalate_grade(
                cfg,
                client,
                secondary,
                rubric,
                assignment_prompt,
                ctx,
                result,
            )
            if result.get("_used_openai_arbitration"):
                result["_model_used"] = f"openai:{cfg.OPENAI_MODEL}"
            else:
                result["_model_used"] = single_label
            result["_models_used"] = [result["_model_used"]]
        else:
            result = _average_grading_results(grading_results)

    result = coerce_grading_output_shape(result)
    _attach_modality_to_result(result, ctx)
    return result


def run_standalone_grading_pipeline(
    cfg,
    artifacts_bytes: dict,
    title: str,
    rubric_text: str | None,
    answer_key_text: str | None,
    rubric_file_excerpt: str | None,
    answer_key_file_excerpt: str | None,
    grading_instructions: str | None = None,
):
    """
    Course-independent grading: inferred modality, default rubric if none, merged text context.
    grading_instructions: optional instructor prompt (focus, constraints) fed into the grader prompt.
    """
    modality = infer_modality_from_artifacts(artifacts_bytes)
    merged_rubric_note_parts = []
    if rubric_text and rubric_text.strip():
        merged_rubric_note_parts.append(rubric_text.strip())
    if rubric_file_excerpt and rubric_file_excerpt.strip():
        merged_rubric_note_parts.append("Rubric (from uploaded file):\n" + rubric_file_excerpt.strip())
    merged_rubric = "\n\n".join(merged_rubric_note_parts) if merged_rubric_note_parts else None

    merged_ak_parts = []
    if answer_key_text and answer_key_text.strip():
        merged_ak_parts.append(answer_key_text.strip())
    if answer_key_file_excerpt and answer_key_file_excerpt.strip():
        merged_ak_parts.append("Answer key (from uploaded file):\n" + answer_key_file_excerpt.strip())
    merged_ak = "\n\n".join(merged_ak_parts) if merged_ak_parts else None

    desc_parts = []
    base_title = (title or "Standalone submission").strip()
    if base_title:
        desc_parts.append(base_title)
    if grading_instructions and str(grading_instructions).strip():
        desc_parts.append(
            "Instructor grading instructions:\n" + str(grading_instructions).strip()
        )
    description = "\n\n".join(desc_parts) if desc_parts else "Standalone autograder submission"

    pseudo = SimpleNamespace(
        modality=modality,
        rubric=list(DEFAULT_STANDALONE_RUBRIC),
        title=title or "Standalone submission",
        description=description,
    )
    return run_grading_pipeline(
        cfg,
        pseudo,
        artifacts_bytes,
        rubric_text=merged_rubric,
        answer_key_text=merged_ak,
    )
