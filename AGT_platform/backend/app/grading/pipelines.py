import hashlib
import json
import logging
from collections import defaultdict
from types import SimpleNamespace
from typing import Any

import numpy as np

from .agent import grade, plan
from .llm_router import (
    build_grading_clients,
    maybe_escalate_grade,
    openai_client_if_configured,
)
from .output_schema import coerce_grading_output_shape
from .modality_resolution import (
    augment_prompt_for_modality_profile,
    infer_modality_from_artifacts,
    resolve_modality_profile,
)
from .notebook_grader_context import build_notebook_grader_overlay
from .submission_text import submission_text_from_artifacts
from .tools import extract_from_ipynb, extract_text_from_pdf, run_python_tests, transcribe_video_stub

_log = logging.getLogger(__name__)

# Fallback rubric rows when no course rubric is attached (standalone pipeline, local tests).
# Shape matches course DB / API rows: ``criterion`` + ``max_score`` (see ``courses.py``).
DEFAULT_STANDALONE_RUBRIC: tuple[dict[str, Any], ...] = (
    {"criterion": "Correctness", "max_score": 10.0},
    {"criterion": "Completeness", "max_score": 10.0},
    {"criterion": "Clarity and presentation", "max_score": 5.0},
)


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


def _grade_submission_context(
    ctx: dict[str, Any],
    cfg,
    *,
    artifacts_bytes: dict[str, bytes] | None = None,
    assignment_title: str = "",
) -> dict[str, Any]:
    """
    Shallow copy of ``ctx`` with bounded artifact / tool_result strings for ``grade()``.

    Full notebook sources can make ``json.dumps(submission)`` huge; small local models
    then return non-schema JSON (e.g. ``status`` / ``result`` blobs). Staged mode uses
    ``normalized_to_json_str`` caps instead; legacy / entropy / chunk_entropy use this.

    For ``.ipynb``, :func:`build_notebook_grader_overlay` replaces raw code/markdown with
    an ordered ``notebook_qa`` list (question/response/code + ``pair_id``) under the same
    character budget, optionally structured by OpenAI when configured.
    """
    budget = int(getattr(cfg, "STAGED_PROMPT_MAX_CHARS", 28000) or 28000)
    budget = max(8000, min(budget, 200_000))
    per_field = max(4000, budget // 2)
    suffix = (
        "\n\n[... truncated for grader payload; full text is still used for modality "
        "detection and RAG export ...]"
    )
    trim_len = max(1000, per_field - len(suffix))

    arts_in = ctx.get("artifacts") or {}
    arts = dict(arts_in)
    nb_overlay: dict[str, Any] | None = None
    if artifacts_bytes and "ipynb" in artifacts_bytes:
        prof = ctx.get("_modality_resolution") or {}
        nb_overlay = build_notebook_grader_overlay(
            artifacts_bytes,
            cfg,
            modality_subtype=str(prof.get("modality_subtype") or ""),
            assignment_title=(assignment_title or "").strip(),
            budget_chars=budget,
        )
        if nb_overlay:
            arts.update(nb_overlay)

    for key in ("code", "markdown", "text"):
        if nb_overlay and key in ("code", "markdown"):
            continue
        s = arts.get(key)
        if isinstance(s, str) and len(s) > per_field:
            arts[key] = s[:trim_len] + suffix

    tr_in = ctx.get("tool_results") or {}
    tr = dict(tr_in)
    tests = tr.get("tests")
    if isinstance(tests, dict):
        ser = json.dumps(tests, default=str)
        if len(ser) > per_field:
            tr["tests"] = {
                "truncated": True,
                "char_len": len(ser),
                "preview": ser[:trim_len] + suffix,
            }

    return {**ctx, "artifacts": arts, "tool_results": tr}


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


def run_grading_pipeline(
    cfg,
    assignment,
    artifacts_bytes: dict,
    *,
    rubric_text: str | None = None,
    answer_key_text: str | None = None,
):
    """
    Grade a submission using **OpenAI only** (``OPENAI_API_KEY`` / ``OPENAI_MODEL``).

    ``GRADING_MODEL_2`` / ``GRADING_MODEL_3`` may add additional ``openai:`` models; Ollama
    specs are ignored. ``GRADING_PIPELINE_MODE`` (``staged`` / ``chunk_entropy``) and
    ``GRADING_ENTROPY_MODE`` are no longer supported and are ignored.

    Optional ``rubric_text`` / ``answer_key_text`` add instructor context to the grader prompt.
    Results may include ``_modality`` from :func:`resolve_modality_profile`.
    """
    client = openai_client_if_configured(cfg)
    if client is None:
        raise RuntimeError(
            "Course grading requires OPENAI_API_KEY (Ollama-backed grading has been removed)."
        )
    secondary = client
    modality = getattr(assignment, "modality", None) or infer_modality_from_artifacts(
        artifacts_bytes
    )
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
                ctx["tool_results"]["tests"] = run_python_tests(
                    artifacts_bytes["py"], "submission.py"
                )
            if "ipynb" in artifacts_bytes:
                nb = extract_from_ipynb(artifacts_bytes["ipynb"])
                ctx["artifacts"]["code"] = nb["code"]
                ctx["artifacts"]["markdown"] = nb["markdown"]
        elif tool == "transcribe_video":
            if "mp4" in artifacts_bytes:
                ctx["tool_results"]["transcript"] = transcribe_video_stub(
                    artifacts_bytes["mp4"]
                )

    _ensure_submission_artifacts_in_ctx(ctx, artifacts_bytes)

    plain_profile = _plaintext_for_modality_profile(artifacts_bytes, ctx)
    profile = resolve_modality_profile(assignment, artifacts_bytes, plain_profile)
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
    if mode in ("staged", "chunk_entropy") or getattr(cfg, "GRADING_ENTROPY_MODE", False):
        _log.warning(
            "Ignoring GRADING_PIPELINE_MODE=%r and GRADING_ENTROPY_MODE=%r (OpenAI-only path).",
            mode,
            getattr(cfg, "GRADING_ENTROPY_MODE", False),
        )

    grading_clients = build_grading_clients(cfg)
    if not grading_clients:
        raise RuntimeError(
            "No OpenAI grading clients configured (check OPENAI_API_KEY and OPENAI_MODEL)."
        )
    ctx_llm = _grade_submission_context(
        ctx,
        cfg,
        artifacts_bytes=artifacts_bytes,
        assignment_title=str(getattr(assignment, "title", "") or ""),
    )
    grading_results: list[tuple[dict, str]] = []

    for grading_client, model_label in grading_clients:
        try:
            res = grade(
                client=grading_client,
                rubric=rubric,
                assignment_prompt=assignment_prompt,
                submission_context=ctx_llm,
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
            submission_context=ctx_llm,
        )
        om = (cfg.OPENAI_MODEL or "gpt-4o-mini").strip()
        result["_model_used"] = f"openai:{om}"
        result["_models_used"] = [f"openai:{om}"]
    elif len(grading_results) == 1:
        result = grading_results[0][0]
        single_label = grading_results[0][1]
        result = maybe_escalate_grade(
            cfg,
            client,
            secondary,
            rubric,
            assignment_prompt,
            ctx_llm,
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
