import logging
from collections import defaultdict
from types import SimpleNamespace

from .agent import plan, grade
from .llm_router import (
    build_grading_clients,
    maybe_escalate_grade,
    openai_client_if_configured,
    primary_ollama_client,
)
from .tools import extract_text_from_pdf, extract_from_ipynb, run_python_tests, transcribe_video_stub

_log = logging.getLogger(__name__)

DEFAULT_STANDALONE_RUBRIC = [
    {"criterion": "Clarity", "max_score": 25},
    {"criterion": "Correctness", "max_score": 25},
    {"criterion": "Completeness", "max_score": 25},
    {"criterion": "Organization", "max_score": 25},
]


def infer_modality_from_artifacts(artifacts_bytes: dict) -> str:
    """Infer assignment modality from submission file keys (extension buckets)."""
    if "mp4" in artifacts_bytes:
        return "video"
    if "ipynb" in artifacts_bytes:
        return "notebook"
    if "py" in artifacts_bytes or "zip" in artifacts_bytes:
        return "code"
    if "pdf" in artifacts_bytes:
        return "written"
    if "txt" in artifacts_bytes:
        return "text"
    if "png" in artifacts_bytes or "jpg" in artifacts_bytes or "docx" in artifacts_bytes:
        return "written"
    return "written"


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

    overall_scores = []
    overall_confidences = []
    summaries = []
    for res, _label in results:
        ov = res.get("overall", {}) or {}
        overall_scores.append(float(ov.get("score", 0)))
        overall_confidences.append(float(ov.get("confidence", 0)))
        summaries.append(ov.get("summary", "") or "")

    avg_overall_score = round(sum(overall_scores) / n, 2)
    avg_overall_confidence = round(sum(overall_confidences) / n, 2)

    criteria_by_name: dict[str, list[dict]] = defaultdict(list)
    for res, _label in results:
        for c in res.get("criteria", []) or []:
            criteria_by_name[c.get("name", "unknown")].append(c)

    merged_criteria = []
    for name in sorted(criteria_by_name.keys()):
        entries = criteria_by_name[name]
        m = len(entries)
        avg_score = round(sum(float(e.get("score", 0)) for e in entries) / m, 2)
        avg_conf = round(sum(float(e.get("confidence", 0)) for e in entries) / m, 2)
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
    artifacts_bytes: {kind: bytes}

    Heavy inference runs on GPU workers (Ollama). OpenAI is optional server-side escalation.
    Optional rubric_text / answer_key_text add instructor context to the grader prompt.

    When GRADING_MODEL_2 / GRADING_MODEL_3 are configured, multiple LLMs grade independently
    and the final score is the arithmetic average of their grades.
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

    # Ensure content exists even if planner didn't choose tool
    if modality == "text" and "text" not in ctx["artifacts"]:
        if "txt" in artifacts_bytes:
            ctx["artifacts"]["text"] = artifacts_bytes["txt"].decode("utf-8", errors="ignore")

    rubric = getattr(assignment, "rubric", None) or []
    assignment_prompt = _build_assignment_prompt(assignment, rubric_text, answer_key_text)

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
            _log.warning("Grading failed for model %s, skipping", model_label, exc_info=True)

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
