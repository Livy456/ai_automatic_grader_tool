from types import SimpleNamespace

from .agent import plan, grade
from .llm_router import (
    maybe_escalate_grade,
    openai_client_if_configured,
    primary_ollama_client,
)
from .tools import extract_text_from_pdf, extract_from_ipynb, run_python_tests, transcribe_video_stub

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

    result = grade(
        client=client,
        rubric=rubric,
        assignment_prompt=assignment_prompt,
        submission_context=ctx,
    )
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
        result["_model_used"] = f"ollama:{cfg.OLLAMA_MODEL}"
    return result


def run_standalone_grading_pipeline(
    cfg,
    artifacts_bytes: dict,
    title: str,
    rubric_text: str | None,
    answer_key_text: str | None,
    rubric_file_excerpt: str | None,
    answer_key_file_excerpt: str | None,
):
    """
    Course-independent grading: inferred modality, default rubric if none, merged text context.
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

    pseudo = SimpleNamespace(
        modality=modality,
        rubric=list(DEFAULT_STANDALONE_RUBRIC),
        title=title or "Standalone submission",
        description=title or "Standalone autograder submission",
    )
    return run_grading_pipeline(
        cfg,
        pseudo,
        artifacts_bytes,
        rubric_text=merged_rubric,
        answer_key_text=merged_ak,
    )
