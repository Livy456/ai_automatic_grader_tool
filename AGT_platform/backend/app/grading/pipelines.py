from .agent import plan, grade
from .llm_router import (
    maybe_escalate_grade,
    openai_client_if_configured,
    primary_ollama_client,
)
from .tools import extract_text_from_pdf, extract_from_ipynb, run_python_tests, transcribe_video_stub


def run_grading_pipeline(cfg, assignment, artifacts_bytes: dict):
    """
    artifacts_bytes: {kind: bytes}

    Heavy inference runs on GPU workers (Ollama). OpenAI is optional server-side escalation.
    """
    client = primary_ollama_client(cfg)
    secondary = openai_client_if_configured(cfg)
    p = plan(client, assignment.modality)

    ctx = {"modality": assignment.modality, "artifacts": {}, "tool_results": {}}

    for step in p.get("plan", []):
        tool = step.get("tool","none")

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
    if assignment.modality == "text" and "text" not in ctx["artifacts"]:
        if "txt" in artifacts_bytes:
            ctx["artifacts"]["text"] = artifacts_bytes["txt"].decode("utf-8", errors="ignore")

    result = grade(
        client=client,
        rubric=assignment.rubric,
        assignment_prompt=assignment.description or assignment.title,
        submission_context=ctx,
    )
    result = maybe_escalate_grade(
        cfg,
        client,
        secondary,
        assignment.rubric,
        assignment.description or assignment.title,
        ctx,
        result,
    )
    if result.get("_used_openai_arbitration"):
        result["_model_used"] = f"openai:{cfg.OPENAI_MODEL}"
    else:
        result["_model_used"] = f"ollama:{cfg.OLLAMA_MODEL}"
    return result
