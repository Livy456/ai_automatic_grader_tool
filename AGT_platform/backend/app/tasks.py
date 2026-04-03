from datetime import datetime

from celery import Celery
from sqlalchemy.orm import selectinload

from .config import Config
from .extensions import SessionLocal, engine, init_db
from .grading.pipelines import run_grading_pipeline, run_standalone_grading_pipeline
from .grading.tools import extract_text_from_pdf
from .models import (
    AIScore,
    Assignment,
    StandaloneAIScore,
    StandaloneArtifact,
    StandaloneSubmission,
    Submission,
)
from .storage import get_object_bytes

celery_app = Celery(__name__)

_cfg = Config()
celery_app.conf.broker_url = _cfg.REDIS_URL
celery_app.conf.result_backend = _cfg.REDIS_URL
celery_app.conf.task_routes = {
    "grade_submission": {"queue": "gpu"},
    "grade_standalone_submission": {"queue": "gpu"},
}
# Bound prefetch so one worker does not hoard many large grading tasks in memory.
celery_app.conf.worker_prefetch_multiplier = max(1, _cfg.CELERY_WORKER_PREFETCH)


def init_celery(app):
    celery_app.conf.broker_url = app.config["REDIS_URL"]
    celery_app.conf.result_backend = app.config["REDIS_URL"]


def _ensure_db():
    if engine is None:
        init_db(Config().DATABASE_URL)


@celery_app.task(name="grade_submission", bind=True, max_retries=2)
def grade_submission(self, submission_id: int):
    """
    GPU-queue grading. Idempotent: only one successful transition from queued → grading
    per submission; duplicate deliveries no-op after work started.
    """
    _ensure_db()
    cfg = Config()
    db = SessionLocal()
    sub = None
    try:
        sub = (
            db.query(Submission)
            .options(selectinload(Submission.artifacts))
            .filter_by(id=submission_id)
            .with_for_update()
            .first()
        )
        if not sub:
            return
        # Another worker already owns or finished
        if sub.status in ("grading", "graded", "needs_review"):
            return
        if sub.status == "error":
            return
        if sub.status != "queued":
            # e.g. still uploading — do not grade
            return

        sub.status = "grading"
        sub.updated_at = datetime.utcnow()
        db.commit()

        assignment = db.query(Assignment).get(sub.assignment_id)
        if not assignment:
            sub.status = "error"
            db.commit()
            return

        artifacts = {}
        for art in sub.artifacts:
            data = get_object_bytes(cfg, art.s3_key)
            if art.kind.endswith("pdf"):
                artifacts["pdf"] = data
            if art.kind.endswith("txt"):
                artifacts["txt"] = data
            if art.kind.endswith("ipynb"):
                artifacts["ipynb"] = data
            if art.kind.endswith("py"):
                artifacts["py"] = data
            if art.kind.endswith("mp4"):
                artifacts["mp4"] = data

        result = run_grading_pipeline(cfg, assignment, artifacts)
        model_used = (result.pop("_model_used", None) or cfg.OLLAMA_MODEL)[:200]
        result.pop("_used_openai_arbitration", None)

        criteria = result.get("criteria", [])
        overall = result.get("overall", {})
        flags = set(result.get("flags", []))

        # Idempotent persistence: delete prior AI scores for this submission then insert
        db.query(AIScore).filter_by(submission_id=sub.id).delete()
        for c in criteria:
            db.add(
                AIScore(
                    submission_id=sub.id,
                    criterion=c["name"],
                    score=c["score"],
                    confidence=c.get("confidence", 0.5),
                    rationale=c.get("rationale", ""),
                    evidence=c.get("evidence", {}),
                    model=model_used,
                )
            )

        sub = db.query(Submission).get(submission_id)
        sub.final_score = overall.get("score", 0)
        sub.final_feedback = overall.get("summary", "")
        low_conf = any(float(c.get("confidence", 0)) < 0.70 for c in criteria)
        if low_conf or "needs_review" in flags:
            sub.status = "needs_review"
        else:
            sub.status = "graded"
        sub.updated_at = datetime.utcnow()
        db.commit()
    except Exception:
        db.rollback()
        if sub:
            s2 = db.query(Submission).get(submission_id)
            if s2 and s2.status == "grading":
                s2.status = "error"
                s2.updated_at = datetime.utcnow()
                db.commit()
        raise
    finally:
        db.close()


def _artifact_bucket_key(kind: str, filename: str) -> str | None:
    k = (kind or "").lower().split(".")[-1]
    fn = (filename or "").lower()
    for ext in ("pdf", "txt", "ipynb", "py", "mp4", "zip", "png", "jpg", "jpeg", "docx"):
        if k == ext or fn.endswith(f".{ext}"):
            if ext == "jpeg":
                return "jpg"
            return ext
    return None


def _excerpt_file_bytes(filename: str, data: bytes) -> str:
    if not data:
        return ""
    fn = (filename or "").lower()
    if fn.endswith(".pdf") or (len(data) > 4 and data[:4] == b"%PDF"):
        try:
            return extract_text_from_pdf(data)
        except Exception:
            return ""
    return data.decode("utf-8", errors="ignore")[:80000]


@celery_app.task(name="grade_standalone_submission", bind=True, max_retries=2)
def grade_standalone_submission(self, submission_id: int):
    _ensure_db()
    cfg = Config()
    db = SessionLocal()
    sub = None
    try:
        sub = (
            db.query(StandaloneSubmission)
            .filter_by(id=submission_id)
            .with_for_update()
            .first()
        )
        if not sub:
            return
        if sub.status in ("grading", "graded", "needs_review"):
            return
        if sub.status == "error":
            return
        if sub.status != "queued":
            return

        sub.status = "grading"
        sub.updated_at = datetime.utcnow()
        db.commit()

        arts = db.query(StandaloneArtifact).filter_by(submission_id=sub.id).all()
        main: dict[str, bytes] = {}
        rubric_ex = ""
        answer_ex = ""
        for art in arts:
            data = get_object_bytes(cfg, art.s3_key)
            if art.kind in ("rubric", "answer_key"):
                ex = _excerpt_file_bytes(art.filename, data)
                if art.kind == "rubric":
                    rubric_ex = (rubric_ex + "\n\n" + ex).strip() if rubric_ex else ex
                else:
                    answer_ex = (answer_ex + "\n\n" + ex).strip() if answer_ex else ex
                continue
            bkey = _artifact_bucket_key(art.kind, art.filename)
            if bkey:
                main[bkey] = data

        result = run_standalone_grading_pipeline(
            cfg,
            main,
            sub.title or "Untitled",
            sub.rubric_text,
            sub.answer_key_text,
            rubric_ex or None,
            answer_ex or None,
        )
        model_used = (result.pop("_model_used", None) or cfg.OLLAMA_MODEL)[:200]
        result.pop("_used_openai_arbitration", None)

        criteria = result.get("criteria", [])
        overall = result.get("overall", {})
        flags = set(result.get("flags", []))

        db.query(StandaloneAIScore).filter_by(submission_id=sub.id).delete()
        for c in criteria:
            db.add(
                StandaloneAIScore(
                    submission_id=sub.id,
                    criterion=c["name"],
                    score=c["score"],
                    confidence=c.get("confidence", 0.5),
                    rationale=c.get("rationale", ""),
                    evidence=c.get("evidence", {}),
                    model=model_used,
                )
            )

        sub = db.query(StandaloneSubmission).get(submission_id)
        sub.final_score = overall.get("score", 0)
        sub.final_feedback = overall.get("summary", "")
        low_conf = any(float(c.get("confidence", 0)) < 0.70 for c in criteria)
        if low_conf or "needs_review" in flags:
            sub.status = "needs_review"
        else:
            sub.status = "graded"
        sub.updated_at = datetime.utcnow()
        db.commit()
    except Exception:
        db.rollback()
        if sub:
            s2 = db.query(StandaloneSubmission).get(submission_id)
            if s2 and s2.status == "grading":
                s2.status = "error"
                s2.updated_at = datetime.utcnow()
                db.commit()
        raise
    finally:
        db.close()
