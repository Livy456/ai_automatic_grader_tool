from celery import Celery
from .config import Config
from .extensions import SessionLocal
from .models import Submission, Assignment, AIScore
from .storage import client
from .grading.pipelines import run_grading_pipeline

celery_app = Celery(__name__)

def init_celery(app):
    celery_app.conf.broker_url = app.config["REDIS_URL"]
    celery_app.conf.result_backend = app.config["REDIS_URL"]

@celery_app.task(name="grade_submission")
def grade_submission(submission_id: int):
    from flask import current_app
    cfg = current_app.config_obj  # set in app factory

    db = SessionLocal()
    try:
        sub = db.query(Submission).get(submission_id)
        if not sub:
            return
        sub.status = "grading"; db.commit()

        assignment = db.query(Assignment).get(sub.assignment_id)

        # fetch artifacts from storage
        s3 = client(cfg)
        artifacts = {}
        for art in sub.artifacts:
            resp = s3.get_object(cfg.S3_BUCKET, art.s3_key)
            data = resp.read()
            # normalize kinds
            if art.kind.endswith("pdf"): artifacts["pdf"] = data
            if art.kind.endswith("txt"): artifacts["txt"] = data
            if art.kind.endswith("ipynb"): artifacts["ipynb"] = data
            if art.kind.endswith("py"): artifacts["py"] = data
            if art.kind.endswith("mp4"): artifacts["mp4"] = data

        result = run_grading_pipeline(cfg, assignment, artifacts)

        # store criterion scores
        criteria = result.get("criteria", [])
        overall = result.get("overall", {})
        flags = set(result.get("flags", []))

        for c in criteria:
            db.add(AIScore(
                submission_id=sub.id,
                criterion=c["name"],
                score=c["score"],
                confidence=c.get("confidence", 0.5),
                rationale=c.get("rationale",""),
                evidence=c.get("evidence", {}),
                model=cfg.OLLAMA_MODEL
            ))

        sub.final_score = overall.get("score", 0)
        sub.final_feedback = overall.get("summary", "")

        # basic “needs review” policy for reflective/critical criteria
        low_conf = any(float(c.get("confidence",0)) < 0.70 for c in criteria)
        if low_conf or "needs_review" in flags:
            sub.status = "needs_review"
        else:
            sub.status = "graded"

        db.commit()
    except Exception:
        db.rollback()
        if "sub" in locals() and sub:
            sub.status = "error"
            db.commit()
        raise
    finally:
        db.close()
