"""
Submissions: production path is browser → S3 (presigned PUT) → finalize → Celery.

Multipart upload to Flask is opt-in (ALLOW_FLASK_MULTIPART_UPLOAD=true) for local dev only.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from flask import Blueprint, jsonify, request
from sqlalchemy.orm import selectinload
from werkzeug.utils import secure_filename

from app.audit import log_event
from app.config import Config
from app.extensions import SessionLocal
from app.models import AIScore, Assignment, Enrollment, Submission, SubmissionArtifact
from app.rbac import require_auth
from app.storage import object_exists, presigned_put_url, upload_from_werkzeug_file
from app.tasks import grade_submission

bp = Blueprint("submissions", __name__)

_SUBMITTER_ROLES = frozenset({"student", "teacher", "admin"})


def _is_authorized_submitter(db, user: dict, assignment_id: int) -> bool:
    """
    True if the user may create a submission for this assignment.
    Admins bypass enrollment; teachers and students must be enrolled in the course.
    """
    a = db.query(Assignment).get(assignment_id)
    if not a:
        return False
    if user.get("role") == "admin":
        return True
    return (
        db.query(Enrollment)
        .filter_by(user_id=user["id"], course_id=a.course_id)
        .first()
        is not None
    )


@bp.post("/api/submissions/direct-upload/start")
@require_auth
def direct_upload_start():
    """
    Create submission + artifact rows and return presigned PUT URLs.
    Browser uploads bytes directly to S3 (no large body through Flask).
    """
    user = request.user
    if user["role"] not in _SUBMITTER_ROLES:
        return jsonify({"error": "submission not permitted for this role"}), 403

    body = request.get_json(silent=True) or {}
    assignment_id = body.get("assignment_id")
    files = body.get("files")
    if assignment_id is None or not files or not isinstance(files, list):
        return jsonify({"error": "assignment_id and files[] required"}), 400

    assignment_id = int(assignment_id)
    cfg = Config()
    db = SessionLocal()
    try:
        if not _is_authorized_submitter(db, user, assignment_id):
            return jsonify({"error": "not enrolled or not authorized"}), 403

        sub = Submission(
            assignment_id=assignment_id,
            student_id=user["id"],
            status="uploading",
        )
        db.add(sub)
        db.flush()

        prefix = cfg.UPLOADS_S3_PREFIX.rstrip("/")
        uploads_out = []
        for spec in files:
            raw_name = (spec.get("filename") or "").strip()
            filename = secure_filename(raw_name)
            if not filename:
                continue
            content_type = (spec.get("content_type") or "application/octet-stream").strip()
            kind = filename.split(".")[-1].lower()
            key = f"{prefix}/{assignment_id}/submissions/{sub.id}/{uuid.uuid4().hex}_{filename}"
            art = SubmissionArtifact(
                submission_id=sub.id, kind=kind, s3_key=key
            )
            db.add(art)
            db.flush()
            url = presigned_put_url(cfg, key, content_type)
            uploads_out.append(
                {
                    "artifact_id": art.id,
                    "s3_key": key,
                    "upload_url": url,
                    "content_type": content_type,
                }
            )

        if not uploads_out:
            db.rollback()
            return jsonify({"error": "no valid files"}), 400

        db.commit()
        db.refresh(sub)
        log_event(
            user["id"],
            "CREATE_SUBMISSION_PRESIGN",
            "Submission",
            sub.id,
            {"assignment_id": assignment_id, "n_files": len(uploads_out)},
        )
        return jsonify(
            {
                "submission_id": sub.id,
                "status": sub.status,
                "uploads": uploads_out,
            }
        )
    finally:
        db.close()


@bp.post("/api/submissions/<int:submission_id>/finalize")
@require_auth
def direct_upload_finalize(submission_id: int):
    """Verify S3 objects exist, then enqueue grading at most once (row lock)."""
    user = request.user
    cfg = Config()
    db = SessionLocal()
    try:
        if user["role"] not in _SUBMITTER_ROLES:
            return jsonify({"error": "submission not permitted for this role"}), 403

        sub = (
            db.query(Submission)
            .options(selectinload(Submission.artifacts))
            .filter_by(id=submission_id)
            .with_for_update()
            .first()
        )
        if not sub:
            return jsonify({"error": "not found"}), 404
        if sub.student_id is None:
            return jsonify({"error": "forbidden"}), 403
        if sub.student_id != user["id"]:
            return jsonify({"error": "forbidden"}), 403

        if sub.grading_dispatch_at is not None:
            db.commit()
            return jsonify(
                {
                    "submission_id": sub.id,
                    "status": sub.status,
                    "already_enqueued": True,
                }
            )

        if sub.status in ("queued", "grading", "graded", "needs_review", "error"):
            return jsonify(
                {
                    "submission_id": sub.id,
                    "status": sub.status,
                    "already_finalized": True,
                }
            )

        if sub.status not in ("uploading", "uploaded"):
            return jsonify({"error": f"invalid state: {sub.status}"}), 409

        for art in sub.artifacts:
            if not object_exists(cfg, art.s3_key):
                return jsonify({"error": f"missing object: {art.s3_key}"}), 400

        sub.status = "uploaded"
        sub.updated_at = datetime.utcnow()
        db.flush()

        sub.status = "queued"
        sub.grading_dispatch_at = datetime.utcnow()
        try:
            task = grade_submission.delay(sub.id)
        except Exception:
            db.rollback()
            return jsonify({"error": "failed to enqueue grading job"}), 503
        sub.grading_celery_task_id = task.id
        sub.updated_at = datetime.utcnow()
        db.commit()

        log_event(
            user["id"],
            "FINALIZE_SUBMISSION",
            "Submission",
            sub.id,
            {"celery_task_id": task.id},
        )
        return jsonify(
            {
                "submission_id": sub.id,
                "status": "queued",
                "celery_task_id": task.id,
            }
        )
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@bp.post("/api/submissions")
@require_auth
def submit():
    """Legacy multipart → Flask → S3. Disabled in production (use direct-upload flow)."""
    cfg = Config()
    if not cfg.ALLOW_FLASK_MULTIPART_UPLOAD:
        return (
            jsonify(
                {
                    "error": "multipart upload disabled",
                    "hint": "Use POST /api/submissions/direct-upload/start then S3 PUT then finalize",
                }
            ),
            410,
        )

    user = request.user
    assignment_id = int(request.form["assignment_id"])
    db = SessionLocal()
    cfg = Config()

    try:
        sub = Submission(
            assignment_id=assignment_id,
            student_id=user["id"],
            status="queued",
        )
        db.add(sub)
        db.commit()
        db.refresh(sub)

        for f in request.files.getlist("files"):
            filename = secure_filename(f.filename or "")
            if not filename:
                continue
            kind = filename.split(".")[-1].lower()
            key = (
                f"{cfg.UPLOADS_S3_PREFIX.rstrip('/')}/{assignment_id}/submissions/{sub.id}/"
                f"{uuid.uuid4().hex}_{filename}"
            )
            upload_from_werkzeug_file(cfg, f, key)
            db.add(SubmissionArtifact(submission_id=sub.id, kind=kind, s3_key=key))

        sub.grading_dispatch_at = datetime.utcnow()
        task = grade_submission.delay(sub.id)
        sub.grading_celery_task_id = task.id
        db.commit()

        log_event(
            user["id"], "CREATE_SUBMISSION", "Submission", sub.id, {"assignment_id": assignment_id}
        )
        return jsonify({"submission_id": sub.id, "status": sub.status, "celery_task_id": task.id})
    finally:
        db.close()


@bp.get("/api/submissions/<int:submission_id>")
@require_auth
def get_submission(submission_id: int):
    user = request.user
    db = SessionLocal()
    try:
        sub = db.query(Submission).get(submission_id)
        if not sub:
            return jsonify({"error": "not found"}), 404

        if (
            user["role"] == "student"
            and sub.student_id is not None
            and sub.student_id != user["id"]
        ):
            return jsonify({"error": "forbidden"}), 403

        scores = db.query(AIScore).filter_by(submission_id=sub.id).all()
        log_event(user["id"], "VIEW_SUBMISSION", "Submission", sub.id, {})
        return jsonify(
            {
                "id": sub.id,
                "status": sub.status,
                "final_score": float(sub.final_score) if sub.final_score is not None else None,
                "final_feedback": sub.final_feedback,
                "grading_dispatch_at": sub.grading_dispatch_at.isoformat()
                if sub.grading_dispatch_at
                else None,
                "ai_scores": [
                    {
                        "criterion": s.criterion,
                        "score": float(s.score),
                        "confidence": float(s.confidence),
                        "rationale": s.rationale,
                    }
                    for s in scores
                ],
            }
        )
    finally:
        db.close()
