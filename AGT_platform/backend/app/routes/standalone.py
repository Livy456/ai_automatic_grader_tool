"""
Standalone autograder: submissions not tied to courses (no enrollment).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta

from flask import Blueprint, jsonify, request
from sqlalchemy.orm import selectinload
from werkzeug.utils import secure_filename

from app.audit import log_event
from app.config import Config
from app.extensions import SessionLocal
from app.models import StandaloneAIScore, StandaloneArtifact, StandaloneSubmission
from app.rbac import require_auth
from app.storage import object_exists, presigned_put_url
from app.tasks import grade_standalone_submission

bp = Blueprint("standalone", __name__)

_MAX_FILES = 20
_MAX_TITLE_LEN = 512
_STANDALONE_RATE_WINDOW_HOURS = 1
_STANDALONE_RATE_MAX = 10


def _kind_for_spec(spec: dict, default: str) -> str:
    raw = (spec.get("artifact_kind") or spec.get("kind") or default).strip().lower()
    if raw in ("submission", "rubric", "answer_key"):
        return raw
    return default


def _storage_kind_for_file(spec: dict, filename: str) -> str:
    """S3 artifact kind: extension bucket or rubric/answer_key."""
    role = _kind_for_spec(spec, "submission")
    if role == "rubric":
        return "rubric"
    if role == "answer_key":
        return "answer_key"
    ext = (filename.rsplit(".", 1)[-1] if "." in filename else "").lower()
    return ext or "bin"


def _can_access_submission(user: dict, row: StandaloneSubmission) -> bool:
    if user.get("role") == "admin":
        return True
    return int(row.user_id) == int(user["id"])


@bp.post("/api/standalone/submissions/start")
@require_auth
def standalone_start():
    user = request.user
    body = request.get_json(silent=True) or {}
    title = (body.get("title") or "").strip()
    if not title:
        return jsonify({"error": "title is required"}), 400
    if len(title) > _MAX_TITLE_LEN:
        return jsonify({"error": "title too long"}), 400

    files = body.get("files")
    if not files or not isinstance(files, list):
        return jsonify({"error": "files[] required"}), 400
    if len(files) > _MAX_FILES:
        return jsonify({"error": f"at most {_MAX_FILES} files"}), 400

    rubric_text = (body.get("rubric_text") or "").strip() or None
    answer_key_text = (body.get("answer_key_text") or "").strip() or None

    cfg = Config()
    db = SessionLocal()
    try:
        since = datetime.utcnow() - timedelta(hours=_STANDALONE_RATE_WINDOW_HOURS)
        recent = (
            db.query(StandaloneSubmission)
            .filter(
                StandaloneSubmission.user_id == user["id"],
                StandaloneSubmission.created_at >= since,
                StandaloneSubmission.status != "deleted",
            )
            .count()
        )
        if recent >= _STANDALONE_RATE_MAX:
            return (
                jsonify(
                    {
                        "error": "rate limit",
                        "detail": f"max {_STANDALONE_RATE_MAX} standalone submissions per {_STANDALONE_RATE_WINDOW_HOURS}h",
                    }
                ),
                429,
            )

        sub = StandaloneSubmission(
            user_id=user["id"],
            title=title,
            status="uploading",
            rubric_text=rubric_text,
            answer_key_text=answer_key_text,
        )
        db.add(sub)
        db.flush()

        uploads_out = []
        for spec in files:
            raw_name = (spec.get("filename") or "").strip()
            filename = secure_filename(raw_name)
            if not filename:
                continue
            content_type = (spec.get("content_type") or "application/octet-stream").strip()
            skind = _storage_kind_for_file(spec, filename)
            key = (
                f"standalone/{user['id']}/submissions/{sub.id}/"
                f"{uuid.uuid4().hex}_{filename}"
            )
            art = StandaloneArtifact(
                submission_id=sub.id,
                kind=skind,
                s3_key=key,
                filename=filename,
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
            "CREATE_STANDALONE_SUBMISSION",
            "StandaloneSubmission",
            sub.id,
            {"n_files": len(uploads_out)},
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


@bp.post("/api/standalone/submissions/<int:submission_id>/finalize")
@require_auth
def standalone_finalize(submission_id: int):
    user = request.user
    cfg = Config()
    db = SessionLocal()
    try:
        sub = (
            db.query(StandaloneSubmission)
            .options(selectinload(StandaloneSubmission.artifacts))
            .filter_by(id=submission_id)
            .with_for_update()
            .first()
        )
        if not sub or sub.status == "deleted":
            return jsonify({"error": "not found"}), 404
        if not _can_access_submission(user, sub):
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
            task = grade_standalone_submission.delay(sub.id)
        except Exception:
            db.rollback()
            return jsonify({"error": "failed to enqueue grading job"}), 503
        sub.grading_celery_task_id = task.id
        sub.updated_at = datetime.utcnow()
        db.commit()

        log_event(
            user["id"],
            "FINALIZE_STANDALONE",
            "StandaloneSubmission",
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


@bp.get("/api/standalone/submissions")
@require_auth
def standalone_list():
    user = request.user
    page = int(request.args.get("page") or 1)
    per_page = min(int(request.args.get("per_page") or 20), 100)
    if page < 1:
        page = 1

    db = SessionLocal()
    try:
        q = (
            db.query(StandaloneSubmission)
            .filter(
                StandaloneSubmission.user_id == user["id"],
                StandaloneSubmission.status != "deleted",
            )
        )
        total = q.count()
        rows = (
            q.order_by(StandaloneSubmission.created_at.desc())
            .offset((page - 1) * per_page)
            .limit(per_page)
            .all()
        )
        items = [
            {
                "id": r.id,
                "title": r.title,
                "status": r.status,
                "final_score": float(r.final_score) if r.final_score is not None else None,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]
        return jsonify(
            {"items": items, "total": total, "page": page, "per_page": per_page}
        )
    finally:
        db.close()


@bp.get("/api/standalone/submissions/<int:submission_id>")
@require_auth
def standalone_get(submission_id: int):
    user = request.user
    db = SessionLocal()
    try:
        sub = db.query(StandaloneSubmission).filter_by(id=submission_id).first()
        if not sub:
            return jsonify({"error": "not found"}), 404
        if sub.status == "deleted" and user.get("role") != "admin":
            return jsonify({"error": "not found"}), 404
        if not _can_access_submission(user, sub):
            return jsonify({"error": "forbidden"}), 403

        scores = db.query(StandaloneAIScore).filter_by(submission_id=sub.id).all()
        log_event(user["id"], "VIEW_STANDALONE", "StandaloneSubmission", sub.id, {})
        return jsonify(
            {
                "id": sub.id,
                "title": sub.title,
                "status": sub.status,
                "final_score": float(sub.final_score) if sub.final_score is not None else None,
                "final_feedback": sub.final_feedback,
                "grading_dispatch_at": sub.grading_dispatch_at.isoformat()
                if sub.grading_dispatch_at
                else None,
                "created_at": sub.created_at.isoformat() if sub.created_at else None,
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


@bp.delete("/api/standalone/submissions/<int:submission_id>")
@require_auth
def standalone_delete(submission_id: int):
    user = request.user
    db = SessionLocal()
    try:
        sub = (
            db.query(StandaloneSubmission)
            .filter_by(id=submission_id)
            .with_for_update()
            .first()
        )
        if not sub or sub.status == "deleted":
            return jsonify({"error": "not found"}), 404
        if not _can_access_submission(user, sub):
            return jsonify({"error": "forbidden"}), 403

        sub.status = "deleted"
        sub.updated_at = datetime.utcnow()
        db.commit()
        log_event(user["id"], "DELETE_STANDALONE", "StandaloneSubmission", sub.id, {})
        return jsonify({"ok": True})
    finally:
        db.close()
