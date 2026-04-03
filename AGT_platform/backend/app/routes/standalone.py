"""
Public standalone autograder: no JWT required.

Creates rows in assignments (course_id NULL), submissions (student_id optional),
submission_artifacts, and ai_scores via the existing grade_submission Celery task.
"""
from __future__ import annotations

import copy
import uuid
from datetime import datetime, timedelta

from flask import Blueprint, jsonify, request
from sqlalchemy.orm import selectinload
from werkzeug.utils import secure_filename

from app.audit import log_event
from app.config import Config
from app.extensions import SessionLocal
from app.grading.pipelines import DEFAULT_STANDALONE_RUBRIC
from app.models import AIScore, Assignment, Submission, SubmissionArtifact
from app.rbac import get_user_from_token
from app.storage import object_exists, presigned_put_url
from app.tasks import grade_submission

bp = Blueprint("standalone", __name__)

_MAX_FILES = 20
_MAX_TITLE_LEN = 512
_STANDALONE_RATE_WINDOW_HOURS = 1
_STANDALONE_RATE_MAX = 10


def _client_ip() -> str:
    xff = (request.headers.get("X-Forwarded-For") or "").split(",")[0].strip()
    if xff:
        return xff[:64]
    return (request.remote_addr or "")[:64]


def _optional_user() -> dict | None:
    return get_user_from_token()


def _assignment_for_sub(db, sub: Submission) -> Assignment | None:
    return db.query(Assignment).get(sub.assignment_id)


def _is_public_autograder_submission(sub: Submission, db) -> bool:
    a = _assignment_for_sub(db, sub)
    return bool(a and a.course_id is None)


def _can_view_submission(sub: Submission, db, user: dict | None) -> bool:
    a = _assignment_for_sub(db, sub)
    if not a:
        return False
    if a.course_id is None:
        return True
    if not user:
        return False
    if user.get("role") == "admin":
        return True
    return sub.student_id is not None and int(sub.student_id) == int(user["id"])


def _can_mutate_public_submission(sub: Submission, user: dict | None, db) -> bool:
    if not _is_public_autograder_submission(sub, db):
        return False
    if user:
        if user.get("role") == "admin":
            return True
        if sub.student_id is not None and int(sub.student_id) == int(user["id"]):
            return True
    if sub.student_id is None:
        ip = _client_ip()
        return bool(sub.submitter_ip and ip and sub.submitter_ip == ip)
    return False


def _kind_for_spec(spec: dict, default: str) -> str:
    raw = (spec.get("artifact_kind") or spec.get("kind") or default).strip().lower()
    if raw in ("submission", "rubric", "answer_key"):
        return raw
    return default


def _storage_kind_for_file(spec: dict, filename: str) -> str:
    role = _kind_for_spec(spec, "submission")
    if role == "rubric":
        return "rubric"
    if role == "answer_key":
        return "answer_key"
    ext = (filename.rsplit(".", 1)[-1] if "." in filename else "").lower()
    return ext or "bin"


def _parse_enqueue_grading(body: dict) -> bool:
    if body.get("enqueue_grading") is False:
        return False
    if body.get("defer_grading") is True:
        return False
    return True


@bp.post("/api/standalone/submissions/start")
def standalone_start():
    """Create assignment (no course) + submission + artifact rows; return presigned PUT URLs."""
    user = _optional_user()
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
    grading_instructions = (body.get("grading_instructions") or "").strip() or None

    cfg = Config()
    client_ip = _client_ip()
    db = SessionLocal()
    try:
        since = datetime.utcnow() - timedelta(hours=_STANDALONE_RATE_WINDOW_HOURS)
        if not user:
            recent_ip = (
                db.query(Submission)
                .join(Assignment)
                .filter(
                    Assignment.course_id.is_(None),
                    Submission.submitter_ip == client_ip,
                    Submission.student_id.is_(None),
                    Submission.created_at >= since,
                    Submission.status != "deleted",
                )
                .count()
            )
            if recent_ip >= _STANDALONE_RATE_MAX:
                return (
                    jsonify(
                        {
                            "error": "rate limit",
                            "detail": f"max {_STANDALONE_RATE_MAX} anonymous autograder uploads per {_STANDALONE_RATE_WINDOW_HOURS}h from this network",
                        }
                    ),
                    429,
                )
        else:
            recent = (
                db.query(Submission)
                .join(Assignment)
                .filter(
                    Assignment.course_id.is_(None),
                    Submission.student_id == user["id"],
                    Submission.created_at >= since,
                    Submission.status != "deleted",
                )
                .count()
            )
            if recent >= _STANDALONE_RATE_MAX:
                return (
                    jsonify(
                        {
                            "error": "rate limit",
                            "detail": f"max {_STANDALONE_RATE_MAX} autograder uploads per {_STANDALONE_RATE_WINDOW_HOURS}h",
                        }
                    ),
                    429,
                )

        assignment = Assignment(
            course_id=None,
            title=title,
            description=title,
            modality="written",
            rubric=copy.deepcopy(DEFAULT_STANDALONE_RUBRIC),
            created_at=datetime.utcnow(),
            due_date=None,
            grader_rubric_text=rubric_text,
            grader_answer_key_text=answer_key_text,
            grader_instructions=grading_instructions,
        )
        db.add(assignment)
        db.flush()

        sub = Submission(
            assignment_id=assignment.id,
            student_id=int(user["id"]) if user else None,
            status="uploading",
            submitter_ip=client_ip or None,
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
            skind = _storage_kind_for_file(spec, filename)
            key = f"{prefix}/{assignment.id}/submissions/{sub.id}/{uuid.uuid4().hex}_{filename}"
            art = SubmissionArtifact(submission_id=sub.id, kind=skind, s3_key=key)
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
            user["id"] if user else None,
            "CREATE_PUBLIC_AUTOGRADER",
            "Submission",
            sub.id,
            {"assignment_id": assignment.id, "n_files": len(uploads_out)},
        )
        return jsonify(
            {
                "submission_id": sub.id,
                "assignment_id": assignment.id,
                "status": sub.status,
                "uploads": uploads_out,
            }
        )
    finally:
        db.close()


@bp.post("/api/standalone/submissions/<int:submission_id>/finalize")
def standalone_finalize(submission_id: int):
    user = _optional_user()
    cfg = Config()
    body = request.get_json(silent=True) or {}
    enqueue_grading = _parse_enqueue_grading(body)
    db = SessionLocal()
    try:
        sub = (
            db.query(Submission)
            .options(selectinload(Submission.artifacts))
            .filter_by(id=submission_id)
            .with_for_update()
            .first()
        )
        if not sub or sub.status == "deleted":
            return jsonify({"error": "not found"}), 404
        if not _is_public_autograder_submission(sub, db):
            return jsonify({"error": "not found"}), 404
        if not _can_mutate_public_submission(sub, user, db):
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

        if not enqueue_grading:
            db.commit()
            log_event(
                user["id"] if user else None,
                "FINALIZE_PUBLIC_AUTOGRADER_UPLOAD",
                "Submission",
                sub.id,
                {"enqueue_grading": False},
            )
            return jsonify(
                {
                    "submission_id": sub.id,
                    "status": "uploaded",
                    "enqueue_grading": False,
                }
            )

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
            user["id"] if user else None,
            "FINALIZE_PUBLIC_AUTOGRADER",
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


@bp.patch("/api/standalone/submissions/<int:submission_id>/context")
def standalone_patch_context(submission_id: int):
    user = _optional_user()
    body = request.get_json(silent=True) or {}
    db = SessionLocal()
    try:
        sub = (
            db.query(Submission)
            .filter_by(id=submission_id)
            .with_for_update()
            .first()
        )
        if not sub or sub.status == "deleted":
            return jsonify({"error": "not found"}), 404
        if not _is_public_autograder_submission(sub, db):
            return jsonify({"error": "not found"}), 404
        if not _can_mutate_public_submission(sub, user, db):
            return jsonify({"error": "forbidden"}), 403
        if sub.status != "uploaded" or sub.grading_dispatch_at is not None:
            return (
                jsonify(
                    {
                        "error": "invalid state",
                        "detail": "context can only be edited after upload and before grading is queued",
                    }
                ),
                409,
            )

        a = _assignment_for_sub(db, sub)
        if not a:
            return jsonify({"error": "not found"}), 404

        if "rubric_text" in body:
            v = body.get("rubric_text")
            a.grader_rubric_text = (str(v).strip() if v is not None else "") or None
        if "answer_key_text" in body:
            v = body.get("answer_key_text")
            a.grader_answer_key_text = (str(v).strip() if v is not None else "") or None
        if "grading_instructions" in body:
            v = body.get("grading_instructions")
            a.grader_instructions = (str(v).strip() if v is not None else "") or None

        sub.updated_at = datetime.utcnow()
        db.commit()
        log_event(
            user["id"] if user else None,
            "PATCH_PUBLIC_AUTOGRADER_CONTEXT",
            "Submission",
            sub.id,
            {},
        )
        return jsonify({"submission_id": sub.id, "ok": True})
    finally:
        db.close()


@bp.post("/api/standalone/submissions/<int:submission_id>/context_files/presign")
def standalone_presign_context_files(submission_id: int):
    user = _optional_user()
    body = request.get_json(silent=True) or {}
    files = body.get("files")
    if not files or not isinstance(files, list):
        return jsonify({"error": "files[] required"}), 400

    cfg = Config()
    db = SessionLocal()
    try:
        sub = (
            db.query(Submission)
            .options(selectinload(Submission.artifacts))
            .filter_by(id=submission_id)
            .with_for_update()
            .first()
        )
        if not sub or sub.status == "deleted":
            return jsonify({"error": "not found"}), 404
        if not _is_public_autograder_submission(sub, db):
            return jsonify({"error": "not found"}), 404
        if not _can_mutate_public_submission(sub, user, db):
            return jsonify({"error": "forbidden"}), 403
        if sub.status != "uploaded" or sub.grading_dispatch_at is not None:
            return jsonify({"error": "invalid state for context file upload"}), 409

        if len(sub.artifacts) + len(files) > _MAX_FILES:
            return jsonify({"error": f"at most {_MAX_FILES} files per submission"}), 400

        a = _assignment_for_sub(db, sub)
        if not a:
            return jsonify({"error": "not found"}), 404

        prefix = cfg.UPLOADS_S3_PREFIX.rstrip("/")
        uploads_out = []
        for spec in files:
            raw_name = (spec.get("filename") or "").strip()
            filename = secure_filename(raw_name)
            if not filename:
                continue
            role = _kind_for_spec(spec, "rubric")
            if role not in ("rubric", "answer_key"):
                return jsonify({"error": "context files must be rubric or answer_key"}), 400
            content_type = (spec.get("content_type") or "application/octet-stream").strip()
            skind = _storage_kind_for_file(spec, filename)
            key = f"{prefix}/{a.id}/submissions/{sub.id}/{uuid.uuid4().hex}_{filename}"
            art = SubmissionArtifact(submission_id=sub.id, kind=skind, s3_key=key)
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
        log_event(
            user["id"] if user else None,
            "PRESIGN_PUBLIC_AUTOGRADER_CONTEXT",
            "Submission",
            sub.id,
            {"n_files": len(uploads_out)},
        )
        return jsonify(
            {
                "submission_id": sub.id,
                "uploads": uploads_out,
            }
        )
    finally:
        db.close()


@bp.post("/api/standalone/submissions/<int:submission_id>/enqueue_grading")
def standalone_enqueue_grading(submission_id: int):
    user = _optional_user()
    cfg = Config()
    db = SessionLocal()
    try:
        sub = (
            db.query(Submission)
            .options(selectinload(Submission.artifacts))
            .filter_by(id=submission_id)
            .with_for_update()
            .first()
        )
        if not sub or sub.status == "deleted":
            return jsonify({"error": "not found"}), 404
        if not _is_public_autograder_submission(sub, db):
            return jsonify({"error": "not found"}), 404
        if not _can_mutate_public_submission(sub, user, db):
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

        if sub.status != "uploaded":
            return jsonify({"error": f"expected status uploaded, got {sub.status}"}), 409

        for art in sub.artifacts:
            if not object_exists(cfg, art.s3_key):
                return jsonify({"error": f"missing object: {art.s3_key}"}), 400

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
            user["id"] if user else None,
            "ENQUEUE_PUBLIC_AUTOGRADER",
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


@bp.get("/api/standalone/submissions")
def standalone_list():
    user = _optional_user()
    page = int(request.args.get("page") or 1)
    per_page = min(int(request.args.get("per_page") or 20), 100)
    if page < 1:
        page = 1

    ip = _client_ip()
    db = SessionLocal()
    try:
        q = (
            db.query(Submission)
            .join(Assignment)
            .filter(
                Assignment.course_id.is_(None),
                Submission.status != "deleted",
            )
        )
        if user:
            q = q.filter(Submission.student_id == user["id"])
        else:
            if not ip:
                return jsonify({"items": [], "total": 0, "page": page, "per_page": per_page})
            q = q.filter(
                Submission.student_id.is_(None),
                Submission.submitter_ip == ip,
            )

        total = q.count()
        rows = (
            q.order_by(Submission.created_at.desc())
            .offset((page - 1) * per_page)
            .limit(per_page)
            .all()
        )
        items = []
        for r in rows:
            a = _assignment_for_sub(db, r)
            items.append(
                {
                    "id": r.id,
                    "title": a.title if a else "—",
                    "status": r.status,
                    "final_score": float(r.final_score) if r.final_score is not None else None,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
            )
        return jsonify(
            {"items": items, "total": total, "page": page, "per_page": per_page}
        )
    finally:
        db.close()


@bp.get("/api/standalone/submissions/<int:submission_id>")
def standalone_get(submission_id: int):
    user = _optional_user()
    db = SessionLocal()
    try:
        sub = db.query(Submission).filter_by(id=submission_id).first()
        if not sub or sub.status == "deleted":
            return jsonify({"error": "not found"}), 404
        if not _can_view_submission(sub, db, user):
            return jsonify({"error": "forbidden"}), 403

        a = _assignment_for_sub(db, sub)
        scores = db.query(AIScore).filter_by(submission_id=sub.id).all()
        log_event(user["id"] if user else None, "VIEW_PUBLIC_AUTOGRADER", "Submission", sub.id, {})
        return jsonify(
            {
                "id": sub.id,
                "title": a.title if a else "—",
                "status": sub.status,
                "final_score": float(sub.final_score) if sub.final_score is not None else None,
                "final_feedback": sub.final_feedback,
                "grading_instructions": a.grader_instructions if a else None,
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
def standalone_delete(submission_id: int):
    user = _optional_user()
    db = SessionLocal()
    try:
        sub = (
            db.query(Submission)
            .filter_by(id=submission_id)
            .with_for_update()
            .first()
        )
        if not sub or sub.status == "deleted":
            return jsonify({"error": "not found"}), 404
        if not _is_public_autograder_submission(sub, db):
            return jsonify({"error": "not found"}), 404
        if not _can_mutate_public_submission(sub, user, db):
            return jsonify({"error": "forbidden"}), 403

        sub.status = "deleted"
        sub.updated_at = datetime.utcnow()
        db.commit()
        log_event(user["id"] if user else None, "DELETE_PUBLIC_AUTOGRADER", "Submission", sub.id, {})
        return jsonify({"ok": True})
    finally:
        db.close()
