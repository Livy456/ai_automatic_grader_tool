from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename

from sqlalchemy.orm import Session

from .config import Config
from .extensions import SessionLocal
from .models import AssignmentUpload
from .storage import upload_from_werkzeug_file


bp = Blueprint("assignments", __name__, url_prefix="/api")


# -----------------------------
# Helpers
# -----------------------------

def _db() -> Session:
    """
    Create a per-request SQLAlchemy session.
    We keep it simple for now and always close it in route handlers.
    """
    if SessionLocal is None:
        raise RuntimeError("Database not initialized. Did you call init_db() in create_app()?")

    return SessionLocal()


def _now() -> datetime:
    return datetime.now()


def _allowed_file(filename: str) -> bool:
    """
    You can tighten this. For now we accept many modalities.
    """
    allowed = {
        ".ipynb", ".py", ".pdf",
        ".png", ".jpg", ".jpeg", ".webp",
        ".mp4", ".mov", ".webm",
        ".txt", ".md",
        ".csv", ".json",
    }
    ext = os.path.splitext(filename.lower())[1]
    return ext in allowed


def _save_upload_to_s3(file_storage) -> Tuple[str, str]:
    """
    Stream file to S3 (AWS or MinIO). Returns (assignment_upload_uuid, s3_object_key).
    storage_uri in DB holds the object key (not s3://).
    """
    filename = secure_filename(file_storage.filename or "upload.bin")
    if not _allowed_file(filename):
        raise ValueError(f"File type not allowed: {filename}")

    assignment_id = str(uuid.uuid4())
    cfg = Config()
    key = f"ingest/assignment-uploads/{assignment_id}/{filename}"
    upload_from_werkzeug_file(cfg, file_storage, key)
    return assignment_id, key


def _assignment_to_dict(a: AssignmentUpload) -> Dict[str, Any]:
    return {
        "id": a.id,
        "filename": a.filename,
        "status": a.status,
        "suggested_grade": a.suggested_grade,
        "feedback": a.feedback,
        "created_at": a.created_at.isoformat() if a.created_at else None,
        "updated_at": a.updated_at.isoformat() if a.updated_at else None,
    }


# -----------------------------
# Routes
# -----------------------------

@bp.get("/assignments")
def list_assignments():
    """
    GET /api/assignments -> list most recent assignments.
    """
    db = _db()
    try:
        items = (
            db.query(AssignmentUpload)
            .order_by(AssignmentUpload.created_at.desc())
            .limit(100)
            .all()
        )
        return jsonify([_assignment_to_dict(a) for a in items])
    finally:
        db.close()


@bp.post("/assignments")
def create_assignment():
    """
    POST /api/assignments
    - Accepts multipart/form-data with "file"
    - Stores file in S3 (see storage.py / S3_BUCKET_SETUP.md)
    - Inserts Assignment row into Postgres
    """
    if "file" not in request.files:
        return jsonify({"error": "Missing file field 'file' in form-data"}), 400

    f = request.files["file"]
    if not f or not f.filename:
        return jsonify({"error": "Empty file upload"}), 400

    try:
        assignment_id, storage_uri = _save_upload_to_s3(f)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    db = _db()
    try:
        a = AssignmentUpload(
            id=assignment_id,
            filename=secure_filename(f.filename),
            storage_uri=storage_uri,
            status="uploaded",
            suggested_grade=None,
            feedback=None,
            created_at=_now(),
            updated_at=_now(),
        )
        db.add(a)
        db.commit()
        return jsonify({"id": a.id}), 201
    finally:
        db.close()


@bp.get("/assignments/<assignment_id>")
def get_assignment(assignment_id: str):
    """
    GET /api/assignments/<id> -> fetch status/result
    """
    db = _db()
    try:
        a: Optional[AssignmentUpload] = db.query(AssignmentUpload).filter(AssignmentUpload.id == assignment_id).first()
        if not a:
            return jsonify({"error": "Assignment not found"}), 404
        return jsonify(_assignment_to_dict(a))
    finally:
        db.close()


@bp.post("/assignments/<assignment_id>/grade")
def grade_assignment(assignment_id: str):
    """
    POST /api/assignments/<id>/grade -> start AI job (sync for now)
    """
    db = _db()
    try:
        a: Optional[AssignmentUpload] = db.query(AssignmentUpload).filter(AssignmentUpload.id == assignment_id).first()
        if not a:
            return jsonify({"error": "Assignment not found"}), 404

        if a.status in ("grading", "graded"):
            # idempotent: don't re-run unless you explicitly add a "regrade" endpoint
            return jsonify({"ok": True, "status": a.status})

        # mark grading
        a.status = "grading"
        a.updated_at = _now()
        db.commit()

        # Upload-only flow: no autograde here (course submissions use Celery + multimodal pipeline).
        a.suggested_grade = None
        a.feedback = (
            "No automatic grade for this upload endpoint. "
            "Use course assignment submissions for AI grading."
        )
        a.status = "graded"

        a.updated_at = _now()
        db.commit()

        return jsonify({"ok": True})
    finally:
        db.close()
