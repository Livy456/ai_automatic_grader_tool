# I MIGHT END UP MOVING THIS TO MY ROUTES FOLDER!!!
from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

from flask import Blueprint, jsonify, request, current_app
from werkzeug.utils import secure_filename

from sqlalchemy.orm import Session

from .extensions import SessionLocal
from .models import Assignment  # you'll add this model (see below)


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
    return datetime.utcnow()


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


def _storage_mode() -> str:
    """
    'minio' or 'local'
    """
    return current_app.config.get("STORAGE_MODE", "local").lower()


def _save_file_local(file_storage) -> Tuple[str, str]:
    """
    Save file to local disk (backend/uploads/<assignment_id>/filename)
    Returns: (storage_uri, sha placeholder)
    """
    upload_root = current_app.config.get("UPLOAD_ROOT", os.path.join(os.getcwd(), "uploads"))
    os.makedirs(upload_root, exist_ok=True)

    filename = secure_filename(file_storage.filename or "upload.bin")
    if not _allowed_file(filename):
        raise ValueError(f"File type not allowed: {filename}")

    assignment_id = str(uuid.uuid4())
    assignment_dir = os.path.join(upload_root, assignment_id)
    os.makedirs(assignment_dir, exist_ok=True)

    path = os.path.join(assignment_dir, filename)
    file_storage.save(path)

    storage_uri = f"file://{path}"
    return assignment_id, storage_uri


def _grade_stub(assignment: Assignment) -> Tuple[int, str]:
    """
    Synchronous grading placeholder.

    Replace this with your real workflow:
    - detect modality based on filename or stored mime
    - extract text / run notebook sandbox / transcribe video, etc.
    - call your local SLM (e.g., via Ollama)
    - compute suggested grade + feedback
    """
    # Extremely minimal heuristic example:
    name = (assignment.filename or "").lower()
    if name.endswith(".ipynb"):
        return 92, "Notebook executed successfully. Tests passed. Minor style issues."
    if name.endswith(".py"):
        return 88, "Python script runs. Core functionality correct; add docstrings and edge-case handling."
    if name.endswith(".pdf"):
        return 85, "PDF submission parsed. Good structure; strengthen the argument with citations and clearer conclusions."
    if name.endswith((".png", ".jpg", ".jpeg", ".webp")):
        return 90, "Image submission reviewed. Meets rubric criteria; minor improvements suggested in labeling."
    if name.endswith((".mp4", ".mov", ".webm")):
        return 87, "Video submission reviewed. Clear explanation; consider adding a brief summary slide at the end."
    return 80, "Submission received. Basic rubric checks passed. Improve clarity and completeness."


def _assignment_to_dict(a: Assignment) -> Dict[str, Any]:
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
            db.query(Assignment)
            .order_by(Assignment.created_at.desc())
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
    - Stores file (local for now)
    - Inserts Assignment row into Postgres
    """
    if "file" not in request.files:
        return jsonify({"error": "Missing file field 'file' in form-data"}), 400

    f = request.files["file"]
    if not f or not f.filename:
        return jsonify({"error": "Empty file upload"}), 400

    try:
        assignment_id, storage_uri = _save_file_local(f)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    db = _db()
    try:
        a = Assignment(
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
        a: Optional[Assignment] = db.query(Assignment).filter(Assignment.id == assignment_id).first()
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
        a: Optional[Assignment] = db.query(Assignment).filter(Assignment.id == assignment_id).first()
        if not a:
            return jsonify({"error": "Assignment not found"}), 404

        if a.status in ("grading", "graded"):
            # idempotent: don't re-run unless you explicitly add a "regrade" endpoint
            return jsonify({"ok": True, "status": a.status})

        # mark grading
        a.status = "grading"
        a.updated_at = _now()
        db.commit()

        # run grading
        try:
            grade, feedback = _grade_stub(a)
            a.suggested_grade = int(grade)
            a.feedback = feedback
            a.status = "graded"
        except Exception as e:
            a.status = "failed"
            a.feedback = f"Grading failed: {e!r}"

        a.updated_at = _now()
        db.commit()

        return jsonify({"ok": True})
    finally:
        db.close()
