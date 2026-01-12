# OUTDATED VERSION OF MY ROUTES ASSIGNMENT!!!

from flask import Blueprint, request, jsonify
from app.extensions import SessionLocal
from app.models import Assignment
from app.rbac import require_auth, require_role

bp = Blueprint("assignments", __name__)

@bp.get("/api/assignments")
@require_auth
def list_assignments():
    db = SessionLocal()
    try:
        items = db.query(Assignment).order_by(Assignment.id.desc()).all()
        return jsonify([{
            "id": a.id, "course_id": a.course_id, "title": a.title,
            "modality": a.modality
        } for a in items])
    finally:
        db.close()

@bp.post("/api/assignments")
@require_role("teacher", "admin")
def create_assignment():
    payload = request.json
    db = SessionLocal()
    try:
        a = Assignment(
            course_id=payload["course_id"],
            title=payload["title"],
            description=payload.get("description",""),
            modality=payload["modality"],
            rubric=payload.get("rubric", [])
        )
        db.add(a); db.commit(); db.refresh(a)
        return jsonify({"id": a.id})
    finally:
        db.close()
