from flask import Blueprint, jsonify, request
from app.rbac import require_role
from app.extensions import SessionLocal
from app.models import User, AuditLog, Course, Enrollment, Assignment

bp = Blueprint("admin", __name__)

@bp.get("/api/admin/users")
@require_role("admin")
def users():
    db = SessionLocal()
    try:
        items = db.query(User).order_by(User.id.desc()).all()
        return jsonify([{"id":u.id, "email":u.email, "name": u.name, "role":u.role} for u in items])
    finally:
        db.close()

@bp.post("/api/admin/users/<int:user_id>/role")
@require_role("admin")
def set_role(user_id: int):
    role = request.json["role"]
    if role not in ["student", "teacher", "admin"]:
        return jsonify({"error":"invalid role"}), 400

    db = SessionLocal()
    try:
        u = db.query(User).get(user_id)
        if not u:
            return jsonify({"error":"not found"}), 404
        u.role = role
        db.commit()
        return jsonify({"ok": True})
    finally:
        db.close()

@bp.get("/api/admin/courses")
@require_role("admin")
def list_courses():
    db = SessionLocal()
    try:
        courses = db.query(Course).order_by(Course.id.desc()).all()
        return jsonify([{"id": c.id, "code": c.code, "title": c.title} for c in courses])
    finally:
        db.close()

@bp.post("/api/admin/courses")
@require_role("admin")
def create_course():
    payload = request.json
    db = SessionLocal()
    try:
        c = Course(code=payload["code"], title=payload["title"])
        db.add(c); db.commit(); db.refresh(c)
        return jsonify({"id": c.id})
    finally:
        db.close()

@bp.post("/api/admin/enrollments")
@require_role("admin")
def create_enrollment():
    """
    Admin can enroll users into courses, assign student/teacher role at course-level.
    """
    payload = request.json
    db = SessionLocal()
    try:
        e = Enrollment(
            course_id=int(payload["course_id"]),
            user_id=int(payload["user_id"]),
            role=payload["role"],
        )
        db.add(e); db.commit(); db.refresh(e)
        return jsonify({"id": e.id})
    finally:
        db.close()

@bp.get("/api/admin/assignments")
@require_role("admin")
def list_assignments():
    db = SessionLocal()
    try:
        items = db.query(Assignment).order_by(Assignment.id.desc()).limit(200).all()
        return jsonify([{
            "id": a.id,
            "course_id": a.course_id,
            "title": a.title,
            "modality": a.modality
        } for a in items])
    finally:
        db.close()

@bp.get("/api/admin/audit")
@require_role("admin")
def audit():
    db = SessionLocal()
    try:
        logs = db.query(AuditLog).order_by(AuditLog.id.desc()).limit(300).all()
        return jsonify([{
            "time": l.created_at.isoformat(),
            "actor_user_id": l.actor_user_id,
            "action": l.action,
            "target_type": l.target_type,
            "target_id": l.target_id,
            "metadata": l.metadata
        } for l in logs])
    finally:
        db.close()



