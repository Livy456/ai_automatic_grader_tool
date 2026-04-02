from datetime import datetime

from flask import Blueprint, jsonify, request
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError

from app.audit import log_event
from app.extensions import SessionLocal
from app.models import Assignment, AuditLog, Course, Enrollment, IssuedJwt, User
from app.rbac import require_role

bp = Blueprint("admin", __name__)


@bp.get("/api/admin/users")
@require_role("admin")
def users():
    db = SessionLocal()
    try:
        items = db.query(User).order_by(User.id.desc()).all()
        return jsonify(
            [{"id": u.id, "email": u.email, "name": u.name, "role": u.role} for u in items]
        )
    finally:
        db.close()


@bp.post("/api/admin/users/<int:user_id>/role")
@require_role("admin")
def set_role(user_id: int):
    role = request.json["role"]
    if role not in ["student", "teacher", "admin"]:
        return jsonify({"error": "invalid role"}), 400

    db = SessionLocal()
    try:
        u = db.query(User).get(user_id)
        if not u:
            return jsonify({"error": "not found"}), 404
        u.role = role
        for tok in db.query(IssuedJwt).filter_by(user_id=user_id).all():
            tok.revoked_at = datetime.utcnow()
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
        count_rows = (
            db.query(Enrollment.course_id, func.count(Enrollment.id))
            .group_by(Enrollment.course_id)
            .all()
        )
        count_map = {cid: int(n) for cid, n in count_rows}
        return jsonify(
            [
                {
                    "id": c.id,
                    "code": c.code,
                    "title": c.title,
                    "description": c.description,
                    "enrollment_count": count_map.get(c.id, 0),
                }
                for c in courses
            ]
        )
    finally:
        db.close()


@bp.post("/api/admin/courses")
@require_role("admin")
def create_course():
    payload = request.json or {}
    db = SessionLocal()
    try:
        code = (payload.get("code") or "").strip()
        title = (payload.get("title") or "").strip()
        if not code or not title:
            return jsonify({"error": "code and title required"}), 400
        desc = payload.get("description")
        description = desc.strip() if isinstance(desc, str) else None
        c = Course(code=code, title=title, description=description)
        db.add(c)
        db.commit()
        db.refresh(c)
        log_event(
            request.user["id"],
            "CREATE_COURSE",
            "Course",
            c.id,
            {"code": c.code, "title": c.title},
        )
        return jsonify({"id": c.id}), 201
    finally:
        db.close()


@bp.get("/api/admin/courses/<int:course_id>/enrollments")
@require_role("admin")
def list_course_enrollments(course_id: int):
    db = SessionLocal()
    try:
        c = db.query(Course).get(course_id)
        if not c:
            return jsonify({"error": "not found"}), 404
        rows = (
            db.query(Enrollment, User)
            .join(User, Enrollment.user_id == User.id)
            .filter(Enrollment.course_id == course_id)
            .all()
        )
        return jsonify(
            [
                {
                    "enrollment_id": e.id,
                    "user_id": u.id,
                    "email": u.email,
                    "name": u.name or "",
                    "role": e.role,
                }
                for e, u in rows
            ]
        )
    finally:
        db.close()


@bp.post("/api/admin/enrollments")
@require_role("admin")
def create_enrollment():
    payload = request.json or {}
    db = SessionLocal()
    try:
        role = payload.get("role")
        if role not in ("student", "teacher"):
            return jsonify({"error": "role must be student or teacher"}), 400
        try:
            course_id = int(payload["course_id"])
            user_id = int(payload["user_id"])
        except (KeyError, TypeError, ValueError):
            return jsonify({"error": "course_id and user_id required"}), 400

        if not db.query(Course).get(course_id):
            return jsonify({"error": "course not found"}), 404
        if not db.query(User).get(user_id):
            return jsonify({"error": "user not found"}), 404

        e = Enrollment(course_id=course_id, user_id=user_id, role=role)
        try:
            db.add(e)
            db.commit()
            db.refresh(e)
        except IntegrityError:
            db.rollback()
            return jsonify({"error": "user already enrolled in this course"}), 409

        log_event(
            request.user["id"],
            "ENROLL_USER",
            "Enrollment",
            e.id,
            {"course_id": e.course_id, "user_id": e.user_id, "role": e.role},
        )
        return jsonify({"id": e.id}), 201
    finally:
        db.close()


@bp.delete("/api/admin/enrollments/<int:enrollment_id>")
@require_role("admin")
def delete_enrollment(enrollment_id: int):
    db = SessionLocal()
    try:
        e = db.query(Enrollment).get(enrollment_id)
        if not e:
            return jsonify({"error": "not found"}), 404
        meta = {"course_id": e.course_id, "user_id": e.user_id}
        db.delete(e)
        db.commit()
        log_event(
            request.user["id"],
            "REMOVE_ENROLLMENT",
            "Enrollment",
            enrollment_id,
            meta,
        )
        return jsonify({"ok": True})
    finally:
        db.close()


@bp.get("/api/admin/assignments")
@require_role("admin")
def list_assignments():
    db = SessionLocal()
    try:
        items = db.query(Assignment).order_by(Assignment.id.desc()).limit(200).all()
        return jsonify(
            [
                {
                    "id": a.id,
                    "course_id": a.course_id,
                    "title": a.title,
                    "modality": a.modality,
                }
                for a in items
            ]
        )
    finally:
        db.close()


@bp.get("/api/admin/audit")
@require_role("admin")
def audit():
    db = SessionLocal()
    try:
        logs = db.query(AuditLog).order_by(AuditLog.id.desc()).limit(300).all()
        return jsonify(
            [
                {
                    "time": l.created_at.isoformat(),
                    "actor_user_id": l.actor_user_id,
                    "action": l.action,
                    "target_type": l.target_type,
                    "target_id": l.target_id,
                    "metadata": l.event_metadata,
                }
                for l in logs
            ]
        )
    finally:
        db.close()
