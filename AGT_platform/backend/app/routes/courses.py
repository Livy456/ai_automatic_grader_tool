from datetime import datetime

from flask import Blueprint, jsonify, request

from app.audit import log_event
from app.extensions import SessionLocal
from app.models import Assignment, Course, Enrollment, User
from app.rbac import require_auth, require_role

bp = Blueprint("courses", __name__)

MODALITIES = frozenset({"code", "written", "notebook", "video", "image"})


def _parse_due_date(raw):
    if raw is None or raw == "":
        return None
    if isinstance(raw, str):
        try:
            # ISO 8601 from datetime-local or full ISO
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _user_can_view_course(db, user_id: int, role: str, course_id: int) -> bool:
    if role == "admin":
        return True
    return (
        db.query(Enrollment)
        .filter_by(user_id=user_id, course_id=course_id)
        .first()
        is not None
    )


def _user_is_course_teacher(db, user_id: int, course_id: int) -> bool:
    row = (
        db.query(Enrollment)
        .filter_by(user_id=user_id, course_id=course_id, role="teacher")
        .first()
    )
    return row is not None


def _normalize_rubric(raw):
    if raw is None:
        return []
    if not isinstance(raw, list):
        return None
    out = []
    for item in raw:
        if not isinstance(item, dict):
            return None
        crit = item.get("criterion")
        mx = item.get("max_score")
        if not isinstance(crit, str) or not crit.strip():
            return None
        try:
            mx_num = float(mx)
        except (TypeError, ValueError):
            return None
        out.append({"criterion": crit.strip(), "max_score": mx_num})
    return out


@bp.get("/api/courses")
@require_auth
def list_courses():
    db = SessionLocal()
    try:
        role = request.user["role"]
        user_id = request.user["id"]
        if role == "admin":
            courses = db.query(Course).order_by(Course.id).all()
            return jsonify(
                [
                    {
                        "id": c.id,
                        "code": c.code,
                        "title": c.title,
                        "enrollment_role": None,
                    }
                    for c in courses
                ]
            )
        enrollments = db.query(Enrollment).filter_by(user_id=user_id).all()
        by_course = {e.course_id: e.role for e in enrollments}
        course_ids = list(by_course.keys())
        if not course_ids:
            return jsonify([])
        courses = (
            db.query(Course).filter(Course.id.in_(course_ids)).order_by(Course.id).all()
        )
        return jsonify(
            [
                {
                    "id": c.id,
                    "code": c.code,
                    "title": c.title,
                    "enrollment_role": by_course.get(c.id),
                }
                for c in courses
            ]
        )
    finally:
        db.close()


@bp.get("/api/courses/<int:course_id>")
@require_auth
def get_course(course_id: int):
    db = SessionLocal()
    try:
        user_id = request.user["id"]
        role = request.user["role"]
        c = db.query(Course).get(course_id)
        if not c:
            return jsonify({"error": "not found"}), 404
        if not _user_can_view_course(db, user_id, role, course_id):
            return jsonify({"error": "forbidden"}), 403
        rows = (
            db.query(Enrollment, User)
            .join(User, Enrollment.user_id == User.id)
            .filter(Enrollment.course_id == course_id)
            .all()
        )
        enrollments = [
            {
                "user_id": u.id,
                "email": u.email,
                "name": u.name or "",
                "role": e.role,
            }
            for e, u in rows
        ]
        return jsonify(
            {
                "id": c.id,
                "code": c.code,
                "title": c.title,
                "enrollments": enrollments,
            }
        )
    finally:
        db.close()


def _serialize_assignment(a: Assignment):
    created = a.created_at.isoformat() if a.created_at else None
    due = a.due_date.isoformat() if getattr(a, "due_date", None) else None
    return {
        "id": a.id,
        "course_id": a.course_id,
        "title": a.title,
        "description": a.description or "",
        "modality": a.modality,
        "rubric": a.rubric if a.rubric is not None else [],
        "due_date": due,
        "created_at": created,
    }


@bp.get("/api/courses/<int:course_id>/assignments")
@require_auth
def list_assignments(course_id: int):
    db = SessionLocal()
    try:
        user_id = request.user["id"]
        role = request.user["role"]
        c = db.query(Course).get(course_id)
        if not c:
            return jsonify({"error": "not found"}), 404
        if not _user_can_view_course(db, user_id, role, course_id):
            return jsonify({"error": "forbidden"}), 403
        items = (
            db.query(Assignment)
            .filter_by(course_id=course_id)
            .order_by(Assignment.id)
            .all()
        )
        return jsonify([_serialize_assignment(a) for a in items])
    finally:
        db.close()


@bp.post("/api/courses/<int:course_id>/assignments")
@require_role("teacher", "admin")
def create_assignment(course_id: int):
    payload = request.get_json(silent=True) or {}
    db = SessionLocal()
    try:
        user_id = request.user["id"]
        role = request.user["role"]
        c = db.query(Course).get(course_id)
        if not c:
            return jsonify({"error": "course not found"}), 404
        if role != "admin" and not _user_is_course_teacher(db, user_id, course_id):
            return jsonify({"error": "forbidden"}), 403

        title = payload.get("title")
        if not isinstance(title, str) or not title.strip():
            return jsonify({"error": "title is required"}), 400
        title = title.strip()
        if len(title) > 255:
            return jsonify({"error": "title too long"}), 400

        modality = payload.get("modality")
        if modality not in MODALITIES:
            return jsonify({"error": "invalid modality"}), 400

        description = payload.get("description", "")
        if description is None:
            description = ""
        if not isinstance(description, str):
            return jsonify({"error": "invalid description"}), 400

        rubric = _normalize_rubric(payload.get("rubric", []))
        if rubric is None:
            return jsonify({"error": "invalid rubric"}), 400

        due_raw = payload.get("due_date")
        due_date = _parse_due_date(due_raw)

        a = Assignment(
            course_id=course_id,
            title=title,
            description=description,
            modality=modality,
            rubric=rubric,
            created_at=datetime.utcnow(),
            due_date=due_date,
        )
        db.add(a)
        db.commit()
        db.refresh(a)
        log_event(
            request.user["id"],
            "CREATE_ASSIGNMENT",
            "Assignment",
            a.id,
            {"course_id": course_id, "title": a.title},
        )
        return (
            jsonify({"id": a.id, "title": a.title, "course_id": course_id}),
            201,
        )
    finally:
        db.close()


@bp.route(
    "/api/courses/<int:course_id>/assignments/<int:assignment_id>",
    methods=["PUT", "PATCH"],
)
@require_role("teacher", "admin")
def update_assignment(course_id: int, assignment_id: int):
    payload = request.get_json(silent=True) or {}
    db = SessionLocal()
    try:
        user_id = request.user["id"]
        role = request.user["role"]
        c = db.query(Course).get(course_id)
        if not c:
            return jsonify({"error": "course not found"}), 404
        if role != "admin" and not _user_is_course_teacher(db, user_id, course_id):
            return jsonify({"error": "forbidden"}), 403

        a = db.query(Assignment).get(assignment_id)
        if not a or a.course_id != course_id:
            return jsonify({"error": "not found"}), 404

        if "title" in payload:
            title = payload["title"]
            if not isinstance(title, str) or not title.strip():
                return jsonify({"error": "invalid title"}), 400
            if len(title.strip()) > 255:
                return jsonify({"error": "title too long"}), 400
            a.title = title.strip()

        if "description" in payload:
            d = payload["description"]
            if d is not None and not isinstance(d, str):
                return jsonify({"error": "invalid description"}), 400
            a.description = d if isinstance(d, str) else ""

        if "modality" in payload:
            if payload["modality"] not in MODALITIES:
                return jsonify({"error": "invalid modality"}), 400
            a.modality = payload["modality"]

        if "rubric" in payload:
            rubric = _normalize_rubric(payload["rubric"])
            if rubric is None:
                return jsonify({"error": "invalid rubric"}), 400
            a.rubric = rubric

        if "due_date" in payload:
            a.due_date = _parse_due_date(payload["due_date"])

        db.commit()
        db.refresh(a)
        log_event(
            request.user["id"],
            "UPDATE_ASSIGNMENT",
            "Assignment",
            a.id,
            {"course_id": course_id},
        )
        return jsonify({"id": a.id, "title": a.title})
    finally:
        db.close()
