from flask import Blueprint, jsonify, request
from app.rbac import require_role
from app.extensions import SessionLocal
from app.models import User, AuditLog

bp = Blueprint("admin", __name__)

@bp.get("/api/admin/users")
@require_role("admin")
def users():
    db = SessionLocal()
    try:
        items = db.query(User).all()
        return jsonify([{"id":u.id,"email":u.email,"role":u.role} for u in items])
    finally:
        db.close()

@bp.post("/api/admin/users/<int:user_id>/role")
@require_role("admin")
def set_role(user_id: int):
    role = request.json["role"]
    db = SessionLocal()
    try:
        u = db.query(User).get(user_id)
        if not u: return jsonify({"error":"not found"}), 404
        u.role = role
        db.commit()
        return jsonify({"ok": True})
    finally:
        db.close()

@bp.get("/api/admin/audit")
@require_role("admin")
def audit():
    db = SessionLocal()
    try:
        logs = db.query(AuditLog).order_by(AuditLog.id.desc()).limit(200).all()
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
