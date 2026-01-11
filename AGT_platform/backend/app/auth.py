from flask import Blueprint, redirect, request, jsonify
from authlib.integrations.flask_client import OAuth
import jwt, os, time
from .extensions import SessionLocal
from .models import User

bp = Blueprint("auth", __name__)
oauth = OAuth()

def init_oauth(app):
    oauth.init_app(app)
    oauth.register(
        name="campus",
        server_metadata_url=app.config["OIDC_DISCOVERY_URL"],
        client_id=app.config["OIDC_CLIENT_ID"],
        client_secret=app.config["OIDC_CLIENT_SECRET"],
        client_kwargs={"scope": "openid email profile"},
    )

def _issue_token(user: User):
    payload = {
        "id": user.id,
        "email": user.email,
        "role": user.role,
        "iat": int(time.time()),
        "exp": int(time.time()) + 60*60*8,
    }
    return jwt.encode(payload, os.getenv("SECRET_KEY","dev_secret"), algorithm="HS256")

@bp.get("/api/auth/login")
def login():
    redirect_uri = request.host_url.rstrip("/") + "/api/auth/callback"
    return oauth.campus.authorize_redirect(redirect_uri)

@bp.get("/api/auth/callback")
def callback():
    token = oauth.campus.authorize_access_token()
    userinfo = token.get("userinfo") or oauth.campus.parse_id_token(token)

    email = userinfo.get("email")
    name = userinfo.get("name") or userinfo.get("preferred_username") or email

    if not email:
        return jsonify({"error":"missing email from idp"}), 400

    # Provision user (default role: student unless admin seeds or teacher enrollment exists)
    db = SessionLocal()
    try:
        user = db.query(User).filter_by(email=email).one_or_none()
        if not user:
            user = User(email=email, name=name, role="student")
            db.add(user); db.commit(); db.refresh(user)
        jwt_token = _issue_token(user)
    finally:
        db.close()

    # return token for frontend
    return redirect(f"http://localhost:5173/login#token={jwt_token}")

@bp.get("/api/auth/me")
def me():
    # frontend calls this after storing token
    return jsonify({"ok": True})
