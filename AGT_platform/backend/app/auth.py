# backend/app/auth.py
from flask import Blueprint, redirect, request, jsonify, current_app
from authlib.integrations.flask_client import OAuth
import jwt, os, time
from .extensions import SessionLocal
from .models import User  # you’ll add fields below
from datetime import datetime

bp = Blueprint("auth", __name__)
oauth = OAuth()

def init_oauth(app):
    oauth.init_app(app)
    
    # Register Microsoft OAuth
    if app.config.get("MICROSOFT_CLIENT_ID") and app.config.get("MICROSOFT_CLIENT_SECRET"):
        oauth.register(
            name="microsoft",
            client_id=app.config["MICROSOFT_CLIENT_ID"],
            client_secret=app.config["MICROSOFT_CLIENT_SECRET"],
            server_metadata_url="https://login.microsoftonline.com/common/v2.0/.well-known/openid-configuration",
            client_kwargs={"scope": "openid email profile"},
        )
    
    # Register Google OAuth
    if app.config.get("GOOGLE_CLIENT_ID") and app.config.get("GOOGLE_CLIENT_SECRET"):
        oauth.register(
            name="google",
            client_id=app.config["GOOGLE_CLIENT_ID"],
            client_secret=app.config["GOOGLE_CLIENT_SECRET"],
            server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
            client_kwargs={"scope": "openid email profile"},
        )

def _issuer_for_domain(domain: str) -> str | None:
    """
    MVP: map known domains -> discovery url.
    Replace with DB lookup later.
    """
    domain = domain.lower().strip()

    mapping = {
        "mit.edu": "",
        "gsu.edu": "",
        "qcc.edu": "",
        "harvard.edu": ""
        # "mit.edu": "https://<mit-issuer>/.well-known/openid-configuration",
        # "gsu.edu": "https://<gsu-issuer>/.well-known/openid-configuration",
    }
    return mapping.get(domain)

def _register_dynamic_client(discovery_url: str):
    """
    Authlib needs a client name; we can reuse 'campus' but re-register with new metadata.
    """
    oauth.register(
        name="campus",
        server_metadata_url=discovery_url,
        client_id=current_app.config["OIDC_CLIENT_ID"],
        client_secret=current_app.config["OIDC_CLIENT_SECRET"],
        client_kwargs={"scope": "openid email profile"},
    )

def _is_college_email(email: str) -> bool:
    """
    Validate that the email is from a college/university domain.
    Common patterns: .edu, .ac.uk, .ac.za, etc.
    Also check for common non-educational domains to reject.
    """
    if not email or "@" not in email:
        return False
    
    domain = email.split("@", 1)[1].lower()
    
    # Reject common personal email providers
    personal_domains = [
        "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", 
        "icloud.com", "aol.com", "mail.com", "protonmail.com",
        "yandex.com", "zoho.com", "gmx.com"
    ]
    if domain in personal_domains:
        return False
    
    # Accept .edu domains (US colleges)
    if domain.endswith(".edu"):
        return True
    
    # Accept .ac.* domains (many countries use .ac for academic institutions)
    if ".ac." in domain:
        return True
    
    # Accept .edu.* domains (some countries use .edu.*)
    if domain.startswith("edu.") or ".edu." in domain:
        return True
    
    # For other domains, we'll be permissive but log for review
    # In production, you might want a whitelist
    return True

def _issue_token(user: User):
    payload = {
        "id": user.id,
        "email": user.email,
        "role": user.role,
        "iat": int(time.time()),
        "exp": int(time.time()) + 60 * 60 * 8,
    }
    return jwt.encode(payload, os.getenv("SECRET_KEY", "dev_secret"), algorithm="HS256")

@bp.post("/api/auth/discover")
def discover():
    """
    Frontend sends { email }.
    We derive domain and return whether we can SSO it.
    """
    data = request.get_json(force=True)
    email = (data.get("email") or "").strip().lower()
    if "@" not in email:
        return jsonify({"error": "valid email required"}), 400

    domain = email.split("@", 1)[1]
    discovery_url = _issuer_for_domain(domain)

    if not discovery_url:
        # fallback path: Google/Microsoft, or request manual admin add
        return jsonify({
            "supported": False,
            "domain": domain,
            "message": "School not configured yet. Use Google/Microsoft login or ask admin to add your school."
        }), 200

    return jsonify({"supported": True, "domain": domain}), 200

@bp.get("/api/auth/login")
def login():
    """
    Start OIDC. Requires ?domain=mit.edu OR ?email=x@mit.edu
    """
    domain = request.args.get("domain")
    email = request.args.get("email")

    if not domain and email and "@" in email:
        domain = email.split("@", 1)[1]

    if not domain:
        return jsonify({"error": "missing domain or email"}), 400

    discovery_url = _issuer_for_domain(domain)
    if not discovery_url:
        return jsonify({"error": f"unknown institution domain: {domain}"}), 400

    _register_dynamic_client(discovery_url)

    redirect_uri = request.host_url.rstrip("/") + "/api/auth/callback"
    # persist domain in state via session param
    return oauth.campus.authorize_redirect(redirect_uri, domain=domain)

@bp.get("/api/auth/login/microsoft")
def login_microsoft():
    """Start Microsoft OAuth flow"""
    if not current_app.config.get("MICROSOFT_CLIENT_ID"):
        return jsonify({"error": "Microsoft OAuth not configured"}), 500
    
    redirect_uri = request.host_url.rstrip("/") + "/api/auth/callback/microsoft"
    return oauth.microsoft.authorize_redirect(redirect_uri)

@bp.get("/api/auth/login/google")
def login_google():
    """Start Google OAuth flow"""
    if not current_app.config.get("GOOGLE_CLIENT_ID"):
        return jsonify({"error": "Google OAuth not configured"}), 500
    
    redirect_uri = request.host_url.rstrip("/") + "/api/auth/callback/google"
    return oauth.google.authorize_redirect(redirect_uri)

def _handle_oauth_callback(provider_name: str):
    """Common handler for OAuth callbacks"""
    try:
        provider = getattr(oauth, provider_name)
        token = provider.authorize_access_token()
        userinfo = token.get("userinfo") or provider.parse_id_token(token)
    except Exception as e:
        return jsonify({"error": f"OAuth callback failed: {str(e)}"}), 400

    email = (userinfo.get("email") or "").lower().strip()
    name = userinfo.get("name") or userinfo.get("preferred_username") or email
    
    if not email:
        return jsonify({"error": "missing email from identity provider"}), 400
    
    # Validate college email
    if not _is_college_email(email):
        frontend_base = current_app.config.get("FRONTEND_BASE_URL", "http://localhost:5173")
        return redirect(f"{frontend_base}/login?error=Please use your college email address")

    domain = email.split("@", 1)[1] if "@" in email else None

    db = SessionLocal()
    try:
        user = db.query(User).filter_by(email=email).one_or_none()
        now = datetime.utcnow()

        if not user:
            # First login provisioning
            user = User(
                email=email,
                name=name,
                role="student",                 # default
                institution_domain=domain,
                first_login_at=now,
                last_login_at=now,
            )
            db.add(user)
            db.commit()
            db.refresh(user)
        else:
            user.last_login_at = now
            if not getattr(user, "institution_domain", None):
                user.institution_domain = domain
            db.commit()

        jwt_token = _issue_token(user)
    finally:
        db.close()

    frontend_base = current_app.config.get("FRONTEND_BASE_URL", "http://localhost:5173")
    return redirect(f"{frontend_base}/login#token={jwt_token}")

@bp.get("/api/auth/callback")
def callback():
    """Legacy callback for campus SSO"""
    token = oauth.campus.authorize_access_token()
    userinfo = token.get("userinfo") or oauth.campus.parse_id_token(token)

    email = (userinfo.get("email") or "").lower().strip()
    name = userinfo.get("name") or userinfo.get("preferred_username") or email
    if not email:
        return jsonify({"error": "missing email from idp"}), 400

    domain = email.split("@", 1)[1] if "@" in email else None

    db = SessionLocal()
    try:
        user = db.query(User).filter_by(email=email).one_or_none()
        now = datetime.utcnow()

        if not user:
            # First login provisioning
            user = User(
                email=email,
                name=name,
                role="student",                 # default
                institution_domain=domain,
                first_login_at=now,
                last_login_at=now,
            )
            db.add(user)
            db.commit()
            db.refresh(user)
        else:
            user.last_login_at = now
            if not getattr(user, "institution_domain", None):
                user.institution_domain = domain
            db.commit()

        jwt_token = _issue_token(user)
    finally:
        db.close()

    frontend_base = current_app.config.get("FRONTEND_BASE_URL", "http://localhost:5173")
    return redirect(f"{frontend_base}/login#token={jwt_token}")

@bp.get("/api/auth/callback/microsoft")
def callback_microsoft():
    """Microsoft OAuth callback"""
    return _handle_oauth_callback("microsoft")

@bp.get("/api/auth/callback/google")
def callback_google():
    """Google OAuth callback"""
    return _handle_oauth_callback("google")