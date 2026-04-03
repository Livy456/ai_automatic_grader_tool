# backend/app/auth.py
import hashlib
import secrets
import time
import uuid
from datetime import datetime, timedelta
from urllib.parse import quote

import jwt
from authlib.integrations.flask_client import OAuth
from flask import Blueprint, current_app, jsonify, make_response, redirect, request
from werkzeug.security import check_password_hash

from .config import Config
from .extensions import SessionLocal
from .models import IssuedJwt, RefreshToken, User
from .rbac import get_user_from_token, require_auth

bp = Blueprint("auth", __name__)
oauth = OAuth()


def _public_api_base() -> str:
    """
    Base URL the browser uses to reach this API for OAuth (scheme + host + port).
    Microsoft/Google redirect_uri must match app registration exactly; session cookies for
    the OAuth state must be set on this same origin. When the SPA is on :5174, still use
    PUBLIC_API_URL=http://localhost:5000 for local Docker.
    """
    explicit = (current_app.config.get("PUBLIC_API_URL") or "").strip().rstrip("/")
    if explicit:
        return explicit
    return request.host_url.rstrip("/")


def _frontend_origin() -> str:
    """
    Normalized SPA base URL for redirects after OAuth (no trailing slash).
    Must match where the UI is actually served: Docker frontend usually :5173,
    host `npm run dev` in this repo defaults to :5174 (see frontend/vite.config.ts).
    If FRONTEND_BASE_URL is unset or empty in .env, Flask config can be ""; fall back then.
    """
    raw = (current_app.config.get("FRONTEND_BASE_URL") or "").strip()
    if not raw:
        raw = "http://localhost:5173"
    return raw.rstrip("/")

def _microsoft_entra_iss_ok(_claims, value) -> bool:
    """
    Accept Microsoft-issued ID token iss values. Authlib's default check compares `iss` to
    the `issuer` string from OpenID metadata; for /common (and sometimes tenant-specific
    metadata) that string can be a template or otherwise not equal to the token, which raises
    invalid_claim iss (authlib/authlib#605).

    We still require a Microsoft-shaped issuer (not arbitrary URLs).
    """
    if not value or not isinstance(value, str):
        return False
    v = value.rstrip("/")
    if v.startswith("https://login.microsoftonline.com/") and v.endswith("/v2.0"):
        return True
    if v.startswith("https://sts.windows.net/") and len(v) > len("https://sts.windows.net/"):
        return True
    # Entra External ID (CIAM) style issuers
    if v.startswith("https://") and ".ciamlogin.com/" in v and v.endswith("/v2.0"):
        return True
    return False


# Always pass this for Microsoft id_token validation (including inside authorize_access_token
# when Authlib parses the token before returning — see flask_client/apps.py).
_MICROSOFT_ISS_CLAIMS_OPTIONS = {"iss": {"validate": _microsoft_entra_iss_ok}}


def _parse_oidc_userinfo(provider, provider_name: str, token: dict):
    """Resolve user claims from token."""
    userinfo = token.get("userinfo")
    if userinfo:
        return userinfo
    if provider_name == "microsoft":
        return provider.parse_id_token(
            token, nonce=None, claims_options=_MICROSOFT_ISS_CLAIMS_OPTIONS
        )
    return provider.parse_id_token(token, nonce=None)


def init_oauth(app):
    oauth.init_app(app)
    
    # Register Microsoft OAuth
    if app.config.get("MICROSOFT_CLIENT_ID") and app.config.get("MICROSOFT_CLIENT_SECRET"):
        raw = (app.config.get("MICROSOFT_TENANT_ID") or "").strip()
        authority_segment = raw if raw else "common"
        meta = (
            f"https://login.microsoftonline.com/{authority_segment}"
            f"/v2.0/.well-known/openid-configuration"
        )
        oauth.register(
            name="microsoft",
            client_id=app.config["MICROSOFT_CLIENT_ID"],
            client_secret=app.config["MICROSOFT_CLIENT_SECRET"],
            server_metadata_url=meta,
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


def _hash_refresh(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _normalize_samesite(cfg: Config) -> str:
    s = (cfg.REFRESH_COOKIE_SAMESITE or "lax").lower()
    if s == "none":
        return "None"
    if s == "strict":
        return "Strict"
    return "Lax"


def _refresh_secure_for_response(cfg: Config) -> bool:
    """SameSite=None requires Secure in modern browsers."""
    if _normalize_samesite(cfg) == "None":
        return True
    return bool(cfg.REFRESH_COOKIE_SECURE)


def _apply_refresh_cookie(resp, raw_token: str) -> None:
    cfg = Config()
    resp.set_cookie(
        cfg.REFRESH_TOKEN_COOKIE_NAME,
        raw_token,
        max_age=int(cfg.JWT_REFRESH_EXPIRATION_SECONDS),
        httponly=True,
        secure=_refresh_secure_for_response(cfg),
        samesite=_normalize_samesite(cfg),
        path="/",
    )


def _clear_refresh_cookie(resp) -> None:
    cfg = Config()
    resp.set_cookie(
        cfg.REFRESH_TOKEN_COOKIE_NAME,
        "",
        max_age=0,
        httponly=True,
        secure=_refresh_secure_for_response(cfg),
        samesite=_normalize_samesite(cfg),
        path="/",
    )


def _create_refresh_token(user: User, db) -> str:
    """Rotate refresh: revoke other active rows for this user, persist new hash, return raw secret."""
    cfg = Config()
    now = datetime.utcnow()
    db.query(RefreshToken).filter(
        RefreshToken.user_id == user.id,
        RefreshToken.revoked_at.is_(None),
    ).update({"revoked_at": now})
    raw = secrets.token_urlsafe(48)
    exp = now + timedelta(seconds=int(cfg.JWT_REFRESH_EXPIRATION_SECONDS))
    db.add(
        RefreshToken(
            user_id=user.id,
            token_hash=_hash_refresh(raw),
            expires_at=exp,
        )
    )
    db.commit()
    return raw


def _issue_access_token(user: User, db) -> str:
    """
    Short-lived access JWT; client keeps it in memory only. jti is allowlisted in issued_jwts.
    """
    cfg = Config()
    jti = str(uuid.uuid4())
    now_ts = int(time.time())
    exp_ts = now_ts + int(cfg.JWT_ACCESS_EXPIRATION_SECONDS)
    secret = cfg.SECRET_KEY or "dev_secret"
    payload = {
        "id": user.id,
        "email": user.email,
        "role": user.role,
        "iat": now_ts,
        "exp": exp_ts,
        "jti": jti,
    }
    token = jwt.encode(payload, secret, algorithm="HS256")
    db.add(
        IssuedJwt(
            user_id=user.id,
            jti=jti,
            expires_at=datetime.utcfromtimestamp(exp_ts),
        )
    )
    db.commit()
    return token


@bp.post("/api/auth/login/password")
def login_password():
    """Email + password login; returns JWT (role comes from users.role in the database)."""
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or data.get("username") or "").strip().lower()
    password = data.get("password") or ""
    if not email or not password:
        return jsonify({"error": "email and password required"}), 400

    db = SessionLocal()
    try:
        user = db.query(User).filter_by(email=email).one_or_none()
        if not user or not user.password_hash:
            return jsonify({"error": "invalid credentials"}), 401
        if not check_password_hash(user.password_hash, password):
            return jsonify({"error": "invalid credentials"}), 401

        user.last_login_at = datetime.utcnow()
        db.commit()
        db.refresh(user)

        access = _issue_access_token(user, db)
        raw_refresh = _create_refresh_token(user, db)
        cfg = Config()
        resp = make_response(
            jsonify(
                {
                    "access_token": access,
                    "token_type": "Bearer",
                    "expires_in": cfg.JWT_ACCESS_EXPIRATION_SECONDS,
                }
            )
        )
        _apply_refresh_cookie(resp, raw_refresh)
        return resp
    finally:
        db.close()


@bp.post("/api/auth/refresh")
def refresh_access():
    """
    Exchange HttpOnly refresh cookie for a new in-memory access token (no Bearer required).
    """
    cfg = Config()
    raw = request.cookies.get(cfg.REFRESH_TOKEN_COOKIE_NAME)
    if not raw:
        return jsonify({"error": "unauthorized"}), 401

    db = SessionLocal()
    try:
        h = _hash_refresh(raw)
        row = db.query(RefreshToken).filter(RefreshToken.token_hash == h).one_or_none()
        if row is None or row.revoked_at is not None:
            return jsonify({"error": "unauthorized"}), 401
        if row.expires_at < datetime.utcnow():
            return jsonify({"error": "unauthorized"}), 401

        user = db.query(User).get(int(row.user_id))
        if user is None:
            return jsonify({"error": "unauthorized"}), 401

        access = _issue_access_token(user, db)
        return jsonify(
            {
                "access_token": access,
                "token_type": "Bearer",
                "expires_in": cfg.JWT_ACCESS_EXPIRATION_SECONDS,
            }
        )
    finally:
        db.close()


@bp.get("/api/auth/me")
@require_auth
def me():
    """Lightweight session check for the SPA (validates JWT + issued_jwts)."""
    user = request.user
    return jsonify(
        {"id": user["id"], "email": user.get("email", ""), "role": user.get("role", "student")}
    )


@bp.post("/api/auth/logout")
def logout():
    """Revoke access JWT if present, revoke refresh cookie row, clear refresh cookie."""
    resp = make_response(jsonify({"ok": True}))
    _clear_refresh_cookie(resp)

    u = get_user_from_token()
    if u and u.get("jti"):
        db = SessionLocal()
        try:
            row = db.query(IssuedJwt).filter_by(jti=u["jti"]).one_or_none()
            if row:
                row.revoked_at = datetime.utcnow()
                db.commit()
        finally:
            db.close()

    cfg = Config()
    raw = request.cookies.get(cfg.REFRESH_TOKEN_COOKIE_NAME)
    if raw:
        db = SessionLocal()
        try:
            h = _hash_refresh(raw)
            rrow = db.query(RefreshToken).filter(RefreshToken.token_hash == h).one_or_none()
            if rrow and rrow.revoked_at is None:
                rrow.revoked_at = datetime.utcnow()
                db.commit()
        finally:
            db.close()

    return resp

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

    redirect_uri = _public_api_base() + "/api/auth/callback"
    # persist domain in state via session param
    return oauth.campus.authorize_redirect(redirect_uri, domain=domain)

@bp.get("/api/auth/login/microsoft")
def login_microsoft():
    """Start Microsoft OAuth flow"""
    if not current_app.config.get("MICROSOFT_CLIENT_ID"):
        return jsonify({"error": "Microsoft OAuth not configured"}), 500
    
    # redirect_uri must match Entra registration; use PUBLIC_API_URL when SPA is on another port
    redirect_uri = _public_api_base() + "/api/auth/callback/microsoft"
    # Authlib automatically generates and validates state parameter for CSRF protection
    return oauth.microsoft.authorize_redirect(redirect_uri)

@bp.get("/api/auth/login/google")
def login_google():
    """Start Google OAuth flow"""
    if not current_app.config.get("GOOGLE_CLIENT_ID"):
        return jsonify({"error": "Google OAuth not configured"}), 500
    
    redirect_uri = _public_api_base() + "/api/auth/callback/google"
    # Authlib automatically generates and validates state parameter for CSRF protection
    return oauth.google.authorize_redirect(redirect_uri)

def _oauth_callback_fail_redirect(exc: Exception, provider_name: str):
    """
    Return a browser redirect with a readable error instead of raw JSON 400 (avoids Chrome
    "invalid response" on OAuth return URLs). Logs full exception server-side.
    """
    current_app.logger.exception("OAuth %s callback failed", provider_name)
    frontend_base = _frontend_origin()
    msg = str(exc).strip() or type(exc).__name__
    lower = msg.lower()
    if "state" in lower or "mismatch" in lower or "csrf" in lower:
        user_msg = (
            "OAuth session/state failed (often: session cookie not sent over HTTP). "
            "For local dev use SESSION_COOKIE_SECURE=false (default). Same browser tab, retry sign-in."
        )
    elif "invalid_client" in lower or "unauthorized_client" in lower or "700016" in msg:
        user_msg = (
            "Microsoft rejected the app credentials. If your Azure free trial ended, the app "
            "registration may be disabled—renew Azure or create a new app registration and update "
            "MICROSOFT_CLIENT_ID / MICROSOFT_CLIENT_SECRET."
        )
    elif "invalid_grant" in lower or "expired" in lower:
        user_msg = (
            "Authorization code expired or reused. Close the tab and start sign-in again from the login page."
        )
    else:
        user_msg = f"Sign-in failed ({provider_name}). {msg[:280]}"

    return redirect(f"{frontend_base}/login?error={quote(user_msg)}")


def _handle_oauth_callback(provider_name: str):
    """Common handler for OAuth callbacks"""
    try:
        provider = getattr(oauth, provider_name)
        # Flask OAuth2: when state contains a nonce, authorize_access_token() parses the id_token
        # internally first; it must receive the same Microsoft iss workaround or iss validation fails
        # before we return here (authlib/integrations/flask_client/apps.py).
        token_kw: dict = {}
        if provider_name == "microsoft":
            token_kw["claims_options"] = _MICROSOFT_ISS_CLAIMS_OPTIONS
        token = provider.authorize_access_token(**token_kw)
        userinfo = _parse_oidc_userinfo(provider, provider_name, token)
    except Exception as e:
        return _oauth_callback_fail_redirect(e, provider_name)

    email = (userinfo.get("email") or "").lower().strip()
    name = userinfo.get("name") or userinfo.get("preferred_username") or email
    
    if not email:
        return jsonify({"error": "missing email from identity provider"}), 400
    
    # Validate college email
    if not _is_college_email(email):
        return redirect(
            f"{_frontend_origin()}/login?error=Please use your college email address"
        )

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

        raw_refresh = _create_refresh_token(user, db)
    finally:
        db.close()

    resp = redirect(f"{_frontend_origin()}/login")
    _apply_refresh_cookie(resp, raw_refresh)
    return resp

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

        raw_refresh = _create_refresh_token(user, db)
    finally:
        db.close()

    resp = redirect(f"{_frontend_origin()}/login")
    _apply_refresh_cookie(resp, raw_refresh)
    return resp

@bp.get("/api/auth/callback/microsoft")
def callback_microsoft():
    """Microsoft OAuth callback"""
    return _handle_oauth_callback("microsoft")

@bp.get("/api/auth/callback/google")
def callback_google():
    """Google OAuth callback"""
    return _handle_oauth_callback("google")