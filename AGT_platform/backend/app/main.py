from flask import Flask
from flask_cors import CORS
from .config import Config
# from extensions import init_db, Base, engine

# from app import extensions
# from app.extensions import Base
from app.extensions import init_db
from app.models import Base

from .auth import bp as auth_bp, init_oauth
from .tasks import init_celery
from .routes.health import bp as health_bp
# from .routes.assignments import bp as assignments_bp # OLD VERSION OF THE ASSIGNMENTS
from .routes.submissions import bp as submissions_bp
from .routes.admin import bp as admin_bp
from .routes.courses import bp as courses_bp
from .routes.standalone import bp as standalone_bp
from .routes_assignments import bp as assignments_bp
from .routes.assignment_materials import bp as assignment_materials_bp


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    app.config_obj = Config()  # for celery task access
    _cfg = Config()
    # Production: keep API bodies small (JSON + presign metadata only). Large files go to S3.
    if _cfg.ALLOW_FLASK_MULTIPART_UPLOAD:
        app.config["MAX_CONTENT_LENGTH"] = _cfg.MAX_UPLOAD_BYTES
    else:
        app.config["MAX_CONTENT_LENGTH"] = _cfg.WEB_MAX_BODY_BYTES
    
    # Flask session cookie: used only for OAuth (Authlib) state/CSRF during provider redirects.
    # Authenticated API access uses Authorization: Bearer <JWT>, not session cookies.
    # SECRET_KEY comes from Config / .env.local + .env (see app.config.from_object above).
    if not (app.config.get("SECRET_KEY") or "").strip():
        app.config["SECRET_KEY"] = "dev_secret"
    # Match Config (default False for local http:// OAuth); production sets SESSION_COOKIE_SECURE=true
    app.config["SESSION_COOKIE_SECURE"] = bool(_cfg.SESSION_COOKIE_SECURE)
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

    # Explicit allow_headers so preflight allows Authorization (cross-origin SPA → API).
    # See docs/BUG_REPORT_ADMIN_WRITE_401.md
    CORS(
        app,
        supports_credentials=True,
        origins=app.config["CORS_ORIGINS"],
        allow_headers=["Content-Type", "Authorization"],
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    )

    # init_db(app.config["DATABASE_URL"])
    # Base.metadata.create_all(bind=engine)
    
    # extensions.init_db(app.config["DATABASE_URL"])
    # Base.metadata.create_all(bind=extensions.engine)

    print("DATABASE_URL =", app.config.get("DATABASE_URL"))
    print("DATABASE_URL_ACTUAL: ", app.config["DATABASE_URL"])
    engine = init_db(app.config["DATABASE_URL"])
    #Base.metadata.create_all(bind=engine) # affects alembic migration, will remove later!!

    init_oauth(app)
    init_celery(app)

    app.register_blueprint(auth_bp)
    app.register_blueprint(health_bp)
    app.register_blueprint(assignments_bp)
    app.register_blueprint(assignment_materials_bp)
    app.register_blueprint(submissions_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(courses_bp)
    app.register_blueprint(standalone_bp)
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=Config.FLASK_PORT, debug=True)
