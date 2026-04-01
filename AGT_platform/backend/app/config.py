import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent  # backend/
load_dotenv(BASE_DIR / ".env.local")
load_dotenv(BASE_DIR / ".env", override=True)


def _env_str(key: str) -> str:
    """Always use empty-string default with os.getenv (never None)."""
    return os.getenv(key, "")


def _env_int(key: str, *, default: int) -> int:
    v = os.getenv(key, "").strip()
    return default if not v else int(v, 10)


def _env_bool(key: str) -> bool:
    return os.getenv(key, "").strip().lower() == "true"


class Config:
    SECRET_KEY = _env_str("SECRET_KEY")
    JWT_EXPIRATION_SECONDS = _env_int("JWT_EXPIRATION_SECONDS", default=8 * 3600)
    DATABASE_URL = _env_str("DATABASE_URL")
    REDIS_URL = _env_str("REDIS_URL")
    FRONTEND_BASE_URL = _env_str("FRONTEND_BASE_URL")

    # deployment_tier: web | gpu — informational; used in logs and optional guards
    DEPLOYMENT_TIER = _env_str("DEPLOYMENT_TIER").strip().lower() or "web"

    # S3-compatible storage (AWS S3 or MinIO). Empty S3_ENDPOINT = native AWS.
    S3_ENDPOINT = _env_str("S3_ENDPOINT").strip()
    S3_ACCESS_KEY = _env_str("S3_ACCESS_KEY")
    S3_SECRET_KEY = _env_str("S3_SECRET_KEY")
    S3_BUCKET = _env_str("S3_BUCKET")
    S3_REGION = _env_str("S3_REGION").strip() or "us-east-1"
    S3_SECURE = _env_bool("S3_SECURE")
    S3_ADDRESSING_STYLE = _env_str("S3_ADDRESSING_STYLE").strip()
    AWS_REGION = _env_str("AWS_REGION").strip() or S3_REGION

    # Prefix for student submission objects (unique keys still include ids/uuids).
    UPLOADS_S3_PREFIX = _env_str("UPLOADS_S3_PREFIX").strip() or "assignments/by-id"

    MAX_UPLOAD_MB = _env_int("MAX_UPLOAD_MB", default=1024)
    MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

    # Cap request body size on the API when not using multipart uploads (JSON-only ingress).
    WEB_MAX_BODY_MB = _env_int("WEB_MAX_BODY_MB", default=4)
    WEB_MAX_BODY_BYTES = WEB_MAX_BODY_MB * 1024 * 1024

    S3_INLINE_UPLOAD_MAX_BYTES = _env_int(
        "S3_INLINE_UPLOAD_MAX_BYTES", default=32 * 1024 * 1024
    )
    S3_UPLOAD_SPOOL_MAX_MEMORY_BYTES = _env_int(
        "S3_UPLOAD_SPOOL_MAX_MEMORY_BYTES", default=16 * 1024 * 1024
    )

    # Presigned PUT lifetime (browser → S3 direct upload).
    S3_PRESIGN_PUT_EXPIRES = _env_int("S3_PRESIGN_PUT_EXPIRES", default=3600)

    # Production: false — browser uses presigned S3 only. Dev docker: set true to use multipart to Flask.
    ALLOW_FLASK_MULTIPART_UPLOAD = _env_bool("ALLOW_FLASK_MULTIPART_UPLOAD")

    OIDC_CLIENT_ID = _env_str("OIDC_CLIENT_ID")
    OIDC_CLIENT_SECRET = _env_str("OIDC_CLIENT_SECRET")
    OIDC_DISCOVERY_URL = _env_str("OIDC_DISCOVERY_URL")
    OIDC_REDIRECT_URI = _env_str("OIDC_REDIRECT_URI")

    MICROSOFT_CLIENT_ID = _env_str("MICROSOFT_CLIENT_ID")
    MICROSOFT_CLIENT_SECRET = _env_str("MICROSOFT_CLIENT_SECRET")

    GOOGLE_CLIENT_ID = _env_str("GOOGLE_CLIENT_ID")
    GOOGLE_CLIENT_SECRET = _env_str("GOOGLE_CLIENT_SECRET")

    # GPU / private inference (workers only in split prod; never expose Ollama publicly).
    OLLAMA_BASE_URL = _env_str("OLLAMA_BASE_URL")
    OLLAMA_MODEL = _env_str("OLLAMA_MODEL")
    # Optional override when Ollama is on a different private host than default compose DNS.
    INTERNAL_OLLAMA_URL = _env_str("INTERNAL_OLLAMA_URL").strip() or _env_str(
        "OLLAMA_BASE_URL"
    ).strip()

    OPENAI_API_KEY = _env_str("OPENAI_API_KEY")
    OPENAI_MODEL = _env_str("OPENAI_MODEL").strip() or "gpt-4o-mini"
    # If true, re-run or arbitrate grading with OpenAI when local model confidence is low.
    ESCALATE_TO_OPENAI = _env_bool("ESCALATE_TO_OPENAI")

    WHISPER_ENABLED = _env_bool("WHISPER_ENABLED")

    # Celery worker tuning (documented for ops; worker command line should match).
    CELERY_WORKER_CONCURRENCY = _env_int("CELERY_WORKER_CONCURRENCY", default=1)
    CELERY_WORKER_PREFETCH = _env_int("CELERY_WORKER_PREFETCH", default=1)

    _cors = _env_str("CORS_ORIGINS").strip()
    if not _cors:
        _cors = (
            "http://localhost:5173,http://127.0.0.1:5173,"
            "https://dia-ai-grader.com,https://www.dia-ai-grader.com,"
            "https://api.dia-ai-grader.com"
        )
    CORS_ORIGINS = [o.strip() for o in _cors.split(",") if o.strip()]
