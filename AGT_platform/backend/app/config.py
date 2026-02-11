import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent  # points to backend/
load_dotenv(BASE_DIR / ".env")

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev_secret")
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://dev:dev@localhost:5432/ai_grader")
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    FRONTEND_BASE_URL = os.getenv("FRONTEND_BASE_URL", "http://172.20.29.253:5173/")

    # S3 compatible storage
    S3_ENDPOINT = os.getenv("S3_ENDPOINT")
    S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
    S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")
    S3_BUCKET = os.getenv("S3_BUCKET", "ai-grader")
    S3_REGION = os.getenv("S3_REGION", "us-east-1")
    S3_SECURE = os.getenv("S3_SECURE", "false").lower() == "true"

    # OIDC / SSO
    OIDC_CLIENT_ID = os.getenv("OIDC_CLIENT_ID")
    OIDC_CLIENT_SECRET = os.getenv("OIDC_CLIENT_SECRET")
    OIDC_DISCOVERY_URL = os.getenv("OIDC_DISCOVERY_URL")
    OIDC_REDIRECT_URI = os.getenv("OIDC_REDIRECT_URI")
    
    # Microsoft OAuth
    MICROSOFT_CLIENT_ID = os.getenv("MICROSOFT_CLIENT_ID")
    MICROSOFT_CLIENT_SECRET = os.getenv("MICROSOFT_CLIENT_SECRET")
    
    # Google OAuth
    GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
    GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")

    # AI
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    WHISPER_ENABLED = os.getenv("WHISPER_ENABLED", "false").lower() == "true"

    #DEBUGGING !!
    print("DATABASE_URL =", os.getenv("DATABASE_URL"))
