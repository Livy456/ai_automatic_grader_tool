from minio import Minio
from .config import Config
import uuid

def client(cfg: Config):
    return Minio(
        cfg.S3_ENDPOINT.replace("http://","").replace("https://",""),
        access_key=cfg.S3_ACCESS_KEY,
        secret_key=cfg.S3_SECRET_KEY,
        secure=cfg.S3_SECURE
    )

def put_object(cfg: Config, data_stream, length: int, content_type: str, prefix: str):
    c = client(cfg)
    key = f"{prefix}/{uuid.uuid4().hex}"
    c.put_object(cfg.S3_BUCKET, key, data_stream, length, content_type=content_type)
    return key

def get_presigned_url(cfg: Config, key: str, method="GET", expires=3600):
    c = client(cfg)
    return c.presigned_get_object(cfg.S3_BUCKET, key, expires=expires)
