"""
S3-compatible object storage via boto3.

Works with AWS S3 (leave S3_ENDPOINT unset) or MinIO / other S3 APIs (set S3_ENDPOINT).

Large uploads use multipart transfers (see TransferConfig). Uploads stream through a spooled
temp file so memory stays bounded for big student/teacher files.
"""
from __future__ import annotations

import tempfile
import uuid
from typing import BinaryIO, Optional

import boto3
from botocore.client import Config as BotoClientConfig
from botocore.exceptions import ClientError
from boto3.s3.transfer import TransferConfig

from .config import Config

# Multipart: good default for notebooks, videos, large PDFs
_TRANSFER = TransferConfig(
    multipart_threshold=8 * 1024 * 1024,
    multipart_chunksize=8 * 1024 * 1024,
    max_concurrency=10,
    use_threads=True,
)


def _addressing_style(cfg: Config) -> str:
    raw = (cfg.S3_ADDRESSING_STYLE or "").strip().lower()
    if raw in ("path", "virtual"):
        return raw
    return "path" if cfg.S3_ENDPOINT else "virtual"


def s3_client(cfg: Config):
    """Low-level boto3 S3 client (AWS or MinIO)."""
    kwargs: dict = {
        "region_name": cfg.AWS_REGION or cfg.S3_REGION or "us-east-1",
        "config": BotoClientConfig(
            signature_version="s3v4",
            s3={"addressing_style": _addressing_style(cfg)},
        ),
    }
    if cfg.S3_ACCESS_KEY:
        kwargs["aws_access_key_id"] = cfg.S3_ACCESS_KEY
    if cfg.S3_SECRET_KEY:
        kwargs["aws_secret_access_key"] = cfg.S3_SECRET_KEY
    if cfg.S3_ENDPOINT:
        kwargs["endpoint_url"] = cfg.S3_ENDPOINT
    use_ssl = cfg.S3_SECURE
    kwargs["use_ssl"] = use_ssl
    return boto3.client("s3", **kwargs)


def _upload_fileobj(
    cfg: Config,
    fileobj: BinaryIO,
    key: str,
    content_type: Optional[str],
    extra_args: Optional[dict] = None,
) -> str:
    client = s3_client(cfg)
    extra = dict(extra_args or {})
    if content_type:
        extra["ContentType"] = content_type
    upload_kw = {"Config": _TRANSFER}
    if extra:
        upload_kw["ExtraArgs"] = extra
    client.upload_fileobj(fileobj, cfg.S3_BUCKET, key, **upload_kw)
    return key


def upload_from_werkzeug_file(cfg: Config, file_storage, key: str) -> str:
    """
    Stream upload from Flask/Werkzeug FileStorage into S3 without loading whole file into RAM.
    Uses a SpooledTemporaryFile (spills to disk after max_size bytes in memory).
    """
    content_type = file_storage.mimetype or "application/octet-stream"
    max_mem = cfg.S3_UPLOAD_SPOOL_MAX_MEMORY_BYTES
    spool: tempfile.SpooledTemporaryFile = tempfile.SpooledTemporaryFile(max_size=max_mem)
    try:
        while True:
            chunk = file_storage.stream.read(1024 * 1024)
            if not chunk:
                break
            spool.write(chunk)
        spool.seek(0)
        return _upload_fileobj(cfg, spool, key, content_type)
    finally:
        spool.close()


def put_object(
    cfg: Config,
    data_stream: BinaryIO,
    length: int,
    content_type: str,
    prefix: str,
) -> str:
    """
    Upload from a bounded stream (e.g. BytesIO). Key: {prefix}/{uuid_hex}.
    Prefer upload_from_werkzeug_file for arbitrary large uploads.
    """
    key = f"{prefix}/{uuid.uuid4().hex}"
    if length <= cfg.S3_INLINE_UPLOAD_MAX_BYTES:
        body = data_stream.read(length)
        s3_client(cfg).put_object(
            Bucket=cfg.S3_BUCKET,
            Key=key,
            Body=body,
            ContentType=content_type or "application/octet-stream",
        )
        return key
    return _upload_fileobj(cfg, data_stream, key, content_type)


def get_object_bytes(cfg: Config, key: str) -> bytes:
    """Download full object (used by Celery grading)."""
    r = s3_client(cfg).get_object(Bucket=cfg.S3_BUCKET, Key=key)
    return r["Body"].read()


def get_presigned_url(cfg: Config, key: str, method: str = "GET", expires: int = 3600) -> str:
    client = s3_client(cfg)
    m = method.upper()
    if m == "GET":
        return client.generate_presigned_url(
            "get_object",
            Params={"Bucket": cfg.S3_BUCKET, "Key": key},
            ExpiresIn=expires,
        )
    if m == "PUT":
        return client.generate_presigned_url(
            "put_object",
            Params={"Bucket": cfg.S3_BUCKET, "Key": key},
            ExpiresIn=expires,
        )
    raise ValueError(f"unsupported presign method: {method}")


def presigned_put_url(
    cfg: Config,
    key: str,
    content_type: str,
    expires: Optional[int] = None,
) -> str:
    """
    Browser → S3 direct upload. Client must send the same Content-Type header on PUT.
    Keeps large files off the Flask host (production ingress is metadata + presign only).
    """
    exp = expires if expires is not None else cfg.S3_PRESIGN_PUT_EXPIRES
    client = s3_client(cfg)
    ct = content_type or "application/octet-stream"
    return client.generate_presigned_url(
        "put_object",
        Params={
            "Bucket": cfg.S3_BUCKET,
            "Key": key,
            "ContentType": ct,
        },
        ExpiresIn=exp,
        HttpMethod="PUT",
    )


def object_exists(cfg: Config, key: str) -> bool:
    """Return True if object is present (used to finalize direct uploads)."""
    try:
        s3_client(cfg).head_object(Bucket=cfg.S3_BUCKET, Key=key)
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound", "404 Not Found"):
            return False
        raise
