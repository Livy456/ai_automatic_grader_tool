"""
Build embedding vectors for submission text (Ollama /api/embeddings, OpenAI, or hash fallback).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import requests

from ..config import Config

_log = logging.getLogger(__name__)


def deterministic_hash_embedding(text: str, dimensions: int = 256) -> list[float]:
    """
    Offline-stable pseudo-embedding (not semantic). Used when no API is available.
    Fills a vector in blocks via SHA-256 streams; values vectorized as uint16 → float.
    """
    seed = hashlib.sha256(text.encode("utf-8", errors="replace")).digest()
    out = np.empty(dimensions, dtype=np.float64)
    filled = 0
    counter = 0
    scale = 1.0 / 65535.0
    while filled < dimensions:
        block = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
        counter += 1
        need = min(dimensions - filled, len(block) // 2)
        u16 = np.frombuffer(block, dtype=np.uint16, count=need)
        out[filled : filled + need] = u16.astype(np.float64) * scale - 0.5
        filled += need
    return out.tolist()


def compute_submission_embedding(text: str, cfg: Config) -> tuple[list[float], str]:
    """
    Return (vector, source_description).

    Tries Ollama ``/api/embeddings``, then OpenAI ``text-embedding-3-small``, then hash fallbacks.
    """
    max_c = int(getattr(cfg, "RAG_EMBED_MAX_CHARS", 24000))
    snippet = (text or "")[:max_c]

    base = (cfg.INTERNAL_OLLAMA_URL or cfg.OLLAMA_BASE_URL or "").strip().rstrip("/")
    embed_model = (getattr(cfg, "OLLAMA_EMBEDDINGS_MODEL", "") or "nomic-embed-text").strip()
    if base and snippet:
        try:
            r = requests.post(
                f"{base}/api/embeddings",
                json={"model": embed_model, "prompt": snippet},
                timeout=120,
            )
            if r.status_code == 404:
                raise RuntimeError("legacy /api/embeddings not available")
            r.raise_for_status()
            emb = r.json().get("embedding")
            if isinstance(emb, list) and emb:
                return [float(x) for x in emb], f"ollama:{embed_model}"
        except Exception as exc:
            _log.warning("Ollama embedding failed (%s); trying /api/embed or fallbacks", exc)
        try:
            r2 = requests.post(
                f"{base}/api/embed",
                json={"model": embed_model, "input": snippet},
                timeout=120,
            )
            r2.raise_for_status()
            data = r2.json()
            vecs = data.get("embeddings")
            if isinstance(vecs, list) and vecs and isinstance(vecs[0], list):
                return [float(x) for x in vecs[0]], f"ollama_embed:{embed_model}"
            emb_one = data.get("embedding")
            if isinstance(emb_one, list) and emb_one:
                return [float(x) for x in emb_one], f"ollama_embed:{embed_model}"
        except Exception as exc:
            _log.warning("Ollama /api/embed failed (%s); trying fallbacks", exc)

    key = (cfg.OPENAI_API_KEY or "").strip()
    if key and snippet:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=key)
            resp = client.embeddings.create(
                model="text-embedding-3-small",
                input=snippet[:8000],
            )
            vec = list(resp.data[0].embedding)
            return vec, "openai:text-embedding-3-small"
        except Exception as exc:
            _log.warning("OpenAI embedding failed (%s); using hash embedding", exc)

    dim = 256
    return deterministic_hash_embedding(snippet, dim), "deterministic_hash:sha256×256"


def save_rag_embedding_bundle(
    out_dir: Path,
    *,
    assignment_stem: str,
    artifacts_keys: list[str],
    plaintext_chars: int,
    embedding: list[float],
    embedding_source: str,
    parsed_preview: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write ``<stem>_embedding.json`` and ``<stem>_parsed_preview.txt`` under ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_path = out_dir / f"{assignment_stem}_parsed_preview.txt"
    preview_path.write_text(parsed_preview[:50000], encoding="utf-8")
    payload = {
        "assignment_stem": assignment_stem,
        "artifacts_keys": artifacts_keys,
        "plaintext_char_count": plaintext_chars,
        "embedding_dimension": len(embedding),
        "embedding_source": embedding_source,
        "embedding": embedding,
        "extra": extra or {},
    }
    json_path = out_dir / f"{assignment_stem}_embedding.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return json_path
