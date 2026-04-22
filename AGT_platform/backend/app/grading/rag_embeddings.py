"""
Build embedding vectors for submission text (SentenceTransformers, OpenAI, Ollama, or hash fallback).
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np
import requests

from ..config import Config

_log = logging.getLogger(__name__)

_st_lock = threading.Lock()
_st_models: dict[str, Any] = {}


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


def _openai_embed_snippet(snippet: str, cfg: Config) -> tuple[list[float], str] | None:
    key = (cfg.OPENAI_API_KEY or "").strip()
    if not key or not snippet:
        return None
    model = (
        (getattr(cfg, "OPENAI_TRIO_RAG_EMBEDDING_MODEL", "") or "").strip()
        or "text-embedding-3-small"
    )
    try:
        from openai import OpenAI

        client = OpenAI(api_key=key)
        resp = client.embeddings.create(
            model=model,
            input=snippet[:8000],
        )
        vec = list(resp.data[0].embedding)
        return vec, f"openai:{model}"
    except Exception as exc:
        _log.warning("OpenAI embedding failed (%s); trying other fallbacks", exc)
        return None


def _get_sentence_transformer(model_name: str) -> Any:
    """Lazy singleton per model id (thread-safe)."""
    with _st_lock:
        if model_name not in _st_models:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise RuntimeError(
                    "sentence-transformers is not installed. "
                    "pip install sentence-transformers"
                ) from e
            _log.info("Loading SentenceTransformer %r …", model_name)
            _st_models[model_name] = SentenceTransformer(model_name)
        return _st_models[model_name]


def sentence_transformers_embed_text(text: str, cfg: Config) -> tuple[list[float], str] | None:
    """
    Encode a single text chunk with :class:`sentence_transformers.SentenceTransformer`.

    Returns ``None`` on empty input, import failure, or encode errors (caller may fall back).
    """
    snippet = (text or "").strip()
    if not snippet:
        return None
    model_name = (getattr(cfg, "SENTENCE_TRANSFORMERS_MODEL", "") or "").strip()
    if not model_name:
        model_name = "all-MiniLM-L6-v2"
    try:
        model = _get_sentence_transformer(model_name)
    except Exception as exc:
        _log.warning("SentenceTransformer load failed for %r: %s", model_name, exc)
        return None
    try:
        vec = model.encode(
            snippet,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        if getattr(vec, "ndim", 0) > 1:
            vec = vec[0]
        out = np.asarray(vec, dtype=np.float64).ravel()
        if out.size < 8:
            return None
        return out.tolist(), f"sentence_transformers:{model_name}"
    except Exception as exc:
        _log.warning("SentenceTransformer encode failed: %s", exc)
        return None


def _ollama_embed_snippet(snippet: str, cfg: Config) -> tuple[list[float], str] | None:
    base = (cfg.INTERNAL_OLLAMA_URL or cfg.OLLAMA_BASE_URL or "").strip().rstrip("/")
    embed_model = (getattr(cfg, "OLLAMA_EMBEDDINGS_MODEL", "") or "nomic-embed-text").strip()
    if not base or not snippet:
        return None
    try:
        r = requests.post(
            f"{base}/api/embeddings",
            json={"model": embed_model, "prompt": snippet},
            timeout=120,
        )
        if r.status_code == 404:
            _log.debug(
                "Ollama /api/embeddings 404; trying /api/embed or fallbacks (install embedding model or use RAG_EMBED_ORDER=openai_first)"
            )
            raise RuntimeError("legacy /api/embeddings not available")
        r.raise_for_status()
        emb = r.json().get("embedding")
        if isinstance(emb, list) and emb:
            return [float(x) for x in emb], f"ollama:{embed_model}"
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            _log.debug("Ollama /api/embeddings HTTP 404; trying /api/embed")
        else:
            _log.warning("Ollama embedding failed (%s); trying /api/embed or fallbacks", exc)
    except Exception as exc:
        if "404" in str(exc).lower() or "not available" in str(exc).lower():
            _log.debug("Ollama embedding path unavailable (%s); trying /api/embed or fallbacks", exc)
        else:
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
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            _log.debug(
                "Ollama /api/embed HTTP 404 — use a current Ollama build with embed API, "
                "or set OPENAI_API_KEY and RAG_EMBED_ORDER=openai_first (default when key is set)"
            )
        else:
            _log.warning("Ollama /api/embed failed (%s); trying other fallbacks", exc)
    except Exception as exc:
        _log.debug("Ollama /api/embed failed (%s); trying other fallbacks", exc)
    return None


def compute_submission_embedding(text: str, cfg: Config) -> tuple[list[float], str]:
    """
    Return (vector, source_description).

    Primary path is ``RAG_EMBEDDING_BACKEND``:

    - ``sentence_transformers`` (default): local :class:`sentence_transformers.SentenceTransformer`
      (``SENTENCE_TRANSFORMERS_MODEL``, default ``all-MiniLM-L6-v2``). On failure, falls back
      to the OpenAI/Ollama order from ``RAG_EMBED_ORDER``, then deterministic hash.
    - ``openai``: OpenAI Embeddings API (``OPENAI_TRIO_RAG_EMBEDDING_MODEL``, requires
      ``OPENAI_API_KEY``); on failure falls back to sentence_transformers then hash.
    - ``ollama``: legacy behavior — order from ``RAG_EMBED_ORDER`` (OpenAI + Ollama), then hash.
    """
    max_c = int(getattr(cfg, "RAG_EMBED_MAX_CHARS", 24000))
    snippet = (text or "")[:max_c]

    backend = (getattr(cfg, "RAG_EMBEDDING_BACKEND", "") or "sentence_transformers").strip().lower()
    if backend not in ("ollama", "sentence_transformers", "openai"):
        _log.warning("Unknown RAG_EMBEDDING_BACKEND=%r; using sentence_transformers", backend)
        backend = "sentence_transformers"

    if backend == "openai":
        hit = _openai_embed_snippet(snippet, cfg)
        if hit:
            return hit
        _log.warning(
            "RAG_EMBEDDING_BACKEND=openai failed; falling back to sentence_transformers"
        )
        hit = sentence_transformers_embed_text(snippet, cfg)
        if hit:
            return hit
        dim = 256
        return deterministic_hash_embedding(snippet, dim), "deterministic_hash:sha256×256"

    if backend == "sentence_transformers":
        hit = sentence_transformers_embed_text(snippet, cfg)
        if hit:
            return hit
        _log.warning(
            "RAG_EMBEDDING_BACKEND=sentence_transformers failed; falling back to "
            "OpenAI/Ollama per RAG_EMBED_ORDER"
        )

    order = (getattr(cfg, "RAG_EMBED_ORDER", "auto") or "auto").strip().lower()
    key_ok = bool((cfg.OPENAI_API_KEY or "").strip())
    base = (cfg.INTERNAL_OLLAMA_URL or cfg.OLLAMA_BASE_URL or "").strip().rstrip("/")
    has_ollama = bool(base)

    methods: list[str]
    if order == "openai_only":
        methods = ["openai"]
    elif order == "ollama_only":
        methods = ["ollama"]
    elif order == "openai_first":
        methods = ["openai", "ollama"]
    elif order == "ollama_first":
        methods = ["ollama", "openai"]
    elif key_ok:
        methods = ["openai", "ollama"]
    else:
        methods = ["ollama", "openai"]

    for m in methods:
        if m == "openai" and key_ok:
            hit = _openai_embed_snippet(snippet, cfg)
            if hit:
                return hit
        elif m == "ollama" and has_ollama:
            hit = _ollama_embed_snippet(snippet, cfg)
            if hit:
                return hit

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
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return json_path
