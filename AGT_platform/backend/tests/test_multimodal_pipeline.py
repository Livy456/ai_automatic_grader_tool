"""
Tests for the multimodal grading pipeline.

When integration logs show ``grading_llm_sample_failed`` or JSON/HTTP errors, those are
**LLM inference** failures (Ollama HTTP or local Hugging Face ``transformers``) during
per-chunk grading, not notebook or RAG chunking.
Chunking is recorded under ``pipeline_audit`` → ``chunking``; failures there would indicate
a chunking issue.

- **Unit tests** (routing, entropy, mock pipeline): fast, no LLM required.
- **Integration test** (``LocalAssignmentGradingTests``): grades every file in
  ``assignments_to_grade/`` using the real rubric from ``rubric/default.json``.
  By default the **chat / structure LLM** is **OpenAI** (``gpt-5.4-nano``) when
  ``OPENAI_API_KEY`` is set in the environment; otherwise **Hugging Face** Maverick
  (``Llama-4-Maverick-17B-128E-Instruct:fp8``). Set ``MULTIMODAL_INTEGRATION_LLM_BACKEND=huggingface``
  or ``=ollama`` to force those backends. **RAG embeddings** default to SentenceTransformers.
  Run ``pytest -rs`` for full ``SkipTest`` reasons; use
  ``--log-cli-level=WARNING`` for phased ``[integration]`` diagnostics.

  For **each** assignment (integration class): resolve answer key, build
  ``*_multimodal_units.json`` + embeddings once, run the pipeline from that cache,
  validate (including ``_agentic_workflow``), then write ``grading_output/<stem>_grade_output.json``.

  Optional **answer_key/** (same stem as assignment: ``<stem>.txt``, ``.md``, or ``.json``)
  is loaded into ``modality_hints["answer_key_plaintext"]`` for grading prompts.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import requests

from app.config import Config
from app.grading.llm_router import (
    build_multimodal_grading_clients,
    huggingface_grading_model_id,
    multimodal_llm_backend_uses_huggingface,
    multimodal_llm_backend_uses_openai,
    multimodal_structure_llm_trace_label,
)
from app.grading.grading_units import build_grading_units_from_chunks
from app.grading.modality_resolution import resolve_modality_profile
from app.grading.multimodal import (
    MultimodalGradingConfig,
    MultimodalGradingPipeline,
    Modality,
    PipelineArtifactStore,
    TaskType,
    create_multimodal_pipeline_from_app_config,
    multimodal_assignment_to_grading_dict,
)
from app.grading.multimodal.pipeline import build_envelope_from_plaintext
from app.grading.multimodal.ingestion import ingest_raw_submission
from app.grading.multimodal.schemas import (
    GradingChunk,
    RubricType,
    SampledChunkGrade,
)
from app.grading.multimodal.model_runner import MockChunkModelRunner
from app.grading.multimodal.rubric_router import RubricRouteResult, route_rubric
from app.grading.multimodal.entropy import semantic_entropy_from_cluster_counts
from app.grading.multimodal.parser import parse_chunk_grade_json
from app.grading.multimodal.answer_key_chunk_enrich import (
    enrich_chunks_with_per_question_answer_key,
    split_answer_key_sections,
)
from app.grading.multimodal.chunk_cache import save_grading_chunks_cache
from app.grading.multimodal.notebook_chunker import build_notebook_qa_chunks
from app.grading.multimodal.prompts_chunk import build_chunk_grading_prompt
from app.grading.multimodal.rag_embeddings import (
    build_multimodal_grading_chunks,
    enrich_chunks_with_rag_embeddings,
)
from app.grading.answer_key_resolve import resolve_answer_key_plaintext
from app.grading.output_schema import validate_grading_output
from app.grading.rag_embeddings import (
    compute_submission_embedding,
    deterministic_hash_embedding,
    save_rag_embedding_bundle,
    sentence_transformers_embed_text,
)
from app.grading.submission_chunks import build_submission_chunks, write_chunks_json
from app.grading.submission_text import submission_text_from_artifacts

_log = logging.getLogger(__name__)


def _integration_log(msg: str, *args: object) -> None:
    """Structured integration diagnostics (use ``pytest --log-cli-level=WARNING``)."""
    _log.warning("[integration] " + msg, *args)


def _hf_integration_preflight(cfg: Config) -> tuple[bool, str]:
    """
    Verify optional HF deps + hub token can read the gated model card (no full weight download).
    """
    _integration_log("HF preflight: import torch …")
    try:
        import torch  # noqa: F401

        _integration_log("HF preflight: torch OK (%s)", torch.__version__)
    except ImportError as e:
        return (
            False,
            f"NO_HF_DEPS: torch missing ({e!r}). "
            "Install: pip install -r requirements-huggingface.txt",
        )

    _integration_log("HF preflight: import transformers …")
    try:
        import transformers

        _integration_log("HF preflight: transformers OK (%s)", transformers.__version__)
    except ImportError as e:
        return (
            False,
            f"NO_HF_DEPS: transformers missing ({e!r}). "
            "Install: pip install -r requirements-huggingface.txt",
        )

    _integration_log("HF preflight: import huggingface_hub …")
    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        return (
            False,
            f"NO_HF_DEPS: huggingface_hub missing ({e!r}). "
            "pip install huggingface_hub",
        )

    repo_id = huggingface_grading_model_id(cfg)
    token = (
        (getattr(cfg, "HUGGINGFACE_HUB_TOKEN", "") or "").strip()
        or (getattr(cfg, "HF_TOKEN", "") or "").strip()
    )
    if not token:
        return (
            False,
            "NO_HF_TOKEN: set HUGGINGFACE_HUB_TOKEN or HF_TOKEN (gated meta-llama/*). "
            "Or run with MULTIMODAL_INTEGRATION_LLM_BACKEND=ollama if you only use Ollama.",
        )

    _integration_log(
        "HF preflight: HfApi.model_info(%r) (validates token + repo visibility) …",
        repo_id,
    )
    try:
        api = HfApi(token=token)
        api.model_info(repo_id)
    except Exception as e:
        return (
            False,
            f"HF_HUB_MODEL_CHECK_FAILED: {type(e).__name__}: {e!s}. "
            f"Repo={repo_id!r}. Accept the model license on Hugging Face if required.",
        )

    _integration_log("HF preflight: hub access OK for %r", repo_id)
    return True, ""


REPO_ROOT = Path(__file__).resolve().parents[3]
ASSIGNMENTS_DIR = REPO_ROOT / "assignments_to_grade"
RUBRIC_DIR = REPO_ROOT / "rubric"
OUTPUT_DIR = REPO_ROOT / "grading_output"
RAG_DIR = REPO_ROOT / "RAG_embedding"
ANSWER_KEY_DIR = REPO_ROOT / "answer_key"

_SUPPORTED_SUFFIXES = {".ipynb", ".py", ".pdf", ".txt", ".md", ".mp4"}
# Tabular / data files in ``assignments_to_grade/`` are for ``dataset_resolve`` — not separate submissions.
_DATA_ONLY_SUFFIXES = frozenset({".csv", ".tsv", ".xlsx"})
_SUFFIX_TO_ARTIFACT_KEY = {
    ".ipynb": "ipynb",
    ".py": "py",
    ".pdf": "pdf",
    ".txt": "txt",
    ".md": "md",
    ".mp4": "mp4",
}

_SECTION_NAME_TO_RUBRIC_TYPE = {
    "Scaffolded Coding": RubricType.PROGRAMMING_SCAFFOLDED,
    "Free Response": RubricType.FREE_RESPONSE,
    "Open-Ended EDA": RubricType.EDA_VISUALIZATION,
    "Mock Interview / Oral Assessment": RubricType.ORAL_INTERVIEW,
}


# ---------------------------------------------------------------------------
# Rubric helpers
# ---------------------------------------------------------------------------

def _max_points_from_range(points_range: object) -> float:
    if points_range is None:
        return 10.0
    s = str(points_range).strip().replace(" ", "")
    if "-" in s:
        parts = s.split("-", 1)
        try:
            return float(parts[1])
        except (IndexError, ValueError):
            pass
    try:
        return float(s)
    except ValueError:
        return 10.0


def _flatten_sections_rubric(raw: dict) -> list[dict]:
    out: list[dict] = []
    for sec in raw.get("sections") or []:
        if not isinstance(sec, dict):
            continue
        sec_name = str(sec.get("name") or "Section").strip()
        for c in sec.get("criteria") or []:
            if not isinstance(c, dict):
                continue
            cname = str(c.get("name") or "Criterion").strip()
            max_pts = _max_points_from_range(c.get("points_range"))
            levels = c.get("levels")
            desc = json.dumps(levels, ensure_ascii=False) if isinstance(levels, dict) else ""
            label = f"{sec_name} — {cname}" if sec_name else cname
            out.append({
                "name": label, "max_points": max_pts,
                "criterion": label, "max_score": max_pts,
                "description": desc[:8000],
            })
    return out


def _build_rubric_rows_by_type(rubric_json: dict) -> dict[RubricType, list[dict]]:
    by_type: dict[RubricType, list[dict]] = {}
    for sec in rubric_json.get("sections") or []:
        if not isinstance(sec, dict):
            continue
        sec_name = str(sec.get("name") or "").strip()
        rt = _SECTION_NAME_TO_RUBRIC_TYPE.get(sec_name)
        if rt is None:
            continue
        rows: list[dict] = []
        for c in sec.get("criteria") or []:
            if not isinstance(c, dict):
                continue
            name = str(c.get("name") or "Criterion").strip()
            max_pts = _max_points_from_range(c.get("points_range"))
            levels = c.get("levels")
            desc = json.dumps(levels, ensure_ascii=False) if isinstance(levels, dict) else ""
            rows.append({
                "name": name, "max_points": max_pts,
                "criterion": name, "max_score": max_pts,
                "description": desc,
            })
        by_type[rt] = rows
    return by_type


def _flat_rubric_from_by_type(by_type: dict[RubricType, list[dict]]) -> list[dict]:
    out: list[dict] = []
    for rows in by_type.values():
        out.extend(rows)
    return out


# ---------------------------------------------------------------------------
# File / config helpers
# ---------------------------------------------------------------------------

def _assignment_groups() -> dict[str, list[Path]]:
    if not ASSIGNMENTS_DIR.is_dir():
        return {}
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(ASSIGNMENTS_DIR.iterdir()):
        if not p.is_file() or p.name.startswith("."):
            continue
        suf = p.suffix.lower()
        if suf in _DATA_ONLY_SUFFIXES:
            continue
        if suf not in _SUPPORTED_SUFFIXES:
            continue
        groups[p.stem].append(p)
    return dict(groups)


def _build_artifacts(paths: list[Path]) -> dict[str, bytes]:
    artifacts: dict[str, bytes] = {}
    for p in paths:
        key = _SUFFIX_TO_ARTIFACT_KEY.get(p.suffix.lower())
        if not key:
            continue
        if key in artifacts:
            raise ValueError(f"Duplicate artifact key {key!r} for {paths}")
        artifacts[key] = p.read_bytes()
    return artifacts


def _load_rubric_json() -> dict | None:
    path = RUBRIC_DIR / "default.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _ollama_reachable(cfg: Config) -> bool:
    base = (cfg.INTERNAL_OLLAMA_URL or cfg.OLLAMA_BASE_URL or "").strip().rstrip("/")
    if not base:
        return False
    try:
        return requests.get(f"{base}/api/tags", timeout=4).status_code == 200
    except OSError:
        return False


def _ollama_chat_smoke(cfg: Config) -> tuple[bool, str]:
    """Optional heavy check: /api/chat on ``OLLAMA_MODEL`` (large models may cold-load slowly)."""
    base = (cfg.INTERNAL_OLLAMA_URL or cfg.OLLAMA_BASE_URL or "").strip().rstrip("/")
    model = (cfg.OLLAMA_MODEL or "llama3.2:3b").strip()
    timeout = int(getattr(cfg, "OLLAMA_CHAT_TIMEOUT_SEC", 120))
    try:
        r = requests.post(
            f"{base}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": 'Reply: {"ok":true}'}],
                "stream": False,
                "keep_alive": getattr(cfg, "OLLAMA_KEEP_ALIVE", "5m"),
            },
            timeout=timeout,
        )
        if r.status_code != 200:
            return False, f"HTTP {r.status_code} for {model!r}"
        body = r.json()
        if not (body.get("message") or {}).get("content", "").strip():
            return False, f"Empty content from {model!r}"
        return True, ""
    except requests.exceptions.ReadTimeout:
        return False, f"Timeout ({timeout}s) for {model!r}"
    except requests.exceptions.RequestException as e:
        return False, str(e)


def _ollama_embedding_smoke(cfg: Config) -> tuple[bool, str]:
    """
    Fast Ollama check for multimodal integration on the RAG model (default
    ``nomic-embed-text``). Matches :func:`app.grading.rag_embeddings._ollama_embed_snippet`:
    try legacy ``POST /api/embeddings`` first, then ``POST /api/embed`` (current Ollama).
    """
    base = (cfg.INTERNAL_OLLAMA_URL or cfg.OLLAMA_BASE_URL or "").strip().rstrip("/")
    embed_model = (getattr(cfg, "OLLAMA_EMBEDDINGS_MODEL", "") or "nomic-embed-text").strip()
    if not base or not embed_model:
        return False, "missing OLLAMA_BASE_URL or OLLAMA_EMBEDDINGS_MODEL"

    def _ok_embedding_list(obj: object) -> bool:
        return isinstance(obj, list) and len(obj) >= 8

    try:
        r = requests.post(
            f"{base}/api/embeddings",
            json={"model": embed_model, "prompt": "ping"},
            timeout=45,
        )
        if r.status_code == 200:
            emb = r.json().get("embedding")
            if _ok_embedding_list(emb):
                return True, ""
        # Many Ollama builds return 404 on /api/embeddings but serve /api/embed.
        if r.status_code != 200:
            _integration_log(
                "ollama_embedding_smoke: /api/embeddings HTTP %s for model=%r — trying /api/embed",
                r.status_code,
                embed_model,
            )
    except requests.exceptions.RequestException as e:
        _integration_log(
            "ollama_embedding_smoke: /api/embeddings request error (%s) — trying /api/embed",
            e,
        )

    embed_candidates = [embed_model]
    if ":" not in embed_model:
        embed_candidates.append(f"{embed_model}:latest")

    last_embed_err = ""
    for m in embed_candidates:
        try:
            r2 = requests.post(
                f"{base}/api/embed",
                json={"model": m, "input": "ping"},
                timeout=45,
            )
            if r2.status_code != 200:
                last_embed_err = (
                    f"HTTP {r2.status_code} /api/embed model={m!r} body={r2.text[:200]!r}"
                )
                continue
            data = r2.json()
            vecs = data.get("embeddings")
            if isinstance(vecs, list) and vecs and isinstance(vecs[0], list):
                if _ok_embedding_list(vecs[0]):
                    return True, ""
            emb_one = data.get("embedding")
            if _ok_embedding_list(emb_one):
                return True, ""
            last_embed_err = f"bad /api/embed payload for {m!r}"
        except requests.exceptions.RequestException as e:
            last_embed_err = str(e)
    return False, last_embed_err or "Ollama /api/embed failed for all model name candidates"


def _rag_embedding_smoke(cfg: Config) -> tuple[bool, str]:
    """
    Match :func:`compute_submission_embedding` gating: SentenceTransformers, OpenAI, or Ollama.
    """
    backend = (getattr(cfg, "RAG_EMBEDDING_BACKEND", "") or "").strip().lower()
    if backend == "sentence_transformers":
        hit = sentence_transformers_embed_text("ping", cfg)
        if hit:
            return True, ""
        return (
            False,
            "sentence_transformers path failed (pip install sentence-transformers; "
            "check SENTENCE_TRANSFORMERS_MODEL)",
        )
    if backend == "openai":
        from app.grading.rag_embeddings import _openai_embed_snippet

        if not (cfg.OPENAI_API_KEY or "").strip():
            return False, "RAG_EMBEDDING_BACKEND=openai requires OPENAI_API_KEY"
        hit = _openai_embed_snippet("ping", cfg)
        if hit:
            return True, ""
        return False, "OpenAI embeddings.create failed (check key, model, network)"
    return _ollama_embedding_smoke(cfg)


def _ollama_model_compact_key(name: str) -> str:
    """Normalize model id for fuzzy match (Ollama tags differ in casing / punctuation)."""
    base = (name or "").strip().split(":", 1)[0]
    return re.sub(r"[^a-z0-9]+", "", base.lower())


def _ollama_show_succeeds(base: str, model: str, *, timeout: float = 25.0) -> bool:
    ok, _ = _ollama_show_detail(base, model, timeout=timeout)
    return ok


def _ollama_show_detail(
    base: str, model: str, *, timeout: float = 25.0
) -> tuple[bool, str]:
    try:
        r = requests.post(
            f"{base}/api/show",
            json={"model": (model or "").strip()},
            timeout=timeout,
        )
        if r.status_code == 200:
            return True, ""
        return False, f"HTTP {r.status_code} body={r.text[:400]!r}"
    except requests.RequestException as e:
        return False, str(e)


def _resolve_ollama_grading_model_name(
    base: str, preferred: str, local_models: set[str]
) -> tuple[str | None, str]:
    """
    Return ``(ollama_model_name, note)`` for ``/api/chat`` — Ollama only accepts names that
    pass ``POST /api/show``. ``preferred`` may differ from the tag shown in ``ollama list``
    (e.g. Meta download name vs Modelfile name).
    """
    pref = (preferred or "").strip()
    if not pref:
        return None, "empty OLLAMA_MODEL / MULTIMODAL_INTEGRATION_OLLAMA_MODEL"

    candidates: list[str] = []
    for p in (pref, pref.lower()):
        if not p:
            continue
        candidates.append(p)
        if ":" not in p:
            candidates.append(f"{p}:latest")

    seen: set[str] = set()
    ordered: list[str] = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            ordered.append(c)

    last_err = ""
    _to = 25.0
    for c in ordered:
        ok, err = _ollama_show_detail(base, c, timeout=_to)
        if ok:
            return c, "" if c == pref else f"resolved {pref!r} → {c!r} via /api/show"
        last_err = err or last_err

    pref_key = _ollama_model_compact_key(pref)
    tag_keys = [(tag, _ollama_model_compact_key(tag)) for tag in sorted(local_models)]
    for tag, tk in tag_keys:
        if tk and tk == pref_key:
            ok, err = _ollama_show_detail(base, tag, timeout=_to)
            if ok:
                return (
                    tag,
                    f"resolved {pref!r} → {tag!r} (normalized id == + /api/show)",
                )
            last_err = err or last_err
    for tag, tk in tag_keys:
        if not tk or tk == pref_key:
            continue
        if pref_key in tk or tk in pref_key:
            ok, err = _ollama_show_detail(base, tag, timeout=_to)
            if ok:
                return (
                    tag,
                    f"resolved {pref!r} → {tag!r} (normalized id fuzzy + /api/show)",
                )
            last_err = err or last_err

    preview = ", ".join(sorted(local_models)[:12])
    more = " …" if len(local_models) > 12 else ""
    err_tail = f" Last /api/show error: {last_err}" if last_err else ""
    return (
        None,
        f"GRADING_MODEL_NOT_IN_OLLAMA: {pref!r} not registered (see `ollama list`).{err_tail} "
        f"Tags sample: [{preview}]{more}. "
        f"Fix: `ollama pull <name>` or Modelfile `FROM /path/to.gguf` then `ollama create`. "
        f"Or set MULTIMODAL_INTEGRATION_OLLAMA_MODEL to the exact tag.",
    )


def _configure_for_integration_test(cfg: Config) -> None:
    """Override Config in-place for the integration test.

    **Chunking** (priority order):
      1. **Notebook cell-order** — ipynb files are parsed directly by
         ``build_notebook_qa_chunks``, preserving cell order for accurate
         question/answer pairing.  No LLM required.
      2. **LLM QA segmentation** — when ``MULTIMODAL_OLLAMA_QA_SEGMENT=on``, non-notebook
         files use the **structure LLM** (Hugging Face or Ollama per
         ``MULTIMODAL_INTEGRATION_LLM_BACKEND``).
      3. **Structured heuristic** — deterministic PDF reflow + journal-style boundaries when
         the above is off or fails/timeouts.

    **Grading** (default ``openai`` when ``OPENAI_API_KEY`` is set, else ``huggingface``):
      **OpenAI** path uses ``OPENAI_MULTIMODAL_GRADING_MODEL`` / ``MULTIMODAL_INTEGRATION_OPENAI_GRADING_MODEL``.
      **Hugging Face** uses ``Llama-4-Maverick-17B-128E-Instruct:fp8`` (override with
      ``MULTIMODAL_INTEGRATION_HF_MODEL_ID``). Set ``MULTIMODAL_INTEGRATION_LLM_BACKEND=ollama``
      to use ``MULTIMODAL_INTEGRATION_OLLAMA_MODEL`` against ``ollama list`` instead.

    **RAG embeddings**: default **SentenceTransformers** ``all-MiniLM-L6-v2`` (override with
    ``MULTIMODAL_INTEGRATION_SENTENCE_TRANSFORMERS_MODEL``). Set
    ``MULTIMODAL_INTEGRATION_RAG_EMBEDDING_BACKEND=ollama`` to smoke Ollama
    ``MULTIMODAL_INTEGRATION_EMBEDDINGS_MODEL`` instead. Fallback order:
    ``MULTIMODAL_INTEGRATION_RAG_EMBED_ORDER`` (default ``ollama_only``).

    Default **2** multimodal samples per chunk (``MULTIMODAL_LOCAL_TEST_GRADING_SAMPLES``).
    ``MULTIMODAL_OLLAMA_QA_SEGMENT`` is **off** for deterministic ipynb runs.

    All overrides are **unconditional** so that stale values from parent
    ``.env`` files or host env vars never leak through.
    """
    # --- Chunking: skip Ollama QA segmentation for faster, deterministic ipynb tests ---
    os.environ["MULTIMODAL_OLLAMA_QA_SEGMENT"] = "off"

    # --- Memory: one model loaded at a time (no swap on 24 GB) ---
    # keep_alive="5m" lets consecutive samples reuse the loaded model across reps.
    os.environ["OLLAMA_MAX_LOADED_MODELS"] = "1"
    cfg.OLLAMA_KEEP_ALIVE = "5m"

    cfg.OLLAMA_BASE_URL = cfg.OLLAMA_BASE_URL or "http://localhost:11434"
    cfg.INTERNAL_OLLAMA_URL = cfg.INTERNAL_OLLAMA_URL or cfg.OLLAMA_BASE_URL

    # --- Chat / structure LLM: OpenAI (default when OPENAI_API_KEY) | Hugging Face | Ollama ---
    raw_backend = os.getenv("MULTIMODAL_INTEGRATION_LLM_BACKEND", "").strip().lower()
    if not raw_backend:
        raw_backend = (
            "openai" if os.getenv("OPENAI_API_KEY", "").strip() else "huggingface"
        )
    normalized = {"hf": "huggingface"}.get(raw_backend, raw_backend)
    cfg.MULTIMODAL_LLM_BACKEND = normalized

    if multimodal_llm_backend_uses_huggingface(cfg):
        cfg.HUGGINGFACE_GRADING_MODEL_ID = (
            os.getenv(
                "MULTIMODAL_INTEGRATION_HF_MODEL_ID",
                "Llama-4-Maverick-17B-128E-Instruct:fp8",
            ).strip()
        )
        tok = (
            os.getenv("HUGGINGFACE_HUB_TOKEN", "").strip()
            or os.getenv("HF_TOKEN", "").strip()
        )
        if tok:
            cfg.HUGGINGFACE_HUB_TOKEN = tok
            cfg.HF_TOKEN = tok
        cfg.OLLAMA_MODEL = (
            os.getenv("MULTIMODAL_INTEGRATION_OLLAMA_AUX_MODEL", "llama3.2:3b").strip()
            or "llama3.2:3b"
        )
        cfg.OPENAI_API_KEY = ""
    elif multimodal_llm_backend_uses_openai(cfg):
        cfg.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
        gmod = os.getenv("MULTIMODAL_INTEGRATION_OPENAI_GRADING_MODEL", "").strip()
        if gmod:
            cfg.OPENAI_MULTIMODAL_GRADING_MODEL = gmod
        fr = os.getenv("MULTIMODAL_INTEGRATION_OPENAI_FRONTLOAD", "").strip().lower()
        cfg.MULTIMODAL_OPENAI_TRIO_RAG_FRONTLOAD = fr or "auto"
        cfg.OLLAMA_MODEL = (
            os.getenv("MULTIMODAL_INTEGRATION_OLLAMA_AUX_MODEL", "llama3.2:3b").strip()
            or "llama3.2:3b"
        )
    else:
        cfg.OLLAMA_MODEL = (
            os.getenv(
                "MULTIMODAL_INTEGRATION_OLLAMA_MODEL",
                "Llama-4-Maverick-17B-128E-Instruct:fp8",
            ).strip()
        )
        cfg.OPENAI_API_KEY = ""

    cfg.GRADING_MODEL_2 = ""
    cfg.GRADING_MODEL_3 = ""

    cfg.OLLAMA_CHAT_TIMEOUT_SEC = max(
        int(getattr(cfg, "OLLAMA_CHAT_TIMEOUT_SEC", 300)),
        600,
    )

    cfg.MULTIMODAL_SAMPLES_PER_MODEL = int(
        os.getenv("MULTIMODAL_LOCAL_TEST_GRADING_SAMPLES", "2") or 2
    )
    cfg.GRADING_SAMPLE_TEMPERATURE = 0.3

    # RAG vectors (not the chat LLM): SentenceTransformers by default; Ollama optional fallback.
    cfg.RAG_EMBEDDING_BACKEND = (
        os.getenv(
            "MULTIMODAL_INTEGRATION_RAG_EMBEDDING_BACKEND",
            "sentence_transformers",
        )
        .strip()
        .lower()
        or "sentence_transformers"
    )
    cfg.SENTENCE_TRANSFORMERS_MODEL = (
        os.getenv(
            "MULTIMODAL_INTEGRATION_SENTENCE_TRANSFORMERS_MODEL",
            "all-MiniLM-L6-v2",
        )
        .strip()
        or "all-MiniLM-L6-v2"
    )
    cfg.OLLAMA_EMBEDDINGS_MODEL = (
        os.getenv("MULTIMODAL_INTEGRATION_EMBEDDINGS_MODEL", "nomic-embed-text").strip()
        or "nomic-embed-text"
    )
    cfg.RAG_EMBED_ORDER = (
        os.getenv("MULTIMODAL_INTEGRATION_RAG_EMBED_ORDER", "ollama_only").strip().lower()
        or "ollama_only"
    )

    # Optional LLM trio labeling: off unless MULTIMODAL_INTEGRATION_LLM_TRIO_CHUNKING=1.
    cfg.MULTIMODAL_LLM_TRIO_CHUNKING = (
        os.getenv("MULTIMODAL_INTEGRATION_LLM_TRIO_CHUNKING", "").strip().lower()
        in ("1", "true", "yes")
    )
    if cfg.MULTIMODAL_LLM_TRIO_CHUNKING:
        trio_m = os.getenv("MULTIMODAL_INTEGRATION_TRIO_CHUNKING_MODEL", "").strip()
        if trio_m:
            cfg.MULTIMODAL_TRIO_CHUNKING_MODEL = trio_m
        elif multimodal_llm_backend_uses_huggingface(cfg):
            cfg.MULTIMODAL_TRIO_CHUNKING_MODEL = ""
        else:
            cfg.MULTIMODAL_TRIO_CHUNKING_MODEL = cfg.OLLAMA_MODEL


# ---------------------------------------------------------------------------
# Unit tests (fast, no LLM)
# ---------------------------------------------------------------------------

_MOCK_STEM = "test_multimodal_pipeline_mock"
_SAMPLE_PLAINTEXT = (
    "=== NOTEBOOK MARKDOWN (ipynb) ===\n"
    "# Part 1. Reflection on Data Ethics\n\n"
    "Data ethics is important because it ensures that the data we collect "
    "and use is handled responsibly.\n\n"
    "=== NOTEBOOK CODE (ipynb) ===\n"
    "import pandas as pd\ndf = pd.read_csv('data.csv')\nprint(df.head())\n"
)

_FREE_RESPONSE_RUBRIC = [
    {"name": "Conceptual Correctness", "max_points": 4, "criterion": "Conceptual Correctness",
     "max_score": 4, "description": "Accuracy and depth."},
    {"name": "Evidence & Justification", "max_points": 3, "criterion": "Evidence & Justification",
     "max_score": 3, "description": "Concrete examples and citations."},
    {"name": "Depth of Understanding", "max_points": 2, "criterion": "Depth of Understanding",
     "max_score": 2, "description": "Connections beyond surface-level."},
    {"name": "Clarity", "max_points": 1, "criterion": "Clarity",
     "max_score": 1, "description": "Clear and well-organized."},
]

_SCAFFOLDED_RUBRIC = [
    {"name": "Functional Correctness", "max_points": 4, "criterion": "Functional Correctness",
     "max_score": 4, "description": "Output correctness."},
    {"name": "Logical Implementation", "max_points": 3, "criterion": "Logical Implementation",
     "max_score": 3, "description": "Algorithm fit."},
    {"name": "Code Quality", "max_points": 2, "criterion": "Code Quality",
     "max_score": 2, "description": "Readability."},
    {"name": "Edge Case Awareness", "max_points": 1, "criterion": "Edge Case Awareness",
     "max_score": 1, "description": "Robustness."},
]

_EDA_RUBRIC_STUB = [
    {"name": "Problem Framing", "max_points": 3, "criterion": "Problem Framing",
     "max_score": 3, "description": "Scope."},
]


def _make_sample_json(norm: float, *, confidence_note: str = "") -> str:
    criteria = [
        {"name": "Functional Correctness", "score": round(norm * 4), "max_points": 4},
        {"name": "Logical Implementation", "score": round(norm * 3), "max_points": 3},
        {"name": "Code Quality", "score": round(norm * 2), "max_points": 2},
        {"name": "Edge Case Awareness", "score": round(norm * 1), "max_points": 1},
    ]
    total = sum(c["score"] for c in criteria)
    max_total = sum(c["max_points"] for c in criteria)
    return json.dumps({
        "rubric_type": "programming_scaffolded",
        "criterion_scores": criteria,
        "criterion_justifications": [
            f"Score {c['score']}/{c['max_points']} — evidence for {c['name']}."
            for c in criteria
        ],
        "total_score": total,
        "normalized_score": round(total / max_total, 4) if max_total else 0,
        "confidence_note": confidence_note or "Graded from student evidence.",
        "review_flag": False,
    })


class AnswerKeyResolveTests(unittest.TestCase):
    def test_fuzzy_match_student_prefix_vs_answer_key_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "Week 1 PSet Part 1 [Answer_Key].txt").write_text(
                "expected solution body", encoding="utf-8"
            )
            text, fn = resolve_answer_key_plaintext(
                "[Student 1] Week 1 PSet Part 1", root
            )
            self.assertEqual(text, "expected solution body")
            self.assertTrue(fn.endswith(".txt"))

    def test_exact_stem_preferred(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            stem = "My Assignment"
            (root / f"{stem}.md").write_text("exact", encoding="utf-8")
            (root / "My Assignment Alt.txt").write_text("other", encoding="utf-8")
            text, fn = resolve_answer_key_plaintext(stem, root)
            self.assertEqual(text, "exact")
            self.assertTrue(fn.endswith(".md"))


class MultimodalRoutingTests(unittest.TestCase):
    def test_deterministic_routing_programming(self) -> None:
        ch = GradingChunk(
            chunk_id="c1", assignment_id="a1", student_id="s1", question_id="q1",
            modality=Modality.CODE, task_type=TaskType.SCAFFOLDED_CODING,
            extracted_text="print(1)",
        )
        route_rubric(ch, rubric_rows_by_type={})
        self.assertEqual(ch.rubric_type, RubricType.PROGRAMMING_SCAFFOLDED)

    def test_ipynb_chunk_free_response_task_never_gets_free_response_rubric(self) -> None:
        rows = {
            RubricType.PROGRAMMING_SCAFFOLDED: _SCAFFOLDED_RUBRIC,
            RubricType.EDA_VISUALIZATION: _EDA_RUBRIC_STUB,
            RubricType.FREE_RESPONSE: _FREE_RESPONSE_RUBRIC,
        }
        ch = GradingChunk(
            chunk_id="c1",
            assignment_id="a1",
            student_id="s1",
            question_id="q1",
            modality=Modality.NOTEBOOK,
            task_type=TaskType.FREE_RESPONSE_SHORT,
            extracted_text="# 1.1\nimport csv\n",
            evidence={"chunker": "notebook_cell_order"},
        )
        route_rubric(ch, rubric_rows_by_type=rows)
        self.assertIn(
            ch.rubric_type,
            (RubricType.PROGRAMMING_SCAFFOLDED, RubricType.EDA_VISUALIZATION),
        )
        self.assertNotEqual(ch.rubric_type, RubricType.FREE_RESPONSE)

    def test_ipynb_chunk_unknown_task_defaults_to_scaffolded_or_eda(self) -> None:
        rows = {
            RubricType.PROGRAMMING_SCAFFOLDED: _SCAFFOLDED_RUBRIC,
            RubricType.EDA_VISUALIZATION: _EDA_RUBRIC_STUB,
        }
        ch = GradingChunk(
            chunk_id="c1",
            assignment_id="a1",
            student_id="s1",
            question_id="q1",
            modality=Modality.NOTEBOOK,
            task_type=TaskType.UNKNOWN,
            extracted_text="import csv\n",
            evidence={"chunker": "notebook_cell_order"},
        )
        route_rubric(ch, rubric_rows_by_type=rows)
        self.assertEqual(ch.rubric_type, RubricType.PROGRAMMING_SCAFFOLDED)

    def test_ipynb_chunk_with_plt_routes_to_open_ended_eda_rubric(self) -> None:
        rows = {
            RubricType.PROGRAMMING_SCAFFOLDED: _SCAFFOLDED_RUBRIC,
            RubricType.EDA_VISUALIZATION: _EDA_RUBRIC_STUB,
        }
        ch = GradingChunk(
            chunk_id="c1",
            assignment_id="a1",
            student_id="s1",
            question_id="q1",
            modality=Modality.NOTEBOOK,
            task_type=TaskType.UNKNOWN,
            extracted_text="import matplotlib.pyplot as plt\nplt.scatter(x, y)\n",
            evidence={"chunker": "notebook_cell_order"},
        )
        route_rubric(ch, rubric_rows_by_type=rows)
        self.assertEqual(ch.rubric_type, RubricType.EDA_VISUALIZATION)

    def test_classifier_free_response_overridden_for_trio_frontload_chunk(self) -> None:
        rows = {
            RubricType.PROGRAMMING_SCAFFOLDED: _SCAFFOLDED_RUBRIC,
            RubricType.EDA_VISUALIZATION: _EDA_RUBRIC_STUB,
            RubricType.FREE_RESPONSE: _FREE_RESPONSE_RUBRIC,
        }

        def bad_classifier(_ch: GradingChunk) -> RubricRouteResult:
            return RubricRouteResult(
                RubricType.FREE_RESPONSE, "classifier_says_prose", True
            )

        ch = GradingChunk(
            chunk_id="c1",
            assignment_id="a1",
            student_id="s1",
            question_id="q1",
            modality=Modality.UNKNOWN,
            task_type=TaskType.UNKNOWN,
            extracted_text="import csv\n",
            evidence={"_openai_trio_rag_frontload": True},
        )
        route_rubric(ch, classifier=bad_classifier, rubric_rows_by_type=rows)
        self.assertNotEqual(ch.rubric_type, RubricType.FREE_RESPONSE)

    def test_semantic_entropy_two_clusters(self) -> None:
        h = semantic_entropy_from_cluster_counts({"A": 1, "B": 1})
        self.assertGreater(h, 0.0)


class MultimodalPipelineRunTests(unittest.TestCase):
    """Core pipeline run with mock runner — no LLM, fast."""

    def _run_pipeline(self):
        env = build_envelope_from_plaintext(
            assignment_id=_MOCK_STEM, student_id="test_student",
            plaintext=_SAMPLE_PLAINTEXT,
            modality_hints={
                "modality": "notebook",
                "task_type": "scaffolded_coding",
                "answer_key_plaintext": "Sample reference: cite evidence from the notebook.",
            },
        )
        cfg = MultimodalGradingConfig(
            confidence_ai_auto_accept_min=0.5, confidence_ai_caution_min=0.25,
            score_spread_high=2.0,
        )
        runner = MockChunkModelRunner(responses=[
            _make_sample_json(0.80, confidence_note="Strong evidence."),
            _make_sample_json(0.75, confidence_note="References reading."),
        ])
        pipe = MultimodalGradingPipeline(
            cfg,
            runner,
            rubric_rows_by_type={
                RubricType.PROGRAMMING_SCAFFOLDED: _SCAFFOLDED_RUBRIC,
                RubricType.EDA_VISUALIZATION: _EDA_RUBRIC_STUB,
                RubricType.FREE_RESPONSE: _FREE_RESPONSE_RUBRIC,
            },
        )
        art = PipelineArtifactStore()
        result = pipe.run(env, artifacts=art)
        return result, art

    def test_full_run_mock_runner(self) -> None:
        result, art = self._run_pipeline()
        self.assertIsNotNone(result.assignment_normalized_score)
        self.assertTrue(result.chunk_results)
        for ch in result.chunk_results:
            self.assertIn("confidence_trace", ch.stage_artifacts)
        self.assertIn("pipeline_audit", result.stage_artifacts)

    def test_evidence_and_justification_in_result(self) -> None:
        result, _ = self._run_pipeline()
        for ch in result.chunk_results:
            aux = ch.auxiliary or {}
            self.assertIn("criterion_justifications", aux)
            self.assertIn("confidence_note", aux)

    def test_assignment_overall_score_is_mean_of_question_overall_scores(self) -> None:
        result, _ = self._run_pipeline()
        grading_dict = multimodal_assignment_to_grading_dict(
            result, rubric=_SCAFFOLDED_RUBRIC
        )
        qg = grading_dict.get("question_grades") or []
        self.assertTrue(qg)
        scores = [float((x.get("overall") or {}).get("score", 0)) for x in qg]
        expected = sum(scores) / len(scores)
        self.assertAlmostEqual(
            float(grading_dict["overall"]["score"]),
            expected,
            places=5,
        )
        for x in qg:
            self.assertNotIn(
                "normalized_chunk_estimate",
                x.get("overall") or {},
            )

    def test_chunk_prompt_includes_reference_answer_key(self) -> None:
        ch = GradingChunk(
            chunk_id="s1:a1:pair_1",
            assignment_id="a1",
            student_id="s1",
            question_id="pair_1",
            modality=Modality.NOTEBOOK,
            task_type=TaskType.FREE_RESPONSE_SHORT,
            extracted_text="Student says ethics matter.",
            rubric_type=RubricType.FREE_RESPONSE,
            rubric_rows=list(_FREE_RESPONSE_RUBRIC),
        )
        raw = build_chunk_grading_prompt(
            ch,
            task_description="Week 1 journal",
            answer_key_text="Sample: cite dataset limitations and fairness.",
            dataset_context_text="col1,col2\n1,2\n",
        )
        data = json.loads(raw)
        self.assertIn("reference_answer_key", data)
        self.assertIn("Sample:", data["reference_answer_key"])
        self.assertIn("Answer-key grounding", data["instructions"])
        self.assertIn("matched_dataset_preview", data)
        self.assertIn("col1", data["matched_dataset_preview"])

    def test_chunk_prompt_includes_matched_answer_key_when_present(self) -> None:
        ch = GradingChunk(
            chunk_id="s1:a1:pair_1",
            assignment_id="a1",
            student_id="s1",
            question_id="1.1",
            modality=Modality.NOTEBOOK,
            task_type=TaskType.FREE_RESPONSE_SHORT,
            extracted_text="Student says ethics matter.",
            rubric_type=RubricType.FREE_RESPONSE,
            rubric_rows=list(_FREE_RESPONSE_RUBRIC),
            evidence={
                "answer_key_unit": {
                    "snippet": "Reference: cite dataset limitations.",
                    "parsed": {"match_method": "test"},
                }
            },
        )
        raw = build_chunk_grading_prompt(
            ch,
            task_description="Week 1",
            answer_key_text="Global key text.",
        )
        data = json.loads(raw)
        self.assertIn("matched_answer_key_for_question", data)
        self.assertIn("Reference:", data["matched_answer_key_for_question"])


# ---------------------------------------------------------------------------
# Chunking accuracy tests (no LLM — verifies notebook cell-order chunker)
# ---------------------------------------------------------------------------

@unittest.skipUnless(
    ASSIGNMENTS_DIR.is_dir(),
    "Requires assignments_to_grade/ at the repository root.",
)
class NotebookChunkingAccuracyTests(unittest.TestCase):
    """Verify that the notebook-aware chunker accurately splits ipynb
    assignments into question/answer pairs by preserving cell order.

    No LLM calls — tests the deterministic regex-based chunker only.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.groups = _assignment_groups()
        if not cls.groups:
            raise unittest.SkipTest(f"No files in {ASSIGNMENTS_DIR}.")

    def _get_ipynb_paths(self) -> list[tuple[str, Path]]:
        out: list[tuple[str, Path]] = []
        for stem, paths in self.groups.items():
            for p in paths:
                if p.suffix.lower() == ".ipynb":
                    out.append((stem, p))
        return out

    def test_notebook_chunker_produces_chunks(self) -> None:
        """Each ipynb must produce at least one chunk with non-empty text."""
        from app.grading.multimodal.notebook_chunker import build_notebook_qa_chunks

        ipynb_items = self._get_ipynb_paths()
        if not ipynb_items:
            self.skipTest("No .ipynb files in assignments_to_grade/.")

        for stem, path in ipynb_items:
            with self.subTest(assignment=stem):
                raw = path.read_bytes()
                chunks = build_notebook_qa_chunks(
                    raw, assignment_id=stem, student_id="chunk_test",
                )
                self.assertGreater(
                    len(chunks), 0,
                    f"Notebook chunker produced 0 chunks for {stem!r}",
                )
                for ch in chunks:
                    self.assertTrue(
                        ch.extracted_text.strip(),
                        f"Empty extracted_text in chunk {ch.chunk_id!r}",
                    )

    def test_every_chunk_has_student_content(self) -> None:
        """Each chunk should contain student work, not just a question prompt."""
        from app.grading.multimodal.notebook_chunker import build_notebook_qa_chunks

        ipynb_items = self._get_ipynb_paths()
        if not ipynb_items:
            self.skipTest("No .ipynb files in assignments_to_grade/.")

        for stem, path in ipynb_items:
            with self.subTest(assignment=stem):
                raw = path.read_bytes()
                chunks = build_notebook_qa_chunks(
                    raw, assignment_id=stem, student_id="chunk_test",
                )
                for ch in chunks:
                    ev = ch.evidence or {}
                    resp_preview = ev.get("response_preview", "")
                    self.assertTrue(
                        resp_preview.strip(),
                        f"[{stem}] Chunk {ch.question_id!r} has a question "
                        f"but no student response content.",
                    )

    def test_structured_notebook_question_ids(self) -> None:
        """Notebooks with ### Question X.Y headers produce numeric question IDs."""
        from app.grading.multimodal.notebook_chunker import build_notebook_qa_chunks

        for stem, path in self._get_ipynb_paths():
            with self.subTest(assignment=stem):
                raw = path.read_bytes()
                nb = json.loads(raw)
                md_sources = [
                    "".join(c.get("source", []))
                    if isinstance(c.get("source"), list)
                    else str(c.get("source", ""))
                    for c in nb.get("cells", [])
                    if c.get("cell_type") == "markdown"
                ]
                has_question_headers = any(
                    re.search(
                        r"#{1,4}\s*(?:Question|Problem)\s+\d", s, re.IGNORECASE
                    )
                    for s in md_sources
                )
                if not has_question_headers:
                    continue

                chunks = build_notebook_qa_chunks(
                    raw, assignment_id=stem, student_id="chunk_test",
                )
                numeric_ids = [
                    ch.question_id
                    for ch in chunks
                    if re.match(r"\d+\.", ch.question_id)
                ]
                self.assertGreater(
                    len(numeric_ids), 0,
                    f"[{stem}] Has ### Question headers but chunker "
                    f"produced no numeric question IDs. Got: "
                    f"{[ch.question_id for ch in chunks]}",
                )

    def test_pipeline_uses_notebook_chunker_for_ipynb(self) -> None:
        """When ipynb bytes are in the envelope, the pipeline uses notebook_cell_order."""
        from app.grading.multimodal.rag_embeddings import build_multimodal_grading_chunks

        ipynb_items = self._get_ipynb_paths()
        if not ipynb_items:
            self.skipTest("No .ipynb files in assignments_to_grade/.")

        stem, path = ipynb_items[0]
        raw = path.read_bytes()
        envelope = ingest_raw_submission(
            assignment_id=stem, student_id="chunk_test",
            artifacts={"ipynb": raw},
            extracted_plaintext="(unused for notebook chunker)",
            modality_hints={"modality_subtype": "notebook"},
        )
        cfg = Config()
        chunks, mode = build_multimodal_grading_chunks(envelope, cfg)
        self.assertEqual(mode, "notebook_cell_order")
        self.assertGreater(len(chunks), 0)
        _log.warning(
            "Pipeline notebook chunker: %d chunks for %s (mode=%s)",
            len(chunks), stem, mode,
        )


class AnswerKeyChunkEnrichTests(unittest.TestCase):
    """Per-question answer key snippets + embeddings on :class:`GradingChunk`."""

    def test_split_answer_key_on_headings(self) -> None:
        ak = "## 1.1 Alpha\nBody one.\n\n## 1.2 Beta\nBody two."
        sec = split_answer_key_sections(ak)
        self.assertGreaterEqual(len(sec), 2)
        heads = [h for h, _ in sec]
        self.assertTrue(any("1.1" in h for h in heads))

    def test_enrich_adds_answer_key_unit(self) -> None:
        from app.config import Config

        cfg = Config()
        ch = GradingChunk(
            chunk_id="s:a:1.1",
            assignment_id="a",
            student_id="s",
            question_id="1.1",
            modality=Modality.NOTEBOOK,
            task_type=TaskType.FREE_RESPONSE_SHORT,
            extracted_text="Student response here.",
            evidence={"question_text": "### Question 1.1\nDo X."},
        )
        ak = (
            "## 1.1 Alpha\nReference solution for 1.1 with code:\n```python\nx=1\n```\n\n"
            "## 1.2 Beta\nOther."
        )
        enrich_chunks_with_per_question_answer_key([ch], ak, cfg)
        unit = (ch.evidence or {}).get("answer_key_unit")
        self.assertIsInstance(unit, dict)
        self.assertIn("snippet", unit)
        self.assertIn("Alpha", unit["snippet"])
        rag = unit.get("answer_key_rag")
        self.assertIsInstance(rag, dict)
        self.assertGreater(rag.get("embedding_dimension", 0), 0)


class NotebookChunkerInstructorScaffoldTests(unittest.TestCase):
    """Student code after END OF INSTRUCTOR must land in ``response_parts`` (evidence)."""

    def test_scaffold_code_cell_includes_student_tail(self) -> None:
        nb = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {},
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["### Question 1.1\n", "\n", "Compute the answer.\n"],
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": [
                        "# DO NOT MODIFY\n",
                        "template = 1\n",
                        "# END OF INSTRUCTOR CODE\n",
                        "student_answer = 42\n",
                        "print(student_answer)\n",
                    ],
                },
            ],
        }
        raw = json.dumps(nb).encode("utf-8")
        chunks = build_notebook_qa_chunks(
            raw, assignment_id="hw1", student_id="s1",
        )
        self.assertTrue(chunks, "expected at least one chunk")
        joined = "\n".join(c.extracted_text for c in chunks)
        self.assertIn("student_answer", joined)


# ---------------------------------------------------------------------------
# Integration: grade ALL local assignments with real LLMs
# ---------------------------------------------------------------------------

@unittest.skipUnless(
    ASSIGNMENTS_DIR.is_dir() and RUBRIC_DIR.is_dir(),
    "Requires assignments_to_grade/ and rubric/ at the repository root.",
)
class LocalAssignmentGradingTests(unittest.TestCase):
    """Grade every assignment in ``assignments_to_grade/`` using real LLM calls.

    If this class is skipped, run ``pytest -rs tests/test_multimodal_pipeline.py`` to print
    full skip reasons. Codes include ``NO_OLLAMA``, ``NO_RAG_EMBED``, ``NO_HF_TOKEN``,
    ``NO_HF_DEPS``, ``HF_HUB_MODEL_CHECK_FAILED``, ``GRADING_MODEL_NOT_IN_OLLAMA`` (Ollama
    mode only), ``CHAT_SMOKE_FAIL``.

    **Default:** ``MULTIMODAL_INTEGRATION_LLM_BACKEND=huggingface`` — grading uses
    ``transformers`` + ``Llama-4-Maverick-17B-128E-Instruct:fp8`` (HF repo); **RAG** uses
    **SentenceTransformers** RAG (or Ollama if ``MULTIMODAL_INTEGRATION_RAG_EMBEDDING_BACKEND=ollama``).
    Set ``MULTIMODAL_INTEGRATION_LLM_BACKEND=ollama`` for an
    all-Ollama grader (``MULTIMODAL_INTEGRATION_OLLAMA_MODEL``).

    Use ``pytest --log-cli-level=WARNING`` to see phased ``[integration]`` logs before each
    gate or skip.

    For each assignment the test runs **sequentially**:

    1. Extract plaintext; resolve modality; **require** a matched answer key.
    2. Build **once** per stem: notebook multimodal units + per-unit embeddings, saved to
       ``RAG_embedding/<stem>_multimodal_units.json`` (refreshed when
       ``MULTIMODAL_REFRESH_UNIT_CACHE=1``). Assignment-level embedding JSON is still written.
    3. Run the multimodal pipeline with ``multimodal_chunk_cache_path`` so chunking /
       unit vectorization is **not** repeated inside ``pipeline.run``.
    4. Validate JSON (including ``_agentic_workflow``), write ``grading_output/<stem>_grade_output.json``.

    Optional: set ``MULTIMODAL_INTEGRATION_WRITE_LEGACY_CHUNKS=1`` to also emit legacy
    ``*_chunks.json`` from ``build_submission_chunks`` (slower).
    """

    @classmethod
    def setUpClass(cls) -> None:
        _integration_log(
            "========== LocalAssignmentGradingTests.setUpClass START =========="
        )
        _integration_log(
            "env MULTIMODAL_INTEGRATION_LLM_BACKEND=%r (unset → openai if OPENAI_API_KEY else huggingface)",
            os.getenv("MULTIMODAL_INTEGRATION_LLM_BACKEND", ""),
        )

        if os.getenv("SKIP_LOCAL_LLM_TESTS", "").strip().lower() in ("1", "true", "yes"):
            _integration_log("SKIP_LOCAL_LLM_TESTS is set → skipping integration class")
            raise unittest.SkipTest("SKIP_LOCAL_LLM_TESTS is set.")

        _integration_log("phase=rubric: load rubric/default.json …")
        cls.rubric_raw = _load_rubric_json()
        if cls.rubric_raw is None:
            _integration_log("SKIP: rubric/default.json missing or invalid JSON")
            raise unittest.SkipTest("rubric/default.json not found or invalid.")
        cls.rubric_by_type = _build_rubric_rows_by_type(cls.rubric_raw)
        cls.rubric_flat = _flat_rubric_from_by_type(cls.rubric_by_type)
        cls.rubric_flat_sectioned = _flatten_sections_rubric(cls.rubric_raw)
        if not cls.rubric_flat:
            _integration_log("SKIP: no criteria parsed from rubric")
            raise unittest.SkipTest("No criteria parsed from rubric.")

        _integration_log("phase=fixtures: scan %s …", ASSIGNMENTS_DIR)
        cls.groups = _assignment_groups()
        if not cls.groups:
            _integration_log("SKIP: no assignment files under %s", ASSIGNMENTS_DIR)
            raise unittest.SkipTest(f"No files in {ASSIGNMENTS_DIR}.")

        _integration_log("phase=config: build Config + _configure_for_integration_test …")
        cls.cfg = Config()
        _configure_for_integration_test(cls.cfg)
        _integration_log(
            "phase=config done: MULTIMODAL_LLM_BACKEND=%r chat_grading=%r OLLAMA_BASE=%r "
            "RAG_EMBEDDING_BACKEND=%r SENTENCE_TRANSFORMERS_MODEL=%r OLLAMA_EMBEDDINGS_MODEL=%r",
            getattr(cls.cfg, "MULTIMODAL_LLM_BACKEND", ""),
            huggingface_grading_model_id(cls.cfg)
            if multimodal_llm_backend_uses_huggingface(cls.cfg)
            else (cls.cfg.OLLAMA_MODEL or ""),
            (cls.cfg.INTERNAL_OLLAMA_URL or cls.cfg.OLLAMA_BASE_URL or "").strip(),
            getattr(cls.cfg, "RAG_EMBEDDING_BACKEND", ""),
            getattr(cls.cfg, "SENTENCE_TRANSFORMERS_MODEL", ""),
            getattr(cls.cfg, "OLLAMA_EMBEDDINGS_MODEL", "") or "nomic-embed-text",
        )

        _integration_log(
            "phase=rag_embeddings: smoke (backend=%r) …",
            getattr(cls.cfg, "RAG_EMBEDDING_BACKEND", ""),
        )
        ok_emb, emb_detail = _rag_embedding_smoke(cls.cfg)
        if not ok_emb:
            msg = (
                "[integration] NO_RAG_EMBED: smoke failed (RAG_EMBEDDING_BACKEND="
                f"{getattr(cls.cfg, 'RAG_EMBEDDING_BACKEND', '')!r}): {emb_detail}. "
                "Install sentence-transformers for ST, set RAG_EMBEDDING_BACKEND=openai with "
                "OPENAI_API_KEY, or set RAG_EMBEDDING_BACKEND=ollama and ensure Ollama /api/embed "
                "works for OLLAMA_EMBEDDINGS_MODEL."
            )
            _integration_log("SKIP: %s", msg)
            raise unittest.SkipTest(msg)
        _integration_log("phase=rag_embeddings: OK")

        _rag_be = (getattr(cls.cfg, "RAG_EMBEDDING_BACKEND", "") or "").strip().lower()
        skip_ollama_ping = (
            multimodal_llm_backend_uses_openai(cls.cfg) and _rag_be != "ollama"
        ) or (
            multimodal_llm_backend_uses_huggingface(cls.cfg)
            and ok_emb
            and _rag_be in ("sentence_transformers", "openai")
        )
        if skip_ollama_ping:
            _integration_log(
                "phase=ollama_daemon: skipped (HF grading + RAG backend=%r)", _rag_be
            )
        else:
            _integration_log("phase=ollama_daemon: GET /api/tags (reachability) …")
            if not _ollama_reachable(cls.cfg):
                msg = (
                    "[integration] NO_OLLAMA: daemon not HTTP 200 at "
                    f"{(cls.cfg.INTERNAL_OLLAMA_URL or cls.cfg.OLLAMA_BASE_URL or '').strip()!r}. "
                    "Start `ollama serve` or fix OLLAMA_BASE_URL / INTERNAL_OLLAMA_URL."
                )
                _integration_log("SKIP: %s", msg)
                raise unittest.SkipTest(msg)

        base = (cls.cfg.INTERNAL_OLLAMA_URL or cls.cfg.OLLAMA_BASE_URL or "").strip().rstrip(
            "/"
        )

        if multimodal_llm_backend_uses_huggingface(cls.cfg):
            _integration_log(
                "phase=hf_grading: skipping Ollama /api/show grader check (chat via transformers)"
            )
            ok_hf, hf_detail = _hf_integration_preflight(cls.cfg)
            if not ok_hf:
                msg = "[integration] " + hf_detail
                _integration_log("SKIP: %s", hf_detail)
                raise unittest.SkipTest(msg)
        elif multimodal_llm_backend_uses_openai(cls.cfg):
            _integration_log("phase=openai_grading: OpenAI API (skip HF + Ollama grader checks)")
            if not (cls.cfg.OPENAI_API_KEY or "").strip():
                msg = (
                    "[integration] MULTIMODAL_LLM_BACKEND=openai requires OPENAI_API_KEY "
                    "in the environment (set it or use MULTIMODAL_INTEGRATION_LLM_BACKEND=huggingface)."
                )
                _integration_log("SKIP: %s", msg)
                raise unittest.SkipTest(msg)
        else:
            _integration_log(
                "phase=ollama_grader: resolve MULTIMODAL_INTEGRATION_OLLAMA_MODEL via /api/show …"
            )
            preferred = (cls.cfg.OLLAMA_MODEL or "").strip()
            try:
                tags_resp = requests.get(f"{base}/api/tags", timeout=8).json()
                local_models = {m["name"] for m in tags_resp.get("models", [])}
            except (requests.RequestException, KeyError, ValueError, TypeError) as e:
                msg = f"Could not read Ollama /api/tags for grading model check: {e}"
                _integration_log("SKIP: %s", msg)
                raise unittest.SkipTest(msg) from e

            resolved, resolve_note = _resolve_ollama_grading_model_name(
                base, preferred, local_models
            )
            if not resolved:
                msg = "[integration] " + resolve_note
                _integration_log("SKIP: %s", resolve_note)
                raise unittest.SkipTest(msg)
            if resolve_note:
                _integration_log("%s", resolve_note)
            cls.cfg.OLLAMA_MODEL = resolved
            _integration_log("phase=ollama_grader: resolved OLLAMA_MODEL=%r", resolved)

        if os.getenv("MULTIMODAL_INTEGRATION_CHAT_SMOKE", "").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            if multimodal_llm_backend_uses_openai(cls.cfg):
                _integration_log(
                    "phase=ollama_chat_smoke: skipped (OpenAI multimodal grader; no Ollama smoke)"
                )
            else:
                if multimodal_llm_backend_uses_huggingface(cls.cfg):
                    _integration_log(
                        "phase=ollama_chat_smoke: MULTIMODAL_INTEGRATION_CHAT_SMOKE on — "
                        "smoking auxiliary OLLAMA_MODEL=%r (not the HF grader)",
                        (cls.cfg.OLLAMA_MODEL or "").strip(),
                    )
                ok_chat, chat_detail = _ollama_chat_smoke(cls.cfg)
                if not ok_chat:
                    msg = (
                        f"[integration] CHAT_SMOKE_FAIL: /api/chat on OLLAMA_MODEL failed: {chat_detail}. "
                        "Unset MULTIMODAL_INTEGRATION_CHAT_SMOKE to skip this check."
                    )
                    _integration_log("SKIP: %s", msg)
                    raise unittest.SkipTest(msg)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        RAG_DIR.mkdir(parents=True, exist_ok=True)

        _integration_log("phase=grading_clients: build_multimodal_grading_clients …")
        clients = build_multimodal_grading_clients(cls.cfg)
        cls._model_labels = [label for _, label in clients]
        if len(clients) != 1:
            msg = (
                f"Expected 1 grading client for single-model multimodal, got {len(clients)}: "
                f"{cls._model_labels}. Clear GRADING_MODEL_2/GRADING_MODEL_3."
            )
            _integration_log("SKIP: %s", msg)
            raise unittest.SkipTest(msg)
        _integration_log(
            "phase=grading_clients: OK — %d model(s): %s, multimodal_samples_per_model=%d",
            len(clients),
            ", ".join(cls._model_labels),
            cls.cfg.MULTIMODAL_SAMPLES_PER_MODEL,
        )
        _integration_log(
            "========== LocalAssignmentGradingTests.setUpClass END (will run tests) =========="
        )

    def test_grade_all_assignments(self) -> None:
        graded_stems: list[str] = []

        for stem, paths in sorted(self.groups.items()):
            with self.subTest(assignment=stem):
                _log.warning(
                    "--- %r (%d file(s)): save RAG → grade → save output ---",
                    stem,
                    len(paths),
                )

                # -- 1. Artifacts & plaintext --
                artifacts = _build_artifacts(paths)
                plain = submission_text_from_artifacts(artifacts)
                self.assertGreater(len(plain.strip()), 0, f"Empty text for {stem!r}")

                assignment_ns = SimpleNamespace(
                    modality=None, rubric=self.rubric_flat_sectioned,
                    title=stem, description=f"Local fixture: {stem}",
                )
                modality_profile = resolve_modality_profile(assignment_ns, artifacts, plain)

                answer_key_full, answer_key_match = resolve_answer_key_plaintext(
                    stem, ANSWER_KEY_DIR
                )
                ak_prompt_cap = int(
                    os.getenv("MULTIMODAL_INTEGRATION_ANSWER_KEY_CHARS", "18000") or 18000
                )
                ak_prompt_cap = max(4000, min(ak_prompt_cap, 100_000))
                answer_key_text = (
                    answer_key_full[:ak_prompt_cap]
                    if len(answer_key_full) > ak_prompt_cap
                    else answer_key_full
                )
                if len(answer_key_full) > ak_prompt_cap:
                    _log.warning(
                        "  [%s] Answer key truncated for grading prompt (%d → %d chars)",
                        stem,
                        len(answer_key_full),
                        ak_prompt_cap,
                    )
                if not str(answer_key_text or "").strip():
                    self.skipTest(
                        f"No answer key matched for {stem!r} under {ANSWER_KEY_DIR}; "
                        "integration requires resolve_answer_key_plaintext to succeed."
                    )
                _log.warning(
                    "  [%s] Answer key (%d chars) matched=%r",
                    stem,
                    len(answer_key_text),
                    answer_key_match,
                )

                max_units = int(os.getenv("MULTIMODAL_LOCAL_TEST_MAX_GRADING_UNITS", "6") or 6)
                mm_units_path = RAG_DIR / f"{stem}_multimodal_units.json"
                refresh_units = os.getenv("MULTIMODAL_REFRESH_UNIT_CACHE", "").lower() in (
                    "1",
                    "true",
                    "yes",
                )
                env_chunk = ingest_raw_submission(
                    assignment_id=stem,
                    student_id="local_test",
                    artifacts=artifacts,
                    extracted_plaintext=plain,
                    modality_hints={
                        "modality_subtype": str(
                            modality_profile.get("modality_subtype") or ""
                        ),
                        "max_grading_units": max_units,
                        "answer_key_plaintext": answer_key_text,
                        "answer_key_matched_file": answer_key_match,
                    },
                )
                if refresh_units or not mm_units_path.is_file():
                    mm_chunks, mm_mode = build_multimodal_grading_chunks(
                        env_chunk, self.cfg
                    )
                    self.assertTrue(mm_chunks, f"No multimodal chunks for {stem!r} ({mm_mode})")
                    enrich_chunks_with_rag_embeddings(mm_chunks, self.cfg)
                    save_grading_chunks_cache(mm_units_path, mm_chunks)
                    _log.warning(
                        "  [%s] Built multimodal unit cache (%s, %d units, mode=%s)",
                        stem,
                        mm_units_path.name,
                        len(mm_chunks),
                        mm_mode,
                    )
                else:
                    _log.warning(
                        "  [%s] Reusing multimodal unit cache %s",
                        stem,
                        mm_units_path.name,
                    )

                # -- 2. Optional legacy plaintext chunks JSON (off by default for speed) --
                if os.getenv("MULTIMODAL_INTEGRATION_WRITE_LEGACY_CHUNKS", "").lower() in (
                    "1",
                    "true",
                    "yes",
                ):
                    chunks = build_submission_chunks(
                        plain,
                        assignment_title=stem,
                        modality_subtype=str(
                            modality_profile.get("modality_subtype") or ""
                        ),
                        max_chunk_chars=None,
                    )
                    self.assertGreater(len(chunks), 0, f"No legacy chunks for {stem!r}")
                    chunks_path = RAG_DIR / f"{stem}_chunks.json"
                    write_chunks_json(
                        chunks_path,
                        chunks=chunks,
                        assignment_title=stem,
                        source_file=",".join(p.name for p in paths),
                        profile=modality_profile,
                    )
                    self.assertTrue(chunks_path.is_file(), f"Missing {chunks_path}")

                vec, vec_src = compute_submission_embedding(plain, self.cfg)
                emb_path = save_rag_embedding_bundle(
                    RAG_DIR,
                    assignment_stem=stem,
                    artifacts_keys=sorted(artifacts.keys()),
                    plaintext_chars=len(plain),
                    embedding=vec,
                    embedding_source=vec_src,
                    parsed_preview=plain[:8000],
                    extra={"paths": [p.name for p in paths]},
                )
                preview_path = RAG_DIR / f"{stem}_parsed_preview.txt"
                self.assertTrue(emb_path.is_file(), f"Missing {emb_path}")
                self.assertTrue(preview_path.is_file(), f"Missing {preview_path}")
                _log.warning(
                    "  [%s] Assignment embedding saved (%s, %s)",
                    stem,
                    emb_path.name,
                    preview_path.name,
                )

                # -- 3. Multimodal grading pipeline (LLM) — reads cached units + embeddings --
                task_desc = f"Local fixture: {stem}"
                llm_grading_instr = self.rubric_raw.get("llm_grading_instructions")
                if isinstance(llm_grading_instr, str) and llm_grading_instr.strip():
                    task_desc += "\n\n" + llm_grading_instr.strip()

                mm_cfg = MultimodalGradingConfig(require_answer_key=True)
                pipeline = create_multimodal_pipeline_from_app_config(
                    self.cfg,
                    multimodal_cfg=mm_cfg,
                    rubric_rows_by_type=self.rubric_by_type,
                    task_description=task_desc,
                )
                envelope = ingest_raw_submission(
                    assignment_id=stem, student_id="local_test",
                    artifacts=artifacts,
                    extracted_plaintext=plain,
                    modality_hints={
                        "modality_subtype": str(modality_profile.get("modality_subtype") or ""),
                        "max_grading_units": max_units,
                        "answer_key_plaintext": answer_key_text,
                        "answer_key_matched_file": answer_key_match,
                        "answer_key_dir": str(ANSWER_KEY_DIR),
                        "multimodal_chunk_cache_path": str(mm_units_path),
                    },
                )
                result = pipeline.run(envelope)
                self.assertTrue(result.chunk_results, f"No chunk results for {stem!r}")

                # -- 4. Build grading JSON, validate, persist to grading_output/ --
                grading_dict = multimodal_assignment_to_grading_dict(
                    result, rubric=self.rubric_flat,
                    modality_profile=modality_profile,
                )
                awf = grading_dict.get("_agentic_workflow")
                self.assertIsInstance(awf, list)
                self.assertGreaterEqual(len(awf), 3)
                flat_allowed = frozenset(
                    str(r["name"]).strip()
                    for r in self.rubric_flat
                    if isinstance(r, dict) and str(r.get("name") or "").strip()
                )
                validated = validate_grading_output(
                    grading_dict, allowed_criterion_names=flat_allowed
                )
                out_path = OUTPUT_DIR / f"{stem}_grade_output.json"
                out_path.write_text(
                    json.dumps(grading_dict, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8",
                )
                self.assertTrue(out_path.is_file(), f"Missing {out_path}")
                _log.warning("  [%s] Grading output saved: %s", stem, out_path.name)

                # -- 5. Assertions (after RAG + grade artifacts are on disk) --
                self.assertIn("score", validated["overall"])
                self.assertEqual(validated["overall"].get("max_score"), 1.0)
                self.assertIn("question_grades", validated)
                for row in validated.get("criteria") or []:
                    if isinstance(row, dict) and row.get("name"):
                        self.assertIn(
                            row["name"], flat_allowed,
                            f"[{stem}] assignment criteria must be rubric-backed: {row!r}",
                        )
                for idx, qg in enumerate(
                    validated.get("question_grades") or [], start=1
                ):
                    cid = str(qg.get("chunk_id") or "")
                    self.assertTrue(
                        cid.endswith(f":pair_{idx}"),
                        f"[{stem}] chunk_id should end with :pair_{idx}, got {cid!r}",
                    )
                    self.assertEqual(
                        (qg.get("overall") or {}).get("max_score"),
                        1.0,
                        f"[{stem}] question overall.max_score",
                    )
                for qg in validated.get("question_grades") or []:
                    for c in (qg.get("criteria") or []) if isinstance(qg, dict) else []:
                        if isinstance(c, dict) and c.get("name"):
                            self.assertIn(
                                c["name"], flat_allowed,
                                f"[{stem}] chunk criteria must be rubric-backed: {c!r}",
                            )

                # Check justifications + evidence are present and content-specific
                chunk_summaries: list[str] = []
                all_justifications: list[str] = []
                all_evidence: list[str] = []
                for qg in validated["question_grades"]:
                    summary = qg.get("overall", {}).get("summary", "")
                    chunk_summaries.append(summary)
                    self.assertNotIn(
                        "ai_confidence", qg,
                        f"[{stem}] question_grade should not have top-level ai_confidence",
                    )
                    for c in qg.get("criteria") or []:
                        j = c.get("justification", "")
                        if j:
                            all_justifications.append(j)
                        ev = c.get("evidence", "")
                        if ev:
                            all_evidence.append(ev)

                n_crit = len(validated.get("criteria") or [])
                _log.warning(
                    "  %s: score=%.3f, chunks=%d, criteria=%d, justifications=%d, evidence=%d",
                    stem, validated["overall"]["score"],
                    len(result.chunk_results), n_crit,
                    len(all_justifications), len(all_evidence),
                )

                degraded_llm = len(all_justifications) == 0 and len(all_evidence) == 0
                strict_quality = os.getenv(
                    "MULTIMODAL_INTEGRATION_STRICT_QUALITY", "0"
                ).strip().lower() in ("1", "true", "yes")
                if degraded_llm:
                    _log.warning(
                        "  [%s] Degraded run: no criterion justifications/evidence (Ollama "
                        "timeouts/500/invalid JSON). Set MULTIMODAL_INTEGRATION_STRICT_QUALITY=1 "
                        "to fail on empty LLM text when models are healthy.",
                        stem,
                    )
                if strict_quality and not degraded_llm:
                    if len(result.chunk_results) > 1:
                        self.assertGreater(
                            len(set(chunk_summaries)), 1,
                            f"[{stem}] All {len(chunk_summaries)} chunks have identical "
                            f"evidence summaries — LLM should produce unique per-chunk text.",
                        )
                    self.assertGreater(
                        len(all_justifications), 0,
                        f"[{stem}] No justification text in criterion rows. See {out_path}.",
                    )
                    if len(all_justifications) > 1:
                        unique_ratio = len(set(all_justifications)) / len(
                            all_justifications
                        )
                        self.assertGreater(
                            unique_ratio,
                            0.3,
                            f"[{stem}] Only {unique_ratio:.0%} of justifications are unique.",
                        )

                graded_stems.append(stem)

        self.assertEqual(
            len(graded_stems), len(self.groups),
            f"Graded {len(graded_stems)} of {len(self.groups)} assignments.",
        )


class ParseChunkGradeRubricAlignmentTests(unittest.TestCase):
    """Chunk JSON is aligned to routed rubric rows; placeholder keys are dropped."""

    def test_drop_placeholder_criterion_names(self) -> None:
        rubric_rows = [
            {"name": "Conceptual Correctness", "max_points": 4},
            {"name": "Clarity", "max_points": 1},
        ]
        raw = json.dumps(
            {
                "rubric_type": "free_response",
                "criterion_scores": [
                    {
                        "name": "criterion_1",
                        "score": 50,
                        "max_points": 100,
                        "evidence": "",
                        "reasoning": "",
                        "justification": "",
                    },
                    {
                        "name": "Conceptual Correctness",
                        "score": 3.5,
                        "max_points": 4,
                        "evidence": "e1",
                        "reasoning": "r1",
                        "justification": "j1",
                    },
                ],
                "criterion_justifications": ["", ""],
                "total_score": 3.5,
                "normalized_score": 0.7,
                "confidence_note": "",
                "review_flag": False,
            }
        )
        parsed, warns = parse_chunk_grade_json(raw, rubric_rows=rubric_rows)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        names = [c.name for c in parsed.criterion_scores]
        self.assertEqual(names, ["Conceptual Correctness", "Clarity"])
        self.assertEqual(parsed.criterion_scores[0].score, 3.5)
        self.assertAlmostEqual(parsed.criterion_scores[0].calibrated_credit, 0.90, places=5)
        self.assertTrue(any("dropped_unknown_criterion" in w for w in warns))


class MultimodalHuggingFaceRoutingTests(unittest.TestCase):
    """Routing for ``MULTIMODAL_LLM_BACKEND`` without loading torch/transformers."""

    def test_compute_submission_embedding_sentence_transformers(self) -> None:
        cfg = Config()
        cfg.RAG_EMBEDDING_BACKEND = "sentence_transformers"
        cfg.SENTENCE_TRANSFORMERS_MODEL = "all-MiniLM-L6-v2"
        cfg.OPENAI_API_KEY = ""
        vec, src = compute_submission_embedding(
            "Unit test snippet for SentenceTransformers RAG.", cfg
        )
        self.assertGreater(len(vec), 64)
        if not src.startswith("sentence_transformers:"):
            self.skipTest(
                "sentence-transformers not usable here (install backend deps); "
                f"got source={src!r}"
            )
        self.assertIn("all-MiniLM", src)

    def test_multimodal_openai_trio_rag_frontload_enabled_respects_off(self) -> None:
        from app.grading.multimodal.openai_trio_rag_frontload import (
            multimodal_openai_trio_rag_frontload_enabled,
        )

        cfg = Config()
        cfg.OPENAI_API_KEY = "sk-test"
        cfg.MULTIMODAL_OPENAI_TRIO_RAG_FRONTLOAD = "off"
        self.assertFalse(multimodal_openai_trio_rag_frontload_enabled(cfg))
        cfg.MULTIMODAL_OPENAI_TRIO_RAG_FRONTLOAD = "auto"
        self.assertTrue(multimodal_openai_trio_rag_frontload_enabled(cfg))
        cfg.MULTIMODAL_OPENAI_TRIO_RAG_FRONTLOAD = ""
        self.assertTrue(multimodal_openai_trio_rag_frontload_enabled(cfg))

    def test_compute_submission_embedding_openai_backend(self) -> None:
        from unittest.mock import patch

        cfg = Config()
        cfg.RAG_EMBEDDING_BACKEND = "openai"
        cfg.OPENAI_API_KEY = "sk-test"
        cfg.OPENAI_TRIO_RAG_EMBEDDING_MODEL = "text-embedding-3-small"

        def fake_openai(snippet: str, _cfg: Config):
            return [0.01] * 16, "openai:text-embedding-3-small"

        with patch(
            "app.grading.rag_embeddings._openai_embed_snippet",
            side_effect=fake_openai,
        ):
            vec, src = compute_submission_embedding("hello", cfg)
        self.assertEqual(len(vec), 16)
        self.assertTrue(src.startswith("openai:"))

    def test_build_multimodal_matches_build_grading_when_backend_ollama(self) -> None:
        from app.grading.llm_router import (
            build_grading_clients,
            build_multimodal_grading_clients,
        )

        cfg = Config()
        cfg.MULTIMODAL_LLM_BACKEND = "ollama"
        cfg.OLLAMA_BASE_URL = "http://127.0.0.1:11434"
        cfg.OLLAMA_MODEL = "llama3.2:3b"
        a = [lbl for _, lbl in build_grading_clients(cfg)]
        b = [lbl for _, lbl in build_multimodal_grading_clients(cfg)]
        self.assertEqual(a, b)

    def test_huggingface_grading_model_id_maps_llama_package_descriptor(self) -> None:
        from app.grading.llm_router import huggingface_grading_model_id

        cfg = Config()
        cfg.HUGGINGFACE_GRADING_MODEL_ID = "Llama-4-Maverick-17B-128E-Instruct:fp8"
        self.assertEqual(
            huggingface_grading_model_id(cfg),
            "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        )

    def test_multimodal_structure_trace_label_huggingface(self) -> None:
        cfg = Config()
        cfg.MULTIMODAL_LLM_BACKEND = "hf"
        cfg.HUGGINGFACE_GRADING_MODEL_ID = ""
        self.assertEqual(
            multimodal_structure_llm_trace_label(cfg),
            "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        )

    def test_multimodal_structure_trace_label_openai(self) -> None:
        cfg = Config()
        cfg.MULTIMODAL_LLM_BACKEND = "openai"
        cfg.OPENAI_MULTIMODAL_GRADING_MODEL = "gpt-5.4-nano"
        self.assertEqual(multimodal_structure_llm_trace_label(cfg), "gpt-5.4-nano")

    def test_build_multimodal_grading_clients_openai_backend(self) -> None:
        cfg = Config()
        cfg.MULTIMODAL_LLM_BACKEND = "openai"
        cfg.OPENAI_API_KEY = "sk-test"
        cfg.OPENAI_MULTIMODAL_GRADING_MODEL = "gpt-5.4-nano"
        clients = build_multimodal_grading_clients(cfg)
        self.assertEqual(len(clients), 1)
        self.assertEqual(clients[0][1], "openai:gpt-5.4-nano")

    def test_refine_trio_invokes_structure_client(self) -> None:
        from unittest.mock import MagicMock, patch

        from app.grading.multimodal.rag_embeddings import refine_chunks_trio_with_ollama

        mock_client = MagicMock()
        mock_client.chat_json.return_value = {
            "question": "Q",
            "student_response": "S",
            "instructor_context": "",
        }
        cfg = Config()
        cfg.MULTIMODAL_LLM_TRIO_CHUNKING = True
        ch = GradingChunk(
            chunk_id="t1:a1:pair_1",
            assignment_id="a1",
            student_id="t1",
            question_id="pair_1",
            modality=Modality.WRITTEN,
            task_type=TaskType.FREE_RESPONSE_SHORT,
            extracted_text="combined blob",
            evidence={"trio": {}},
        )
        with patch(
            "app.grading.multimodal.rag_embeddings._multimodal_structure_chat_client",
            return_value=(mock_client, "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"),
        ):
            refine_chunks_trio_with_ollama([ch], cfg)
        mock_client.chat_json.assert_called_once()
        trio = (ch.evidence or {}).get("trio") or {}
        self.assertEqual(trio.get("question"), "Q")
        self.assertEqual(trio.get("student_response"), "S")

    def test_openai_trio_rag_frontload_chunks_embed_and_cost_audit(self) -> None:
        from unittest.mock import MagicMock, patch

        from app.grading.multimodal.ingestion import ingest_raw_submission
        from app.grading.multimodal.openai_trio_rag_frontload import (
            run_openai_trio_rag_frontload,
        )

        cfg = Config()
        cfg.OPENAI_API_KEY = "sk-test"
        cfg.OPENAI_TRIO_RAG_CHAT_MODEL = "gpt-test"
        cfg.OPENAI_TRIO_RAG_EMBEDDING_MODEL = "text-embedding-3-small"

        inst = MagicMock()
        inst.chat_json_with_usage.return_value = (
            {
                "units": [
                    {
                        "question_id": "1",
                        "question": "What is 2+2?",
                        "student_response": "4",
                        "answer_key_segment": "4",
                        "extracted_text": "What is 2+2?\n4",
                    }
                ]
            },
            {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        )

        def fake_embed(texts, **kwargs):
            return [[0.1, 0.2, 0.3] for _ in range(len(texts))], 80

        with patch(
            "app.grading.multimodal.openai_trio_rag_frontload.OpenAIJsonClient",
            return_value=inst,
        ), patch(
            "app.grading.multimodal.openai_trio_rag_frontload._openai_embed_batch",
            side_effect=fake_embed,
        ):
            env = ingest_raw_submission(
                assignment_id="a1",
                student_id="s1",
                artifacts={},
                extracted_plaintext="dummy submission text for grading.",
            )
            chunks, audit = run_openai_trio_rag_frontload(env, cfg, "## 1\n2+2 = 4")

        self.assertTrue(audit.get("ok"))
        self.assertEqual(len(chunks), 1)
        ev = chunks[0].evidence or {}
        self.assertTrue(ev.get("_openai_trio_rag_frontload"))
        seg = ev.get("trio_segment_rag") or {}
        self.assertEqual((seg.get("question") or {}).get("embedding_dimension"), 3)
        self.assertIn("rag_embedding_bundle", ev)
        self.assertIn("cost_usd", audit)
        self.assertIn("per_chunk_avg_cost_usd", audit)
        inst.chat_json_with_usage.assert_called_once()

    def test_openai_trio_rag_frontload_multiple_windows_merge_dupes(self) -> None:
        from unittest.mock import MagicMock, patch

        from app.grading.multimodal.ingestion import ingest_raw_submission
        from app.grading.multimodal.openai_trio_rag_frontload import (
            run_openai_trio_rag_frontload,
        )

        cfg = Config()
        cfg.OPENAI_API_KEY = "sk-test"
        cfg.OPENAI_TRIO_RAG_CHAT_MODEL = "gpt-test"
        cfg.OPENAI_TRIO_RAG_EMBEDDING_MODEL = "text-embedding-3-small"
        cfg.MULTIMODAL_OPENAI_TRIO_WINDOW_CHARS = 12
        cfg.MULTIMODAL_OPENAI_TRIO_WINDOW_OVERLAP_CHARS = 2

        inst = MagicMock()
        one_unit = {
            "question_id": "w",
            "question": "q",
            "student_response": "s",
            "answer_key_segment": "a",
            "extracted_text": "q\ns",
        }
        inst.chat_json_with_usage.return_value = (
            {"units": [one_unit]},
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

        def fake_embed(texts, **kwargs):
            return [[0.1, 0.2] for _ in range(len(texts))], 10

        long_text = "x" * 30
        with patch(
            "app.grading.multimodal.openai_trio_rag_frontload.OpenAIJsonClient",
            return_value=inst,
        ), patch(
            "app.grading.multimodal.openai_trio_rag_frontload._openai_embed_batch",
            side_effect=fake_embed,
        ):
            env = ingest_raw_submission(
                assignment_id="a1",
                student_id="s1",
                artifacts={},
                extracted_plaintext=long_text,
            )
            chunks, audit = run_openai_trio_rag_frontload(env, cfg, "## key")

        self.assertTrue(audit.get("ok"))
        self.assertGreater(int(audit.get("trio_window_count") or 0), 1)
        self.assertGreater(inst.chat_json_with_usage.call_count, 1)
        self.assertEqual(len(chunks), 1)


if __name__ == "__main__":
    unittest.main()
