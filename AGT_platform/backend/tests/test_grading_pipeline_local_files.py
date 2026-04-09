"""
Integration test: grade local fixtures from the repository root.

Layout (at repo root, next to ``AGT_platform/``):

- ``assignments_to_grade/`` — submission files (any supported modality; grouped by shared basename).
- ``rubric/`` — **either** a single generic rubric **or** per-assignment rubrics.
- ``grading_output/`` — written as ``<basename>_grade_output.json`` (pipeline JSON only).

**Generic rubric (grades all assignments):** the first match wins, in order:

1. ``rubric/default.{json,md,txt}``
2. ``rubric/generic.{json,md,txt}``
3. ``rubric/rubric.{json,md,txt}``

Use one basename only (e.g. only ``default.json`` + optional ``default.md``). JSON defines
criteria rows; ``.md`` / ``.txt`` with the same basename add ``rubric_text``. Prose-only
generic files use :data:`DEFAULT_STANDALONE_RUBRIC` as row templates.

**Per-assignment rubric (fallback):** if no generic files exist, use
``rubric/<assignment_basename>.{json,md,txt}`` as before.

Requires a running Ollama (or configured OpenAI for staged/escalation paths) matching ``Config``.
"""

from __future__ import annotations

import json
import os
import unittest

import requests
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from app.config import Config
from app.grading.modality_resolution import resolve_modality_profile
from app.grading.output_schema import validate_grading_output
from app.grading.pipelines import DEFAULT_STANDALONE_RUBRIC, run_grading_pipeline
from app.grading.rag_embeddings import compute_submission_embedding, save_rag_embedding_bundle
from app.grading.submission_chunks import build_submission_chunks, write_chunks_json
from app.grading.submission_text import submission_text_from_artifacts

# Repo root: .../ai-automatic-grader-tool (contains AGT_platform/, assignments_to_grade/, ...)
REPO_ROOT = Path(__file__).resolve().parents[3]
ASSIGNMENTS_DIR = REPO_ROOT / "assignments_to_grade"
RUBRIC_DIR = REPO_ROOT / "rubric"
OUTPUT_DIR = REPO_ROOT / "grading_output"
RAG_DIR = REPO_ROOT / "RAG_embedding"

_GENERIC_BASENAMES = ("default", "generic", "rubric")

_SUPPORTED_SUFFIXES = {".ipynb", ".py", ".pdf", ".txt", ".md", ".mp4"}
_SUFFIX_TO_ARTIFACT_KEY = {
    ".ipynb": "ipynb",
    ".py": "py",
    ".pdf": "pdf",
    ".txt": "txt",
    ".md": "md",
    ".mp4": "mp4",
}


def _fixtures_layout_ok() -> bool:
    return ASSIGNMENTS_DIR.is_dir() and RUBRIC_DIR.is_dir()


def _skip_if_no_local_llm() -> None:
    if os.getenv("SKIP_LOCAL_LLM_TESTS", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        raise unittest.SkipTest(
            "SKIP_LOCAL_LLM_TESTS is set; skipping integration calls to Ollama."
        )


def _ollama_reachable_quick(cfg: Config) -> bool:
    base = (cfg.INTERNAL_OLLAMA_URL or cfg.OLLAMA_BASE_URL or "").strip().rstrip("/")
    if not base:
        return False
    try:
        r = requests.get(f"{base}/api/tags", timeout=4)
        return r.status_code == 200
    except OSError:
        return False


def _ollama_primary_chat_smoke(cfg: Config) -> tuple[bool, str]:
    """
    Fail fast if /api/chat cannot complete quickly.

    ``/api/tags`` can return 200 while embeddings return 404 (hash fallback is fine) and
    while /api/chat hangs until OLLAMA_CHAT_TIMEOUT_SEC — this smoke avoids ~300s waits
    per failing call when the daemon or model is misconfigured.
    """
    base = (cfg.INTERNAL_OLLAMA_URL or cfg.OLLAMA_BASE_URL or "").strip().rstrip("/")
    if not base:
        return False, "OLLAMA_BASE_URL / INTERNAL_OLLAMA_URL is empty"
    model = (cfg.OLLAMA_MODEL or "llama3.2:3b").strip()
    raw = os.getenv("LOCAL_LLM_SMOKE_TIMEOUT_SEC", "").strip()
    smoke_to = int(raw) if raw else 25
    smoke_to = max(5, min(smoke_to, 120))
    try:
        r = requests.post(
            f"{base}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "user", "content": 'Reply with exactly the JSON object: {"ok":true}'}
                ],
                "stream": False,
            },
            timeout=smoke_to,
        )
        if r.status_code != 200:
            return False, (
                f"/api/chat returned HTTP {r.status_code} for model {model!r}. "
                f"Body (truncated): {r.text[:300]!r}. Try: ollama pull {model}"
            )
        body = r.json()
        content = (body.get("message") or {}).get("content") or ""
        if not content.strip():
            return False, f"/api/chat returned empty content for model {model!r}"
        return True, ""
    except requests.exceptions.ReadTimeout:
        return False, (
            f"/api/chat timed out after {smoke_to}s (LOCAL_LLM_SMOKE_TIMEOUT_SEC). "
            f"Is model {model!r} pulled and loaded? Is the GPU busy? "
            f"(Full grading uses OLLAMA_CHAT_TIMEOUT_SEC, often 300s.)"
        )
    except requests.exceptions.RequestException as e:
        return False, f"/api/chat request failed: {e}"


def _assignment_groups() -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(ASSIGNMENTS_DIR.iterdir()):
        if not p.is_file() or p.name.startswith("."):
            continue
        if p.suffix.lower() not in _SUPPORTED_SUFFIXES:
            continue
        groups[p.stem].append(p)
    return dict(groups)


def _max_points_from_range(points_range: object) -> float:
    """e.g. \"0-4\" -> 4, \"0-10\" -> 10; fallback 10."""
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
    """Convert nested ``sections[].criteria[]`` (MEng-style) into flat grader rows."""
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
            desc_parts: list[str] = []
            if isinstance(levels, dict):
                desc_parts.append(json.dumps(levels, ensure_ascii=False))
            desc = "\n".join(desc_parts)[:8000]
            label = f"{sec_name} — {cname}" if sec_name else cname
            out.append(
                {
                    "name": label,
                    "max_points": max_pts,
                    "criterion": label,
                    "max_score": max_pts,
                    "description": desc,
                }
            )
    return out


def _parse_rubric_json_document(data: str) -> tuple[list[dict], str | None]:
    """
    Parse rubric JSON into criterion rows and optional instructor prose.

    Supports: top-level list; ``rubric`` / ``criteria`` / ``items`` arrays; and
    ``sections`` + per-criterion objects (plus ``llm_grading_instructions``).
    """
    try:
        raw = json.loads(data)
    except json.JSONDecodeError as e:
        raise ValueError(
            "rubric/default.json must be valid JSON only. Do not append free-text after "
            "the closing brace — put long LLM instructions in the "
            "`llm_grading_instructions` string field inside the JSON, or use default.md."
        ) from e

    extra_prose: str | None = None
    rubric_list: list[dict] = []

    if isinstance(raw, list):
        rubric_list = [x for x in raw if isinstance(x, dict)]
    elif isinstance(raw, dict):
        instr = raw.get("llm_grading_instructions")
        if isinstance(instr, str) and instr.strip():
            extra_prose = instr.strip()
        if isinstance(raw.get("sections"), list):
            rubric_list = _flatten_sections_rubric(raw)
        for key in ("rubric", "criteria", "items"):
            if rubric_list:
                break
            chunk = raw.get(key)
            if isinstance(chunk, list):
                rubric_list = [x for x in chunk if isinstance(x, dict)]
                break

    if not rubric_list:
        rubric_list = [dict(x) for x in DEFAULT_STANDALONE_RUBRIC]
    return rubric_list, extra_prose


def _collect_prose_for_basename(basename: str) -> str | None:
    parts: list[str] = []
    for ext in (".md", ".txt"):
        path = RUBRIC_DIR / f"{basename}{ext}"
        if path.is_file():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                parts.append(text)
    if not parts:
        return None
    return "\n\n".join(parts)


def _try_load_generic_rubric() -> tuple[list[dict], str | None] | None:
    """
    Load a single generic rubric shared by all local assignments.

    Returns ``None`` if no ``default`` / ``generic`` / ``rubric`` files are present.
    """
    for base in _GENERIC_BASENAMES:
        json_path = RUBRIC_DIR / f"{base}.json"
        prose = _collect_prose_for_basename(base)

        if json_path.is_file():
            text = json_path.read_text(encoding="utf-8")
            rubric_list, json_prose = _parse_rubric_json_document(text)
            prose_parts: list[str] = []
            if json_prose:
                prose_parts.append(json_prose)
            if prose:
                prose_parts.append(prose)
            combined = "\n\n".join(prose_parts) if prose_parts else None
            return rubric_list, combined

        if prose is not None:
            rubric_list = [dict(x) for x in DEFAULT_STANDALONE_RUBRIC]
            return rubric_list, prose

    return None


def _load_rubric_for_stem(stem: str) -> tuple[list[dict], str | None]:
    """
    Resolve ``rubric/<stem>.{json,md,txt}`` (per-assignment rubric).
    """
    rubric_list: list[dict] = [dict(x) for x in DEFAULT_STANDALONE_RUBRIC]
    rubric_text: str | None = None

    json_path = RUBRIC_DIR / f"{stem}.json"
    if json_path.is_file():
        rubric_list, json_prose = _parse_rubric_json_document(
            json_path.read_text(encoding="utf-8")
        )
        if json_prose:
            rubric_text = json_prose

    for ext in (".md", ".txt"):
        prose_path = RUBRIC_DIR / f"{stem}{ext}"
        if prose_path.is_file():
            extra = prose_path.read_text(encoding="utf-8").strip()
            if extra:
                rubric_text = (
                    (rubric_text + "\n\n" + extra) if rubric_text else extra
                )

    return rubric_list, rubric_text


def _rubric_files_exist_for_stem(stem: str) -> bool:
    return any(
        (RUBRIC_DIR / f"{stem}{ext}").is_file()
        for ext in (".json", ".md", ".txt")
    )


def _build_artifacts(paths: list[Path]) -> dict[str, bytes]:
    artifacts: dict[str, bytes] = {}
    for p in paths:
        key = _SUFFIX_TO_ARTIFACT_KEY.get(p.suffix.lower())
        if not key:
            continue
        if key in artifacts:
            raise ValueError(
                f"Duplicate artifact key {key!r} for assignment files: {paths}"
            )
        artifacts[key] = p.read_bytes()
    return artifacts


@unittest.skipUnless(
    _fixtures_layout_ok(),
    "Create directories assignments_to_grade/ and rubric/ at the repository root.",
)
class TestGradingPipelineLocalFiles(unittest.TestCase):
    """Run :func:`run_grading_pipeline` on local assignment + rubric fixtures."""

    def test_grade_local_assignments_write_json(self) -> None:
        groups = _assignment_groups()
        if not groups:
            self.skipTest(
                f"No supported files under {ASSIGNMENTS_DIR} "
                f"(suffixes: {sorted(_SUPPORTED_SUFFIXES)})."
            )

        generic_rubric = _try_load_generic_rubric()
        if generic_rubric is None and not any(
            _rubric_files_exist_for_stem(stem) for stem in groups
        ):
            self.skipTest(
                "Add a generic rubric (rubric/default.json or generic.md, etc.) "
                "or per-assignment rubric files under rubric/."
            )

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        RAG_DIR.mkdir(parents=True, exist_ok=True)
        _skip_if_no_local_llm()
        cfg = Config()
        if not _ollama_reachable_quick(cfg):
            raise unittest.SkipTest(
                "Ollama unreachable at INTERNAL_OLLAMA_URL / OLLAMA_BASE_URL "
                "(try `ollama serve`, fix URL for laptop vs GPU host, or set "
                "SKIP_LOCAL_LLM_TESTS=true to skip)."
            )
        ok_chat, chat_detail = _ollama_primary_chat_smoke(cfg)
        if not ok_chat:
            raise unittest.SkipTest(
                "Ollama chat smoke test failed — integration would hang on slow/failing "
                f"/api/chat. Detail: {chat_detail}\n"
                "Embeddings: 404 on /api/embeddings or /api/embed is OK (hash fallback); "
                "for real vectors run `ollama pull nomic-embed-text` (or set "
                "OLLAMA_EMBEDDINGS_MODEL) on a current Ollama build.\n"
                "Skip this test with SKIP_LOCAL_LLM_TESTS=true if you have no local GPU server."
            )

        for stem, paths in sorted(groups.items()):
            with self.subTest(assignment=stem):
                if generic_rubric is not None:
                    rubric_list, rubric_text = generic_rubric
                else:
                    if not _rubric_files_exist_for_stem(stem):
                        self.fail(
                            f"No generic rubric and missing per-assignment rubric for "
                            f"{stem!r}: add rubric/{stem}.json (and optionally .md/.txt), "
                            f"or add rubric/default.json (or generic.* / rubric.*) for all."
                        )
                    rubric_list, rubric_text = _load_rubric_for_stem(stem)

                artifacts = _build_artifacts(paths)

                assignment = SimpleNamespace(
                    modality=None,
                    rubric=rubric_list,
                    title=stem,
                    description=f"Local grading fixture: {stem}",
                )

                # Unified plaintext + embedding export (PDF, ipynb, txt, etc.)
                plain = submission_text_from_artifacts(artifacts)
                self.assertGreater(
                    len(plain.strip()),
                    0,
                    f"Parsed submission text empty for {stem!r}; check artifact bytes and formats.",
                )
                modality_profile = resolve_modality_profile(
                    assignment, artifacts, plain[:12000]
                )
                chunk_cap = int(getattr(cfg, "RAG_EMBED_MAX_CHARS", 4000) or 4000)
                chunks = build_submission_chunks(
                    plain,
                    assignment_title=stem,
                    modality_subtype=str(modality_profile.get("modality_subtype") or ""),
                    max_chunk_chars=max(2000, min(chunk_cap, 12000)),
                )
                write_chunks_json(
                    RAG_DIR / f"{stem}_chunks.json",
                    chunks=chunks,
                    assignment_title=stem,
                    source_file=",".join(p.name for p in paths),
                    profile=modality_profile,
                )

                vec, vec_src = compute_submission_embedding(plain, cfg)
                self.assertGreater(len(vec), 0)
                save_rag_embedding_bundle(
                    RAG_DIR,
                    assignment_stem=stem,
                    artifacts_keys=sorted(artifacts.keys()),
                    plaintext_chars=len(plain),
                    embedding=vec,
                    embedding_source=vec_src,
                    parsed_preview=plain[:8000],
                    extra={"paths": [str(p.name) for p in paths]},
                )

                result = run_grading_pipeline(
                    cfg,
                    assignment,
                    artifacts,
                    rubric_text=rubric_text,
                    answer_key_text=None,
                )

                validated = validate_grading_output(result)
                self.assertIn("score", validated["overall"])

                out_path = OUTPUT_DIR / f"{stem}_grade_output.json"
                out_path.write_text(
                    json.dumps(result, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8",
                )
                self.assertTrue(out_path.is_file())
                self.assertIn("overall", result)
                self.assertIn("criteria", result)


if __name__ == "__main__":
    unittest.main()
