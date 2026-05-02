"""
Integration test: grade local fixtures from the repository root.

Layout (at repo root, next to ``AGT_platform/``):

- ``assignments_to_grade/`` — submission files (any supported modality; grouped by shared basename).
- ``rubric/`` — **either** a single generic rubric **or** per-assignment rubrics.
- ``multi-modal_assignments_to_grade/`` — multimodal pipeline JSON as
  ``<basename>_grade_output.json``.
- ``multi-modal_RAG_embeding/`` — per-assignment chunk JSON and RAG embedding bundles
  (same filenames as before: ``<stem>_chunks.json``, ``<stem>_embedding.json``, etc.).

**Generic rubric (grades all assignments):** the first match wins, in order:

1. ``rubric/default.{json,md,txt}``
2. ``rubric/generic.{json,md,txt}``
3. ``rubric/rubric.{json,md,txt}``

Use one basename only (e.g. only ``default.json`` + optional ``default.md``). JSON defines
criteria rows; ``.md`` / ``.txt`` with the same basename add ``rubric_text``. Prose-only
generic files use :data:`DEFAULT_STANDALONE_RUBRIC` as row templates.

**Per-assignment rubric (fallback):** if no generic files exist, use
``rubric/<assignment_basename>.{json,md,txt}`` as before.

Grading uses :class:`app.grading.multimodal.MultimodalGradingPipeline` with
:class:`app.grading.multimodal.MultiModelChunkRunner` (``MULTIMODAL_SAMPLES_PER_MODEL`` /
``GRADING_SAMPLE_TEMPERATURE``). Per-chunk grading uses **OpenAI only**
(``OPENAI_API_KEY`` + ``OPENAI_MULTIMODAL_GRADING_MODEL``); optional ``GRADING_MODEL_2`` /
``GRADING_MODEL_3`` must be ``openai:…`` specs. This test no longer requires Ollama.

**Why this test can run a long time (it is usually not stuck):** the multimodal pipeline
issues **OpenAI** ``chat_json`` calls **sequentially** — for each grading chunk,
``len(build_multimodal_grading_clients(cfg)) * MULTIMODAL_SAMPLES_PER_MODEL`` requests. Long
submissions split into many chunks; optional extra grading models multiply the call count;
``MULTIMODAL_SAMPLES_PER_MODEL`` > 1 multiplies it again.

**Fast defaults for this test** (so ``pytest`` finishes in minutes, not hours):

- ``MULTIMODAL_LOCAL_TEST_MAX_ASSIGNMENTS`` defaults to **1** (first basename only).
  Set to ``0`` or ``all`` to grade every assignment under ``assignments_to_grade/``.
- ``MULTIMODAL_LOCAL_TEST_GRADING_SAMPLES`` defaults to **1** (overrides ``MULTIMODAL_SAMPLES_PER_MODEL`` for
  this test). Set to ``from_config`` to use ``MULTIMODAL_SAMPLES_PER_MODEL`` from ``.env``,
  or set an explicit integer 1–16.
- ``MULTIMODAL_LOCAL_TEST_MAX_GRADING_UNITS`` defaults to **8** (caps multimodal chunk
  units per assignment). Set to ``0`` or ``all`` for no cap.

Raise provider timeouts in ``.env`` only if your OpenAI calls are slow (rare for small fixtures).
"""

from __future__ import annotations

import json
import logging
import os
import unittest

import requests
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from app.config import Config
from app.grading.grading_units import build_grading_units_from_chunks
from app.grading.modality_resolution import resolve_modality_profile
from app.grading.multimodal import (
    create_multimodal_pipeline_from_app_config,
    multimodal_assignment_to_grading_dict,
)
from app.grading.multimodal.ingestion import ingest_raw_submission
from app.grading.multimodal.schemas import RubricType
from app.grading.answer_key_resolve import resolve_answer_key_plaintext
from app.grading.output_schema import validate_grading_output
from app.grading.pipelines import DEFAULT_STANDALONE_RUBRIC
from app.grading.rag_embeddings import compute_submission_embedding, save_rag_embedding_bundle
from app.grading.submission_chunks import build_submission_chunks, write_chunks_json
from app.grading.submission_text import submission_text_from_artifacts

# Repo root: .../ai-automatic-grader-tool (contains AGT_platform/, assignments_to_grade/, ...)
REPO_ROOT = Path(__file__).resolve().parents[3]
ASSIGNMENTS_DIR = REPO_ROOT / "assignments_to_grade"
RUBRIC_DIR = REPO_ROOT / "rubric"
OUTPUT_DIR = REPO_ROOT / "grading_output"
RAG_DIR = REPO_ROOT / "RAG_embedding"
ANSWER_KEY_DIR = REPO_ROOT / "answer_key"

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
            "SKIP_LOCAL_LLM_TESTS is set; skipping integration calls to remote LLM APIs."
        )


def _ollama_reachable_quick(cfg: Config) -> bool:
    """Legacy helper (multimodal grading no longer requires Ollama)."""
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
    # Default 60s: first load of llama3.2 on CPU often exceeds 25s without being "broken".
    smoke_to = int(raw) if raw else 60
    smoke_to = max(5, min(smoke_to, 180))
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


_SECTION_NAME_TO_RUBRIC_TYPE = {
    "Scaffolded Coding": RubricType.PROGRAMMING_SCAFFOLDED,
    "Free Response": RubricType.FREE_RESPONSE,
    "Open-Ended EDA": RubricType.EDA_VISUALIZATION,
    "Mock Interview / Oral Assessment": RubricType.ORAL_INTERVIEW,
}


def _build_rubric_rows_by_type(rubric_json: dict) -> dict[RubricType, list[dict]]:
    """Map each rubric section to its RubricType with only that section's criteria."""
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
                "name": name,
                "max_points": max_pts,
                "criterion": name,
                "max_score": max_pts,
                "description": desc,
            })
        by_type[rt] = rows
    return by_type


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


def _try_load_generic_rubric_raw_json() -> dict | None:
    """Return the raw parsed JSON dict from the first generic rubric found (or None)."""
    for base in _GENERIC_BASENAMES:
        json_path = RUBRIC_DIR / f"{base}.json"
        if json_path.is_file():
            try:
                return json.loads(json_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return None
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


def _multimodal_local_test_max_assignments() -> int | None:
    """
    None = grade all basenames. Default when env unset: 1 (first basename only).
    Set ``MULTIMODAL_LOCAL_TEST_MAX_ASSIGNMENTS=0`` or ``all`` for no limit.
    """
    raw = os.getenv("MULTIMODAL_LOCAL_TEST_MAX_ASSIGNMENTS", "").strip().lower()
    if raw in ("",):
        return 1
    if raw in ("0", "all", "none"):
        return None
    try:
        return max(1, int(raw, 10))
    except ValueError:
        return 1


def _multimodal_local_test_max_grading_units() -> int | None:
    """
    Cap units passed to multimodal grading (``max_grading_units`` hint).

    None = no cap. Default when env unset: 8. Set ``0`` or ``all`` for no cap.
    """
    raw = os.getenv("MULTIMODAL_LOCAL_TEST_MAX_GRADING_UNITS", "").strip().lower()
    if raw in ("",):
        return 8
    if raw in ("0", "all", "none"):
        return None
    try:
        return max(1, int(raw, 10))
    except ValueError:
        return 8


def _multimodal_local_test_grading_samples_override(cfg: Config) -> None:
    """
    Per-test override for ``MULTIMODAL_SAMPLES_PER_MODEL`` (instance attribute only).

    Default when env unset: **1** (fast). Set ``from_config`` to use ``Config`` from ``.env``.

    .. note::

       Meaningful semantic entropy needs **several** stochastic samples from the same
       model (or multiple models). With k=1, entropy is always 0 and ai_confidence is
       always 1.0 — by design for fast test iteration. For a real signal, set
       ``MULTIMODAL_LOCAL_TEST_GRADING_SAMPLES=5`` (or use ``from_config`` with production
       ``MULTIMODAL_SAMPLES_PER_MODEL`` in ``.env``).
    """
    raw = os.getenv("MULTIMODAL_LOCAL_TEST_GRADING_SAMPLES", "").strip()
    if raw == "":
        cfg.MULTIMODAL_SAMPLES_PER_MODEL = 1
        return
    if raw.lower() in ("from_config", "use_config", "config"):
        return
    try:
        k = int(raw, 10)
    except ValueError:
        cfg.MULTIMODAL_SAMPLES_PER_MODEL = 1
        return
    cfg.MULTIMODAL_SAMPLES_PER_MODEL = max(1, min(k, 16))


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
    """Run the multimodal grading pipeline on local assignment + rubric fixtures."""

    def test_grade_local_assignments_write_json(self) -> None:
        from app.grading.llm_router import build_multimodal_grading_clients

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

        max_assign = _multimodal_local_test_max_assignments()
        if max_assign is not None and len(groups) > max_assign:
            sorted_pairs = sorted(groups.items())
            groups = dict(sorted_pairs[:max_assign])
            logging.getLogger(__name__).warning(
                "Grading %s of %s assignment basename(s). Default cap: "
                "MULTIMODAL_LOCAL_TEST_MAX_ASSIGNMENTS=1. Set to 0 or all to grade every file.",
                len(groups),
                len(sorted_pairs),
            )

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        RAG_DIR.mkdir(parents=True, exist_ok=True)
        _skip_if_no_local_llm()
        cfg = Config()
        _multimodal_local_test_grading_samples_override(cfg)
        if not build_multimodal_grading_clients(cfg):
            raise unittest.SkipTest(
                "Multimodal per-chunk grading requires OPENAI_API_KEY (and a valid "
                "OPENAI_MULTIMODAL_GRADING_MODEL). Ollama is not used for this path anymore."
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
                modality_profile = resolve_modality_profile(assignment, artifacts, plain)
                chunks = build_submission_chunks(
                    plain,
                    assignment_title=stem,
                    modality_subtype=str(modality_profile.get("modality_subtype") or ""),
                    max_chunk_chars=None,
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

                task_desc_parts: list[str] = []
                if getattr(assignment, "description", None):
                    task_desc_parts.append(str(assignment.description))
                if rubric_text:
                    task_desc_parts.append(rubric_text)
                task_description = "\n\n".join(task_desc_parts)

                n_models = len(build_multimodal_grading_clients(cfg))
                k_samples = max(1, int(getattr(cfg, "MULTIMODAL_SAMPLES_PER_MODEL", 5)))
                n_grading_units = len(build_grading_units_from_chunks(chunks))
                u_cap = _multimodal_local_test_max_grading_units()
                if u_cap is not None:
                    n_grading_units = min(n_grading_units, u_cap)
                n_calls = n_grading_units * n_models * k_samples
                chat_to = int(getattr(cfg, "OLLAMA_CHAT_TIMEOUT_SEC", 300))
                logging.getLogger(__name__).warning(
                    "Multimodal grading for %r: ~%s grading unit(s) × %s model(s) × %s sample(s)/model "
                    "≈ %s sequential /api/chat calls (up to %ss each).",
                    stem,
                    n_grading_units,
                    n_models,
                    k_samples,
                    n_calls,
                    chat_to,
                )

                raw_json = _try_load_generic_rubric_raw_json()
                if raw_json and isinstance(raw_json.get("sections"), list):
                    rubric_rows_by_type = _build_rubric_rows_by_type(raw_json)
                else:
                    rubric_rows_by_type = {RubricType.FREE_RESPONSE: rubric_list}
                pipeline = create_multimodal_pipeline_from_app_config(
                    cfg,
                    rubric_rows_by_type=rubric_rows_by_type,
                    task_description=task_description,
                )
                modality_hints = {
                    "modality_subtype": str(
                        modality_profile.get("modality_subtype") or ""
                    ),
                    "answer_key_plaintext": resolve_answer_key_plaintext(
                        stem, ANSWER_KEY_DIR
                    )[0],
                }
                unit_cap = _multimodal_local_test_max_grading_units()
                if unit_cap is not None:
                    modality_hints["max_grading_units"] = unit_cap

                envelope = ingest_raw_submission(
                    assignment_id=stem,
                    student_id="local_test",
                    artifacts={k: "<local_fixture>" for k in sorted(artifacts.keys())},
                    extracted_plaintext=plain,
                    modality_hints=modality_hints,
                )
                mm_result = pipeline.run(envelope)
                result = multimodal_assignment_to_grading_dict(
                    mm_result,
                    rubric=rubric_list,
                    modality_profile=modality_profile,
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
