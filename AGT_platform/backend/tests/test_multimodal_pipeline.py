"""
Tests for the multimodal grading pipeline.

- **Unit tests** (routing, entropy, mock pipeline): fast, no LLM required.
- **Integration test** (``LocalAssignmentGradingTests``): grades every file in
  ``assignments_to_grade/`` using the real rubric from ``rubric/default.json``
  and real LLM calls (Ollama llama3.2:3b + gemma3, plus OpenAI when
  ``OPENAI_API_KEY`` is set).  Saves per-assignment:

  - ``grading_output/<stem>_grade_output.json``
  - ``RAG_embedding/<stem>_chunks.json``
  - ``RAG_embedding/<stem>_embedding.json``
  - ``RAG_embedding/<stem>_parsed_preview.txt``
"""

from __future__ import annotations

import json
import logging
import os
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import requests

from app.config import Config
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
from app.grading.multimodal.rubric_router import route_rubric
from app.grading.multimodal.entropy import semantic_entropy_from_cluster_counts
from app.grading.output_schema import validate_grading_output
from app.grading.rag_embeddings import (
    compute_submission_embedding,
    deterministic_hash_embedding,
    save_rag_embedding_bundle,
)
from app.grading.submission_chunks import build_submission_chunks, write_chunks_json
from app.grading.submission_text import submission_text_from_artifacts

_log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
ASSIGNMENTS_DIR = REPO_ROOT / "assignments_to_grade"
RUBRIC_DIR = REPO_ROOT / "rubric"
OUTPUT_DIR = REPO_ROOT / "grading_output"
RAG_DIR = REPO_ROOT / "RAG_embedding"

_SUPPORTED_SUFFIXES = {".ipynb", ".py", ".pdf", ".txt", ".md", ".mp4"}
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
        if p.suffix.lower() not in _SUPPORTED_SUFFIXES:
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
    base = (cfg.INTERNAL_OLLAMA_URL or cfg.OLLAMA_BASE_URL or "").strip().rstrip("/")
    model = (cfg.OLLAMA_MODEL or "llama3.2:3b").strip()
    try:
        r = requests.post(
            f"{base}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": 'Reply: {"ok":true}'}],
                "stream": False,
            },
            timeout=60,
        )
        if r.status_code != 200:
            return False, f"HTTP {r.status_code} for {model!r}"
        body = r.json()
        if not (body.get("message") or {}).get("content", "").strip():
            return False, f"Empty content from {model!r}"
        return True, ""
    except requests.exceptions.ReadTimeout:
        return False, f"Timeout for {model!r}"
    except requests.exceptions.RequestException as e:
        return False, str(e)


def _configure_for_integration_test(cfg: Config) -> None:
    """Override Config in-place for the integration test.

    **Chunking**: llama3.2:3b splits each assignment into Q/A grading units
    via Ollama QA segmentation (``MULTIMODAL_OLLAMA_QA_SEGMENT=on``).

    **Grading**: three Ollama models, no OpenAI:
      1. llama3.2:3b  (primary)
      2. phi3.5:3.8b  (Phi-3.5-Mini-3.8B)
      3. deepseek-r1:1.5b (DeepSeek-R1 1.5B)

    3 samples per model at temperature 0.4 → 9 samples per chunk.
    Timeout raised to 600s for cold model loads on laptops.
    """
    # --- Chunking: llama3.2:3b only ---
    os.environ["MULTIMODAL_OLLAMA_QA_SEGMENT"] = "on"
    os.environ["MULTIMODAL_QA_SEGMENT_MODEL"] = "llama3.2:3b"

    # --- Memory: Option A — one model loaded at a time (no swap on 24 GB) ---
    os.environ["OLLAMA_MAX_LOADED_MODELS"] = "1"
    cfg.OLLAMA_KEEP_ALIVE = "0s"

    # --- Grading: 3 models ---
    if not getattr(cfg, "GRADING_MODEL_2", "").strip():
        cfg.GRADING_MODEL_2 = "ollama:phi3.5:3.8b"
    if not getattr(cfg, "GRADING_MODEL_3", "").strip():
        cfg.GRADING_MODEL_3 = "ollama:deepseek-r1:1.5b"

    cfg.OPENAI_API_KEY = ""

    cfg.OLLAMA_CHAT_TIMEOUT_SEC = max(
        int(getattr(cfg, "OLLAMA_CHAT_TIMEOUT_SEC", 300)),
        600,
    )

    cfg.GRADING_SAMPLES_PER_MODEL = int(
        os.getenv("MULTIMODAL_LOCAL_TEST_GRADING_SAMPLES", "3") or 3
    )
    cfg.GRADING_SAMPLE_TEMPERATURE = 0.4


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


def _make_sample_json(norm: float, *, confidence_note: str = "") -> str:
    criteria = [
        {"name": "Conceptual Correctness", "score": round(norm * 4), "max_points": 4},
        {"name": "Evidence & Justification", "score": round(norm * 3), "max_points": 3},
        {"name": "Depth of Understanding", "score": round(norm * 2), "max_points": 2},
        {"name": "Clarity", "score": round(norm * 1), "max_points": 1},
    ]
    total = sum(c["score"] for c in criteria)
    max_total = sum(c["max_points"] for c in criteria)
    return json.dumps({
        "rubric_type": "free_response",
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


class MultimodalRoutingTests(unittest.TestCase):
    def test_deterministic_routing_programming(self) -> None:
        ch = GradingChunk(
            chunk_id="c1", assignment_id="a1", student_id="s1", question_id="q1",
            modality=Modality.CODE, task_type=TaskType.SCAFFOLDED_CODING,
            extracted_text="print(1)",
        )
        route_rubric(ch, rubric_rows_by_type={})
        self.assertEqual(ch.rubric_type, RubricType.PROGRAMMING_SCAFFOLDED)

    def test_semantic_entropy_two_clusters(self) -> None:
        h = semantic_entropy_from_cluster_counts({"A": 1, "B": 1})
        self.assertGreater(h, 0.0)


class MultimodalPipelineRunTests(unittest.TestCase):
    """Core pipeline run with mock runner — no LLM, fast."""

    def _run_pipeline(self):
        env = build_envelope_from_plaintext(
            assignment_id=_MOCK_STEM, student_id="test_student",
            plaintext=_SAMPLE_PLAINTEXT,
            modality_hints={"modality": "notebook", "task_type": "free_response_short"},
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
            cfg, runner, rubric_rows_by_type={RubricType.FREE_RESPONSE: _FREE_RESPONSE_RUBRIC},
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


# ---------------------------------------------------------------------------
# Integration: grade ALL local assignments with real LLMs
# ---------------------------------------------------------------------------

@unittest.skipUnless(
    ASSIGNMENTS_DIR.is_dir() and RUBRIC_DIR.is_dir(),
    "Requires assignments_to_grade/ and rubric/ at the repository root.",
)
class LocalAssignmentGradingTests(unittest.TestCase):
    """Grade every assignment in ``assignments_to_grade/`` using real LLM calls.

    Uses ``rubric/default.json`` (per-section rubric), Ollama llama3.2:3b + gemma3
    (and OpenAI gpt-4o-mini when ``OPENAI_API_KEY`` is set).

    For each assignment the test:

    1. Extracts plaintext from the submission file (PDF / ipynb).
    2. Builds submission chunks → ``RAG_embedding/<stem>_chunks.json``.
    3. Computes embedding → ``RAG_embedding/<stem>_embedding.json``.
    4. Runs the multimodal pipeline with real LLM grading.
    5. Validates output: unique per-chunk justifications, criteria scores.
    6. Writes ``grading_output/<stem>_grade_output.json``.
    """

    @classmethod
    def setUpClass(cls) -> None:
        if os.getenv("SKIP_LOCAL_LLM_TESTS", "").strip().lower() in ("1", "true", "yes"):
            raise unittest.SkipTest("SKIP_LOCAL_LLM_TESTS is set.")

        cls.rubric_raw = _load_rubric_json()
        if cls.rubric_raw is None:
            raise unittest.SkipTest("rubric/default.json not found or invalid.")
        cls.rubric_by_type = _build_rubric_rows_by_type(cls.rubric_raw)
        cls.rubric_flat = _flat_rubric_from_by_type(cls.rubric_by_type)
        cls.rubric_flat_sectioned = _flatten_sections_rubric(cls.rubric_raw)
        if not cls.rubric_flat:
            raise unittest.SkipTest("No criteria parsed from rubric.")

        cls.groups = _assignment_groups()
        if not cls.groups:
            raise unittest.SkipTest(f"No files in {ASSIGNMENTS_DIR}.")

        cls.cfg = Config()
        _configure_for_integration_test(cls.cfg)

        if not _ollama_reachable(cls.cfg):
            raise unittest.SkipTest(
                "Ollama not reachable. Start with `ollama serve` or set SKIP_LOCAL_LLM_TESTS=true."
            )
        ok, detail = _ollama_chat_smoke(cls.cfg)
        if not ok:
            raise unittest.SkipTest(f"Ollama chat smoke failed: {detail}")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        RAG_DIR.mkdir(parents=True, exist_ok=True)

        from app.grading.llm_router import build_grading_clients
        clients = build_grading_clients(cls.cfg)
        cls._model_labels = [label for _, label in clients]
        _log.warning(
            "Integration test: %d model(s): %s, samples_per_model=%d",
            len(clients),
            ", ".join(cls._model_labels),
            cls.cfg.GRADING_SAMPLES_PER_MODEL,
        )

    def test_grade_all_assignments(self) -> None:
        graded_stems: list[str] = []

        for stem, paths in sorted(self.groups.items()):
            with self.subTest(assignment=stem):
                _log.warning("--- Grading %r (%d file(s)) ---", stem, len(paths))

                # -- 1. Artifacts & plaintext --
                artifacts = _build_artifacts(paths)
                plain = submission_text_from_artifacts(artifacts)
                self.assertGreater(len(plain.strip()), 0, f"Empty text for {stem!r}")

                assignment_ns = SimpleNamespace(
                    modality=None, rubric=self.rubric_flat_sectioned,
                    title=stem, description=f"Local fixture: {stem}",
                )
                modality_profile = resolve_modality_profile(assignment_ns, artifacts, plain)

                # -- 2. Chunks → RAG_embedding/ --
                chunks = build_submission_chunks(
                    plain, assignment_title=stem,
                    modality_subtype=str(modality_profile.get("modality_subtype") or ""),
                    max_chunk_chars=None,
                )
                self.assertGreater(len(chunks), 0, f"No chunks for {stem!r}")
                write_chunks_json(
                    RAG_DIR / f"{stem}_chunks.json",
                    chunks=chunks, assignment_title=stem,
                    source_file=",".join(p.name for p in paths),
                    profile=modality_profile,
                )

                # -- 3. Embedding → RAG_embedding/ --
                vec, vec_src = compute_submission_embedding(plain, self.cfg)
                save_rag_embedding_bundle(
                    RAG_DIR, assignment_stem=stem,
                    artifacts_keys=sorted(artifacts.keys()),
                    plaintext_chars=len(plain), embedding=vec,
                    embedding_source=vec_src,
                    parsed_preview=plain[:8000],
                    extra={"paths": [p.name for p in paths]},
                )

                # -- 4. Pipeline with real LLM calls --
                task_desc = f"Local fixture: {stem}"
                llm_grading_instr = self.rubric_raw.get("llm_grading_instructions")
                if isinstance(llm_grading_instr, str) and llm_grading_instr.strip():
                    task_desc += "\n\n" + llm_grading_instr.strip()

                pipeline = create_multimodal_pipeline_from_app_config(
                    self.cfg,
                    rubric_rows_by_type=self.rubric_by_type,
                    task_description=task_desc,
                )
                envelope = ingest_raw_submission(
                    assignment_id=stem, student_id="local_test",
                    artifacts={k: "<local_fixture>" for k in sorted(artifacts.keys())},
                    extracted_plaintext=plain,
                    modality_hints={
                        "modality_subtype": str(modality_profile.get("modality_subtype") or ""),
                        "max_grading_units": 8,
                    },
                )
                result = pipeline.run(envelope)
                self.assertTrue(result.chunk_results, f"No chunk results for {stem!r}")

                # -- 5. Grading dict → validate --
                grading_dict = multimodal_assignment_to_grading_dict(
                    result, rubric=self.rubric_flat,
                    modality_profile=modality_profile,
                )
                validated = validate_grading_output(grading_dict)
                self.assertIn("score", validated["overall"])
                self.assertIn("question_grades", validated)

                # -- 5b. Always write output before assertions --
                out_path = OUTPUT_DIR / f"{stem}_grade_output.json"
                out_path.write_text(
                    json.dumps(grading_dict, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8",
                )

                # Check justifications are present and content-specific
                chunk_summaries: list[str] = []
                all_justifications: list[str] = []
                for qg in validated["question_grades"]:
                    summary = qg.get("overall", {}).get("summary", "")
                    chunk_summaries.append(summary)
                    for c in qg.get("criteria") or []:
                        j = c.get("justification", "")
                        if j:
                            all_justifications.append(j)

                n_crit = len(validated.get("criteria") or [])
                _log.warning(
                    "  %s: score=%.3f, chunks=%d, criteria=%d, justifications=%d",
                    stem, validated["overall"]["score"],
                    len(result.chunk_results), n_crit, len(all_justifications),
                )

                if len(result.chunk_results) > 1:
                    self.assertGreater(
                        len(set(chunk_summaries)), 1,
                        f"[{stem}] All {len(chunk_summaries)} chunks have identical "
                        f"evidence summaries — LLM should produce unique per-chunk justifications.",
                    )

                self.assertGreater(
                    len(all_justifications), 0,
                    f"[{stem}] No justification text found in any criterion row "
                    f"(criteria_count={n_crit}, chunks={len(result.chunk_results)}). "
                    f"See {out_path} for diagnostic output.",
                )
                if len(all_justifications) > 1:
                    unique_ratio = len(set(all_justifications)) / len(all_justifications)
                    self.assertGreater(
                        unique_ratio, 0.3,
                        f"[{stem}] Only {unique_ratio:.0%} of justifications are unique — "
                        f"LLM is producing duplicate text across chunks.",
                    )

                graded_stems.append(stem)

        self.assertEqual(
            len(graded_stems), len(self.groups),
            f"Graded {len(graded_stems)} of {len(self.groups)} assignments.",
        )


if __name__ == "__main__":
    unittest.main()
