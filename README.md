# ai-automatic-grader-tool

Monorepo for the **AGT** grading platform. The production API, Celery workers, and **multimodal grading pipeline** live under `AGT_platform/backend/` (Python package `app`).

## Backend tests

From the repo root (with dev dependencies installed for the backend):

```bash
cd AGT_platform/backend
pytest tests/ -q
```

Or using root `pyproject.toml` paths:

```bash
pytest -q
```

## Layout (high level)

- `AGT_platform/backend/` — Flask app, SQLAlchemy models, Celery tasks, `app/grading/multimodal/`
- `AGT_platform/frontend/` — web UI
- `assignments_to_grade/`, `rubric/`, `answer_key/` — optional local fixtures for integration-style runs (see backend test docstrings)

The former standalone `assignment-parser` library and `specs/` example bundles were removed; they were not imported by the multimodal pipeline.
