Make sure you are in the **`AGT_platform/backend`** directory for local Python commands (migrations, `app.main`).

1. Install backend dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. If you are using the autograder and not an MIT-affiliated course, consult your school’s IT docs for “OpenID Connect discovery URL” / “Issuer” / “OIDC”, then set **`OIDC_DISCOVERY_URL`** appropriately.

3. **Database migrations (Alembic)** — this repo **already** has `alembic/` and `alembic.ini`. Do **not** run `alembic init` again.

   From **`AGT_platform/backend/`** (so `alembic.ini` is found), with **`DATABASE_URL`** set:

   ```bash
   python -m alembic heads          # should show a single head (e.g. d4e5f6a7b8c9)
   python -m alembic upgrade head
   ```

   Or from the repo root:

   ```bash
   python -m alembic -c AGT_platform/backend/alembic.ini upgrade head
   ```

   With Docker, run migrations **inside** the backend container (see root **`README.md`**).

4. Run the API:

   ```bash
   python -m app.main
   ```

5. Access the backend locally at the host/port your environment configures (see app defaults and `.env`).

**Adding a new migration** (after model changes), from **`backend/`**:

```bash
python -m alembic revision --autogenerate -m "describe_change"
python -m alembic upgrade head
```

Review generated SQL carefully; autogenerate is not always complete.

## Ollama vs Meta llama-models (Llama 4 / Maverick)

Official download instructions: [meta-llama/llama-models — Download](https://github.com/meta-llama/llama-models?tab=readme-ov-file#download).

**Two different CLIs**

- **`llama-model`** (from the `llama-models` PyPI package): `llama-model list`, `llama-model download`, `llama-model verify-download`. This is what the README refers to.
- **`llama`** from **Llama Stack** only supports `llama stack …`. It is **not** the model downloader; `llama model …` will fail with “invalid choice”.

**What your logs usually mean**

1. **`llama download: No module named 'llama_models.cli.model'`** — packaging/import bug in some `llama-models` wheels (wrong import path). Upgrade the package or align with the version pinned in Meta’s repo; the downloader lives under `llama_models`’s CLI, not under `llama stack`.
2. **`peer closed connection … (received N bytes, expected …)`** — the transfer was truncated; any later “100%” per file can still leave **corrupt shards**.
3. **`Not enough disk space. Required: ~454000 MB`** — full Maverick FP8 needs on the order of **hundreds of GB** free for download and extraction; a machine with ~258 GB free cannot complete a full re-download in the default layout.
4. **`llama-model verify-download` → hash mismatch on every file** — on-disk files do **not** match the manifest (incomplete download, bad resume, or wrong tree). Treat the checkpoint directory as **invalid**.

**Recover a bad checkpoint**

Remove the broken tree (example path from a successful-but-unverified run):

```bash
rm -rf ~/.llama/checkpoints/Llama-4-Maverick-17B-128E-Instruct-fp8
```

Then re-download only when you have **enough free space**, a **stable network**, and a **fresh signed URL** from [llama.com downloads](https://www.llama.com/llama-downloads/) if using `--source meta`. Alternatively use **`--source huggingface`** with a Hugging Face token if the README’s HF path fits your disk and access.

**Using a large open-weight model with this backend’s multimodal pipeline**

**Option A — Ollama for chat; RAG via SentenceTransformers (default) or Ollama**

Chunk grading can use **Ollama’s HTTP API** (`OLLAMA_BASE_URL` / `INTERNAL_OLLAMA_URL`, `OLLAMA_MODEL`). It does **not** load `~/.llama/checkpoints` directly. RAG vectors default to **SentenceTransformers** (`RAG_EMBEDDING_BACKEND`, `SENTENCE_TRANSFORMERS_MODEL`); set **`RAG_EMBEDDING_BACKEND=ollama`** to use **`OLLAMA_EMBEDDINGS_MODEL`** (e.g. nomic) instead. To grade with Maverick (or any other large model) through Ollama:

1. Ensure the model is **available to Ollama** on that host (`ollama list` shows it), e.g. via `ollama pull` if published, or import/create from a supported artifact per Ollama docs.
2. Set **`OLLAMA_MODEL`** to that exact tag (and raise **`OLLAMA_CHAT_TIMEOUT_SEC`** for slow GPUs).
3. For **multiple stochastic grades per chunk** with a single model, use **`MULTIMODAL_SAMPLES_PER_MODEL`** (see `app/config.py`). Optional second/third graders: **`GRADING_MODEL_2`** / **`GRADING_MODEL_3`** (`ollama:…` or `openai:…`).

**Integration test env** (`tests/test_multimodal_pipeline.py::LocalAssignmentGradingTests`): defaults to **`MULTIMODAL_INTEGRATION_LLM_BACKEND=huggingface`** (Maverick FP8 on the Hub + `HF_TOKEN`). Set **`MULTIMODAL_INTEGRATION_LLM_BACKEND=ollama`** to exercise the Ollama-only path. Run **`pytest -rs`** for full skip text and **`--log-cli-level=WARNING`** for `[integration]` phase logs.

**Option B — Hugging Face for chat; RAG via SentenceTransformers (default)**

Use local **transformers** for structure + grading; **RAG vectors** use **SentenceTransformers** by default (`all-MiniLM-L6-v2` via `requirements.txt`). Optional Ollama is only needed if you set **`RAG_EMBEDDING_BACKEND=ollama`** or use Ollama-backed extras.

1. `pip install -r requirements.txt -r requirements-huggingface.txt`
2. Set **`MULTIMODAL_LLM_BACKEND=huggingface`** (alias: **`hf`**).
3. **`HUGGINGFACE_HUB_TOKEN`** or **`HF_TOKEN`** with access to the gated repo.
4. **`HUGGINGFACE_GRADING_MODEL_ID`** — optional; defaults to **`meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8`**. The Meta **llama-model** id `Llama-4-Maverick-17B-128E-Instruct:fp8` is accepted and mapped to that repo.
5. **`RAG_EMBEDDING_BACKEND`** / **`SENTENCE_TRANSFORMERS_MODEL`** — defaults match the multimodal pipeline (no Ollama required for embeddings on this path).

Optional trio/QA segmentation and per-chunk grading both use this Hugging Face primary; **`GRADING_MODEL_2` / `GRADING_MODEL_3`** remain `ollama:` or `openai:` if you add them.
