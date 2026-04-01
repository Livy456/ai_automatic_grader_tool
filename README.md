# AI Automatic Grader Tool (AGT)

Monorepo for the **AGT platform**: a React (Vite) frontend, Flask API + Celery worker backend, PostgreSQL, Redis, MinIO (S3-compatible storage), and Ollama for local LLM grading.

Application code lives under **`AGT_platform/`**.

**Production (two-EC2, direct S3 uploads, GPU workers):** see **[`docs/production_architecture_redesign.md`](docs/production_architecture_redesign.md)**.

---

## Prerequisites

- **Docker Desktop** (or Docker Engine + Compose v2) for the full stack
- **Node.js 18+** and **npm** (local frontend dev)
- **Python 3.11+** and **pip** (local backend dev)
- **Git**

---

## Quick start: Docker (full stack)

All Compose commands are run from **`AGT_platform/`** (where `docker-compose.yaml` is). For a Docker Desktop–focused walkthrough (install, UI, troubleshooting), see **[`AGT_platform/DOCKER_DESKTOP.md`](AGT_platform/DOCKER_DESKTOP.md)**.

### 1. Create environment file

Create **`AGT_platform/.env`** (same folder as `docker-compose.yaml`). Compose reads this file for variable substitution and passes it to the **backend**, **worker**, and **frontend** services.

Fill in real values locally; the shape of the file is:

```env
DATABASE_URL=
REDIS_URL=
SECRET_KEY=
JWT_EXPIRATION_SECONDS=

OLLAMA_BASE_URL=
OLLAMA_MODEL=

FRONTEND_BASE_URL=
CORS_ORIGINS=

VITE_API_BASE=
```

Use Docker **service names** as hosts where the stack runs in Compose (see `docker-compose.yaml` for service names and published ports). For `VITE_API_BASE`, point the Vite dev proxy at your API service using the backend service name and port as appropriate for your compose file.

Optional `S3_*` overrides go in the same file if you are not relying entirely on defaults defined in `docker-compose.yaml`.

### 2. Start containers

```bash
cd AGT_platform
docker compose up -d --build
```

### 3. Database migrations

Run Alembic **inside the backend container** once (and after pulling migrations):

```bash
docker compose exec backend python -m alembic upgrade head
```

### 4. Pull an Ollama model (first time)

After the Ollama container is running, pull the model matching your **`OLLAMA_MODEL`** value (container name may match your compose project):

```bash
docker exec -it <ollama_container_name> ollama pull <model_name>
```

### 5. Open the app

| Service | Access |
|--------|--------|
| Frontend (Vite dev in container) | Host port mapped for the frontend service in `docker-compose.yaml` |
| Backend API | Host port mapped for the backend service |
| MinIO console | Host port mapped for the MinIO console; credentials from your compose / env |
| Ollama | Host port mapped for the Ollama service |

Useful commands:

```bash
docker compose logs -f backend
docker compose down
```

---

## Local development: backend (without Docker for Python)

From **`AGT_platform/backend/`**:

1. **Virtual environment (recommended)**

   ```bash
   cd AGT_platform/backend
   python3.11 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Environment variables**

   The app loads **`backend/.env.local`** first, then **`backend/.env`** (overrides). Set at least:

   - `DATABASE_URL=`
   - `REDIS_URL=`
   - `SECRET_KEY=`, `JWT_EXPIRATION_SECONDS=`, `OLLAMA_BASE_URL=`, `S3_*` as needed (empty values use application defaults where the backend supports them)

3. **Migrations**

   ```bash
   python -m alembic upgrade head
   ```

4. **Run the API**

   ```bash
   python -m app.main
   ```

   The Flask app listens on the host and port configured by your environment / defaults.

5. **Celery worker (optional locally)**

   With the same env activated and Redis reachable:

   ```bash
   celery -A app.tasks.celery_app worker --loglevel=INFO -Q gpu,celery
   ```

More backend notes: **`AGT_platform/backend/ReadMe.md`**.

---

## Local development: frontend (without Docker for Node)

From **`AGT_platform/frontend/`**:

1. **Install dependencies**

   ```bash
   cd AGT_platform/frontend
   npm install
   ```

2. **API base for the Vite proxy**

   Create **`AGT_platform/frontend/.env.development.local`** (optional) or export before `npm run dev`:

   ```env
   VITE_API_BASE=
   ```

   Set `VITE_API_BASE` to your Flask base URL (non-empty when developing against a real API). The dev server proxies `/api` to this target. If you leave it empty, the Vite config uses its own fallback behavior.

3. **Run the dev server**

   ```bash
   npm run dev
   ```

   Open the URL and port printed in the terminal (per Vite defaults and your config).

4. **Production build**

   ```bash
   npm run build
   npm run preview
   ```

If `npm run build` fails with a missing module (for example **`jwt-decode`**), install it and retry:

```bash
npm install jwt-decode
```

More frontend notes: **`AGT_platform/frontend/ReadMe.md`**.

---

## Mixed setup (common)

- Run **Postgres, Redis, MinIO, Ollama** with Docker, but run **backend** and **frontend** on the host for faster iteration.
- Set `DATABASE_URL` / `REDIS_URL` to reach those services via **localhost** (or your host) and the ports published in `docker-compose.yaml`.
- Set **`VITE_API_BASE=`** to your local Flask base URL when running the frontend on the host.

---

## Production

Production compose and split web/GPU layouts may differ from `docker-compose.yaml`. See the root **`env.example`** and any deployment docs you keep for your environment (env files with secrets must not be committed).

---

## Repository layout (high level)

```
AGT_platform/
  docker-compose.yaml    # Local full stack
  docker-compose.prod.yaml
  backend/               # Flask app, Alembic, Celery
  frontend/              # Vite + React
```
