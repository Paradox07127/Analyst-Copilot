# EDA Agent Platform

> A local-first, traceable, and safety-bounded workspace for AI-assisted exploratory data analysis.

EDA Agent Platform helps analysts understand CSV data quality, structure, relationships, and business signals. It combines deterministic analysis with constrained LLM workflows and evidence validation: establish reproducible facts first, then extend them through inspectable questions, reports, and chat analysis.

The current package version is `0.2.0`. The implementation lives in `eda_platform/`.
The former `single_eda_agent Ver1.0/` implementation has been removed from the
working tree and remains available only through Git history.

## Contents

- [Capabilities](#capabilities)
- [Technology and requirements](#technology-and-requirements)
- [Quick start](#quick-start)
- [Ways to run the app](#ways-to-run-the-app)
- [User guide](#user-guide)
- [LLM and privacy settings](#llm-and-privacy-settings)
- [Optional: open-ended Python analysis](#optional-open-ended-python-analysis)
- [Artifacts, traceability, and safety boundaries](#artifacts-traceability-and-safety-boundaries)
- [Development and quality checks](#development-and-quality-checks)
- [Project structure](#project-structure)

## Capabilities

| Area | What the platform provides |
|---|---|
| Data ingestion and understanding | Upload one or more CSV files and generate column profiles, quality checks, statistical analysis tables, chart specifications, and raw-data views. |
| Data preparation | Optionally apply non-destructive missing-value and IQR-outlier cleaning before analysis. Source files are never modified in place; cleaning produces a new data version. |
| Multi-table analysis | Discover candidate relationships from names, types, overlap, and uniqueness; validate joins with DuckDB; inspect join multipliers, orphan rates, cardinality, and an ER diagram. |
| Guided investigation | Generate, score, and batch-execute verifiable questions; review findings, statistical tests, and optional lightweight ML baselines with model cards. |
| Reporting | Produce evidence-backed reports with a claim ledger and hard-validator results. Download a self-contained HTML report, or PDF when the optional dependency is available. |
| Conversational analysis | In live LLM mode, use Chat for intent routing, planning, and read-only DuckDB SQL. Open-ended Python analysis is isolated behind a sandbox. |
| Reuse and comparison | Edit semantic knowledge, save validated analysis skills, and compare two runs or fork a one-change variant. |
| Observability | Watch a run's stages and events in the floating Activity panel, then inspect the typed artifacts, validation state, and errors it produced. |

## Technology and requirements

- Python 3.12+
- Node.js 20+ and npm, to build the React workbench (`apps/web`)
- [FastAPI](https://fastapi.tiangolo.com/) for the `/api/v1` contract and [React](https://react.dev/) + [Vite](https://vite.dev/) for the product UI
- Pandas, DuckDB, SciPy, and scikit-learn for processing, querying, statistics, and baseline modeling
- Pydantic for typed artifacts and workflow contracts
- Vega-Lite and `vl-convert-python` for interactive charts and PNG export;
  Matplotlib is available to sandboxed open-ended Python analysis
- `uv` (recommended) or a standard Python virtual environment
- Docker Desktop (optional, for isolated open-ended Python analysis)

Linux, macOS, and Windows are all supported. The platform-specific pieces —
cross-process file locking, the run fence, worker launch and termination, and
process birth identity — each have a Windows implementation alongside the POSIX
one; see `core/file_lock.py`, `core/run_fence.py`, `core/process_control.py`,
and `infrastructure/launch_gate.py`. The automated suite currently runs on
Linux and macOS only, so the Windows branches are reviewed but not yet covered
by CI.

## Quick start

Run the following commands from the repository root.

### 1. Install dependencies

Python (using `uv` is recommended):

```bash
uv sync --extra dev
```

Without `uv`:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

Frontend:

```bash
npm install --prefix apps/web
```

### 2. Configure the local environment (optional)

Offline mode requires no API key and can run immediately. To enable live LLM
reports and Chat, or to put the workspace somewhere other than
`eda_platform/workspace`, copy the example file:

```bash
cp .env.example .env
```

Add the credentials for the chosen provider to `.env`, and set `EDA_WORKSPACE`
to an **absolute** path if you want a different data directory. The real `.env`
file is ignored by Git and must never be committed.

### 3. Build and start

```bash
npm run build --prefix apps/web
uv run python scripts/serve.py            # http://127.0.0.1:8000
```

`serve.py` serves the built React workbench and the API from one origin. This is
the default and recommended way to run the platform. Rebuild the frontend
whenever `apps/web` changes; `serve.py` serves whatever is in `apps/web/dist`.

### 4. First run in the browser

Open <http://127.0.0.1:8000/>. A fresh workspace has no projects yet:

1. **Create a project** from the project list. The display name is free text; the
   project id becomes a directory name, so it is restricted to letters, digits,
   spaces, `_`, `.`, and `-`. Ids that differ only in case are rejected, because
   the same directory would be reused on case-insensitive filesystems.
2. **Upload one or more CSV files** on the new-run page, optionally add business
   context, and choose the LLM mode (`Environment default` uses whatever `.env`
   configures; `Offline` makes no API calls).
3. **Run the analysis.** It executes in a separate worker process, so progress
   keeps streaming into Activity even if you close its panel or navigate away, and the
   run survives closing the browser tab.
4. When the job completes, the run's pages (Data Map, Quality, Report, …) are
   populated. Past runs are listed in the left session rail.

## Ways to run the app

The React UI (`apps/web`) talks to the FastAPI backend under `/api/v1`.

### Option A: single command (default)

```bash
npm run build --prefix apps/web
uv run python scripts/serve.py            # http://127.0.0.1:8000
uv run python scripts/serve.py --port 8321  # alternate port
```

`serve.py` serves the static build and the API from one origin, with an SPA
fallback so deep links like `/projects/<id>/sessions/<id>/data-map` work on refresh.
It binds loopback by default; pass `--host` explicitly for remote access.
The web UI is enabled by passing `serve_web_dist` to `create_app` (which is what
`serve.py` does); a bare `create_app()` (Option B) serves the API only.

### Option B: development (two processes)

```bash
uv run uvicorn eda_platform.api.main:create_app --factory --port 8000
npm run dev --prefix apps/web   # Vite on http://localhost:5173, proxies /api to :8000
```

### Option C: Docker

```bash
cd docker/app
docker compose up --build               # http://127.0.0.1:8000, loopback only
docker compose --profile caddy up --build  # optional Caddy front proxy on :8080
```

The container stores data in a named `workspace` volume and runs as a non-root
user. See [docker/app/](docker/app/) for the Dockerfile, compose file, and an
example Caddyfile (gzip, upload cap, SSE-safe proxying). Open-ended Python
analysis is rejected inside this container because no sandbox backend is
available there; core EDA runs are unaffected.

Public deployment is an explicit security mode; changing only the Caddy domain
is not sufficient. Set `EDA_DEPLOYMENT_MODE=remote`, exact
`EDA_ALLOWED_HOSTS` host names, and exact `EDA_ALLOWED_ORIGINS` browser origins.
Remote unsafe requests require the same allowed origin plus the frontend's
`X-EDA-CSRF` signal. Remote uploads are also rate-limited by the direct client
address (or `X-Forwarded-For` only when the direct proxy IP is explicitly
trusted). Persistent SQLite quotas cover each project's canonical file count,
aggregate upload bytes, and concurrent upload reservations. See
[`.env.example`](.env.example) for defaults and overrides. The default `local`
mode keeps the existing loopback-only, zero-configuration workflow.

> **`EDA_WORKSPACE` must be an absolute path.** Relative values are rejected.
> Without an override, every entry point resolves the same repository-anchored
> `eda_platform/workspace` path, independent of its current working directory.
> This guard prevents API and worker processes from silently creating separate
> workspaces.

## User guide

### Standard workflow

1. Pick or create a project, then open **New session**.
2. Upload one or more CSV files, optionally add **business context**, and choose
   the LLM mode for this run.
3. Start the analysis. Open the draggable Activity button to see the stage
   stepper (Profiling, Quality, Charts, Analysis, Starter questions), and switch
   to the separate Event log section for the raw stream. **Cancel** stops the
   run at the next stage boundary and keeps whatever artifacts were already
   written.
4. Read the results: **Data Map** for run health and per-dataset summaries,
   **Quality** for issues by severity, **Profiles & Charts** for column profiles
   and charts, **Relationships** for the dataset graph.
5. Investigate: approve and run a candidate in **Questions**, collect conclusions
   in **Findings**, organise leads on the **Board**, or ask constrained
   read-only questions in **Chat**.
6. Audit: **Report** for the narrative with its claim ledger, **Artifacts** for
   the raw typed artifacts behind every number.

Every page is addressable — the URL carries project, run, dataset, and table
offset, so refreshing or sharing a link restores the same view. Past runs are in
the left session rail; runs derived from another run (question batches, skill
replays) are hidden from that list but remain reachable by direct link.

### Pages

| Page | Route | Purpose |
|---|---|---|
| Projects | `/projects` | Create a project, or open an existing one. |
| New session | `/projects/:id/new-session` | Upload CSVs, set business context and LLM mode, start the analysis. |
| Data Map | `…/sessions/:id/data-map` | Run health, key indicators, and per-dataset summaries. |
| Table Preview | `…/sessions/:id/table/:datasetId` | Server-paged rows with column types; the offset lives in the URL. |
| Quality | `…/sessions/:id/quality` | Data-quality issues by severity, filterable per dataset. |
| Profiles & Charts | `…/sessions/:id/profiles` | Column profiles and the deterministic chart canvas. |
| Relationships | `…/sessions/:id/relationships` | Dataset graph (join edges follow the relationships API). |
| Cleaning | `…/sessions/:id/cleaning` | Preview a cleaning recipe, approve it, and fork an analysis of the cleaned version. |
| Questions | `…/sessions/:id/questions` | Review candidates, approve one, and execute it as a tracked job. |
| Findings | `…/sessions/:id/findings` | Evidence-backed conclusions with freshness and source links. |
| Knowledge | `…/sessions/:id/semantic` | Edit field meanings, confirm join whitelist entries, accept or reject proposals. |
| Report | `…/sessions/:id/report` | The narrative with its claim ledger and validation state. |
| Artifacts | `…/sessions/:id/artifacts` | Browse the typed artifacts behind every number. |
| Chat | `…/sessions/:id/chat` | Constrained read-only questions; analysis plans require approval before they run. |
| Board | `…/sessions/:id/board` | Organise leads as cards; drag with the mouse or move them with the keyboard. |
| Compare | `/projects/:projectId/compare` | Put two runs of the same project side by side. `…/sessions/:id/compare` redirects here with the run as `?left=`. |
| Skills | `…/sessions/:id/skills` | Browse saved skills and seed templates, and replay one against this run. |

Pre-cleaning profiles, deep-analysis tables, and ML model cards are currently
reachable through **Artifacts** rather than dedicated pages.

### Reports and exports

The Report page shows the narrative, claim ledger, validator results, and limitations together. HTML exports are self-contained for archiving or sharing. PDF export is optional:

```bash
uv sync --extra pdf
brew install pango  # macOS
```

Once installed, download PDF from the **Report** page. If system dependencies are unavailable, HTML export remains available and the interface explains the missing requirement.

## LLM and privacy settings

The provider registry (`core/provider_registry.py`) ships 18 providers: OpenAI,
Anthropic, Gemini, Azure OpenAI, DeepSeek, Qwen, Moonshot, Zhipu, xAI, Mistral,
OpenRouter, Together, Groq, Fireworks, Ollama, LM Studio, a generic
`openai_compatible` entry, and `offline`.

**The React workbench has no settings dialog: the provider, model, and key come
from the process environment or `.env`.** What the UI does expose per run is the
choice between the configured provider and `offline`:

| Mode | Use | Required values |
|---|---|---|
| `offline` | Deterministic local fallback with no external model request. Chat answers are canned. | None |
| Configured provider | Whatever `EDA_LLM_PROVIDER` names in the environment. | API key and model (plus base URL for local/compatible endpoints) |

Restart the server after changing `.env` — the settings are read at process
start, and running workers inherit them.

Common environment variables are listed below; see [`.env.example`](.env.example) for the complete example.

```dotenv
EDA_LLM_PROVIDER=offline
EDA_LLM_API_KEY=
EDA_LLM_BASE_URL=
EDA_LLM_MODEL=
EDA_LLM_TEMPERATURE=0.2
EDA_LLM_MAX_TOKENS=6000
EDA_LLM_TIMEOUT_SECONDS=180
EDA_LLM_STRUCTURED_OUTPUT_MODE=auto
```

How much data may be sent to the LLM is governed by the payload policy:

- `schema_only`: structural metadata only; highest privacy and lowest cost.
- `schema+aggregates`: the default; adds aggregates to improve analytical quality.
- `schema+aggregates+sample`: also sends sample rows; highest information exposure and cost.

Runs started from the React app use the default (`schema+aggregates`); the
policy is not yet exposed as a per-run option there. Choose it according to your
data classification, vendor agreement, and organization policy. API keys stay in
the server process — they are never sent to the browser or written into project
files.

## Optional: open-ended Python analysis

Open-ended Python analysis is used only when a matching Chat request requires it. Deterministic Auto EDA, reports, and read-only SQL do not depend on it. For the recommended isolation boundary, start Docker Desktop and build the image once:

```bash
docker build -t eda-agent-sandbox:py312 docker/eda-agent-sandbox
```

`EDA_SANDBOX_BACKEND=auto` and `docker` both select the Docker-only backend.
There is no Seatbelt or host-subprocess fallback. Before each CodeAgent run, the
broker proves that the Linux container has seccomp, a private cgroup namespace,
no capabilities, `no-new-privileges`, a read-only root filesystem, and no
network. If any check fails, open-ended code is rejected.

Only the requested dataset files are copied into a private per-execution staging
directory and mounted read-only under `/work/inputs`; original uploads, the
workspace, source tree, `.env`, credentials, and Docker socket are never mounted.
Only `/work` and a bounded `/tmp` tmpfs are writable. Outputs are size/count
limited and sealed with SHA-256 manifests after execution.

For deployments that must refuse to start unless this runtime proof succeeds,
set `EDA_SANDBOX_REQUIRED=1`. You can run the same operational check directly:

```bash
uv run python scripts/check_sandbox.py
```

Set `EDA_SANDBOX_DOCKER_IMAGE` to use a custom prebuilt image. See the
[sandbox documentation](docker/eda-agent-sandbox/README.md).

## Artifacts, traceability, and safety boundaries

### Local storage

The default workspace is `eda_platform/workspace/projects/<project_id>/`:

```text
uploads/<dataset_id>/v1/                 # preserved source-data versions
sessions/<session_id>/manifest.json              # session manifest and code version
sessions/<session_id>/trace.jsonl                # step and call trace events
sessions/<session_id>/artifacts/*.json           # typed analysis artifacts
sessions/<session_id>/report/report.md           # Markdown report
sessions/<session_id>/report/report.html         # self-contained HTML report
```

The platform is local-first: uploaded files and generated artifacts remain in the local workspace. Data is sent to a model provider only when a live LLM is enabled and the selected payload policy permits the relevant schema, aggregate, or sample data.

### How conclusions are kept reviewable

- Every LLM-supplied tool parameter is checked against column, range, and enum constraints.
- The Chat SQL path is read-only DuckDB and saves result artifacts for inspection.
- Reports use evidence packs, claim ledgers, and a hard validator; their validation state and limitations appear with the report.
- Each run records its `code_version`, tool calls, model token usage, estimated cost, and failures for reproduction and troubleshooting.

These controls reduce the risk of unsupported conclusions; they do not replace business review, data-owner confirmation, or human judgment in production decisions.

## Development and quality checks

Run the complete local suite:

```bash
scripts/ci_local.sh
```

`ci_local.sh` covers the Python suite, the frontend gates (typecheck, Vitest,
production build), and an OpenAPI drift check. Browser E2E is **not** part of it
— it starts a real server and worker, so it is run separately (see below).

Or run checks individually:

```bash
UV_CACHE_DIR=.uv-cache uv run ruff check .
UV_CACHE_DIR=.uv-cache uv run pyright
UV_CACHE_DIR=.uv-cache uv run pytest
npm run typecheck --prefix apps/web
npm test --prefix apps/web          # Vitest component/state tests
npm run build --prefix apps/web
git diff --check
```

### Browser end-to-end tests

Playwright drives a real Chromium against a real server: `playwright.config.ts`
starts `scripts/serve.py` on a throwaway port pointed at a temporary workspace,
so the suite never touches your own runs. Install the browser once:

```bash
npm run e2e:install --prefix apps/web   # == npx playwright install chromium
```

Then, with a current production build in place:

```bash
npm run build --prefix apps/web
npm run e2e --prefix apps/web
```

The suite covers upload → offline run → live progress → Data Map, table deep
links with an offset, report rendering, the one-shot cleaning approval, and
keyboard-only board reordering.

### API contract

`api/openapi.json` is generated, never hand-edited:

```bash
uv run python scripts/export_openapi.py     # re-export after changing any endpoint
npm run gen:api --prefix apps/web           # regenerate the TypeScript types
```

Without `uv`, replace `uv run` with `.venv/bin/python -m`, for example:

```bash
.venv/bin/python -m pytest
```

The repository also includes offline demos and evaluation scripts:

```bash
uv run python scripts/demo_j3.py
uv run python scripts/evaluate_workflow.py \
  --case eda_platform/tests/evals/workflow_quality/cases/semantic_guardrails.json \
  --input-dir eda_platform/tests/evals/workflow_quality/data \
  --repeat 3
```


## Related documentation

- [Docker sandbox documentation](docker/eda-agent-sandbox/README.md)
