#!/usr/bin/env bash
set -euo pipefail

uv run ruff check
uv run pyright
if command -v docker >/dev/null 2>&1 && docker info --format '{{.ServerVersion}}' >/dev/null 2>&1; then
  docker build -t eda-agent-sandbox:py312 docker/eda-agent-sandbox
else
  printf '%s\n' "Docker unavailable; Docker runtime tests will be skipped."
fi
uv run pytest

# OpenAPI is generated from the FastAPI app; a stale api/openapi.json means
# the frontend types were generated against a contract that no longer exists.
# Re-export into a scratch copy and compare — equivalent to "git diff is empty",
# but it also catches the case where the file is not committed yet.
openapi_before="$(mktemp)"
trap 'rm -f "$openapi_before"' EXIT
cp api/openapi.json "$openapi_before" 2>/dev/null || : > "$openapi_before"
uv run python scripts/export_openapi.py >/dev/null
if ! diff -q "$openapi_before" api/openapi.json >/dev/null 2>&1; then
  printf '%s\n' "OpenAPI drift: api/openapi.json was stale and has been regenerated." >&2
  printf '%s\n' "Re-run 'npm run gen:api --prefix apps/web' and commit both files." >&2
  diff -u "$openapi_before" api/openapi.json | head -60 >&2 || :
  exit 1
fi

# Frontend gates. Browser E2E (npm run e2e) is deliberately not here: it starts a
# real server and worker, so it is run on its own.
if command -v npm >/dev/null 2>&1; then
  [ -d apps/web/node_modules ] || npm ci --prefix apps/web
  npm run typecheck --prefix apps/web
  npm test --prefix apps/web
  npm run build --prefix apps/web
else
  printf '%s\n' "npm unavailable; frontend typecheck/tests/build skipped." >&2
fi

uv run python scripts/evaluate_workflow.py \
  --case eda_platform/tests/evals/workflow_quality/cases/semantic_guardrails.json \
  --input-dir eda_platform/tests/evals/workflow_quality/data \
  --repeat 3
uv run python scripts/evaluate_workflow.py \
  --case eda_platform/tests/evals/workflow_quality/cases/contract_abstention.json \
  --input-dir eda_platform/tests/evals/workflow_quality/data \
  --repeat 3 \
  --baseline eda_platform/tests/evals/workflow_quality/baselines/contract_abstention.json
.venv/bin/python scripts/demo_j1.py
.venv/bin/python scripts/demo_j3.py
