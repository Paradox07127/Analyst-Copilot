#!/usr/bin/env bash
# Test entry point. Default is the narrow, fast path; the full suite is opt-in.
#
# Why not put `-n auto` in pyproject's addopts: xdist pays ~18 worker startups
# per invocation. Measured on this repo (M5 Pro, 18 cores):
#
#     full suite   serial 6m07s   ->  -n 16  52s     (7.1x faster)
#     one file     serial 19.4s   ->  -n auto 21.3s  (10% SLOWER)
#
# Since most runs during development are one file, a global -n would tax the
# common case to speed up the rare one. So parallelism lives here, per mode.
#
#   ./scripts/test.sh              changed frontend tests + last-failed backend
#   ./scripts/test.sh all          everything, parallel  (pre-commit gate)
#   ./scripts/test.sh be [args]    backend, serial, args passed through
#   ./scripts/test.sh be-all       backend only, parallel
#   ./scripts/test.sh fe [args]    frontend, args passed through
#   ./scripts/test.sh lf           backend: only what failed last time
set -euo pipefail
cd "$(dirname "$0")/.."

# loadfile, not the default loadscope: same-file tests share module-level
# fixtures and a tmp workspace, so keeping a file on one worker avoids
# re-running expensive setup and cross-worker interference.
XDIST=(-n auto --dist loadfile)

case "${1:-changed}" in
  all)
    echo "== backend (parallel) =="
    uv run pytest eda_platform/tests/unit "${XDIST[@]}" -q
    echo "== frontend =="
    (cd apps/web && npx vitest run --reporter=basic && npm run typecheck)
    ;;
  be-all)
    uv run pytest eda_platform/tests/unit "${XDIST[@]}" -q
    ;;
  be)
    shift
    uv run pytest "${@:-eda_platform/tests/unit}" -q
    ;;
  lf)
    # --lf alone exits 5 ("no tests ran") when the last run was clean.
    uv run pytest eda_platform/tests/unit --lf -q || [ $? -eq 5 ]
    ;;
  fe)
    shift
    (cd apps/web && npx vitest run "$@" --reporter=basic)
    ;;
  changed)
    # Frontend --changed walks the real module graph, so it is accurate.
    # The backend has no equivalent: only 19% of src modules have a
    # test_<module>.py, so a name-based guess would silently skip coverage.
    # --lf is the honest substitute; use `be <path>` when you know the area.
    echo "== frontend (changed vs HEAD) =="
    (cd apps/web && npx vitest run --changed HEAD --reporter=basic) || true
    echo "== backend (last failed) =="
    uv run pytest eda_platform/tests/unit --lf -q || [ $? -eq 5 ]
    echo
    echo "Narrow run only. Before committing: ./scripts/test.sh all"
    ;;
  *)
    echo "usage: $0 [changed|all|be|be-all|fe|lf]" >&2; exit 2 ;;
esac
