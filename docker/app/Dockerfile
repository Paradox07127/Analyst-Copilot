# syntax=docker/dockerfile:1
# Production image: React static build + FastAPI served from one port.
# Build from the repo root:  docker build -f docker/app/Dockerfile .

FROM node:22-bookworm-slim AS web-build
WORKDIR /build/apps/web
COPY apps/web/package.json apps/web/package-lock.json ./
RUN npm ci
COPY apps/web/ ./
RUN npm run build

FROM python:3.12-slim-bookworm AS runtime
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /usr/local/bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev
COPY eda_platform/src ./eda_platform/src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev
COPY scripts/serve.py ./scripts/serve.py
COPY --from=web-build /build/apps/web/dist ./apps/web/dist

RUN useradd --create-home --uid 10001 eda \
    && mkdir -p /workspace \
    && chown eda:eda /workspace
USER eda

# Note: no sandbox backend is available inside this container, so open-ended
# code execution is rejected by the platform; core EDA runs are unaffected.
ENV EDA_WORKSPACE=/workspace
VOLUME ["/workspace"]
EXPOSE 8000

# 0.0.0.0 is container-internal; compose publishes to host loopback only.
CMD ["uv", "run", "--no-sync", "python", "scripts/serve.py", "--host", "0.0.0.0", "--port", "8000"]
