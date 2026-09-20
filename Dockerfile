# Multi-stage build: dependencies resolve once, the runtime image stays small.
# Package: optcg_forecast

FROM python:3.12-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.16 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependency layer first, so it caches independently of source edits.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev --extra serve

COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra serve


FROM python:3.12-slim-bookworm AS runtime

# Run as a non-root user.
RUN groupadd --gid 1000 app \
 && useradd --uid 1000 --gid app --shell /bin/bash --create-home app

WORKDIR /app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/src /app/src

# The registry is not baked in: the image is the same for every model, and the champion it
# serves is whatever is mounted or fetched at deploy time. Baking a model into an image makes
# "which version is live" a question about image tags instead of about the registry.
ENV MODEL_ROOT=/app/data/models

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER app

# Cloud Run supplies $PORT and expects the container to listen on it.
ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "python -m optcg_forecast.inference.serve --host 0.0.0.0 --port ${PORT}"]
