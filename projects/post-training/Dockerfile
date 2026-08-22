ARG UV_VERSION
ARG PYTHON_VERSION

FROM ghcr.io/astral-sh/uv:${UV_VERSION}-python${PYTHON_VERSION}-trixie-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Copy workspace root manifest and lock, then every package manifest and source
COPY pyproject.toml uv.lock ./
COPY projects/obs-common/pyproject.toml projects/obs-common/
COPY projects/obs-common/src projects/obs-common/src/
COPY projects/ml-common/pyproject.toml projects/ml-common/
COPY projects/ml-common/src projects/ml-common/src/
COPY projects/post-training/pyproject.toml projects/post-training/
COPY projects/post-training/src projects/post-training/src/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --package post-training

FROM python:${PYTHON_VERSION}-slim-trixie

RUN groupadd --gid 1000 appuser && useradd --uid 1000 --gid appuser --no-create-home appuser

WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

USER appuser

ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["python", "-m", "post_training.main"]
