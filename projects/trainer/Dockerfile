ARG UV_VERSION
ARG PYTHON_VERSION

FROM ghcr.io/astral-sh/uv:${UV_VERSION}-python${PYTHON_VERSION}-trixie-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Copy workspace root manifest and lock, then all package manifests and sources
COPY pyproject.toml uv.lock ./
COPY projects/obs_common/pyproject.toml projects/obs_common/
COPY projects/obs_common/src projects/obs_common/src/
COPY projects/data_contracts/pyproject.toml projects/data_contracts/
COPY projects/data_contracts/src projects/data_contracts/src/
COPY projects/ml_common/pyproject.toml projects/ml_common/
COPY projects/ml_common/src projects/ml_common/src/
COPY projects/modeling/pyproject.toml projects/modeling/
COPY projects/modeling/src projects/modeling/src/
COPY projects/trainer/pyproject.toml projects/trainer/
COPY projects/trainer/src projects/trainer/src/

# Cache mount persists downloaded packages across builds without baking them into the image layer
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --package trainer

FROM python:${PYTHON_VERSION}-slim-trixie

RUN groupadd --gid 1000 appuser && useradd --uid 1000 --gid appuser --no-create-home appuser

WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

USER appuser

ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["python", "-m", "trainer.main"]
