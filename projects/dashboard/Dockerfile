ARG UV_VERSION
ARG PYTHON_VERSION

FROM ghcr.io/astral-sh/uv:${UV_VERSION}-python${PYTHON_VERSION}-trixie-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Copy workspace root manifest and lock, then both package manifests and sources.
# Only obs_common is copied in: the dashboard reads a BigQuery table and renders it, so
# ml_common (and through it xgboost and scikit-learn) has no business in this image. Per
# architecture.md, this COPY set is the enforcement point for that boundary.
COPY pyproject.toml uv.lock ./
COPY projects/data_contracts/pyproject.toml projects/data_contracts/
COPY projects/data_contracts/src projects/data_contracts/src/
COPY projects/obs_common/pyproject.toml projects/obs_common/
COPY projects/obs_common/src projects/obs_common/src/
COPY projects/dashboard/pyproject.toml projects/dashboard/
COPY projects/dashboard/src projects/dashboard/src/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --package dashboard

FROM python:${PYTHON_VERSION}-slim-trixie

RUN groupadd --gid 1000 appuser && useradd --uid 1000 --gid appuser --no-create-home appuser

WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --chown=appuser:appuser projects/dashboard/.streamlit ./.streamlit

USER appuser

ENV PATH="/app/.venv/bin:$PATH"

# Streamlit writes ~/.streamlit at first run. appuser is created with --no-create-home, so
# without a writable HOME the server fails at startup rather than falling back silently.
ENV HOME=/tmp

EXPOSE 8080

ENTRYPOINT ["python", "-m", "dashboard.serve"]
