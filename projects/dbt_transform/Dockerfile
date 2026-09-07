ARG UV_VERSION
ARG PYTHON_VERSION

FROM ghcr.io/astral-sh/uv:${UV_VERSION}-python${PYTHON_VERSION}-trixie-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

FROM python:${PYTHON_VERSION}-slim-trixie

RUN groupadd --gid 1000 appuser && useradd --uid 1000 --gid appuser --no-create-home appuser

WORKDIR /app

# dbt writes logs/ and target/ under the working directory at runtime, so appuser
# needs write access to /app itself, not just the files COPY --chown places in it.
RUN chown appuser:appuser /app

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --chown=appuser:appuser dbt_project.yml profiles.yml ./
COPY --chown=appuser:appuser models ./models
COPY --chown=appuser:appuser macros ./macros

USER appuser

ENV PATH="/app/.venv/bin:$PATH"

# `build`, not `run`: dbt build interleaves models and their declared tests in DAG order, so
# a model whose contract is violated fails instead of silently feeding everything downstream.
# The schema.yml tests existed but were never executed under `dbt run` — customer_features'
# grain, event_id uniqueness and the membership_tier domain were documented guarantees that
# nothing enforced. A failure here halts the orchestrator before batch prediction rather than
# after it, which is strictly earlier than the drift monitor's data-quality gate can catch the
# same class of problem (see docs/observability.md).
ENTRYPOINT ["dbt", "build", "--profiles-dir", ".", "--no-use-colors"]
