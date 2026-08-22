# Claude Code Instructions

## Terraform

Every GCP resource must live inside a module under `iac/modules/`. Never write `resource` blocks directly in `iac/main.tf` — root `main.tf` is for module calls, data sources, and provider-level lookups only.

When adding a new GCP service or resource type, create a dedicated module for it first, then call that module from `main.tf`.

## Python

Follow PEP 257 for docstrings. Every public function, method, and class gets a docstring. Use a one-liner when the signature is self-explanatory; add a short multi-line summary only when behaviour is non-obvious. Do not document every parameter — rely on type annotations instead.

```python
def load_events(table_id: str, limit: int = 100) -> list[dict]:
    """Return the most recent CDC events from BigQuery."""
```

## Architecture

Always follow README.md and the files in `docs/` when building. Each doc file owns a specific topic:

| File | Topic |
|------|-------|
| `docs/setup.md` | Prerequisites & setup |
| `docs/architecture.md` | Monorepo structure & directory layout |
| `docs/data-lifecycle.md` | CDC simulation & feature engineering |
| `docs/orchestration.md` | Cloud Workflows DAG & retraining loop |
| `docs/model-development.md` | Model choice, HPO strategy, evaluation criteria, DS local workflow |
| `docs/ml-infrastructure.md` | Pipeline stages, containers, experiment tracking, serving validation |
| `docs/feature-exploration.md` | Feature discovery, selection notebooks & trigger signals |
| `docs/observability.md` | Drift detection & automated remediation |
| `docs/iac.md` | Terraform modules & IaC design |
| `docs/cicd.md` | Cloud Build pipeline steps |

If your changes introduce new components, patterns, data flows, tools, or directory structure, update the relevant `docs/` file. Update `README.md` only if the overview or tech stack changes.
