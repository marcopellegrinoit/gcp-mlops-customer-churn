# Feature Exploration & Selection

Feature selection is not a one-time decision baked into the initial build. As customer behaviour shifts over time, the signals that predicted churn at launch may become weaker and new signals may emerge. This document describes when and how the feature set is re-evaluated.

---

## Role of the Data Scientist

Feature exploration is deliberately kept outside the automated pipeline. A data scientist runs this notebook interactively in their own environment — Vertex AI Workbench, a local Jupyter server, or any equivalent — and the work never touches the production codebase until they are ready to promote.

The workflow is exploratory and iterative: pull the feature table, inspect distributions, check correlations, fit a quick model, inspect SHAP values, drop a column, refit, compare metrics, repeat. The DS treats the `features.customer_features` BigQuery table as a read-only source and accumulates findings in the notebook without modifying any dbt SQL or Terraform until the feature set has been agreed upon.

The feature set is considered **stable** until a trigger fires (see below). The DS does not re-run this notebook on a schedule. Between trigger events, what the system continuously monitors is **feature drift** — whether the statistical distribution of already-selected features shifts relative to the training baseline — not whether the feature set itself is still optimal. Drift detection triggers automated retraining on the same features, not feature re-selection. Re-selection is a human decision made only when the evidence suggests the current feature set is structurally wrong, not merely trained on stale data.

Once the DS is satisfied with a new feature set, they propagate the change through the stack (see *Output & Promotion Path*) and open a pull request. The next scheduled training run picks up the updated features automatically.

---

## Two Phases, Two Notebooks

Feature exploration is split into two distinct notebooks because they answer different questions, query different tables, and produce different outputs. They can be run independently — a trigger signal may call for only one of them.

### Phase 1 — Raw Data Exploration (`notebooks/raw_exploration.ipynb`)

This notebook is the upstream phase. It queries `raw.activity_cdc` directly and asks: **are there signals in the raw events that the current dbt models are not capturing?**

Without this phase, the DS can only prune features that already exist. They cannot discover that, for example, the sequence of event types in the 14 days before a churn signal is more predictive than any rolling average currently engineered. Raw exploration is how the DS identifies new feature candidates and communicates them to the data engineer as concrete engineering requests ("add a feature counting `membership_cancelled` events in the last 14 days").

The notebook covers:

| Step | What it does |
|---|---|
| **Event distribution** | Counts and visualises event type frequencies; highlights rare event types that may carry disproportionate churn signal |
| **Raw field correlations with churn** | Computes point-biserial correlation and mutual information between each raw field and the `churned` label at the event level |
| **Cohort breakdowns** | Cross-tabulates churn rate by `membership_tier`, `region`, and `channel` to surface interaction effects not captured by scalar features |
| **Time-series patterns** | Plots rolling churn rate against event volume over time to identify seasonal or trending patterns |
| **Gap analysis** | Compares raw fields available in `activity_cdc` against the columns currently engineered in dbt — any field with churn signal that has no corresponding engineered feature is a candidate for a new dbt column |
| **Output** | A list of candidate new features with justification, formatted as a concrete engineering request for the data engineer |

### Phase 2 — Feature Selection (`notebooks/feature_selection.ipynb`)

This notebook is the downstream phase. It queries `features.customer_features` and asks: **of the features that already exist in the matrix, which ones are earning their place?**

| Step | What it does |
|---|---|
| **EDA** | Reports class balance, per-feature distributions, and missing value rates with their missingness mechanism (MCAR / MAR / MNAR) |
| **Correlation analysis** | Generates a numeric correlation heatmap; flags pairs with \|r\| > 0.7 as redundancy candidates |
| **Baseline model** | Trains a lightweight XGBoost classifier on all current features with class-imbalance weighting; reports PR-AUC as a sanity check |
| **SHAP importance** | Computes global mean \|SHAP\| values and a beeswarm direction plot for every feature |
| **Selection decision** | Drops features whose mean \|SHAP\| falls below 5% of the maximum importance, cross-referenced against high-correlation pairs |
| **Production comparison** | Compares the current SHAP ranking against the importances logged in Vertex AI Experiments from recent production training runs to quantify how much relevance has shifted |
| **Output** | Prints the recommended feature list and the features to remove, with the dbt and `ml_common` files that must be updated |

---

## Trigger Signals

Two conditions tell the team it's time to revisit the feature set rather than retrain again. Both are computed inside the pipeline and reported by the terminal `notify` stage, which logs them to Cloud Logging against the pipeline run — no email is sent, because `notify` is a deliberate dummy sink (see [observability.md](observability.md)).

### 1 — Consecutive Challenger Rejections

Every time a challenger model fails the champion/challenger gate, `register_or_reject` increments a rejection counter persisted as a small JSON state file in GCS (see [ml-infrastructure.md](ml-infrastructure.md#consecutive-rejection-counter)). When the counter reaches **3 consecutive rejections** (`MAX_CONSECUTIVE_REJECTIONS`), it sets `feature_review_alert: true` on the outcome payload and `notify` escalates its log line from a plain "challenger REJECTED" to "feature review recommended", reporting:

- The number of consecutive failures
- The challenger and champion PR-AUC/F1 metrics from this run

The counter resets to zero whenever a challenger is promoted.

The rationale: if the model keeps failing to improve despite being retrained on fresh data each time, the problem is most likely the feature set rather than the training data volume.

### 2 — SHAP Importance Rank Drift

On every training run the training container logs per-feature mean |SHAP| values to Vertex AI Experiments via the MLflow integration. When a new model is registered as champion, its SHAP importance ranking is stored as the **baseline** in Vertex AI Model Registry metadata.

On every subsequent training run, the `evaluate` stage computes the **Spearman rank correlation** between the current run's SHAP ranking and the stored baseline (`ml_common.evaluate._spearman_rank_correlation`). If this correlation drops below **0.75**, feature relevance has shifted significantly — a signal to open the exploration notebook. The score travels through `register_or_reject` onto the outcome payload as `shap_rank_correlation` and is logged by `notify`.

---

## Output & Promotion Path

Once the notebook has produced a recommended feature list, the data scientist must propagate it through the stack manually:

1. **`projects/ml-common/src/ml_common/preprocess.py`** — update `CATEGORICAL_COLS`/`DROP_COLS` if a categorical or metadata column changed. There is no separate `FEATURE_COLUMNS` manifest: the feature set is implicit in the `features.customer_features` BigQuery schema plus `feature_names()`, and both the training and serving containers consume it from there
2. **`projects/dbt_transform/models/`** — add or remove columns from `int_customer_aggregates.sql` and `customer_features.sql`
3. **`iac/config/bigquery.yaml`** — add or remove column definitions from the `features.customer_features` table schema
4. **`docs/data-lifecycle.md`** — update the feature group and full schema tables

After these changes are merged, the next scheduled training run will pick up the new feature set automatically.

---

## What Does Not Change Automatically

Feature selection is intentionally not automated. The notebook exists to give a data scientist visibility into *why* a feature is being added or dropped — correlation structure, direction of SHAP effects, and comparison against the production baseline — before any change reaches the dbt models or the serving container. Automating this decision would risk silently removing features that are temporarily unimportant due to a data quality incident but structurally meaningful.
