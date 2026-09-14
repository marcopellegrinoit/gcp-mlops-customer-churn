# Model Development

This document covers everything a data scientist needs to understand and iterate on the churn prediction model: framework choice, feature set, local development workflow, HPO strategy, and evaluation criteria. For how the model runs inside the production pipeline, see [ml-infrastructure.md](ml-infrastructure.md).

---

## Model & Framework

The system trains a gradient-boosted decision tree classifier using **XGBoost**. XGBoost is selected for its strong performance on structured tabular data, native support for class-imbalance weighting (`scale_pos_weight`), and compatibility with Vertex AI's custom training infrastructure.

The trained model artifact is serialized and registered in **Vertex AI Model Registry**, providing versioned lineage back to the exact BigQuery feature snapshot and training run that produced it.

---

## Input Feature Matrix

The training pipeline reads directly from `features.customer_features` in BigQuery. The feature set covers four behavioral dimensions:

| Dimension | Example features |
|---|---|
| Transaction behaviour | `avg_monthly_transaction_30d`, `transaction_trend_slope`, `days_since_last_transaction` |
| Payment health | `payment_failure_rate_30d`, `payment_attempts_last_30d`, `payment_failure_rate_90d` |
| Engagement | `avg_engagement_score_30d`, `campaign_participation_rate`, `preferred_channel` |
| Membership tenure | `member_since_days`, `membership_tier`, `renewal_count` |

**Target label**: `churned` (boolean) — whether the customer's most recent recorded event was a churn event.

### Why no sklearn transformation pipeline

Feature transformation (rolling-window aggregation, rate calculations, null-handling decisions) happens entirely upstream in dbt/BigQuery, not in an sklearn `Pipeline` at training time. `prepare_features` in `ml-common` only drops `customer_id`, casts three columns to pandas `category` dtype, and splits off the target — no scaling, no imputation, no encoding.

This is intentional, not a gap:

- **Nulls are preserved as signal.** Some features (e.g. `avg_engagement_score_30d` for offline-only customers, `days_since_last_successful_payment` when no payment has ever succeeded) are null by design, reflecting MAR/MNAR patterns in the underlying behaviour. XGBoost's native missing-value handling splits on missingness directly. An sklearn pipeline would force imputation before fitting, destroying that signal.
- **Tree-based models don't need scaling.** XGBoost is invariant to monotonic feature scaling, so `StandardScaler`/`MinMaxScaler` would add no value.
- **One feature table, no train/serve skew.** Both training (`prepare_features`) and serving (`select_inference_features`) read from the same `features.customer_features` BigQuery table and apply the same lightweight column handling, so there's no risk of a transformer fit on training statistics drifting from production feature distributions.

An sklearn pipeline would be the right call for a linear/distance-based model (which needs scaling) or if transformation logic needed to live in the modelling codebase rather than the data warehouse. Neither applies here — the dbt mart layer is the system of record for feature engineering, and XGBoost consumes its output directly.

---

## Data Scientist Local Development

The `modeling` and `ml-common` packages have **zero GCP dependencies**. A data scientist can install them and iterate locally without cloud credentials, a running pipeline, or a container build.

### Setup

```bash
# from the repo root — installs modeling and its pure ML deps into a local venv
uv sync --package modeling --extra dev
```

### Iterating in notebooks

The primary DS workflow is a Jupyter notebook in `notebooks/` that imports directly from `modeling` and `ml_common`. Data is pulled from BigQuery using [Application Default Credentials](https://cloud.google.com/docs/authentication/application-default-credentials) (a one-time `gcloud auth application-default login`):

```python
from google.cloud import bigquery
from ml_common.preprocess import prepare_features
from modeling.config import get_settings as get_modeling_settings
from modeling.train import train_model

modeling_cfg = get_modeling_settings()

df = bigquery.Client().query("""
    SELECT * EXCEPT (feature_computed_at, latest_event_ts)
    FROM `project.features.customer_features`
    WHERE snapshot_date = '2025-01-01'
    LIMIT 5000
""").to_dataframe()

X, y = prepare_features(df)
model, feat_names, shap = train_model(X, y, {"max_depth": 4, "learning_rate": 0.1, ...}, modeling_cfg.fixed_params)
```

No Dockerfile, no KFP pipeline, no orchestration. The notebook talks directly to BQ and calls the same functions that the pipeline will run in production — including the same settings objects, which default to the deployed values and need no environment to import. To try a different policy for one session, set the corresponding variable before starting Jupyter (`MODELING_HPO_N_TRIALS=20`, `ML_TARGET_RECALL=0.9`, or `MODELING_HPO_SEARCH_SPACE` as JSON) rather than editing the call sites.

### Unit tests

All tests in `projects/modeling/tests/` and `projects/ml_common/tests/` use synthetic DataFrames and run with no GCP credentials:

```bash
uv run --package modeling pytest -m "not integration"
uv run --package ml-common pytest -m "not integration"
```

### What the DS owns

All ML decisions live in `projects/modeling/` and `projects/ml_common/`:

| File | Package | What the DS controls |
|---|---|---|
| `config.py` | `modeling` | Search space (typed: `IntParam`/`FloatParam`), fixed XGBoost params, HPO budget |
| `config.py` | `ml-common` | Promotion gate deltas, target recall, split, data-quality tolerances |
| `preprocess.py` | `ml-common` | Feature encoding, column drops, categorical handling |
| `train.py` | `modeling` | Model fit loop, SHAP computation, out-of-fold threshold selection |
| `hpo.py` | `modeling` | Optuna objective function, CV fold logic |
| `evaluate.py` | `ml-common` | Metrics (at an already-selected threshold), SHAP rank correlation; also owns the pure `select_threshold` function `train.py` calls |

Both `config.py` files are `pydantic-settings` classes: the values in them are the *defaults*, and each is overridable per deployment through an `ML_`- or `MODELING_`-prefixed environment variable (see [architecture.md](architecture.md#data-contracts--configuration)). Changing a default is a code change and goes through review; changing what production runs with today is a Terraform apply against `iac/config/triggers.yaml`. Validation lives with the setting — an inverted search-space range or a target recall above 1 is refused at startup rather than part-way through a paid HPO run.

When the DS merges a change to `modeling` or `ml-common`, the MLE's `trainer` picks it up automatically (same workspace lock file). No coordination is required unless the function signatures change.

---

## Validation Architecture

Because customer churn datasets are heavily skewed towards non-churning profiles, standard random evaluation splits introduce dangerous evaluation biases. The training service implements a rigorous **Stratified K-Fold cross-validation** strategy. This splits the feature matrix into proportional subsets, ensuring that the critical target classification balance is preserved identically across every validation fold.

Cross-validation results are aggregated and logged to Vertex AI Experiments before the final model is trained on the full dataset and submitted for champion/challenger evaluation.

---

## Hyperparameter Optimization (HPO)

Model parameters are systematically optimized via an automated **Bayesian search** pattern utilizing a Tree-structured Parzen Estimator (TPE). The search explores the following parameter space:

| Parameter | Search range | Effect |
|---|---|---|
| `max_depth` | [3, 10] | Controls tree complexity and overfitting risk |
| `learning_rate` | [0.01, 0.3] | Step size for gradient descent |
| `subsample` | [0.5, 1.0] | Row subsampling ratio per tree |
| `colsample_bytree` | [0.5, 1.0] | Feature subsampling ratio per tree |
| `n_estimators` | [50, 500] | Number of boosting rounds |

`scale_pos_weight` is not part of the search space. It is computed directly from each training split as `count(negatives) / count(positives)` and passed as a fixed parameter to XGBoost. Because it is a property of the label distribution rather than a modelling choice, searching it would conflate data characteristics with hyperparameter optimisation.

To shield against imbalanced distribution biases, the HPO objective targets **Precision-Recall AUC (PR-AUC)** rather than generic accuracy. PR-AUC is more informative than ROC-AUC when the positive class (churned customers) is rare.

### Why Optuna over random/grid search

Random and grid search are memoryless — every trial is independent. Optuna uses **TPE (Tree-structured Parzen Estimator)**, a surrogate-based Bayesian approach that learns from past trials and focuses sampling on promising regions of the search space. This finds better hyperparameters in fewer trials, which directly reduces Vertex AI Training cost.

Optuna also supports **early pruning**: trials that are clearly underperforming can be killed mid-training before they complete, saving further compute. Random search always runs every candidate to completion.

### Why not Vertex AI Hyperparameter Tuning Jobs (Vizier)

Vertex AI's managed HPO product is built on Vizier, which has two limitations for this pipeline:

- **No choice of search algorithm.** Vizier exposes its own black-box strategy with no TPE or custom pruning logic — you take its search behaviour as given.
- **Per-trial container spin-up.** Each trial provisions a fresh container, so cold-start/provisioning time is fixed overhead on every trial. This pays off when a single trial trains for a long time (e.g. large deep learning models), but for this project's 6-parameter XGBoost search — where a full fit takes seconds to low minutes — the orchestration overhead would dominate the actual training cost.

Running Optuna in-process inside the training container avoids both: no per-trial provisioning, and full control over the sampler and pruning strategy.

### Why not MLflow / Weights & Biases

MLflow and W&B (with Sweeps as its HPO layer) are the most complete answer for production HPO: integrated experiment tracking, a run-comparison UI, and an artifact/model registry on top of the search itself. They were not adopted here, not on technical merit, but because of the operational cost relative to this project's scope:

- **MLflow** requires a continuously hosted tracking server plus a backend store and artifact store — additional infrastructure to provision, secure, and maintain.
- **W&B** is a third-party SaaS dependency, which adds an external service and data-residency consideration outside the GCP-native stack.

This pipeline already uses Vertex AI Experiments for tracking, so the marginal benefit of either tool doesn't justify the added infrastructure footprint. The gap they would have filled — a dashboard/UI for comparing trials — is the main trade-off of going with Optuna alone.

### Why TPE and not a generic ML surrogate model

Surrogate-based HPO works by fitting a cheap model on `(hyperparameters → validation score)` observations, then using an acquisition function to pick the next trial that best balances exploration vs. exploitation:

```
EI(x) = f(μ(x), σ(x))   # Expected Improvement over best seen so far
```

The critical requirement is **calibrated uncertainty** (`σ`). Without it, the acquisition function has no way to distinguish "confidently good region" from "never explored region" and the search degenerates into local hill-climbing.

Standard ML models (neural networks, gradient boosting) do not provide calibrated uncertainty without additional machinery (ensembles, MC dropout). They also require thousands of training points to generalise reliably — HPO budgets of 50–200 trials are far too small for them to learn a useful landscape before overfitting to noise.

Gaussian Processes (GP) give theoretically optimal uncertainty estimates and are used in tools like `scikit-optimize`, but their O(n³) fitting cost becomes a bottleneck as trial count grows, and they struggle with categorical and conditional hyperparameters (e.g. a parameter that is only valid when another parameter takes a specific value).

TPE sidesteps these limitations by modelling the parameter space as two density estimators — one over configurations that performed well, one over the rest — and sampling from the ratio. It is fast, handles categorical and conditional spaces natively, and is empirically competitive with GP for the trial counts typical in ML HPO.

| Approach | Sample efficiency | Handles categoricals | Scales with trials | Uncertainty |
|---|---|---|---|---|
| Random search | Low | Yes | Yes | None |
| Gaussian Process | Highest | Poor | O(n³) | Calibrated |
| Random Forest (SMAC) | High | Yes | Good | Approximate |
| TPE (Optuna) | High | Yes | Good | Implicit |
| Neural surrogate | High (large n) | Yes | Good | Requires ensembles |

TPE is the practical optimum for HPO budgets in the 50–200 trial range with mixed continuous/categorical search spaces, which is exactly the regime this pipeline operates in.

### Scaling HPO beyond in-process Optuna

The current `hpo.py` calls `study.optimize()` with the default in-memory `InMemoryStorage` and no `n_jobs` — all trials run sequentially inside a single trainer container. This is correct for the current model: 100 trials × 5-fold CV on a tabular XGBoost classifier finishes in minutes, well inside the pipeline's SLA.

This stops being sufficient if either the model or the dataset changes shape:

- **Per-trial cost grows** — e.g. a neural network trained over many epochs, or XGBoost over a dataset large enough that a single fit takes minutes rather than seconds.
- **Accelerators are involved** — GPU/TPU time is expensive per unit time, so parallelizing trials shortens paid wall-clock rather than just wall-clock.

In either case, the fix is to parallelize trials across multiple containers rather than running them sequentially in one. This requires:

1. **Shared storage** — replace `InMemoryStorage` with `optuna.storages.RDBStorage` (Cloud SQL Postgres/MySQL) or `JournalStorage` (GCS-backed, no database to provision) so every worker container attaches to the same named study via `load_if_exists=True`.
2. **`TPESampler(constant_liar=True)`** — without it, concurrent workers can't see each other's in-flight trials and the TPE prior degrades toward random search as parallelism increases. This is the detail that makes "parallel trials" and "the Bayesian prior still works" compatible.
3. **A worker fan-out in the KFP pipeline** — replace the single `hpo` container component with a `dsl.ParallelFor` that launches N `hpo_worker` tasks (each running `n_trials / N` against the shared study), followed by an `hpo_finalize` task that loads the completed study and writes `study.best_params` for the downstream `train` stage, unchanged.
4. **A pruner** (`HyperbandPruner` or `MedianPruner`) reporting intermediate validation metrics via `trial.report()` / `should_prune()` — matters more here than at small scale, since killing a bad trial early on an expensive NN training run saves real compute cost.

This is not implemented today. Before reaching for it, consider whether GPU acceleration (`device="cuda"` for XGBoost) or row subsampling solves the per-trial cost problem without adding shared-storage infrastructure — parallelizing trials means provisioning and operating a new stateful dependency (Cloud SQL, under its own `iac/modules/` per this repo's Terraform convention) for what a one-line training-time change might fix.

---

## Promotion Criteria

The new candidate ("Challenger") is registered for production **if and only if** it outperforms the current champion on both metrics by a predefined margin:

| Metric | Minimum improvement to promote |
|---|---|
| PR-AUC | +2 percentage points |
| F1 Score | +1 percentage point |

Both metrics must clear their floor simultaneously — passing one while regressing the other is a rejection. This prevents single-metric gaming (e.g. a model that maximises PR-AUC by always predicting churn).

If the Challenger fails, it is logged to Vertex AI Experiments for analysis but rejected for deployment. After 3 consecutive rejections the pipeline escalates to a feature review alert, since repeated failures suggest the feature set is stale rather than the training data volume. See [feature-exploration.md](feature-exploration.md) for what to do when this alert fires.

---

## Serving Threshold

XGBoost outputs a continuous probability score between 0 and 1. Converting that score to a binary churn/no-churn prediction requires a decision threshold. The system does **not** use the default 0.5 threshold at serving time.

### Why recall takes priority

A missed churner (false negative) is costlier than a false alarm (false positive): a customer who churns silently generates zero future revenue, while a false alarm at worst triggers an unnecessary retention outreach. The asymmetry means recall must be maximised, even at the expense of precision.

### Threshold selection

The threshold is selected during the **train** pipeline stage, not evaluate, and never from the held-out test set. `modeling.train.select_threshold_via_cv` refits the already-chosen hyperparameters across the same `n_folds` stratified folds used by HPO (no search this time, just refitting), collects each fold's held-out (out-of-fold) predictions into one probability array spanning the whole training set, and selects the most precise threshold that still achieves a **target recall of ≥ 0.80** on those out-of-fold predictions via `ml_common.evaluate.select_threshold`. Recall falls as the threshold rises, so the thresholds satisfying the recall floor are a prefix of the curve and the most precise of them is the *highest* one in that prefix — the strictest operating point that still catches the required share of churners, which is what limits the volume of false-alarm interventions. (Earlier revisions of this paragraph, and of `select_threshold`'s own docstring, described it as the *lowest* such threshold, which is the opposite: the lowest threshold clearing any recall target is the one that flags the entire base.)

This out-of-fold approach is deliberate: if the threshold were instead picked from the model's predictions on the test set — the same test set `evaluate` then reports F1 against — the reported F1 would be optimistically biased, since the operating point would have been chosen specifically to perform well on the exact data used to score it. Selecting it from out-of-fold CV predictions during training keeps the test set strictly held out for unbiased reporting.

The selected threshold is written to the model artifact's `metadata.json` alongside the serialised XGBoost weights, and from there flows two ways: `evaluate` reads it directly (`challenger_meta["threshold"]`) to compute F1 at that fixed operating point; `register_or_reject` bakes it into the registered model's serving container as the `THRESHOLD` env var if promoted. The serving container reads this value at startup and applies it to every inference request.

### Trade-off accepted

Targeting 0.80 recall will lower precision below what a 0.5 threshold would yield. This is intentional and expected. The champion/challenger gate still tracks both PR-AUC and F1 to ensure the model itself improves overall, but the operational threshold is driven by the recall requirement, not by maximising F1.
