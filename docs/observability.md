# Observability & Closed-Loop Remediation

The platform enforces proactive system reliability by continuously cross-examining model operational states.

## Log Severity

Every container entrypoint calls `obs_common.logging.configure_logging()` at startup instead of `logging.basicConfig()`. This is not a stylistic preference — `basicConfig` installs one handler on `sys.stderr`, and every GCP log agent reads an unstructured stderr line as **`ERROR`** severity. Under that default, an ordinary `logging.info` call — an HPO trial score, a promotion decision — arrives in Logs Explorer flagged red, and a `severity=ERROR` alert or log-based metric counts a successful run as hundreds of failures. The `INFO` in a `%(levelname)s`-formatted line is only message text; nothing reads it as the entry's severity.

`configure_logging()` addresses this with two mechanisms, because the two runtimes this repo deploys to classify logs differently:

| Mechanism | Effect |
|-----------|--------|
| **Structured JSON payload** | Each record is emitted as a single-line JSON object carrying a `severity` field. Cloud Run (`data-gen-job`, `drift-monitor-job`) parses this and lifts `severity` onto the entry, so `WARNING` stays `WARNING` rather than collapsing into `ERROR`. Anything passed via `extra={...}` is merged in as a queryable `jsonPayload` field. |
| **Severity-based stream routing** | Records below `WARNING` go to stdout, the rest to stderr. Vertex AI Pipelines does not reliably parse the JSON payload and falls back to classifying by stream, so there the result degrades to an accurate `INFO`/`ERROR` split instead of marking everything an error. |

The practical consequence for the training pipeline: the `hpo` stage's per-trial lines land as `INFO`, and a red entry in a pipeline run's logs once again means something actually failed.

## Drift Detection Mechanics

`drift-monitor-job` (the `drift-monitor` package, `projects/drift_monitor/`) runs once per day as a Cloud Run Job, invoked by `orchestrator-workflow` after batch prediction. It fetches the current champion model from Vertex AI Model Registry, downloads the per-feature baseline distribution frozen into that model's `metadata.json` at training time (`ml_common.drift.compute_baseline_stats`, computed from the training split in `trainer.experiment.run_train_stage`), and compares it against the latest BigQuery feature snapshot.

The comparison calculates the Population Stability Index (PSI) per feature (`ml_common.drift.compute_psi`). The decision — per-feature PSI scores, the threshold each was judged against, which breached, and which carried no signal — is written as JSON to `gs://<project>-pipeline-metadata/drift/latest.json`, since a Cloud Run Job execution has no return-value channel back to its caller.

If no champion is registered yet (first-ever run), the check short-circuits to `drift_detected = false` — there's nothing to compare against.

Wasserstein Distance is not currently computed; PSI alone drives the threshold decision.

### Baseline Shapes: Why Decile Bins Alone Are Not Enough

Numeric features here are mostly **not** continuous. `renewal_count` and `contact_requests_last_30d` are small counts concentrated on zero; `payment_failure_rate_30d` and `campaign_participation_rate` collapse to exactly 0.0 or 1.0 for any customer with a single event; `total_events` is 1 for most customers. Decile binning handles none of these well, so `compute_baseline_stats` freezes one of three shapes per column:

| Baseline `type` | Used for | Frozen content |
|-----------------|----------|----------------|
| `categorical` | `CATEGORICAL_COLS` | Category frequencies |
| `discrete` | Numeric columns with ≤ 20 distinct values | A frequency table over the observed values, keyed `str(float(v))` so it survives the `metadata.json` round-trip |
| `numeric` | Genuinely continuous columns | Decile bin edges **plus the baseline proportion measured in each bucket** |

The stored proportions matter. `np.quantile` on a column with a mass point returns several identical edges, `np.unique` collapses them, and the surviving buckets are then nowhere near equal-frequency. An earlier implementation assumed they were (`expected_pct = 1/n_buckets`), which made the gap between the assumed and actual baseline mass a permanent PSI floor — large enough that a baseline breached the threshold against *the very rows it was built from*, and the orchestrator submitted a retraining pipeline every single night. `test_no_drift_against_own_training_data` in `projects/ml_common/tests/test_drift.py` is the regression guard.

The same collapse had a mirror-image effect: a column that degenerated to a single bucket scored a structural PSI of `0.0` and read as perfectly healthy while being, in fact, unwatched. Every spec now carries `monitored`, and `evaluate_drift` reports those columns in `unmonitored_features` instead of silently counting them as passing.

### Missingness Is Part of the Distribution

Every spec also records `null_rate`, and the comparison carries an extra bucket for it: each side's value buckets are rescaled to its own non-null mass and the null share is appended, giving a proper distribution over *(values…, missing)* on both sides. A feature whose null rate moves is therefore caught even when the values that *are* present look unchanged — which a values-only comparison misses entirely, because it never sees the rows that went missing.

This matters here specifically because the missingness is neither random nor ignorable: `engagement_score` is absent for offline-only customers (**MAR**) and `days_since_last_successful_payment` is absent for exactly those customers who never paid successfully (**MNAR**, and correlated with the target — see [data-lifecycle.md](data-lifecycle.md)). A shift in how often either is null is a real population change, and usually an earlier signal than the values themselves.

The decision JSON reports `feature_null_rates` (baseline vs current per feature) alongside the PSI, because the two call for different responses: "PSI 0.4" and "null rate went 3% → 40%" usually mean a population shift and a broken upstream join respectively.

Baselines frozen before `null_rate` was recorded keep the old behaviour exactly — numeric columns drop nulls, categorical nulls fall into a `"nan"` pseudo-category. Retrofitting a null bucket onto them would compare a live null rate against an expectation of zero and manufacture a breach.

### Per-Feature Thresholds

A daily snapshot is a finite sample, so PSI is never exactly zero even against a perfectly stable population — and the smaller the sample, the larger that noise. A single hardcoded 0.2 applied across every feature ignores this.

`psi_noise_floor` draws repeated multinomial samples of the live snapshot's size from the baseline's own frozen proportions and returns the 99th percentile of the resulting PSI values — the level this feature reaches from sampling noise alone. It needs only the stored proportions, not the original training rows, so it is computed at check time once the live sample size is known. The effective threshold is:

```
threshold = max(PSI_THRESHOLD, psi_noise_floor(baseline_spec, len(snapshot)))
```

A feature therefore alarms only when the shift is both **large enough to matter** (the configured effect size) and **large enough to be distinguishable from noise** (the bootstrap floor). Small snapshots stop firing spuriously without loosening the threshold for features that have the data to support it.

### Persistence: One Night Is Not Evidence

Even a correctly calibrated detector tests ~13 features every night, and a single breach can come from a partial upstream load or an unlucky sample as easily as from a real shift. Acting on one directly is what turns a noisy detector into a nightly retrain loop.

A feature must therefore breach in at least `PERSISTENCE_MIN_BREACHES` of the last `PERSISTENCE_WINDOW` runs (default **2 of 3**) before it gates retraining. This is also what absorbs the multiple-comparison risk of testing every feature independently every day: at a 1-in-100 per-feature false-alarm rate, the same feature misfiring twice in three runs is a ~3-in-10,000 event — a far stronger control than a Bonferroni-corrected threshold, and one that does not require resolving the deep bootstrap tail such a correction would demand.

The decision JSON carries both verdicts: `run_drift_detected` is this run's raw result, `drift_detected` is the persistent one the orchestrator acts on, and `breach_counts` shows how far each breaching feature has progressed toward the rule.

### PSI History (`ml.drift_metrics`)

The persistence rule needs history, and `drift/latest.json` is overwritten daily. Each run therefore appends one row per evaluated feature — plus the `churn_probability` pseudo-feature — to **`ml.drift_metrics`** (`run_ts`, `snapshot_date`, `champion_model`, `feature`, `psi`, `threshold`, `breached`, `monitored`), partitioned by `snapshot_date` and clustered by `feature`.

Reads are scoped to the current `champion_model`: promoting a new model freezes a new baseline, so PSI measured against the previous champion says nothing about the current one and must not carry into the persistence count. The history query deliberately excludes the run in progress — the caller adds it in memory — so the rule never depends on reading back a row it just streamed in.

Both history operations degrade rather than fail. An unreadable table (first run after deploy, a permissions gap) counts zero prior breaches, which delays a genuine alarm by a run but cannot cause a spurious retrain; a failed write loses one run's history without discarding a decision already computed correctly.

## Prediction Score Drift (Monitoring-Only)

Feature-PSI only sees the input side — a model's predicted-score distribution can drift (staleness, degraded calibration) without any single input feature crossing the PSI threshold. `drift-monitor-job` also PSIs the champion's `churn_probability` output the same way it PSIs features: `trainer.experiment.run_train_stage` scores the training data with the final model and folds that distribution into the same `baseline_stats` dict as an extra pseudo-feature (`ml_common.config.CHURN_PROBABILITY_FIELD`), reusing `compute_baseline_stats`/`compute_psi` unmodified.

At check time, `drift_monitor.predictions.fetch_latest_predictions` finds the champion's most recent **successfully completed, daily-scoring** `BatchPredictionJob` — disambiguated from `post_training`'s own evaluation-time batch prediction against the champion by `display_name` (`BATCH_PREDICT_DISPLAY_NAME` env var, must match `submit_batch_predict`'s `displayName` in `orchestrator-workflow.yaml`) — and reads its raw BigQuery output table (`output_info.bigquery_output_dataset` + `bigquery_output_table`). It deliberately keeps reading Vertex's unmanaged, auto-created `predictions_<timestamp>` table directly rather than the clean `ml.predictions` schema declared in `iac/config/bigquery.yaml` (which `orchestrator-workflow`'s `sync_predictions_to_ml_predictions` subworkflow now populates daily, see [ml-infrastructure.md](ml-infrastructure.md) and [orchestration.md](orchestration.md)): `ml.predictions` only carries the latest reconciled value per `(customer_id, snapshot_date)` and doesn't record which specific `BatchPredictionJob`/model produced it, so it can't disambiguate the daily-scoring job from an evaluation run the way `display_name` + `model` filtering on the job resource itself can.

The score check applies the same two-part threshold the feature check does — `max(PSI_THRESHOLD, psi_noise_floor(score_baseline, len(predictions)))`, reported as `score_threshold` — and its PSI is appended to `ml.drift_metrics` under the `churn_probability` feature name so score drift has the same history as any input feature.

This check is **monitoring-only**: `score_psi`/`score_threshold`/`score_drift_detected` are written into the same decision JSON as additive fields, but the feature-only `drift_detected` still exclusively drives automated retraining. A score-only breach still reaches a human — `orchestrator-workflow` sends a `send_drift_alert` email for it (no `PipelineJob` submitted, just a review prompt) — see below. A failure anywhere in this check (BigQuery permission issue, no matching batch-prediction job yet, a malformed `output_info`) degrades silently to "no score signal this run" rather than affecting the feature-PSI result.

## Data Quality vs. Drift

Drift and defects are indistinguishable from a PSI score alone — a broken upstream join, a partial load, or a column that silently became `NULL` moves a distribution exactly as a genuine population shift does. The correct responses are opposite:

| | Cause | Correct response |
|---|---|---|
| **Drift** | The world changed | Retrain on the new distribution |
| **Defect** | The data is wrong | Halt, page a human, fix the pipeline |

Retraining on a defect bakes it into the model. The champion/challenger gate offers no protection, because the challenger is evaluated against a test split drawn from the *same* corrupted snapshot — it can score well and be promoted precisely on the strength of the defect.

`ml_common.data_quality.check_data_quality` therefore runs as a separate gate, comparing the live snapshot against the champion's frozen baseline and against the feature table's own recent history. These are absolute assertions rather than distributional ones, and they fire on first occurrence rather than waiting for the N-of-M persistence rule that governs genuine drift:

| Assertion | Fails when | What it catches |
|---|---|---|
| `row_volume` | Snapshot is under 50% of the median of the last `QUALITY_HISTORY_PARTITIONS` (default 7) partitions | An incomplete upstream load — dbt writes the partition whether or not every source event arrived |
| `duplicate_keys` | Any `customer_id` appears more than once | `stg_activity_cdc`'s event dedup or `customer_features`' `QUALIFY ROW_NUMBER()` stopped holding, so aggregates are computed over duplicated history |
| `missing_columns` | A feature the champion trained on is absent | Schema change upstream |
| `null_rate` | A feature's null rate is more than 25 points above its training baseline | A broken join, distinguished from the genuine missingness shifts that PSI handles |
| `collapsed_column` | A feature that varied at training now has one distinct value | An upstream default written into every row — the case where PSI is least reliable |

A failure forces `drift_detected` to false, records `retrain_suppressed_by_data_quality`, and routes the orchestrator to `alert_data_quality_failure`, which names the failed assertions and states that retraining was withheld. The PSI numbers are still computed and written to `ml.drift_metrics` — they are useful evidence when diagnosing the defect.

This gate is the one check in the job that **fails closed**: if it cannot run at all, the run is treated as having failed it. Every other degradation in this job already biases toward not retraining, and so does this one.

> Placement: the gate currently runs inside `drift-monitor-job`, which is after batch prediction, so a corrupt snapshot is still scored before the failure is caught. Failing fast between `dbt-job` and batch prediction would be better and is the natural next step; it needs a separate Cloud Run job rather than a new branch in an existing one.

## Automated Remediation

`orchestrator-workflow` (see [orchestration.md](orchestration.md)) reads the drift decision from GCS after the drift-monitor job completes. If persistent drift was detected, the workflow itself — not the drift-monitor container — submits the staged KFP template as a Vertex AI `PipelineJob` (`submit_training_pipeline` subworkflow), running under the `vertex-ai-pipeline-sa` service account, using the same `templateUri`/GCS-path conventions as the manual `scripts/submit_dev_pipeline.py`. This closes the MLOps loop: the system senses environmental change, automatically trains a new challenger model on the updated feature distribution, validates it via the champion/challenger gate, and promotes it to production without human intervention. The workflow does not wait for the training pipeline to finish — it's fire-and-forget, since training can run far longer than the daily orchestration DAG should block for.

### Retraining Guards (`last_retrain_status`)

Fire-and-forget submission needs a brake. Drift can outlive a retrain — a genuine shift the challenger keeps failing to beat the promotion gate on, or a defect upstream of the model that no amount of retraining fixes — and without a guard the workflow would submit a fresh `PipelineJob` every night indefinitely. The `last_retrain_status` subworkflow lists the most recent `churn-training-retrain` job and blocks submission on either of two conditions:

| Guard | Condition | Why |
|-------|-----------|-----|
| `in_flight` | Latest job is `PENDING`/`QUEUED`/`RUNNING` | Training outlives this DAG, so yesterday's job is often still going. Stacking doubles Vertex spend and races the two runs against each other over the same `ml.split_assignments` partition. **This is the guard that actually prevents concurrent runs**, and it is independent of elapsed time. |
| `backing_off` | Previous challengers were rejected *and* the last attempt is newer than the back-off window | Throttles **repeated failed attempts**, not retraining in general. |

The back-off is keyed on `consecutive_rejections` rather than a flat interval, because elapsed time is the wrong question — what matters is whether a new run could plausibly produce a different outcome:

| `consecutive_rejections` | Delay before the next attempt |
|---|---|
| 0 (last challenger was promoted) | **none** |
| 1 | 1 day |
| 2 | 2 days |
| 3+ | 7 days |

A flat floor conflated two opposite situations. A retrain that was *promoted* means the loop is working, so the next genuine drift deserves an immediate response; a retrain that was *rejected* means more of the same data will very likely be rejected again. The flat version also let a real population shift arriving two days after a successful promotion go unaddressed for the rest of the window — a worse failure than the spend it was avoiding. Reading the counter is best-effort: an absent or unreadable state file degrades to 0, i.e. "no evidence of repeated failure", which favours retraining rather than suppressing it.

A blocked night is **not** silent: `alert_retrain_suppressed` sends the drift email with the suppression reason, so a suppressed night is distinguishable from a quiet one. Escalation to a human remains `post_training`'s `MAX_CONSECUTIVE_REJECTIONS` feature-review alert.

### Rebuilding a Stale Baseline

Baselines frozen before per-bucket proportions were stored carry only bin edges, and evaluating one would mean re-assuming uniformity — the defect itself. Such specs are reported `unmonitored` rather than acted on, which means an un-migrated champion is not watched.

Waiting for the next promotion is not a migration path: the gate requires the challenger to beat the champion on PR-AUC *and* F1, and a challenger trained on an undrifted distribution generally does not, so a stale baseline can persist indefinitely. `scripts/rebuild_champion_baseline.py` recomputes it in place from `ml.split_assignments` — the permanent record of the exact rows the champion trained on — with no retraining and no approximation:

```bash
PROJECT_ID=<project> uv run scripts/rebuild_champion_baseline.py --dry-run
```

Models trained from now on record `training_snapshot_date` in `metadata.json`, so the script knows which partition to read; for older artifacts it infers the most recent partition at or before the model's registration date, since every training run writes a partition — rejected challengers included — and the newest partition overall is often some later run's.

## Email Notifications

The platform sends transactional email alerts via **SendGrid**, called directly from `orchestrator-workflow` (no separate Cloud Run subscriber service) using the `get_sendgrid_key`/`send_failure_alert`/`send_drift_alert` subworkflows in `workflows/orchestrator-workflow.yaml`.

The SendGrid integration is intentionally fictional by default for this showcase project: the `secret_manager` module (`iac/modules/secret_manager/`) provisions the `sendgrid-api-key` Secret Manager secret container, but it starts with no version — the API key itself is never passed through a Terraform variable or state, by design — and `alert_from_email` defaults to the unverified `noreply@mlops-alerts.com`. So a real send always fails. Every call site (`alert_data_gen_failure`, `alert_dbt_failure`, `alert_batch_predict_failure`, `alert_drift_monitor_failure`, `alert_score_drift_only`, `alert_drift_detected`) wraps its `send_failure_alert`/`send_drift_alert` call in `try`/`except` so that failure is swallowed rather than propagated — alerting is best-effort and never masks or replaces the underlying job failure the workflow raises afterward.

To make delivery actually work: add a version to the already-provisioned secret by hand (`gcloud secrets versions add sendgrid-api-key --data-file=-`, or via the console — the workflow's SA already holds `roles/secretmanager.secretAccessor`, so no IAM change is needed), and set `alert_from_email` in `terraform.tfvars` to a sender address verified in the SendGrid account (SendGrid rejects sends from unverified senders regardless of API key validity), then re-apply. `alert_from_email` flows from Terraform through Cloud Scheduler's job body into the workflow's `args`, the same path `alert_email` already uses.

### Drift Detected

When `orchestrator-workflow` reads `drift_detected = true` from the drift-monitor's decision file, it submits the retraining `PipelineJob` and then sends an email (`send_drift_alert`) to `alert_email` containing:

- **Breached features** — the per-feature PSI scores that exceeded the threshold (`decision.breached_features`)
- **Action taken** — the submitted `PipelineJob`'s resource name, confirming retraining was initiated

### Score Drift Detected (No Retraining)

When feature drift is absent but `score_drift_detected = true` (the champion's `churn_probability` output has drifted from its training-time baseline — see [Prediction Score Drift](#prediction-score-drift-monitoring-only) above), `check_drift` routes to `alert_score_drift_only` instead of `trigger_training`. This calls the same `send_drift_alert` subworkflow with `pipeline_job: null` and no `suppressed_reason`, which branches its message to state the PSI value and threshold breached and that **no retraining was triggered automatically** — this is the only place a human finds out about score drift at all, since it's otherwise just a field in the GCS decision JSON and Cloud Run Job logs.

### Model Retrained & Promoted

No email is sent for training outcomes. The training pipeline's terminal `notify` stage (`post_training.notify`) is **deliberately a dummy**: it logs the outcome — promoted/rejected, model version, metrics, consecutive-rejection count, feature-review flag — to stdout, which Vertex AI Pipelines captures in Cloud Logging against the pipeline run. It publishes nothing and calls no external service, so the stage needs no project, no topic, and no IAM grant.

The reason is that this is a showcase project: a real outcome-notification path needs a real consumer (an inbox that's monitored, a Slack channel, an on-call rotation), and standing up a Pub/Sub topic plus a subscriber service that nothing reads would be infrastructure theatre rather than a working feature. The step still exists as a distinct DAG node so the outcome event has one obvious home: swapping the body of `post_training.notify` for a Pub/Sub publish, a SendGrid call, or a Slack post is a single-file change that needs no edits to the pipeline graph. Drift and failure alerts, which *do* have a live delivery path, are sent directly from `orchestrator-workflow` (above).

The two feature-review escalation signals described in [feature-exploration.md](feature-exploration.md) — the consecutive-rejection counter and SHAP importance rank drift — are computed and carried in that logged payload as `feature_review_alert` and `shap_rank_correlation`; they surface in the pipeline run's logs rather than as an email.
