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

The comparison calculates the Population Stability Index (PSI) per feature (`ml_common.drift.compute_psi`): numeric columns use the baseline's training-time decile bin edges, categorical columns use baseline category frequencies. If any feature's PSI exceeds the configured threshold (default 0.2, `PSI_THRESHOLD` env var), `evaluate_drift` flags `drift_detected = true`. The decision — including the per-feature PSI scores and which ones breached — is written as JSON to `gs://<project>-pipeline-metadata/drift/latest.json`, since a Cloud Run Job execution has no return-value channel back to its caller.

If no champion is registered yet (first-ever run), the check short-circuits to `drift_detected = false` — there's nothing to compare against.

Wasserstein Distance is not currently computed; PSI alone drives the threshold decision.

## Automated Remediation

`orchestrator-workflow` (see [orchestration.md](orchestration.md)) reads the drift decision from GCS after the drift-monitor job completes. If drift was detected, the workflow itself — not the drift-monitor container — submits the staged KFP template as a Vertex AI `PipelineJob` (`submit_training_pipeline` subworkflow), running under the `vertex-ai-pipeline-sa` service account, using the same `templateUri`/GCS-path conventions as the manual `scripts/submit_dev_pipeline.py`. This closes the MLOps loop: the system senses environmental change, automatically trains a new challenger model on the updated feature distribution, validates it via the champion/challenger gate, and promotes it to production without human intervention. The workflow does not wait for the training pipeline to finish — it's fire-and-forget, since training can run far longer than the daily orchestration DAG should block for.

## Email Notifications

The platform sends transactional email alerts via **SendGrid**, called directly from `orchestrator-workflow` (no separate Cloud Run subscriber service) using the `get_sendgrid_key`/`send_failure_alert`/`send_drift_alert` subworkflows in `workflows/orchestrator-workflow.yaml`.

The SendGrid integration is intentionally fictional by default for this showcase project: the `secret_manager` module (`iac/modules/secret_manager/`) provisions the `sendgrid-api-key` Secret Manager secret container, but it starts with no version — the API key itself is never passed through a Terraform variable or state, by design — and `alert_from_email` defaults to the unverified `noreply@mlops-alerts.com`. So a real send always fails. Every call site (`alert_data_gen_failure`, `alert_dbt_failure`, `alert_batch_predict_failure`, `alert_drift_monitor_failure`, `alert_drift_detected`) wraps its `send_failure_alert`/`send_drift_alert` call in `try`/`except` so that failure is swallowed rather than propagated — alerting is best-effort and never masks or replaces the underlying job failure the workflow raises afterward.

To make delivery actually work: add a version to the already-provisioned secret by hand (`gcloud secrets versions add sendgrid-api-key --data-file=-`, or via the console — the workflow's SA already holds `roles/secretmanager.secretAccessor`, so no IAM change is needed), and set `alert_from_email` in `terraform.tfvars` to a sender address verified in the SendGrid account (SendGrid rejects sends from unverified senders regardless of API key validity), then re-apply. `alert_from_email` flows from Terraform through Cloud Scheduler's job body into the workflow's `args`, the same path `alert_email` already uses.

### Drift Detected

When `orchestrator-workflow` reads `drift_detected = true` from the drift-monitor's decision file, it submits the retraining `PipelineJob` and then sends an email (`send_drift_alert`) to `alert_email` containing:

- **Breached features** — the per-feature PSI scores that exceeded the threshold (`decision.breached_features`)
- **Action taken** — the submitted `PipelineJob`'s resource name, confirming retraining was initiated

### Model Retrained & Promoted

No email is sent for training outcomes. The training pipeline's terminal `notify` stage (`post_training.notify`) is **deliberately a dummy**: it logs the outcome — promoted/rejected, model version, metrics, consecutive-rejection count, feature-review flag — to stdout, which Vertex AI Pipelines captures in Cloud Logging against the pipeline run. It publishes nothing and calls no external service, so the stage needs no project, no topic, and no IAM grant.

The reason is that this is a showcase project: a real outcome-notification path needs a real consumer (an inbox that's monitored, a Slack channel, an on-call rotation), and standing up a Pub/Sub topic plus a subscriber service that nothing reads would be infrastructure theatre rather than a working feature. The step still exists as a distinct DAG node so the outcome event has one obvious home: swapping the body of `post_training.notify` for a Pub/Sub publish, a SendGrid call, or a Slack post is a single-file change that needs no edits to the pipeline graph. Drift and failure alerts, which *do* have a live delivery path, are sent directly from `orchestrator-workflow` (above).

The two feature-review escalation signals described in [feature-exploration.md](feature-exploration.md) — the consecutive-rejection counter and SHAP importance rank drift — are computed and carried in that logged payload as `feature_review_alert` and `shap_rank_correlation`; they surface in the pipeline run's logs rather than as an email.
