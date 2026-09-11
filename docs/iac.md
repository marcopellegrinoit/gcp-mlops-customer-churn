# Infrastructure as Code (IaC)

To prevent environment divergence and ensure that staging and production layouts are entirely identical, all core cloud primitives are written as configuration blocks within declarative infrastructure files using Terraform.

## Design Principles

Resource definitions are driven by declarative YAML files under `iac/config/`, decoded at plan time via `yamldecode`, eliminating `.tfvars` sprawl. Each top-level config domain gets its own file, one per resource concern:

| File | Contents |
|------|----------|
| `apis.yaml` | List of GCP APIs to enable |
| `cloud_build.yaml` | GitHub connection details + default Cloud Build SA roles |
| `triggers.yaml` | Cloud Build trigger definitions |
| `artifact_registry.yaml` | Artifact Registry repositories |
| `gcs_buckets.yaml` | GCS bucket definitions |
| `cloud_run_jobs.yaml` | Cloud Run job definitions |
| `workflows.yaml` | Cloud Workflows definitions |
| `bigquery.yaml` | BigQuery dataset and table definitions |
| `vertex_ai_pipeline.yaml` | Dedicated service account roles for the Vertex AI training pipeline |

`locals.tf` decodes each file independently into its own local (e.g. `local.apis`, `local.cloud_build`) and derives consumable shapes like `local.all_datasets` and `local.all_cloud_build_triggers` for `main.tf`. GCP APIs are declared in `apis.yaml` and enabled via `google_project_service` before any other resource, with all modules carrying `depends_on = [google_project_service.apis]` to guarantee API readiness on a fresh project.

Every GCP resource lives inside a dedicated module under `iac/modules/`. The root `iac/main.tf` is reserved for module calls, data sources, and provider-level lookups only — no `resource` blocks directly in root.

## Module Structure

| Module | Purpose |
|--------|---------|
| `artifact_registry/` | Creates `google_artifact_registry_repository` resources |
| `bq_dataset/` | Creates `google_bigquery_dataset` resources |
| `bq_table/` | Creates `google_bigquery_table` resources |
| `cloud_run_job/` | Creates Cloud Run Job + dedicated service account + IAM bindings |
| `cloud_run_service/` | Creates Cloud Run Service + dedicated service account + IAM bindings + IAP |
| `cloud_build/` | Creates Cloud Build v2 repo link, SA IAM bindings, and all triggers |
| `cloud_workflow/` | Creates Cloud Workflows orchestrator definitions |
| `gcs_bucket/` | Creates `google_storage_bucket` resources + optional bucket-level IAM bindings |
| `vertex_ai_pipeline/` | Creates the dedicated SA the training pipeline runs as + its IAM bindings |
| `secret_manager/` | Creates a `google_secret_manager_secret` + initial version |
| `billing_budget/` | Creates a project-scoped `google_billing_budget` + the email notification channels it delivers through |

## Runtime Configuration

Application behaviour is configured through environment variables read by `pydantic-settings` classes in the code (see [architecture.md](architecture.md#data-contracts--configuration)), which makes this repository the single place a knob is turned — no image rebuild, no code change.

Two delivery paths, because the two runtimes differ:

* **Cloud Run services** take `env_vars` from `cloud_run_services.yaml`, through the same `locals.tf` merge. The `dashboard-service` block carries the dashboard's cost guards (`MAX_BYTES_BILLED`, `CACHE_TTL_SECONDS`) and its presentation bands (`HIGH_RISK_THRESHOLD`, `MEDIUM_RISK_THRESHOLD`, `INDICATOR_RATIO`) — see [dashboard.md](dashboard.md#configuration-reference).
* **Cloud Run jobs** take `env_vars` from `cloud_run_jobs.yaml`. `locals.tf` merges `BQ_PROJECT_ID = var.project_id` into every job's map, so no job hardcodes the project. The `drift-monitor-job` block carries the detection policy (`PSI_THRESHOLD`, `PERSISTENCE_WINDOW`, `PERSISTENCE_MIN_BREACHES`, `QUALITY_HISTORY_PARTITIONS`, `ML_MIN_ROW_RATIO`, `ML_MAX_NULL_RATE_INCREASE`); `data-gen-job` carries the simulation's scale and its anomaly-injection rate.
* **Vertex AI Pipelines containers** have no Terraform of their own — the compiled pipeline spec is their deployment descriptor. `training-pipeline-trigger` in `triggers.yaml` therefore declares `substitutions`, which Cloud Build exports into the pipeline-compile step's environment; `training_pipeline.settings_env` reads them and pins the resolved values onto every task's container. The `_ML_*`/`_MODELING_*` substitutions there are the promotion gate and the HPO budget — the platform's most consequential numbers — and they restate the code defaults on purpose, so what production runs with is legible here rather than only in a Python module.

Each value is required to be typed and in range by the settings class that reads it, so a bad value fails the container at startup rather than producing a plausible-looking but wrong run.

## BigQuery Tables (`bq_table` module)

The `bq_table` module accepts an optional `time_partitioning` block in `bigquery.yaml`. When `expiration_ms` is set, BigQuery automatically drops partitions older than that duration — no scheduled query or cleanup job required.

```yaml
time_partitioning:
  type: DAY
  field: event_timestamp
  expiration_ms: 7776000000  # 90 days
```

The `raw.activity_cdc` table and `features.customer_features` both have a 90-day partition expiration. Partitions outside this window are deleted by BigQuery on its own maintenance schedule. This is safe for `customer_features` because nothing ever reads an old `snapshot_date` partition of it: `data_split` freezes the exact rows it used into `ml.split_assignments` (which has no expiration — it's the permanent lineage record, see [ml-infrastructure.md](ml-infrastructure.md)), and `drift-monitor` only ever compares against the latest snapshot.

## Cloud Run Jobs

The `cloud_run_job` module co-locates service account creation and IAM bindings with the job resource, and uses `lifecycle { ignore_changes = [template] }` so CI/CD image updates are not reverted by Terraform. Besides `bq_dataset_roles` and `service_account_project_roles`, it also accepts a `gcs_bucket_roles` map (bucket short key → role), resolved to the project-prefixed bucket name in `main.tf` the same way `vertex_ai_pipeline` resolves its own bucket roles — added for `drift-monitor-job`, which needs to read the champion's training artifacts and write its decision JSON to GCS.

## Cloud Run Services (`cloud_run_service` module)

One service exists: `dashboard-service`, the business-user churn dashboard. The module mirrors `cloud_run_job` — service account, project roles and BigQuery dataset roles created alongside the resource, with `ignore_changes` on the image so CI/CD revisions are not reverted — and adds the three things a *service* needs that a Job does not.

**IAP, without a load balancer.** `iap_enabled = true` on `google_cloud_run_v2_service` protects the `run.app` URL directly. This integration is GA and free; the older topology needed an external Application Load Balancer, forwarding rule and static IP, none of which have a free tier. Two IAM bindings complete it: `google_cloud_run_v2_service_iam_member` grants `roles/run.invoker` to IAP's service agent — the **only** principal that holds it, which is what makes the service unreachable except through IAP — and `google_iap_web_cloud_run_service_iam_member` grants `roles/iap.httpsResourceAccessor` to each member of the `dashboard_viewers` variable.

`dashboard_viewers` is a Terraform variable rather than a field in `cloud_run_services.yaml` on purpose: the YAML files describe the *shape* of the infrastructure, while who may read customer-level data is per-environment. It defaults to `[]`, so the service deploys reachable by nobody.

The service's `ingress` is `INGRESS_TRAFFIC_ALL`, which the direct IAP integration requires — IAP fronts the public `run.app` path. The authorisation boundary is the IAM binding, not the network.

**Scaling that stays free.** `min_instance_count = 0`. Cloud Run's always-free allowance is 180k vCPU-seconds a month against roughly 2.6M for one instance pinned warm through a 730-hour month, so `min_instances = 1` would leave the free tier by about 15×. `startup_cpu_boost` absorbs the resulting cold start. `cpu_idle = false` is required in the other direction — Streamlit serves each session over a long-lived WebSocket from a per-process cache, so throttling CPU between requests would stall it.

`max_instance_count = 1`, and that is the ceiling on the service's whole compute bill: three instances pinned for a month would be roughly 7.8M vCPU-seconds, about 40× the free allowance. One instance serves this platform's readership with room to spare, so a second one starting would be a symptom rather than a need — see [dashboard.md](dashboard.md#the-abuse-surface).

**An explicit concurrency ceiling.** `max_instance_request_concurrency = 20`, against Cloud Run's default of 80. A "request" to a Streamlit container is a session-long WebSocket, and `st.cache_data` hands each script run its own *copy* of the cached snapshot, so concurrent readers are concurrent copies of the scored base inside a 1 GiB container. At 80 the instance OOMs — dropping every live session and discarding the cache, which costs two BigQuery queries to rebuild. At 20 the reader who would have been the last straw gets a clean 429 instead. With `max_instances = 1` this is the service's total simultaneous-reader ceiling rather than a scale-out trigger, which is the intended behaviour here.

**A real startup probe.** Without an explicit probe Cloud Run uses a TCP check that succeeds as soon as the port is bound — before the app can serve — so the first request of a cold start lands on a server that is not ready. The module probes Streamlit's own `/_stcore/health`.

### The one beta-provider resource

`google_project_service_identity` (which provisions IAP's service agent) has no GA equivalent, so `provider.tf` declares `hashicorp/google-beta` for it alone. GCP creates that agent lazily, and without it the invoker binding can reference a principal that does not exist yet and fail the first apply. Nothing else in this configuration uses the beta provider.

The GA provider floor was also raised from `>= 4.0` to `>= 7.0`: `iap_enabled` and `google_iap_web_cloud_run_service_iam_member` both postdate the 4.x line, and on an older provider the IAP field is silently unknown and the service deploys unprotected.

## Cloud Build (2nd-gen API)

Cloud Build triggers use the **2nd-generation** API: a `google_cloudbuildv2_connection` (created once via the GCP Console and referenced as a Terraform data source) links the GitHub repository via `google_cloudbuildv2_repository`, and triggers use `repository_event_config` instead of the legacy `github` block — enabling regional triggers in `europe-west1`.

The default Cloud Build SA is granted the minimum required roles (`artifactregistry.writer`, `run.developer`, `logging.logWriter`) declared in `cloud_build.yaml`. Only provider-level identity and secrets (`project_id`, `region`, `alert_email`) live in `terraform.tfvars`, which is gitignored.

## GCS Buckets (`gcs_bucket` module)

Bucket names in `gcs_buckets.yaml` are short keys (e.g. `training-data`); the module prefixes them with `${project_id}-` to guarantee global uniqueness. Two buckets are currently provisioned:

| Bucket key | Purpose |
|---|---|
| `training-data` | Train/test Parquet exports and model artifacts written by the trainer container during pipeline runs |
| `pipeline-metadata` | Vertex AI Pipelines `pipeline_root`: KFP execution metadata, task I/O, and caching for each training pipeline run |

Cross-module IAM (e.g. granting a Cloud Build SA write access to a specific bucket) can be computed in `locals.tf` and merged onto a bucket's config as an `iam_bindings` list rather than as a `resource` block in root `main.tf`, keeping root limited to module calls per the project's Terraform convention.

An optional `lifecycle_age_days` key on a bucket config enables a `Delete` lifecycle rule for objects older than that many days. `training-data` sets this to 30, since its Parquet exports and model artifacts are regenerated on every pipeline run and don't need indefinite retention. `pipeline-metadata` sets this to 7 for the same storage-cost reason — but KFP's execution cache in Vertex ML Metadata has no matching expiry of its own, so a cache hit can still reference a GCS artifact this rule already deleted. See [Pipeline Stages](ml-infrastructure.md#pipeline-stages) for how the pipeline avoids that.

## Artifact Registry (`artifact_registry` module)

Two repositories are provisioned under `artifact_registry.repositories` in `artifact_registry.yaml`:

| Repository key | Format | Purpose |
|---|---|---|
| `containers` | `DOCKER` | Shared repo for all container images (trainer, dbt, data-gen), differentiated by image name |
| `pipeline-templates` | `KFP` | Compiled Vertex AI Pipeline templates, versioned and pushed via `kfp.registry.RegistryClient` rather than treated as plain files |

The Cloud Build SA's project-level `roles/artifactregistry.writer` (declared in `cloud_build.default_sa_roles`) covers writes to both repositories — no per-repository IAM binding is needed. The module also accepts a `reader_service_account_emails` input that grants `roles/artifactregistry.reader` scoped to that single repository via `google_artifact_registry_repository_iam_member`, for SAs that need to pull images but shouldn't get the project-wide reader role; `main.tf` uses this to grant `vertex-batch-predict-sa` read access to `containers` (see below).

## Vertex AI Pipeline Service Accounts (`vertex_ai_pipeline` module)

The training pipeline (`PipelineJob`) runs under its own dedicated SA rather than the project's default compute SA, scoped to exactly what `training_pipeline.pipeline` touches: `roles/bigquery.dataViewer` on the `features` dataset (export to train/test splits), `roles/storage.objectAdmin` on the `training-data` and `pipeline-metadata` buckets (model artifacts + KFP execution root), `roles/artifactregistry.reader` (pull trainer/post-training images), `roles/bigquery.jobUser`, and `roles/aiplatform.user` (batch prediction, Model Registry get/upload). GCS bucket roles are declared against the short bucket key in `vertex_ai_pipeline.yaml` and resolved to the project-prefixed bucket name in `main.tf` via `module.gcs_bucket[name].bucket_name`, the same cross-module pattern used elsewhere for bucket IAM.

The champion/challenger `ModelBatchPredictOp` tasks run under a second, narrower SA (`vertex-batch-predict-sa`) instead of inheriting the pipeline's own broader SA: it holds `roles/storage.objectAdmin` on the `training-data` bucket (read model artifacts + test instances, write predictions) and a repository-scoped `roles/artifactregistry.reader` on the `containers` repo only (granted via the `artifact_registry` module's `reader_service_account_emails` input, not `vertex_ai_pipeline`'s own role list) so it can pull the custom serving container image it runs as the BatchPredictionJob's `serviceAccount` — without it, Vertex AI cannot pull the image and the job fails immediately with "Model server terminated". It has no BigQuery or `pipeline-metadata` access. Both SAs are created from the same `vertex_ai_pipeline` module — `account_id`/`display_name` are now module inputs rather than hardcoded, so `main.tf` calls it twice (`module.vertex_ai_pipeline` and `module.vertex_ai_batch_predict_sa`), each with its own section in `vertex_ai_pipeline.yaml`. Because the pipeline's own SA is the identity that actually calls `CreateBatchPredictionJob` (as the KFP step executing that task), it needs to impersonate the batch-predict SA to launch jobs under it — the module's `act_as_service_account_emails` input grants that one binding (`roles/iam.serviceAccountUser`, scoped to just that target SA, not project-wide).

`orchestrator-workflow` submits the `PipelineJob` automatically on drift (see below); Cloud Build itself still only compiles and stages the KFP template to the `pipeline-templates` Artifact Registry repo. The manual/dev `scripts/submit_dev_pipeline.py` script (see [ml-infrastructure.md](ml-infrastructure.md)) remains the path for ad hoc runs; both it and the workflow pass the pipeline SA's email as the PipelineJob's own `service_account` (submission-time identity, not a pipeline template parameter — Vertex AI does not pick it up automatically) and separately pass the batch-predict SA's email as the `batch_predict_service_account` runtime pipeline parameter, since `ModelBatchPredictOp` needs it as an explicit task input rather than inheriting the PipelineJob's identity.

## Cloud Workflows Service Account (`cloud_workflow` module)

Like `cloud_run_job`, the `cloud_workflow` module accepts a `gcs_bucket_roles` map for bucket-level grants (`orchestrator-workflow` uses `roles/storage.objectViewer` on `pipeline-metadata` to read the drift monitor's decision JSON). It also accepts `act_as_service_account_emails`, a list of SA emails the workflow's own SA is granted `roles/iam.serviceAccountUser` on — needed because `orchestrator-workflow` submits the training `PipelineJob` under `vertex-ai-pipeline-sa`, not its own identity. `workflows.yaml` sets `act_as_vertex_pipeline_sa: true` to opt the orchestrator workflow into this grant; `main.tf` resolves the flag to `module.vertex_ai_pipeline.service_account_email`.

`google_workflows_workflow.this` sets `source_contents = file(...)` but carries `lifecycle { ignore_changes = [source_contents] }`, the same pattern `cloud_run_job` uses for `template` (see above): Terraform seeds the initial revision on a fresh project and owns the resource's SA/IAM/scheduler, but ongoing content updates are owned exclusively by `workflow-trigger` (see [cicd.md](cicd.md)), which runs `gcloud workflows deploy` on every push to `main` touching `workflows/**`. Without `ignore_changes`, a `terraform apply` from a stale local checkout would silently revert the live workflow to old content — ignoring the attribute makes Cloud Build the single source of truth for the definition post-creation.

The `alert_email` and `alert_from_email` variables both flow the same way: root `main.tf` passes them into the `cloud_workflow` module, which embeds them in the Cloud Scheduler job's HTTP body (`argument = jsonencode({ ... })`) alongside `project_id`/`region`; `orchestrator-workflow.yaml`'s `main.init` step reads them back out of `args`.

## SendGrid Secret (`secret_manager` module)

`secret_manager` is a minimal, general-purpose module (`google_secret_manager_secret` only) used once today, for `sendgrid-api-key`. It deliberately provisions just the empty container, not a `google_secret_manager_secret_version` — the API key is a credential that should never pass through a `.tf` variable or Terraform state, so it's added by hand after `apply` (`gcloud secrets versions add sendgrid-api-key --data-file=-`, or via the console). With no version yet, `orchestrator-workflow`'s `get_sendgrid_key` subworkflow 404s on `versions/latest:access`, which its `try`/`except` wrapper swallows (see [observability.md](observability.md)). The orchestrator workflow's SA already holds project-scoped `roles/secretmanager.secretAccessor` via `workflows.yaml`, so adding a version by hand is the only step needed to make delivery work — no IAM changes, no Terraform changes.

## Budget Alerting (`billing_budget` module)

Every other cost control in this repository is a per-resource ceiling: `max_instance_count` on the Cloud Run service, `MAX_BYTES_BILLED` on every dashboard query, `max_retries` on the jobs, the HPO trial budget in `triggers.yaml`. Each bounds one runaway in isolation, and none of them notices a bill climbing across several at once — or a resource nobody thought to cap. The budget is the backstop that does.

It is a **notification, not a cap.** GCP has no mechanism that stops spend, so what this buys is time-to-discovery: hours instead of a billing cycle. Four rules fire — actual spend at 50%, 90% and 100% of `monthly_budget_amount`, plus *forecasted* spend at 100%. The forecast rule is the one that matters for a platform designed to live inside the free tiers: actual spend crossing 50% tells you the month is already half gone, while a forecast crossing 100% fires on the trend, catching a service that started billing yesterday and will keep billing until someone looks.

The budget is scoped to this project by number (`projects/<number>`), the only form the Budgets API accepts. Unscoped it would cover every project on the billing account, and an alert would say nothing about whether *this* platform is what is costing money.

**The billing account is derived, not configured.** Every project that can run this platform is already attached to one — that attachment is what makes BigQuery and Cloud Run billable at all — so `locals.tf` reads it from `data.google_project.this.billing_account` rather than asking for a value Terraform can already see. `var.billing_account_id` remains as an override for the case where the budget belongs on a different account than the project bills to. The module is still `count = 0` when the derived value is empty, which happens only for a project with no billing account attached — a project that cannot run the rest of this either.

What the budget *does* need is a permission, and it is the one permission in this repository that does not come from the project: a budget is created on the billing account, so the identity running `apply` needs `roles/billing.costsManager` there. A personal project's owner already has it; a deployer service account needs an explicit grant (see [setup.md](setup.md)).

```hcl
# iac/terraform.tfvars — the account is detected, so only these two are worth setting
monthly_budget_amount = 10
budget_alert_emails   = ["you@example.com"]
```

`budget_alert_emails` is additive, not a replacement: each address becomes a `google_monitoring_notification_channel`, and `disable_default_iam_recipients = false` keeps the billing account's own admins on the distribution. Left empty, no `all_updates_rule` is written at all and the Budgets API falls back to emailing those admins — which is why the block is `dynamic` rather than declared with an empty channel list, since an empty list would silence it instead.

## Provisioned Resources

The infrastructure definitions provision and link the following system layers:

* **Storage Frameworks:** Object storage buckets for tracking system configurations, staging compiled pipelines, and caching model parameters, alongside highly scalable analytical data warehouse sets.
* **Compute Runtimes:** Serverless container execution endpoints, managed job schedulers, and scalable orchestration engines.
* **Networking & Protection:** Secure virtual network routing, explicit IAM roles granting minimal required access per service account, and load balancers to distribute internal API requests smoothly.
