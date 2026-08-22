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
| `cloud_build/` | Creates Cloud Build v2 repo link, SA IAM bindings, and all triggers |
| `cloud_workflow/` | Creates Cloud Workflows orchestrator definitions |
| `gcs_bucket/` | Creates `google_storage_bucket` resources + optional bucket-level IAM bindings |
| `vertex_ai_pipeline/` | Creates the dedicated SA the training pipeline runs as + its IAM bindings |
| `secret_manager/` | Creates a `google_secret_manager_secret` + initial version |

## BigQuery Tables (`bq_table` module)

The `bq_table` module accepts an optional `time_partitioning` block in `bigquery.yaml`. When `expiration_ms` is set, BigQuery automatically drops partitions older than that duration — no scheduled query or cleanup job required.

```yaml
time_partitioning:
  type: DAY
  field: event_timestamp
  expiration_ms: 7776000000  # 90 days
```

The `raw.activity_cdc` table has a 90-day partition expiration. Partitions outside this window are deleted by BigQuery on its own maintenance schedule. The `features` dataset has no expiration because its tables are small point-in-time snapshots used for training.

## Cloud Run Jobs

The `cloud_run_job` module co-locates service account creation and IAM bindings with the job resource, and uses `lifecycle { ignore_changes = [template] }` so CI/CD image updates are not reverted by Terraform. Besides `bq_dataset_roles` and `service_account_project_roles`, it also accepts a `gcs_bucket_roles` map (bucket short key → role), resolved to the project-prefixed bucket name in `main.tf` the same way `vertex_ai_pipeline` resolves its own bucket roles — added for `drift-monitor-job`, which needs to read the champion's training artifacts and write its decision JSON to GCS.

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

## Provisioned Resources

The infrastructure definitions provision and link the following system layers:

* **Storage Frameworks:** Object storage buckets for tracking system configurations, staging compiled pipelines, and caching model parameters, alongside highly scalable analytical data warehouse sets.
* **Compute Runtimes:** Serverless container execution endpoints, managed job schedulers, and scalable orchestration engines.
* **Networking & Protection:** Secure virtual network routing, explicit IAM roles granting minimal required access per service account, and load balancers to distribute internal API requests smoothly.
