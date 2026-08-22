data "google_project" "this" {
  project_id = var.project_id
}

resource "google_project_service" "apis" {
  for_each = toset(local.apis.apis)

  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

module "artifact_registry" {
  source   = "./modules/artifact_registry"
  for_each = local.artifact_registry.artifact_registry.repositories

  project_id    = var.project_id
  location      = var.region
  repository_id = each.key
  description   = each.value.description
  format        = each.value.format
  # The batch-predict SA runs the challenger/champion serving container as a custom
  # BatchPredictionJob container, so it must be able to pull the image itself.
  reader_service_account_emails = (
    each.key == "containers" ? [module.vertex_ai_batch_predict_sa.service_account_email] : []
  )

  depends_on = [google_project_service.apis, module.vertex_ai_batch_predict_sa]
}

module "cloud_run_job" {
  source   = "./modules/cloud_run_job"
  for_each = local.all_cloud_run_jobs

  project_id       = var.project_id
  region           = var.region
  job_name         = each.key
  image            = each.value.image
  cpu              = try(each.value.cpu, "1")
  memory           = try(each.value.memory, "512Mi")
  timeout          = try(each.value.timeout, "600s")
  max_retries      = try(each.value.max_retries, 1)
  env_vars         = try(each.value.env_vars, {})
  bq_dataset_roles = try(each.value.bq_dataset_roles, {})
  gcs_bucket_roles = {
    for name, role in try(each.value.gcs_bucket_roles, {}) :
    module.gcs_bucket[name].bucket_name => role
  }
  service_account_project_roles = try(each.value.service_account_project_roles, [])
  deployer_sa_email             = module.cloud_build.sa_email

  depends_on = [google_project_service.apis, module.bq_dataset, module.cloud_build, module.gcs_bucket]
}

module "bq_dataset" {
  source   = "./modules/bq_dataset"
  for_each = local.all_datasets

  project_id                  = var.project_id
  location                    = var.region
  dataset_id                  = each.key
  description                 = each.value.description
  default_table_expiration_ms = try(each.value.default_table_expiration_ms, null)

  depends_on = [google_project_service.apis]
}

module "cloud_build" {
  source = "./modules/cloud_build"

  project_id      = var.project_id
  region          = var.region
  github_owner    = local.cloud_build.cloud_build.github.owner
  github_repo     = local.cloud_build.cloud_build.github.repo
  connection_name = local.cloud_build.cloud_build.github.connection_name
  sa_roles        = try(local.cloud_build.cloud_build.default_sa_roles, [])
  triggers        = local.all_cloud_build_triggers

  depends_on = [google_project_service.apis]
}

module "bq_table" {
  source   = "./modules/bq_table"
  for_each = local.all_tables

  project_id        = var.project_id
  dataset_id        = each.value.dataset_id
  table_id          = split(".", each.key)[1]
  description       = each.value.description
  friendly_name     = try(each.value.friendly_name, null)
  schema            = each.value.schema
  time_partitioning = try(each.value.time_partitioning, null)
  clustering_fields = try(each.value.clustering_fields, null)

  depends_on = [module.bq_dataset]
}

module "gcs_bucket" {
  source   = "./modules/gcs_bucket"
  for_each = local.all_gcs_buckets

  project_id         = var.project_id
  location           = var.region
  bucket_name        = each.key
  force_destroy      = try(each.value.force_destroy, false)
  lifecycle_age_days = try(each.value.lifecycle_age_days, null)
  iam_bindings       = each.value.iam_bindings

  depends_on = [google_project_service.apis, module.cloud_build]
}

module "vertex_ai_batch_predict_sa" {
  source = "./modules/vertex_ai_pipeline"

  project_id       = var.project_id
  account_id       = "vertex-batch-predict-sa"
  display_name     = "Vertex AI batch prediction service account (churn training pipeline)"
  bq_dataset_roles = local.vertex_ai_batch_predict_sa.bq_dataset_roles
  gcs_bucket_roles = {
    for name, role in local.vertex_ai_batch_predict_sa.gcs_bucket_roles :
    module.gcs_bucket[name].bucket_name => role
  }

  depends_on = [google_project_service.apis, module.bq_dataset, module.gcs_bucket]
}

module "vertex_ai_pipeline" {
  source = "./modules/vertex_ai_pipeline"

  project_id                    = var.project_id
  account_id                    = "vertex-ai-pipeline-sa"
  display_name                  = "Vertex AI training pipeline service account"
  service_account_project_roles = local.vertex_ai_pipeline.service_account_project_roles
  bq_dataset_roles              = local.vertex_ai_pipeline.bq_dataset_roles
  gcs_bucket_roles = {
    for name, role in local.vertex_ai_pipeline.gcs_bucket_roles :
    module.gcs_bucket[name].bucket_name => role
  }
  # The pipeline's own SA is what actually calls CreateBatchPredictionJob (as the
  # ModelBatchPredictOp task runs under this SA's PipelineJob execution context), so it
  # needs to impersonate the narrower batch-predict SA to launch jobs under that identity.
  act_as_service_account_emails = [module.vertex_ai_batch_predict_sa.service_account_email]

  depends_on = [google_project_service.apis, module.bq_dataset, module.gcs_bucket, module.vertex_ai_batch_predict_sa]
}

module "sendgrid_secret" {
  source = "./modules/secret_manager"

  project_id = var.project_id
  secret_id  = "sendgrid-api-key"

  depends_on = [google_project_service.apis]
}

module "cloud_workflow" {
  source   = "./modules/cloud_workflow"
  for_each = local.all_workflows

  project_id                    = var.project_id
  region                        = var.region
  workflow_name                 = each.key
  description                   = each.value.description
  schedule                      = try(each.value.schedule, "")
  alert_email                   = var.alert_email
  alert_from_email              = var.alert_from_email
  alerts_enabled                = var.alerts_enabled
  source_contents               = file("${path.root}/../workflows/${each.key}.yaml")
  service_account_project_roles = try(each.value.service_account_project_roles, [])
  bq_dataset_roles              = try(each.value.bq_dataset_roles, {})
  gcs_bucket_roles = {
    for name, role in try(each.value.gcs_bucket_roles, {}) :
    module.gcs_bucket[name].bucket_name => role
  }
  act_as_service_account_emails = (
    try(each.value.act_as_vertex_pipeline_sa, false)
    ? [module.vertex_ai_pipeline.service_account_email]
    : []
  )

  depends_on = [google_project_service.apis, module.bq_dataset, module.cloud_run_job, module.gcs_bucket, module.vertex_ai_pipeline]
}
