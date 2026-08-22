locals {
  apis                       = yamldecode(file("${path.module}/config/apis.yaml"))
  cloud_build                = yamldecode(file("${path.module}/config/cloud_build.yaml"))
  artifact_registry          = yamldecode(file("${path.module}/config/artifact_registry.yaml"))
  gcs_buckets                = yamldecode(file("${path.module}/config/gcs_buckets.yaml"))
  cloud_run_jobs             = yamldecode(file("${path.module}/config/cloud_run_jobs.yaml"))
  workflows                  = yamldecode(file("${path.module}/config/workflows.yaml"))
  bigquery                   = yamldecode(file("${path.module}/config/bigquery.yaml"))
  triggers                   = yamldecode(file("${path.module}/config/triggers.yaml"))
  vertex_ai_pipeline         = yamldecode(file("${path.module}/config/vertex_ai_pipeline.yaml")).vertex_ai_pipeline
  vertex_ai_batch_predict_sa = yamldecode(file("${path.module}/config/vertex_ai_pipeline.yaml")).vertex_ai_batch_predict_sa

  all_datasets = local.bigquery.datasets

  # Flatten nested datasets.tables into a single map keyed by "dataset_id.table_id"
  all_tables = merge([
    for dataset_id, dataset in local.all_datasets : {
      for table_id, table in dataset.tables :
      "${dataset_id}.${table_id}" => merge(table, { dataset_id = dataset_id })
    }
  ]...)

  all_cloud_build_triggers = try(local.triggers.triggers, {})

  # Resolve full Artifact Registry image URIs for each Cloud Run Job
  all_cloud_run_jobs = {
    for job_name, job in local.cloud_run_jobs.cloud_run_jobs :
    job_name => merge(job, {
      image    = try(job.image_override, null)
      env_vars = merge(try(job.env_vars, {}), { BQ_PROJECT_ID = var.project_id })
    })
  }

  all_workflows = try(local.workflows.workflows, {})

  all_gcs_buckets = {
    for name, bucket in try(local.gcs_buckets.gcs_buckets, {}) :
    name => merge(bucket, { iam_bindings = try(bucket.iam_bindings, []) })
  }
}
