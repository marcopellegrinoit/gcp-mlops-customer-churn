{{ config(materialized='table') }}

/*
  The daily scoring input: exactly one snapshot, the newest one.

  Vertex AI's bigquerySource takes a table reference with no row filter, so scoping the
  daily BatchPredictionJob to today's customers has to happen upstream — hence a dedicated
  model rather than a WHERE clause in the workflow. It is rebuilt by the same `dbt run` that
  builds customer_features, so it is always in step with it and needs no separate
  orchestration step.

  Pointing the job at customer_features itself meant re-scoring every retained partition
  every night: ~99% of the work reproduced predictions for customer-days whose features are
  immutable and whose decision window closed weeks ago. Worse than the cost, the MERGE into
  ml.predictions then overwrote those historical rows, so `predicted_at` ("when this
  prediction was written") and `model_version` ("the champion used to score this row") were
  restamped nightly and no longer described the past. Only one champion has ever existed, so
  nothing contradicted itself yet — but the first promotion would have silently rewritten
  every retained day with a model that did not exist on those days.

  The label and the build-metadata timestamps are dropped. instanceConfig.instanceType is
  "object", so every column present here is serialised into the instance sent to the serving
  container; select_inference_features discards anything outside the frozen feature set, but
  shipping the target into a scoring request is not something to rely on being ignored.
  customer_id and snapshot_date stay: the batch job echoes source columns into its output
  table, and sync_predictions_to_ml_predictions keys its MERGE on exactly that pair.
*/

SELECT * EXCEPT (churned, latest_event_ts, feature_computed_at)
FROM {{ ref('customer_features') }}
-- MAX rather than CURRENT_DATE(): customer_features stamps snapshot_date at its own build
-- time, and a dbt run that crosses midnight UTC would otherwise leave this model empty.
WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM {{ ref('customer_features') }})
