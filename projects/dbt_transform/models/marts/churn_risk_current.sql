{{ config(materialized='view') }}

/*
  The business-facing read model behind the churn dashboard: the newest scored snapshot,
  one row per customer, joined back to the features a non-technical reader can act on.

  A view, not a table, because of orchestration ordering. The daily DAG is
  data-gen -> dbt -> batch predict -> sync to ml.predictions -> drift monitor, so at the
  moment dbt runs, today's predictions do not exist yet. Materialising this would freeze
  yesterday's scores into a table named "current" and the dashboard would always trail the
  pipeline by a day. Adding a second dbt invocation after the sync step would fix the
  staleness at the cost of another orchestrator branch to fail in; a view moves the join to
  read time and needs no orchestration at all.

  Cost is bounded by the dashboard caching a whole snapshot in process (see
  projects/dashboard) rather than by anything here: this is queried a handful of times a
  day, not once per interaction.

  The label is deliberately absent. `churned` on customer_features is the *observed*
  outcome; putting it next to a predicted probability in a business UI invites reading the
  two as the same kind of fact. Retrospective accuracy belongs in the model evaluation
  reports, not in the retention worklist.
*/

WITH scored_snapshot AS (
    SELECT MAX(snapshot_date) AS snapshot_date
    FROM {{ source('ml', 'predictions') }}
)

SELECT
    p.customer_id,

    -- Model output
    p.churn_probability,
    p.churn_prediction,
    p.model_version,
    p.snapshot_date,
    p.predicted_at,

    -- Who the customer is: the segmentation axes a retention owner filters on
    f.membership_tier,
    f.region,
    f.preferred_channel,
    f.member_since_days,

    -- What is at stake. The generator emits monthly recurring amounts, so the 30-day
    -- average is a monthly run rate — the dashboard labels it as such rather than
    -- implying an annualised or lifetime figure it has no basis to claim.
    f.avg_transaction_30d AS monthly_value,
    f.avg_transaction_90d AS monthly_value_90d,

    -- Risk indicators. These are the features the dashboard contrasts against the cohort
    -- to explain a score. They correlate with churn; they are NOT the model's attribution
    -- for this customer, and the UI is required to say so.
    f.payment_failure_rate_30d,
    f.payment_failure_rate_90d,
    f.days_since_last_successful_payment,
    f.avg_engagement_30d,
    f.avg_engagement_90d,
    f.contact_requests_last_30d,
    f.campaign_participation_rate,
    f.events_last_30d,
    f.events_last_90d,
    f.renewal_count

FROM {{ source('ml', 'predictions') }} AS p
JOIN {{ ref('customer_features') }} AS f
    USING (customer_id, snapshot_date)
-- Both sides are filtered explicitly rather than relying on the join to narrow them:
-- a join predicate does not prune partitions, and customer_features retains 90 days.
WHERE p.snapshot_date = (SELECT snapshot_date FROM scored_snapshot)
  AND f.snapshot_date = (SELECT snapshot_date FROM scored_snapshot)
