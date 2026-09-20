{{ config(materialized='view') }}

/*
  The dashboard's trend strip: one row per scored day, so a business reader can see whether
  the at-risk population is growing or shrinking rather than only where it stands tonight.

  Deliberately aggregates ml.predictions alone and never joins customer_features. The join
  would be the natural way to add revenue-at-risk-over-time, but it would also scan every
  retained feature partition on every dashboard load — turning the one query in this app
  whose cost grows with retention into the one that runs most often. Portfolio-level value
  lives in churn_risk_current for the day the user is actually working; here the question
  is only "which direction is this going", which counts and mean probability answer.

  HIGH_RISK_FLOOR mirrors the dashboard's own high-risk band. It is duplicated rather than
  shared because SQL and Python cannot read one constant, and the trend line would quietly
  disagree with the table below it if only one side were changed. Both sides are asserted
  against each other in projects/dashboard/tests/test_risk.py.
*/

SELECT
    snapshot_date,
    COUNT(*) AS customers_scored,
    COUNTIF(churn_prediction) AS predicted_churners,
    COUNTIF(churn_probability >= 0.7) AS high_risk_customers,
    AVG(churn_probability) AS mean_churn_probability,
    -- The share the model would act on at its registered decision threshold, which is what
    -- moves when a new champion is promoted with a different threshold.
    SAFE_DIVIDE(COUNTIF(churn_prediction), COUNT(*)) AS predicted_churn_rate,
    -- Surfaces a champion promotion as the discontinuity it usually is: a jump in the
    -- trend line that coincides with a change here is a model change, not a customer change.
    COUNT(DISTINCT model_version) AS model_versions_used,
    MAX(predicted_at) AS last_predicted_at
FROM {{ source('ml', 'predictions') }}
GROUP BY snapshot_date
ORDER BY snapshot_date
