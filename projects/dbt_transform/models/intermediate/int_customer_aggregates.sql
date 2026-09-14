/*
  Every rolling window here is anchored to the snapshot date — the day the features are
  computed — not to each customer's own most recent event.

  The distinction is the whole meaning of these columns. Anchoring per customer made
  "events_last_30d" mean "events in the 30 days before this customer last did anything",
  so a customer dormant for six weeks and one active this morning produced windows over
  completely different calendar periods and were indistinguishable in the feature matrix.
  It also guaranteed every window contained at least one event (the anchor event itself),
  so events_last_30d could never be 0 and dormancy — the single strongest churn signal —
  was unrepresentable.

  days_since_last_successful_payment showed the damage most starkly: measured from the
  customer's own last event, it was 0 for all 11,193 customers that had a value at all,
  while true calendar staleness spanned 16 days. A feature whose entire purpose is recency
  was a constant. Anchoring to the snapshot date makes it mean what its name says.

  CURRENT_DATE() matches the snapshot_date customer_features stamps in the same dbt run, so
  a feature row and its partition key always describe the same day.
*/

WITH source AS (
    SELECT * FROM {{ ref('stg_activity_cdc') }}
),

events_windowed AS (
    SELECT
        s.*,
        DATE_DIFF(CURRENT_DATE(), DATE(s.event_timestamp), DAY) AS days_since_event
    FROM source s
),

last_successful_payment AS (
    SELECT
        customer_id,
        -- Capped at PAYMENT_RECENCY_CAP_DAYS. Uncapped, this column is a clock: it is
        -- measured from CURRENT_DATE(), so every customer without a new successful payment
        -- gains exactly 1 per day and the whole distribution translates right overnight.
        -- Drift PSI compares it against a baseline frozen at training time, so the translation
        -- reads as drift that retraining cannot fix — it re-breaches within two days of every
        -- rebaseline (0.0001 the day the champion trained, 0.72/0.80/0.89 on the three days
        -- after). The mass point is what does the damage: everyone whose last success predates
        -- the CDC history piles up at the maximum, which is simply the dataset's age, and that
        -- pile lands in a bin the baseline has never seen. Capping collapses that pile into one
        -- stable bin so the feature stops moving once the history is older than the cap.
        LEAST(MIN(days_since_event), {{ var('payment_recency_cap_days') }}) AS days_since_last_successful_payment
    FROM events_windowed
    WHERE payment_status = 'success'
    GROUP BY customer_id
)

SELECT
    e.customer_id,

    -- Transaction rolling averages
    AVG(CASE WHEN e.days_since_event <= 30 THEN e.monthly_transaction END)  AS avg_transaction_30d,
    AVG(CASE WHEN e.days_since_event <= 90 THEN e.monthly_transaction END)  AS avg_transaction_90d,

    -- Payment health
    SAFE_DIVIDE(
        COUNTIF(e.days_since_event <= 30 AND e.payment_status = 'failed'),
        COUNTIF(e.days_since_event <= 30)
    )                                                                     AS payment_failure_rate_30d,
    SAFE_DIVIDE(
        COUNTIF(e.days_since_event <= 90 AND e.payment_status = 'failed'),
        COUNTIF(e.days_since_event <= 90)
    )                                                                     AS payment_failure_rate_90d,
    SUM(e.payment_attempts_last_30d)                                      AS total_payment_attempts,

    ANY_VALUE(lsp.days_since_last_successful_payment)                     AS days_since_last_successful_payment,

    -- Engagement rolling averages
    AVG(CASE WHEN e.days_since_event <= 30 THEN e.engagement_score END)  AS avg_engagement_30d,
    AVG(CASE WHEN e.days_since_event <= 90 THEN e.engagement_score END)  AS avg_engagement_90d,

    SAFE_DIVIDE(
        COUNTIF(e.event_type = 'campaign_action'),
        COUNT(*)
    )                                                                     AS campaign_participation_rate,

    APPROX_TOP_COUNT(e.channel, 1)[OFFSET(0)].value                      AS preferred_channel,

    -- Activity volume
    COUNT(*)                                                              AS total_events,
    -- Now genuinely 0 for a customer with no activity in the window. Under the previous
    -- per-customer anchoring these could never fall below 1, because the anchor event was
    -- always inside its own window.
    COUNTIF(e.days_since_event <= 30)                                     AS events_last_30d,
    COUNTIF(e.days_since_event <= 90)                                     AS events_last_90d,
    COUNTIF(e.event_type = 'membership_renewal')                          AS renewal_count

FROM events_windowed e
LEFT JOIN last_successful_payment lsp USING (customer_id)
GROUP BY e.customer_id
