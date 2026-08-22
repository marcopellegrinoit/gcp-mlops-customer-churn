WITH source AS (
    SELECT * FROM {{ ref('stg_activity_cdc') }}
),

customer_max_ts AS (
    SELECT
        customer_id,
        MAX(event_timestamp) AS latest_event_ts
    FROM source
    GROUP BY customer_id
),

events_windowed AS (
    SELECT
        s.*,
        TIMESTAMP_DIFF(d.latest_event_ts, s.event_timestamp, DAY) AS days_since_event
    FROM source s
    JOIN customer_max_ts d USING (customer_id)
),

last_successful_payment AS (
    SELECT
        customer_id,
        MIN(days_since_event) AS days_since_last_successful_payment
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
    COUNTIF(e.days_since_event <= 30)                                     AS events_last_30d,
    COUNTIF(e.days_since_event <= 90)                                     AS events_last_90d,
    COUNTIF(e.event_type = 'membership_renewal')                          AS renewal_count

FROM events_windowed e
LEFT JOIN last_successful_payment lsp USING (customer_id)
GROUP BY e.customer_id
