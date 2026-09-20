WITH deduplicated AS (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY event_id ORDER BY event_timestamp DESC) AS _row_num
    FROM {{ source('raw', 'activity_cdc') }}
    WHERE NOT COALESCE(anomaly_injected, FALSE)
)

SELECT
    event_id,
    event_timestamp,
    customer_id,
    event_type,
    region,
    membership_tier,
    COALESCE(monthly_transaction, 0.0)         AS monthly_transaction,
    payment_status,
    COALESCE(payment_attempts_last_30d, 0)  AS payment_attempts_last_30d,
    engagement_score,
    member_since_days,
    COALESCE(contact_requests_last_30d, 0)  AS contact_requests_last_30d,
    COALESCE(churned, FALSE)                AS churned,
    channel
FROM deduplicated
WHERE _row_num = 1
