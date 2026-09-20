{{ config(
    materialized='incremental',
    partition_by={'field': 'snapshot_date', 'data_type': 'date', 'granularity': 'day'},
    incremental_strategy='insert_overwrite',
    on_schema_change='fail'
) }}

WITH latest_state AS (
    SELECT
        customer_id,
        membership_tier,
        region,
        member_since_days,
        contact_requests_last_30d,
        churned,
        event_timestamp AS latest_event_ts
    FROM {{ ref('stg_activity_cdc') }}
    QUALIFY ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY event_timestamp DESC) = 1
),

aggregates AS (
    SELECT * FROM {{ ref('int_customer_aggregates') }}
)

SELECT
    ls.customer_id,

    -- Identity & tenure
    ls.membership_tier,
    ls.region,
    ls.member_since_days,
    ls.contact_requests_last_30d,

    -- Transaction features
    a.avg_transaction_30d,
    a.avg_transaction_90d,

    -- Payment health
    a.payment_failure_rate_30d,
    a.payment_failure_rate_90d,
    a.total_payment_attempts,
    a.days_since_last_successful_payment,

    -- Engagement
    a.avg_engagement_30d,
    a.avg_engagement_90d,
    a.campaign_participation_rate,
    a.preferred_channel,

    -- Activity
    a.total_events,
    a.events_last_30d,
    a.events_last_90d,
    a.renewal_count,

    -- Metadata
    ls.latest_event_ts,
    CURRENT_TIMESTAMP()                     AS feature_computed_at,

    -- Target label
    ls.churned,

    -- Partition key
    CURRENT_DATE()                          AS snapshot_date

FROM latest_state ls
LEFT JOIN aggregates a USING (customer_id)
