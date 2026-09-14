# Data Lifecycle & ELT Engine

The data engine operates on an automated, event-driven ELT (Extract, Load, Transform) pattern designed to simulate real-world transactional applications.

---

## Synthetic Data Generation (CDC Simulation)

To accurately model an enterprise data stream, a data generator application executes as an ephemeral task on a scheduled frequency. Crucially, **this job simulates a Change Data Capture (CDC) feed from an upstream transactional database** (like PostgreSQL).

Instead of overwriting a static state table, it generates an append-only log of synthetic user activity, billing cycles, profile adjustments, and service interaction events. Appending these raw, immutable telemetry records directly to ingestion tables in BigQuery guarantees point-in-time correctness, prevents historical data leakage, and forces the downstream ELT layer to correctly reconstruct user features. To test system resilience, the generator also exposes configuration variables capable of manually injecting structural feature anomalies or sudden behavioral shifts.

### Configuration Variables

The generator is fully controlled by environment variables injected at Cloud Run Job runtime:

| Variable | Required | Default | Description |
|---|---|---|---|
| `BQ_PROJECT_ID` | Yes | — | GCP project hosting the target BigQuery dataset |
| `BQ_DATASET_ID` | Yes | — | Target dataset (e.g. `raw`) |
| `BQ_TABLE_ID` | Yes | — | Target table (e.g. `activity_cdc`) |
| `BATCH_SIZE` | No | `2000` | Number of events to generate and stream-insert per invocation |
| `ANOMALY_RATE` | No | `0.0` | Fraction of events that carry injected out-of-range feature values (0.0–1.0) |
| `USER_POOL_SIZE` | No | `10000` | Founding customer base, acquired on the pool epoch |
| `DAILY_ACQUISITIONS` | No | `25` | New customers acquired per elapsed day since the pool epoch |

### Customer Pool

Customer identities are **stable across runs**. Every profile is derived deterministically from its index — `uuid5` for the id, an RNG seeded with the index for its traits — so customer *n* is the same customer on every invocation and their events accumulate into a single history. Nothing is persisted between runs to achieve this; the pool is reconstructed identically each time.

This is load-bearing rather than cosmetic. Every column in `customer_features` is a window aggregate over one customer's own history — `avg_transaction_30d`, `events_last_90d`, `payment_failure_rate_30d`, `days_since_last_successful_payment`, `renewal_count`. An earlier version minted a fresh `uuid4()` pool on every run, so no customer ever received a second event. Measured in production, that produced **32,609 customers over 14 days at 1.1 events each, with zero spanning more than one day**: every window aggregate collapsed to a constant or a two-to-three-value discrete, and the `churned` label — read from the customer's most recent event — was a single coin flip uncorrelated with any accumulated behaviour. The model was being fit to noise, and the drift monitor was watching features that could not move.

| Trait | Description | Missing value effect |
|---|---|---|
| `preferred_channel` | The customer's primary interaction channel | Drives the `channel` field on every event |
| `is_offline_only` | `True` when channel is `direct_mail` or `phone` (~33% of customers) | `engagement_score = NULL` on every event — **MAR**: missingness is fully explained by `channel` |
| `always_fails_payments` | `True` for ~8% of customers | `payment_status` is always `failed` or `pending`, never `success` — produces `days_since_last_successful_payment = NULL` in the feature matrix — **MNAR**: missingness correlates with elevated churn |
| `tenure_days_at_join` | Tenure on joining; `None` for ~3% of customers | `member_since_days = NULL` in the feature matrix — **MCAR**: missing at random, no correlation with outcome |
| `membership_tier`, `region` | Fixed per customer | Previously drawn per event, which made a customer appear in a different tier and region on every event — with `customer_features` reading whichever the latest event carried, both columns were noise |

`member_since_days` is `tenure_days_at_join` plus the days since that customer joined, so tenure advances with time instead of being a fixed random draw. Daily acquisition is what keeps the resulting distribution at a steady state: in a closed base, tenure would climb for everyone at once and become a permanent source of drift.

This design ensures that missing values in the feature matrix arise from structurally consistent customer behaviour rather than random per-event noise, making them suitable for imputation strategy selection during model training.

### Churn Is Absorbing

A customer who churns emits a final `membership_cancelled` event and is never selected again. The generator reads back the set of already-churned customers from `raw.activity_cdc` at the start of each run and excludes them, and within a run a customer is removed from the selectable set the moment their event comes back flagged.

Both halves matter, because `customer_features` derives the label from the customer's *most recent* event. Without them, a customer would flicker between churned and not-churned as later events overwrote earlier ones, and the target would describe a moment rather than the customer. `membership_cancelled` is correspondingly reserved for a real churn rather than being one of the event types an active customer emits at random.

A read failure degrades to treating the whole base as active. That keeps generating events for customers who have left — a visible data problem rather than a silent one — which is preferable to failing the run and producing nothing.

### Event Schema

The row is declared once as `data_generator.schema.ActivityCdcRow` and validated before it is streamed, so the generator cannot emit a row that `raw.activity_cdc` will not accept. The categorical vocabularies (`event_type`, `region`, `membership_tier`, `payment_status`, `channel`) are `StrEnum`s in that module rather than loose string lists — the model's categorical splits are frozen against exactly these values at training time, and a value outside them silently becomes NaN at inference.

Numeric ranges are deliberately *not* constrained by the contract: the generator injects out-of-range values on purpose (see [Anomaly Injection](#anomaly-injection)), and bounding them here would reject the very rows the drift and data-quality machinery exists to catch. Validation covers shape and type; plausibility is the data-quality gate's job, further downstream. See [architecture.md](architecture.md#data-contracts--configuration).

Every row written to BigQuery contains the following fields:

| Field | Type | Description |
|---|---|---|
| `event_id` | STRING | Unique UUID for this event |
| `event_timestamp` | TIMESTAMP | UTC time of generation (ISO 8601) |
| `customer_id` | STRING | UUID identifying the synthetic customer |
| `event_type` | STRING | One of: `transaction`, `membership_renewal`, `campaign_action`, `email_engagement`, `event_attended`, `contact_request`, `membership_cancelled` |
| `region` | STRING | One of: `us-east`, `us-west`, `eu-west`, `eu-central`, `apac`, `latam` |
| `membership_tier` | STRING | One of: `supporter`, `friend`, `champion`, `guardian` |
| `monthly_transaction` | FLOAT64 | Sampled from a Normal distribution anchored to the tier's base amount; clipped to ≥ 0 |
| `payment_status` | STRING | One of: `success`, `failed`, `pending` |
| `payment_attempts_last_30d` | INT64 | Sampled from a Poisson(λ=1.5) distribution |
| `engagement_score` | FLOAT64 \| NULL | Sampled from a Beta(α=2, β=5) distribution; range [0, 1]. **NULL** for offline-channel customers (`direct_mail`, `phone`) — MAR |
| `member_since_days` | INT64 \| NULL | Uniform random integer in [1, 3650]. **NULL** for ~3% of customers with no recorded start date — MCAR |
| `contact_requests_last_30d` | INT64 | Sampled from a Poisson(λ=0.3) distribution |
| `churned` | BOOL | Derived probabilistically (see churn logic below) |
| `channel` | STRING | One of: `email`, `web`, `mobile`, `direct_mail`, `in_person`, `phone` |
| `raw_payload` | STRING | JSON-serialised snapshot of all core fields (excludes itself and `anomaly_injected`) |
| `anomaly_injected` | BOOL | `TRUE` when the event carries deliberately out-of-range values |

### Churn Signal Logic

Churn probability is not uniform — it is elevated for customers exhibiting distress signals:

| Condition | Per-event churn hazard |
|---|---|
| `engagement_score IS NULL` **or** `engagement_score < 0.15` **or** `payment_attempts_last_30d > 5` **or** `always_fails_payments = True` | **2%** |
| Otherwise | **0.2%** |

This asymmetry creates a learnable signal for the downstream ML model without making churn trivially predictable.

These are per-event *hazards* on an absorbing state, not the per-event coin flip they replace. A customer now sees many events over their lifetime and each one is a fresh opportunity to churn, so the rates are an order of magnitude lower than the previous 25%/4% — at the old values the entire base would cancel within days.

### Anomaly Injection

When `ANOMALY_RATE > 0`, a fraction of events replace their numerical fields with deliberately out-of-range values. This simulates upstream data quality failures and exercises the drift detection layer:

| Field | Normal range | Injected anomaly values |
|---|---|---|
| `monthly_transaction` | ≥ 0 | `-999.0`, `0.0`, `9999.99` |
| `engagement_score` | [0, 1] | `-1.0`, `5.0`, `99.9` |
| `payment_attempts_last_30d` | ≥ 0 | 50–200 |

Anomalous events are flagged with `anomaly_injected = TRUE`, allowing downstream queries to isolate or filter injected rows independently of the drift detection system.

---

## Feature Engineering Layer

Transformations, aggregations, and windowing functions are completely decoupled from Python code and handled within the data warehouse layer using a containerized instance of `dbt Core`. The dbt project lives at `projects/dbt_transform/` and runs as a Cloud Run Job (`dbt-job`). On each invocation it executes `dbt run --profiles-dir .`, which writes a complete feature snapshot for every customer into a new daily partition of `features.customer_features`.

### Configuration Variables

The dbt job is controlled by environment variables injected at Cloud Run Job runtime:

| Variable | Required | Default | Description |
|---|---|---|---|
| `BQ_PROJECT_ID` | Yes | *(injected by Terraform)* | GCP project hosting BigQuery |
| `BQ_LOCATION` | No | `EU` | BigQuery dataset location for job routing |

### Model Graph & Materialization Strategy

The dbt project executes a six-model graph. Staging and intermediate models use **ephemeral** materialization — they compile into CTEs inlined directly into the final query and create no BigQuery objects. The `customer_features` mart uses **incremental** materialization with `insert_overwrite`: on each run dbt computes a full feature snapshot for all customers and writes it into the partition matching `CURRENT_DATE()`, overwriting that day's partition if it already exists and leaving all previous partitions untouched. This gives a complete, queryable history of feature states keyed by `snapshot_date`.

A **mart** (short for data mart) is the final, consumer-ready layer in a dbt project. Where staging and intermediate models clean and reshape raw data, a mart assembles everything into a purpose-built table that downstream systems — in this case Vertex AI training and the serving endpoint — can query directly without any further transformation.

```mermaid
flowchart TD
    RAW[("raw.activity_cdc<br/>BigQuery source (append-only)")]
    STG["models/staging/stg_activity_cdc<br/>[EPHEMERAL]<br/>Deduplication, anomaly filtering, null coalescing"]
    INT["models/intermediate/int_customer_aggregates<br/>[EPHEMERAL]<br/>30/90-day rolling aggregates,<br/>anchored to snapshot_date"]
    MART[("models/marts/customer_features<br/>[INCREMENTAL, insert_overwrite]<br/>→ features.customer_features<br/>One row per customer per snapshot_date partition")]
    CUR[("models/marts/customer_features_current<br/>[TABLE] → features.customer_features_current<br/>Newest partition, label dropped — the batch-predict input")]
    PRED[("ml.predictions<br/>BigQuery source — written by the workflow's<br/>batch-predict sync, never built by dbt")]
    DASH1["models/marts/churn_risk_current<br/>[VIEW] → features.churn_risk_current<br/>Newest scored snapshot, business columns"]
    DASH2["models/marts/churn_risk_daily<br/>[VIEW] → features.churn_risk_daily<br/>One row per scored day"]

    RAW --> STG
    STG --> INT
    STG --> MART
    INT --> MART
    MART --> CUR
    MART --> DASH1
    PRED --> DASH1
    PRED --> DASH2
```

### Layer 1 — Staging: `stg_activity_cdc`

**Source:** `raw.activity_cdc`  
**File:** `models/staging/stg_activity_cdc.sql`  
**Materialization:** ephemeral

Cleans the raw CDC log before any aggregation:

- **Anomaly filtering** — rows where `anomaly_injected = TRUE` are excluded so that injected out-of-range values never contaminate the feature matrix.
- **Deduplication** — BigQuery streaming inserts can produce short-window duplicates. A `ROW_NUMBER()` window partitioned on `event_id` keeps the latest version of each event.
- **Selective null coalescing** — count fields where `0` is the correct missing-value substitute (`monthly_transaction`, `payment_attempts_last_30d`, `contact_requests_last_30d`) are coalesced to `0`; `churned` to `FALSE`. `engagement_score` and `member_since_days` are intentionally left as `NULL` so their missingness propagates to the feature matrix for ML imputation.
- `raw_payload` and `anomaly_injected` are dropped — they are not needed downstream.

### Layer 2 — Intermediate: `int_customer_aggregates`

**Source:** `stg_activity_cdc`  
**File:** `models/intermediate/int_customer_aggregates.sql`  
**Materialization:** ephemeral

Computes per-customer behavioral aggregates. All rolling windows are **relative to the snapshot date** — `DATE_DIFF(CURRENT_DATE(), DATE(event_timestamp), DAY)` as `days_since_event` — matching the `snapshot_date` that `customer_features` stamps in the same run, so a feature row and its partition key always describe the same day.

They were previously anchored to each customer's own `MAX(event_timestamp)`, which is the difference between a feature that measures recency and one that cannot. Under per-customer anchoring, `events_last_30d` meant "events in the 30 days before this customer last did anything": a customer dormant for six weeks and one active this morning described entirely different calendar periods in the same column and were indistinguishable to the model. Every window also necessarily contained the anchor event itself, so `events_last_30d` could never be 0 and dormancy — the strongest churn signal there is — was unrepresentable.

`days_since_last_successful_payment` showed the cost most plainly. Measured from the customer's own last event it was **0 for all 11,193 customers that had a value at all**, while true calendar staleness spanned 16 days; the column whose entire purpose is recency was a constant, which is also why the drift monitor reported it as carrying no usable signal. Anchored to the snapshot date it takes 14 distinct values across 0–16 days.

Snapshot anchoring fixed the constant, but it introduced the opposite failure: the column became a **clock**. Measured from `CURRENT_DATE()`, every customer without a new successful payment gains exactly 1 per day, so the whole distribution translates right overnight. Drift PSI compares it against a baseline frozen at training time, which makes that translation read as drift — and retraining cannot fix a clock. Observed against the champion trained on `2026-09-11`: PSI **0.0001** the day the baseline was frozen, then **0.72 / 0.80 / 0.89** on the three days after, re-breaching the 0.20 threshold within two days of every rebaseline. Three retrains fired and all were rejected, each challenger landing within +0.005 PR-AUC of the champion.

The mass point is what does the damage. Everyone whose last success predates the CDC history piles up at the maximum — which is simply the dataset's age, since `raw.activity_cdc` starts at `2026-08-22` — and that pile lands in a bin the frozen baseline has never seen. The champion's baseline held **12.8%** at `20`; three days later the same pile sat at `23`, and bin `20` was empty.

`days_since_last_successful_payment` is therefore **capped** at the `payment_recency_cap_days` var (currently **30**, set in `dbt_project.yml`), which collapses that pile into one stable bin so the feature stops moving once the history is older than the cap. Past a month, "stale" carries no further churn signal, so the truncated tail costs nothing. Two consequences worth knowing: the cap only binds once the history exceeds it (from `2026-09-21`, given the `2026-08-22` start), so the mechanical drift continues until then; and changing the var changes feature semantics, so it requires a retrain to take effect. `member_since_days` is anchored the same way and will develop the identical problem as tenure accumulates — it is not capped yet.

### Accumulators and the same failure mode

`days_since_last_successful_payment` was the acute case, but any *lifetime* aggregate over `raw.activity_cdc` has the same defect in slower motion: the log only ever grows, so a lifetime count encodes the dataset's age as much as the customer's behaviour, and can never match a baseline frozen at training time. `total_payment_attempts` and `renewal_count` are therefore computed over `activity_window_days` (90, matching `events_last_90d` and the rate columns) rather than over all history. The change is inert until the CDC history is older than the window — verified as **0 rows differing** at a 23-day history — so it costs nothing now and prevents the drift later.

Two known exceptions, both deliberate:

* **`total_events` is still lifetime.** Windowing it at 90 days would make it identical to `events_last_90d`, and dropping it is destructive: BigQuery cannot drop a column through a schema update, so Terraform would replace `features.customer_features` (the module sets `deletion_protection = false`), taking the historical snapshots that drift baselines and `ml.split_assignments` depend on with it. It needs a migration, not an edit.
* **`member_since_days` is not affected in practice.** It is anchored to `CURRENT_DATE()` and does climb, but its spread is years, so a day's shift is negligible: PSI has stayed at **0.0002** while the payment-recency PSI went to 0.89.

Features that are windowed or expressed as ratios — `events_last_30d`, `events_last_90d`, `payment_failure_rate_*`, `campaign_participation_rate`, `avg_*` — are climbing today only because the history is younger than their windows. They are self-correcting and need no change.

A consequence worth expecting: windows can now legitimately be empty. `events_last_30d = 0` and a `NULL` `avg_transaction_30d` mean "this customer did nothing in that window", which is information, not a defect. (It reads as 0 dormant customers today only because `raw.activity_cdc` holds ~16 days of events so far — nobody *can* be 30-day dormant until the history is older than the window.)

| Feature group | Columns produced |
|---|---|
| **Transaction rolling averages** | `avg_transaction_30d`, `avg_transaction_90d` |
| **Payment health** | `payment_failure_rate_30d`, `payment_failure_rate_90d` (failed / total events in window), `total_payment_attempts` (over `activity_window_days`) |
| **Payment recency** | `days_since_last_successful_payment` — days since last `payment_status = 'success'`, capped at `payment_recency_cap_days` (30) to keep it stationary; **NULL** when no successful payment exists (MNAR — see Customer Pool section) |
| **Engagement rolling averages** | `avg_engagement_30d`, `avg_engagement_90d` |
| **Engagement behaviour** | `campaign_participation_rate` (fraction of events that are `campaign_action`), `preferred_channel` (mode via `APPROX_TOP_COUNT`) |
| **Activity counts** | `total_events` (lifetime — see below), `events_last_30d`, `events_last_90d`, `renewal_count` (over `activity_window_days`) |

### Layer 3 — Mart: `customer_features`

**Sources:** `stg_activity_cdc`, `int_customer_aggregates`  
**File:** `models/marts/customer_features.sql`  
**Materialization:** incremental (`insert_overwrite`, partitioned by `snapshot_date`) → `features.customer_features`

Produces one row per customer by joining the rolling aggregates with the customer's latest state. The latest state is extracted with `QUALIFY ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY event_timestamp DESC) = 1`.

`snapshot_date` partitions expire after 90 days (see [iac.md](iac.md#bigquery-tables-bq_table-module)) so the table doesn't grow unbounded from one full customer snapshot being appended every dbt run. This is safe because nothing reads an old partition: `data_split` freezes the exact rows each training run used into `ml.split_assignments`, which never expires (see [ml-infrastructure.md](ml-infrastructure.md)).

#### Full schema

| Column | Type | Description |
|---|---|---|
| `customer_id` | STRING | Customer UUID — primary key |
| `membership_tier` | STRING | Tier at most recent event (`supporter` / `friend` / `champion` / `guardian`) |
| `region` | STRING | Region at most recent event |
| `member_since_days` | INT64 \| NULL | Tenure in days at most recent event; NULL for ~3% of customers (MCAR) |
| `contact_requests_last_30d` | INT64 | Contact requests at most recent event |
| `avg_transaction_30d` | FLOAT64 | Average monthly transaction over last 30 days |
| `avg_transaction_90d` | FLOAT64 | Average monthly transaction over last 90 days |
| `payment_failure_rate_30d` | FLOAT64 | Fraction of payments that failed in last 30 days |
| `payment_failure_rate_90d` | FLOAT64 | Fraction of payments that failed in last 90 days |
| `total_payment_attempts` | INT64 | Cumulative sum of `payment_attempts_last_30d` across all events |
| `days_since_last_successful_payment` | INT64 \| NULL | Days since last success, capped at 30; NULL = no successful payment on record (MNAR) |
| `avg_engagement_30d` | FLOAT64 \| NULL | Average engagement score over last 30 days; NULL for offline-channel customers (MAR) |
| `avg_engagement_90d` | FLOAT64 \| NULL | Average engagement score over last 90 days; NULL for offline-channel customers (MAR) |
| `campaign_participation_rate` | FLOAT64 | Fraction of all events that are `campaign_action` |
| `preferred_channel` | STRING | Most frequent interaction channel |
| `total_events` | INT64 | Total event count across full history |
| `events_last_30d` | INT64 | Event count in last 30 days |
| `events_last_90d` | INT64 | Event count in last 90 days |
| `renewal_count` | INT64 | Total `membership_renewal` events |
| `latest_event_ts` | TIMESTAMP | Timestamp of the customer's most recent event |
| `feature_computed_at` | TIMESTAMP | UTC time when this row was materialized |
| **`churned`** | **BOOL** | **Target label** — TRUE if most recent event signals churn |
| `snapshot_date` | DATE | Partition key — the calendar date on which dbt ran and produced this snapshot |

This table is the direct input to the Vertex AI training pipeline and the serving endpoint for real-time inference.

### `customer_features_current`

A second marts model holding only the newest partition of `customer_features`, with the label and build-metadata timestamps removed. It exists because Vertex AI's `bigquerySource` takes a table reference with no row filter, so scoping the daily scoring run to one snapshot has to happen in the transformation layer. Built by the same `dbt build` invocation as the mart, so it can never drift out of step with it. See [ml-infrastructure.md](ml-infrastructure.md) for why the daily job must not score the full mart.

### `churn_risk_current` and `churn_risk_daily`

Two marts models backing the business dashboard — both **views**, and both reading `ml.predictions`, which dbt declares as a *source* and never builds or tests. See [dashboard.md](dashboard.md) for the full design.

Views rather than tables, because of where dbt sits in the daily DAG. The orchestrator runs `data-gen → dbt → batch predict → sync → drift monitor`, so at the moment dbt runs, *today's predictions do not exist yet*. Materialising a model called `churn_risk_current` would freeze yesterday's scores into it and leave the dashboard permanently a day behind the pipeline. Adding a second dbt invocation after the sync step would fix that at the cost of another orchestrator branch to fail in; a view moves the join to read time and needs no orchestration at all.

| Model | Grain | What it is for |
|---|---|---|
| `churn_risk_current` | One row per customer, newest scored snapshot | Joins `ml.predictions` to `customer_features` and renames the columns a non-technical reader works with. **Drops `churned`** — an observed outcome must not sit beside a predicted probability in a business UI. |
| `churn_risk_daily` | One row per scored day | Aggregates `ml.predictions` alone. The `customer_features` join would add revenue-over-time and make this the one dashboard query whose cost grows with retention — and it runs on every page load. |

`churn_risk_current` filters **both** sides of its join to the scored snapshot explicitly. A join predicate does not prune partitions, and `customer_features` retains 90 days, so relying on the join to narrow it would scan the lot.

`churn_risk_daily` hardcodes the dashboard's high-risk floor in a `COUNTIF`, because SQL cannot import a Python constant. A test in `projects/dashboard/tests/test_risk.py` reads this file and asserts it still matches the app's default — without it the trend line and the table beneath it could describe different populations while both are labelled "high risk".

### Contract enforcement

The dbt container's entrypoint is `dbt build`, not `dbt run`. `build` interleaves each model with its declared `schema.yml` tests in DAG order, so a model whose contract is violated fails rather than silently feeding everything downstream. Under `dbt run` those tests were declared but never executed — `customer_features`' grain, `event_id` uniqueness and the `membership_tier` domain were documented guarantees that nothing checked. A failure here halts the orchestrator *before* batch prediction, which is earlier than the drift monitor's data-quality gate can catch the same class of problem (see [observability.md](observability.md)).

`customer_features`' uniqueness test is on `(customer_id, snapshot_date)`, not `customer_id` alone. The single-column test used to pass only because the generator minted a fresh customer pool every run, so no customer ever appeared on two days; with stable identities the same customer legitimately recurs across partitions.

> **Data mart vs. feature store:** `customer_features` is a **data mart** — a denormalized, consumer-ready table scoped to a specific use case (churn prediction). The term describes *how data is shaped and where it lives*, nothing more. A **feature store** is a higher-level MLOps infrastructure component built on top of that kind of storage. It adds: (1) **point-in-time correctness** — serving only features that existed before each label's timestamp to prevent training/serving skew; (2) **online + offline serving** — a low-latency store (e.g. Bigtable or Redis) alongside the offline warehouse table; and (3) **feature versioning and reuse** — a central registry that multiple models and teams share. In a real enterprise scenario the correct solution here is **Vertex AI Feature Store**, which provides all three. The partitioned BigQuery table used here is a pragmatic approximation: it achieves reproducibility by pinning a training job to a specific `snapshot_date` partition, but it requires the caller to enforce point-in-time correctness manually and has no online-serving layer. Feature Store was omitted to keep the infrastructure footprint simple and self-contained.

---

## Data Flow Summary

```mermaid
flowchart TD
    SCHED[Cloud Scheduler] -->|triggers on schedule| DG["Cloud Run Job<br/>(data_generator)"]
    DG -->|streams BATCH_SIZE events| RAW[("BigQuery<br/>raw.activity_cdc")]
    RAW -->|triggers Cloud Workflows| DBT["Cloud Run Job<br/>(dbt_transform)"]
    DBT -->|materializes feature matrix| FEAT[("BigQuery<br/>features.customer_features<br/>(90-day partition expiration)")]

    FEAT --> TRAIN[Vertex AI Training Pipeline]
    FEAT --> SERVE[Vertex AI Serving Endpoint]

    TRAIN -->|"data_split freezes a stratified split,<br/>one snapshot_date partition per run"| SPLIT[("BigQuery<br/>ml.split_assignments<br/>(no partition expiration —<br/>permanent lineage record)")]

    SPLIT --> HPO["hpo / train / evaluate<br/>read directly (once per task)"]
    SPLIT --> SCRATCH["prep_test_batch_data materializes<br/>scratch.test_batch_&lt;uuid&gt;<br/>(1-day default_table_expiration_ms)"]
    SCRATCH -->|exports as GCS JSONL| BPSRC[Batch Prediction source]
```

See [ml-infrastructure.md](ml-infrastructure.md#why-the-split-lives-in-bigquery-not-pandasparquet) for why the split is computed and frozen in BigQuery rather than loaded into pandas.
