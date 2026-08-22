# Serverless Customer Churn MLOps Platform

## Overview

This repository contains a production-grade, end-to-end MLOps platform for customer churn prediction built entirely on Google Cloud Platform (GCP). Moving beyond the limitations of static datasets and fragile time-based scheduling, this architecture implements a dynamic, living system capable of synthetic data generation, automated feature engineering, robust machine learning workflows, and closed-loop drift remediation.

The primary engineering objective of this project is to model an enterprise-ready infrastructure ecosystem. The core system isolates application concerns, enforces deterministic dependency management, runs fully serverless to optimize costs, and handles operational failures or data distribution shifts autonomously. All continuous integration and continuous deployment pipelines are natively driven by Google Cloud Build.

---

## Tech Stack

* **Data Warehouse:** Google BigQuery
* **Transformation Engine:** dbt Core (Containerized on Cloud Run)
* **ML Infrastructure:** Vertex AI Pipelines (Kubeflow SDK)
* **Experiment Tracking:** Vertex AI Experiments (native SDK)
* **Orchestration:** Google Cloud Workflows
* **CI/CD Platform:** Google Cloud Build
* **Code Quality:** Ruff (lint & format) enforced by pre-commit hooks
* **Compute Layer:** Cloud Run Jobs & Vertex AI Training
* **Infrastructure as Code:** Terraform
* **Data Generation:** Python (NumPy)
* **Email Delivery:** SendGrid (transactional operational alerts)

---

## System Architecture

The diagram below shows the GCP services this platform is built on and how they connect — from CI/CD, through the daily ingestion/transformation/serving loop, to the drift-triggered retraining pipeline.

```mermaid
flowchart TB
    GH[GitHub Repository] -->|push to main| CB[Cloud Build]

    subgraph CICD["CI/CD"]
        CB
    end

    CB -->|test, build, push images| AR_C[Artifact Registry<br/>containers repo]
    CB -->|compile & stage template| AR_K[Artifact Registry<br/>pipeline-templates repo]
    CB -->|gcloud workflows deploy| WF

    subgraph ORCH["Orchestration"]
        SCHED[Cloud Scheduler] -->|daily trigger| WF[Cloud Workflows<br/>orchestrator-workflow]
    end

    subgraph DATA["Data Ingestion & Transformation"]
        DG[Cloud Run Job<br/>data-gen-job] --> BQ_RAW[(BigQuery<br/>raw.activity_cdc)]
        BQ_RAW --> DBT[Cloud Run Job<br/>dbt-job]
        DBT --> BQ_FEAT[(BigQuery<br/>features.customer_features)]
    end

    WF -->|Task A| DG
    WF -->|Task B| DBT

    subgraph SERVE["Serving & Monitoring"]
        BQ_FEAT --> BP[Vertex AI<br/>Batch Prediction]
        BP --> BQ_PRED[(BigQuery<br/>ml.predictions)]
        BQ_FEAT --> DM[Cloud Run Job<br/>drift-monitor-job]
        DM --> GCS_META[(GCS bucket<br/>pipeline-metadata)]
    end

    WF -->|Task C| BP
    WF -->|Task D| DM
    GCS_META -->|drift decision JSON| WF

    subgraph ML["ML Training Pipeline (Vertex AI Pipelines)"]
        VP[Vertex AI PipelineJob<br/>churn-training-pipeline]
        VP --> VE[Vertex AI Experiments]
        VP --> MR[Vertex AI Model Registry]
        VP --> GCS_TRAIN[(GCS bucket<br/>training-data)]
    end

    WF -->|drift detected:<br/>submit PipelineJob, fire-and-forget| VP
    AR_K -.->|template source| VP
    AR_C -.->|trainer/post-training/serving images| VP
    MR -.->|champion model| BP

    WF -->|failure & drift alerts| SG[SendGrid<br/>transactional email]

    classDef gcp fill:#4285F4,color:#ffffff,stroke:#1a56c4;
    classDef ext fill:#6b7280,color:#ffffff,stroke:#374151;
    class CB,AR_C,AR_K,SCHED,WF,DG,DBT,BP,DM,VP,VE,MR,BQ_RAW,BQ_FEAT,BQ_PRED,GCS_META,GCS_TRAIN gcp;
    class GH,SG ext;
```

Every deployable in this diagram is provisioned by Terraform — see [docs/iac.md](docs/iac.md). For the step-by-step daily and retraining flows, see [docs/orchestration.md](docs/orchestration.md).

---

## Documentation

| Topic | File |
|-------|------|
| Prerequisites & Setup | [docs/setup.md](docs/setup.md) |
| Software Architecture & Monorepo Design | [docs/architecture.md](docs/architecture.md) |
| Data Lifecycle & ELT Engine | [docs/data-lifecycle.md](docs/data-lifecycle.md) |
| Event-Driven Orchestration | [docs/orchestration.md](docs/orchestration.md) |
| Model Development (DS guide) | [docs/model-development.md](docs/model-development.md) |
| ML Infrastructure (MLE guide) | [docs/ml-infrastructure.md](docs/ml-infrastructure.md) |
| Feature Exploration & Selection | [docs/feature-exploration.md](docs/feature-exploration.md) |
| Observability & Drift Remediation | [docs/observability.md](docs/observability.md) |
| Infrastructure as Code (Terraform) | [docs/iac.md](docs/iac.md) |
| CI/CD Deployment Blueprint | [docs/cicd.md](docs/cicd.md) |

---

## License

Licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE.md), provided "as is", without warranty of any kind.

* **Noncommercial use** (personal, research, education, nonprofit/government) — free to use, modify, and redistribute.
* **Commercial or business use** — requires a separate paid license. Contact [marcopellegrinoit](https://github.com/marcopellegrinoit).
