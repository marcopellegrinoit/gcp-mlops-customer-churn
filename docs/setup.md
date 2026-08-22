# Prerequisites & Setup

## Local Tools

Install the following before working with this repository:

| Tool | Purpose | Install |
|------|---------|---------|
| [`gcloud` CLI](https://cloud.google.com/sdk/docs/install) | GCP authentication & resource management | `brew install --cask google-cloud-sdk` |
| [Terraform](https://developer.hashicorp.com/terraform/install) | Infrastructure provisioning | `brew install terraform` |
| [`uv`](https://docs.astral.sh/uv/getting-started/installation/) | Python workspace & dependency management | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| [Docker](https://docs.docker.com/get-docker/) | Building container images locally | Download Docker Desktop |
| [`pre-commit`](https://pre-commit.com/) | Lint & format gate on every commit | `uv tool install pre-commit` |

Authenticate your local environment:

```sh
gcloud auth login
gcloud auth application-default login
gcloud config set project <your-project-id>
```

Install the git hooks once per clone — without this step the config file is inert:

```sh
pre-commit install
```

See [cicd.md](cicd.md#pre-commit-hooks) for what the hooks check and how to run them manually.

---

## GCP Bootstrap (one-time manual steps)

These steps must be completed **before** running `terraform apply` because they involve interactive OAuth flows or billing configuration that cannot be automated.

**1. Create a GCP project and enable billing**
Create a project in the [GCP Console](https://console.cloud.google.com/) and attach a billing account.

**2. Enable the Secret Manager API**
The Cloud Build GitHub connection stores its OAuth token in Secret Manager. Enable the API before creating the connection:

```sh
terraform -chdir=iac apply -target='google_project_service.apis["secretmanager.googleapis.com"]' -auto-approve
```

**3. Create the Cloud Build GitHub connection**
Navigate to **Cloud Build → Settings → Repositories (2nd gen)** in the GCP Console and create a new connection linked to your GitHub account via OAuth. Note the connection name you assign — you will need it in the next step.

---

## Configure & Apply Terraform

**1. Set provider variables**

Edit `iac/terraform.tfvars`:
```hcl
project_id  = "your-gcp-project-id"
region      = "europe-west1"
alert_email = "you@example.com"
```

**2. Set the GitHub connection name**

Edit `iac/config/cloud_build.yaml` and set `cloud_build.github.connection_name` to the name you gave the connection in step 3 above.

**3. Initialise and apply**

```sh
cd iac
terraform init
terraform apply
```

This provisions all GCP resources: BigQuery datasets and tables, Artifact Registry, Cloud Run Jobs, service accounts, IAM bindings, the linked GitHub repository, and Cloud Build triggers.
