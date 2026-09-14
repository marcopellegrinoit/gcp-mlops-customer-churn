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

**4. Create the dashboard's OAuth client (IAP)**

Skip this if your project belongs to a Cloud organization — IAP will use its Google-managed
OAuth client and nothing is needed. **A standalone project must do this**, because the
managed client only admits "users within the organization in which the resource is
contained", and a project with no organization has no such set. IAP then has no client at
all and every request to the dashboard returns `502 — Empty Google Account OAuth client
ID(s)/secret(s)`.

This step is manual for the same reason the GitHub connection above is: it cannot be
automated. The IAP OAuth Admin APIs (`google_iap_brand`, `google_iap_client`,
`gcloud iap oauth-brands`) were permanently shut down in March 2026, and they rejected
projects without an organization even before that.

1. **Configure the consent screen** — **APIs & Services → OAuth consent screen** (recent
   consoles show this as **Google Auth Platform → Branding**). Choose
   **External** (Internal requires an organization). Set an app name and support email.
2. **Add yourself as a test user** (**Audience → Test users**). An External consent screen
   starts in *Testing* mode, and
   in that mode only listed test users can sign in — including you. Skipping this produces
   an access-denied screen that looks exactly like a misconfigured IAM binding.
3. **Create the client** — **APIs & Services → Credentials → Create credentials → OAuth
   client ID → Web application** (or **Google Auth Platform → Clients**). Save it, then
   reopen it and add the authorized redirect URI, which embeds the client's own ID:

   ```
   https://iap.googleapis.com/v1/oauth/clientIds/YOUR_CLIENT_ID:handleRedirect
   ```

4. **Hand the credentials to Terraform** in `iac/terraform.tfvars` (gitignored):

   ```hcl
   dashboard_oauth_client_id     = "1234567890-abc.apps.googleusercontent.com"
   dashboard_oauth_client_secret = "GOCSPX-..."
   ```

Terraform attaches them to IAP via `google_iap_settings`; see [dashboard.md](dashboard.md#authentication).

---

## Configure & Apply Terraform

**1. Set provider variables**

Edit `iac/terraform.tfvars`:
```hcl
project_id  = "your-gcp-project-id"
region      = "europe-west1"
alert_email = "you@example.com"

# Who may open the churn dashboard through IAP. Defaults to [] — the service deploys
# reachable by nobody. A group is the intended form; see dashboard.md.
dashboard_viewers = ["group:retention-team@example.com"]

# Only for a project with no Cloud organization — see step 4 above.
dashboard_oauth_client_id     = ""
dashboard_oauth_client_secret = ""

# Monthly budget alert. The billing account is detected from the project, so there is
# nothing to look up — set the amount and who to tell.
monthly_budget_amount = 100
budget_alert_emails   = ["you@example.com"]
```

The budget is a notification, not a cap — GCP has no control that stops spend. It exists so
a runaway shows up in hours rather than at the end of a billing cycle; the actual ceilings
are per-resource, and [iac.md](iac.md#budget-alerting-billing_budget-module) lists them.

Creating it needs `roles/billing.costsManager` on the billing account, because the budget
lives there rather than on the project. If you own the project you already have it. If
`apply` is run by a deployer service account instead, grant it:

```sh
gcloud billing accounts add-iam-policy-binding "$(gcloud billing projects describe "$PROJECT_ID" --format='value(billingAccountName)' | cut -d/ -f2)" \
  --member="serviceAccount:DEPLOYER_SA_EMAIL" --role="roles/billing.costsManager"
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
