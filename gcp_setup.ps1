param(
  [string]$Project = "utahcoder",
  [string]$Region = "us-central1",
  [string]$Bucket = "utahcoder-vertex-staging-us-central1"
)

Write-Host "=== Neuro-Architect GCP Setup ===" -ForegroundColor Cyan
Write-Host "Project: $Project"; Write-Host "Region: $Region"; Write-Host "Bucket: gs://$Bucket"

# 1) Check Cloud SDK tools
if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
  Write-Error "gcloud CLI not found. Install Google Cloud SDK: https://cloud.google.com/sdk/docs/install and re-run this script."
  exit 1
}
if (-not (Get-Command gsutil -ErrorAction SilentlyContinue)) {
  Write-Error "gsutil not found. It is included with the Google Cloud SDK. Ensure your installation is complete."
  exit 1
}

# 2) Auth and set project
Write-Host "Configuring gcloud..." -ForegroundColor Yellow
& gcloud auth application-default login
if ($LASTEXITCODE -ne 0) { Write-Error "ADC auth failed."; exit 1 }

& gcloud config set project $Project | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Error "Failed to set gcloud project."; exit 1 }

# 3) Enable required APIs
$apis = @(
  "aiplatform.googleapis.com",
  "storage.googleapis.com",
  "iam.googleapis.com",
  "artifactregistry.googleapis.com",
  "compute.googleapis.com",
  "serviceusage.googleapis.com"
)
Write-Host "Enabling APIs: $($apis -join ', ')" -ForegroundColor Yellow
foreach ($api in $apis) {
  & gcloud services enable $api --project $Project | Out-Null
  if ($LASTEXITCODE -ne 0) { Write-Error "Failed to enable API: $api"; exit 1 }
}

# 4) Create the staging bucket if it doesn't exist
Write-Host "Checking staging bucket..." -ForegroundColor Yellow
$bucketExists = $false
try {
  & gsutil ls -b "gs://$Bucket" | Out-Null
  if ($LASTEXITCODE -eq 0) { $bucketExists = $true }
} catch { $bucketExists = $false }

if (-not $bucketExists) {
  Write-Host "Creating bucket gs://$Bucket ..." -ForegroundColor Yellow
  & gsutil mb -p $Project -c STANDARD -l $Region -b on "gs://$Bucket"
  if ($LASTEXITCODE -ne 0) { Write-Error "Failed to create bucket."; exit 1 }
} else {
  Write-Host "Bucket already exists." -ForegroundColor Green
}

# 5) Harden bucket: uniform access + versioning
& gsutil uniformbucketlevelaccess set on "gs://$Bucket"
if ($LASTEXITCODE -ne 0) { Write-Warning "Could not set uniform bucket-level access. You can set it in console if needed." }

& gsutil versioning set on "gs://$Bucket"
if ($LASTEXITCODE -ne 0) { Write-Warning "Could not enable versioning. You can enable it in console if needed." }

Write-Host "\nSetup complete." -ForegroundColor Green
Write-Host "Add the following to your .env if not already present:" -ForegroundColor Cyan
Write-Host "GCP_PROJECT=$Project"
Write-Host "VERTEX_LOCATION=$Region"
Write-Host "GCS_STAGING_BUCKET=gs://$Bucket"
