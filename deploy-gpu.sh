#!/usr/bin/env bash
# deploy-gpu.sh — Fire-and-forget GCE Spot T4 for authentication experiments
# Usage: bash deploy-gpu.sh
set -euo pipefail

PROJECT="${GCP_PROJECT:?Set GCP_PROJECT}"
ZONE="europe-west4-a"
INSTANCE="auth-experiments"
BUCKET="${GCS_BUCKET:?Set GCS_BUCKET}"
MACHINE_TYPE="n1-standard-4"
ACCELERATOR="type=nvidia-tesla-t4,count=1"
IMAGE_FAMILY="pytorch-2-7-cu128-ubuntu-2204-nvidia-570"
IMAGE_PROJECT="deeplearning-platform-release"
BOOT_DISK_SIZE="100GB"
MAX_RUN="129600s"  # 36h hard cap

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CACHE_DIR="${SCRIPT_DIR}/cache"
SCAN_PY="${SCRIPT_DIR}/scan.py"
STARTUP="${SCRIPT_DIR}/startup-gpu.sh"

echo "=== GCE Spot T4 Deploy ==="
echo "Project:  ${PROJECT}"
echo "Zone:     ${ZONE}"
echo "Instance: ${INSTANCE}"

# --- Pre-flight checks ---
if ! command -v gcloud &>/dev/null; then
    echo "ERROR: gcloud CLI not found. Install: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

if ! command -v gsutil &>/dev/null; then
    echo "ERROR: gsutil not found (part of gcloud SDK)"
    exit 1
fi

if [ ! -d "${CACHE_DIR}" ]; then
    echo "ERROR: cache/ directory not found at ${CACHE_DIR}"
    exit 1
fi

if [ ! -f "${SCAN_PY}" ]; then
    echo "ERROR: scan.py not found at ${SCAN_PY}"
    exit 1
fi

if [ ! -f "${STARTUP}" ]; then
    echo "ERROR: startup-gpu.sh not found at ${STARTUP}"
    exit 1
fi

# Check GPU quota
echo ""
echo "--- Checking GPU quota ---"
QUOTA=$(gcloud compute project-info describe --project="${PROJECT}" --format=json 2>/dev/null \
    | python3 -c "import json,sys; data=json.load(sys.stdin); print(next((q['limit'] for q in data.get('quotas',[]) if q['metric']=='GPUS_ALL_REGIONS'),0))" 2>/dev/null || echo "0")
if [ "${QUOTA}" = "0" ] || [ -z "${QUOTA}" ]; then
    echo "ERROR: GPUS_ALL_REGIONS quota is 0. Request increase first:"
    echo "  https://console.cloud.google.com/iam-admin/quotas?project=${PROJECT}"
    echo "  Filter: GPUS_ALL_REGIONS → Edit → Request 1"
    exit 1
fi
echo "GPU quota: ${QUOTA} (OK)"

# --- Step 1: Create GCS bucket (idempotent) ---
echo ""
echo "--- Step 1: GCS bucket ---"
gsutil mb -p "${PROJECT}" -l us-central1 "${BUCKET}" 2>/dev/null || true
echo "Bucket: ${BUCKET} (ready)"

# --- Step 2: Upload cache + scan.py ---
echo ""
echo "--- Step 2: Uploading to GCS (~1.9GB, may take a few minutes) ---"
gsutil -m rsync -r -x '.*\.pyc$|__pycache__/' "${CACHE_DIR}" "${BUCKET}/cache/"
gsutil cp "${SCAN_PY}" "${BUCKET}/scan.py"
echo "Upload complete."

# --- Step 3: Create Spot T4 VM ---
echo ""
echo "--- Step 3: Creating Spot T4 VM ---"

# Check if instance already exists
if gcloud compute instances describe "${INSTANCE}" --zone="${ZONE}" --project="${PROJECT}" &>/dev/null; then
    echo "WARNING: Instance '${INSTANCE}' already exists."
    read -rp "Delete and recreate? [y/N] " yn
    if [[ "${yn}" =~ ^[Yy]$ ]]; then
        gcloud compute instances delete "${INSTANCE}" --zone="${ZONE}" --project="${PROJECT}" --quiet
    else
        echo "Aborted."
        exit 1
    fi
fi

gcloud compute instances create "${INSTANCE}" \
    --project="${PROJECT}" \
    --zone="${ZONE}" \
    --machine-type="${MACHINE_TYPE}" \
    --accelerator="${ACCELERATOR}" \
    --maintenance-policy=TERMINATE \
    --provisioning-model=STANDARD \
    --instance-termination-action=DELETE \
    --max-run-duration="${MAX_RUN}" \
    --image-family="${IMAGE_FAMILY}" \
    --image-project="${IMAGE_PROJECT}" \
    --boot-disk-size="${BOOT_DISK_SIZE}" \
    --boot-disk-type=pd-ssd \
    --scopes=storage-full \
    --metadata-from-file=startup-script="${STARTUP}" \
    --metadata=bucket="${BUCKET}" \
    --tags=auth-experiments

echo ""
echo "=== VM Created ==="
echo ""
echo "--- Monitor ---"
echo "  gcloud compute ssh ${INSTANCE} --zone=${ZONE} -- tail -f /var/log/auth-experiments.log"
echo ""
echo "--- Check status ---"
echo "  gcloud compute instances describe ${INSTANCE} --zone=${ZONE} --format='value(status)'"
echo ""
echo "--- Results (after completion) ---"
echo "  gsutil ls ${BUCKET}/results/"
echo "  gsutil -m rsync -r ${BUCKET}/results/ ${CACHE_DIR}/"
echo ""
echo "--- Emergency stop ---"
echo "  gcloud compute instances delete ${INSTANCE} --zone=${ZONE} --quiet"
