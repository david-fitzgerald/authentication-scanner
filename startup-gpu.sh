#!/usr/bin/env bash
# startup-gpu.sh — GCE VM startup script for authentication experiments
# Runs automatically on boot. Downloads cache, runs experiments, uploads results, self-destructs.
set -euo pipefail

LOG="/var/log/auth-experiments.log"
WORK="/opt/auth"
BUCKET=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/attributes/bucket" \
    -H "Metadata-Flavor: Google")
ZONE=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/zone" \
    -H "Metadata-Flavor: Google" | awk -F/ '{print $NF}')
INSTANCE=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/name" \
    -H "Metadata-Flavor: Google")

exec > >(tee -a "${LOG}") 2>&1

echo "========================================"
echo "  Auth Experiments — $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "  Bucket: ${BUCKET}"
echo "  Instance: ${INSTANCE} (${ZONE})"
echo "========================================"

# --- Error trap: upload logs + self-destruct on failure ---
cleanup() {
    echo ""
    echo "[CLEANUP] Uploading logs to GCS..."
    gsutil cp "${LOG}" "${BUCKET}/results/auth-experiments.log" 2>/dev/null || true
    echo "[CLEANUP] Self-destructing VM..."
    gcloud compute instances delete "${INSTANCE}" --zone="${ZONE}" --quiet 2>/dev/null || true
}
trap cleanup ERR EXIT

# --- Wait for GPU driver ---
echo ""
echo "[GPU] Waiting for NVIDIA driver..."
for i in $(seq 1 60); do
    if nvidia-smi &>/dev/null; then
        echo "[GPU] Driver ready."
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "[GPU] ERROR: Driver not ready after 5 minutes."
        exit 1
    fi
    sleep 5
done

# --- Ensure SSH access ---
apt-get install -yq openssh-server 2>/dev/null || true
systemctl enable ssh
systemctl start ssh

# --- Setup workspace ---
mkdir -p "${WORK}/cache"

# --- Download cache from GCS ---
echo ""
echo "[GCS] Downloading cache..."
gsutil -m rsync -r "${BUCKET}/cache/" "${WORK}/cache/"
gsutil cp "${BUCKET}/scan.py" "${WORK}/scan.py"
echo "[GCS] Download complete."

# --- Install dependencies ---
echo ""
echo "[DEPS] Installing Python packages..."
pip install peft scikit-learn Pillow imagehash 2>&1 | tail -5
echo "[DEPS] Done."

# --- Verify CUDA ---
echo ""
python3 -c "import torch; print(f'[CUDA] PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

# --- Run experiments ---
cd "${WORK}"
RESULTS_DIR="${WORK}/cache"
export PYTHONUNBUFFERED=1

echo ""
echo "========================================"
echo "  Starting experiments — $(date -u '+%H:%M:%S UTC')"
echo "========================================"

# --- Tier 3 experiments (D/A only — C killed, see CLAUDE.md) ---

# Experiment D: Two-phase curriculum (~20-30min)
echo ""
echo "[EXP D] Two-phase LoRA curriculum..."
python3 scan.py --lora-curriculum 2>&1
echo "[EXP D] Done — $(date -u '+%H:%M:%S UTC')"

# Clear GPU memory between experiments
python3 -c "import torch; torch.cuda.empty_cache(); print('[GPU] Cache cleared')"

# Experiment A: Frozen transfer probe (~5min)
echo ""
echo "[EXP A] Frozen transfer probe..."
python3 scan.py --transfer-probe 2>&1
echo "[EXP A] Done — $(date -u '+%H:%M:%S UTC')"

# --- Methodological re-evaluation (nested CV, strict labels) ---

# Re-run probe with nested CV (new methodology)
echo ""
echo "[PROBE] Nested CV probe (entropy, methodological fix)..."
python3 scan.py --probe --entropy 2>&1
echo "[PROBE] Nested CV done — $(date -u '+%H:%M:%S UTC')"

# Also run with strict labels for comparison
echo ""
echo "[PROBE-STRICT] Nested CV probe with strict labels..."
python3 scan.py --probe --entropy --strict-labels 2>&1
echo "[PROBE-STRICT] Done — $(date -u '+%H:%M:%S UTC')"

# Legacy CV for comparison
echo ""
echo "[PROBE-LEGACY] Legacy CV probe (for comparison)..."
python3 scan.py --probe --entropy --legacy-cv 2>&1
echo "[PROBE-LEGACY] Done — $(date -u '+%H:%M:%S UTC')"

# Institution holdout: Met
echo ""
echo "[HOLDOUT-MET] Institution holdout (Met)..."
python3 scan.py --probe --entropy --holdout-source met 2>&1
echo "[HOLDOUT-MET] Done — $(date -u '+%H:%M:%S UTC')"

echo ""
echo "========================================"
echo "  All experiments complete — $(date -u '+%H:%M:%S UTC')"
echo "========================================"

# --- Upload results ---
echo ""
echo "[GCS] Uploading results..."
mkdir -p "${WORK}/results"

# Copy all JSON results and logs
cp "${WORK}"/cache/*.json "${WORK}/results/" 2>/dev/null || true
cp "${LOG}" "${WORK}/results/" 2>/dev/null || true

gsutil -m rsync -r "${WORK}/results/" "${BUCKET}/results/"
# Also upload the full cache (has new embeddings/weights)
gsutil -m rsync -r "${WORK}/cache/" "${BUCKET}/cache-results/"
echo "[GCS] Upload complete."

echo ""
echo "=== SUCCESS — VM will self-destruct ==="
# cleanup trap handles self-destruct via EXIT
