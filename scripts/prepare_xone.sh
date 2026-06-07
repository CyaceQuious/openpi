#!/bin/bash
# Pre-download all external resources needed for X-One pi0.5 training.
# Run this BEFORE submitting sslaunch jobs. No need to set proxy beforehand.
#
# Usage:
#   bash scripts/prepare_xone.sh

set -euo pipefail

OPENPI_DIR="/mnt/vepfs/base2/rongyinze/codebases/openpi"
cd "$OPENPI_DIR"

# --- Phase 0: Python dependencies (needs PyPI proxy) ---
export https_proxy=192.168.48.27:18000
export http_proxy=192.168.48.27:18000

echo "=== 0/3 Installing Python dependencies (proxy ON) ==="
if ! command -v uv &>/dev/null; then
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv &>/dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
fi
GIT_LFS_SKIP_SMUDGE=1 uv sync 2>&1 | tail -5
uv pip uninstall opencv-python opencv-python-headless torchcodec 2>&1 | tail -3
uv pip install --no-deps opencv-python-headless==4.11.0.86 2>&1 | tail -2
uv pip install "nvidia-nccl-cu12>=2.30" 2>&1 | tail -2
echo "  dependencies OK"

# --- Phase 1: GCS downloads (proxy still ON) ---
echo ""
echo "=== 1/3 Downloading GCS resources (proxy ON) ==="
.venv/bin/python -u -c "
from openpi.shared.download import maybe_download

print('Downloading pi05_base checkpoint (~12GB)...')
maybe_download('gs://openpi-assets/checkpoints/pi05_base/params')
print('  checkpoint OK')

print('Downloading pi05_base assets (norm stats)...')
maybe_download('gs://openpi-assets/checkpoints/pi05_base/assets')
print('  assets OK')

print('Downloading PaliGemma tokenizer...')
maybe_download('gs://big_vision/paligemma_tokenizer.model', gs={'token': 'anon'})
print('  tokenizer OK')
"

# --- Phase 2: BOS download (internal network, proxy OFF) ---
unset https_proxy http_proxy

echo ""
echo "=== 2/3 Syncing LeRobot dataset from BOS (proxy OFF) ==="
DATASET_DST="$HOME/.cache/huggingface/lerobot/xone/xone26"
if [ -f "$DATASET_DST/meta/info.json" ]; then
    echo "  Dataset already exists at $DATASET_DST, skipping."
else
    echo "  Downloading from bos://world-model-data/rongyinze/dataset/x-one/lerobotDataset ..."
    conda run --no-capture-output -n base python -u -c "
from megfile import smart_sync
import os
dst = os.path.expanduser('~/.cache/huggingface/lerobot/xone/xone26')
os.makedirs(dst, exist_ok=True)
smart_sync('bos://world-model-data/rongyinze/dataset/x-one/lerobotDataset/', dst + '/')
print('  dataset sync OK')
"
fi

# --- Phase 3: Verify ---
echo ""
echo "=== 3/3 Verification ==="
FAIL=0
check() {
    if [ -e "$1" ]; then
        echo "  OK: $1"
    else
        echo "  MISSING: $1"
        FAIL=1
    fi
}

check "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_base/params/_CHECKPOINT_METADATA"
check "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_base/assets/trossen"
check "$HOME/.cache/openpi/big_vision/paligemma_tokenizer.model"
check "$HOME/.cache/huggingface/lerobot/xone/xone26/meta/info.json"

if [ "$FAIL" -eq 0 ]; then
    echo ""
    echo "All resources ready. You can now submit training jobs."
else
    echo ""
    echo "Some resources are missing. Check errors above."
    exit 1
fi
