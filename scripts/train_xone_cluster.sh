#!/bin/bash
# 8-GPU JAX training script for X-One pi0.5 fine-tuning on sslaunch cluster.
# All external resources must be pre-downloaded via scripts/prepare_xone.sh.
#
# Usage:
#   bash scripts/train_xone_cluster.sh <config_name> <exp_name>
#
# Examples:
#   bash scripts/train_xone_cluster.sh pi05_xone xone26_sft
#   bash scripts/train_xone_cluster.sh pi05_xone_adapt xone26_sft_adapt

set -euo pipefail

CONFIG_NAME="${1:?Usage: $0 <config_name> <exp_name>}"
EXP_NAME="${2:?Usage: $0 <config_name> <exp_name>}_$(date +%Y%m%d_%H%M%S)"

OPENPI_DIR="/mnt/vepfs/base2/rongyinze/codebases/openpi"
cd "$OPENPI_DIR"

# Verify pre-downloaded resources exist (fail fast instead of hanging on download)
OPENPI_CACHE="$HOME/.cache/openpi"
LEROBOT_CACHE="$HOME/.cache/huggingface/lerobot"
# Which LeRobot dataset this run needs (override via -e DATASET_REPO_ID=...)
DATASET_REPO_ID="${DATASET_REPO_ID:-xone/pick_place_dual_hand}"
for f in \
    "$OPENPI_CACHE/openpi-assets/checkpoints/pi05_base/params/_CHECKPOINT_METADATA" \
    "$OPENPI_CACHE/openpi-assets/checkpoints/pi05_base/assets/trossen" \
    "$OPENPI_CACHE/big_vision/paligemma_tokenizer.model" \
    "$LEROBOT_CACHE/$DATASET_REPO_ID/meta/info.json"; do
    if [ ! -e "$f" ]; then
        echo "FATAL: Missing pre-downloaded resource: $f"
        echo "Run 'bash scripts/prepare_xone.sh' first (with proxy enabled)."
        exit 1
    fi
done
echo "Pre-downloaded resources verified."

# Prevent any network access during training
export HF_HUB_OFFLINE=1

# Verify .venv is ready (dependencies pre-installed by prepare_xone.sh)
if [ ! -x "$OPENPI_DIR/.venv/bin/python" ]; then
    echo "FATAL: .venv not found. Run 'bash scripts/prepare_xone.sh' first."
    exit 1
fi

export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1

export BOS_UPLOAD_DIR="${BOS_UPLOAD_DIR:-}"
export UPLOAD_KEEP_PERIOD="${UPLOAD_KEEP_PERIOD:-5000}"
if [ -n "$BOS_UPLOAD_DIR" ]; then
    MEGFILE_BIN="/mnt/vepfs/base2/rongyinze/miniconda3/bin/megfile"
    if [ ! -x "$MEGFILE_BIN" ]; then
        echo "FATAL: megfile CLI not found at $MEGFILE_BIN (needed for BOS upload)"
        exit 1
    fi
    echo "BOS upload:  $BOS_UPLOAD_DIR (period=$UPLOAD_KEEP_PERIOD)"
fi

echo "=========================================="
echo "Config:     $CONFIG_NAME"
echo "Exp:        $EXP_NAME"
echo "GPUs:       $(.venv/bin/python -c 'import jax; print(jax.device_count())' 2>/dev/null || echo 'unknown')"
echo "=========================================="

# Log outside checkpoint dir so --overwrite doesn't delete it
LOG_DIR="$OPENPI_DIR/logs/${CONFIG_NAME}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${EXP_NAME}.log"

# Use .venv/bin/python directly to avoid uv run re-syncing opencv-python
"$OPENPI_DIR/.venv/bin/python" scripts/train.py "$CONFIG_NAME" \
    --exp-name="$EXP_NAME" \
    --fsdp-devices 8 \
    --no-wandb-enabled \
    --overwrite 2>&1 | tee "$LOG_FILE"
