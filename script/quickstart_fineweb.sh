#!/usr/bin/env bash
# ============================================================================
# BeetleLM Quickstart: FineWeb-Scale Training (1 node x 8xA100)
# ============================================================================
# Trains a single 125M-param bilingual model (Dutch-English, balanced)
# on 24B tokens using 8xA100 GPUs with DDP.
#
# Usage:
#   bash script/quickstart_fineweb.sh
#   bash script/quickstart_fineweb.sh --dry_run
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG="${PROJECT_ROOT}/configs/quickstart_fineweb.yaml"

echo "============================================"
echo "BeetleLM FineWeb-Scale Quickstart"
echo "============================================"
echo "Config : ${CONFIG}"
echo "GPUs   : 8"
echo "Model  : PicoDecoder 125M"
echo "Data   : FineWeb Dutch + English (24B tokens, 1 epoch)"
echo "Steps  : 91,552"
echo "Batch  : 512 effective (64 per GPU x 8 GPUs)"
echo "Wall   : ~6 hours on 8xA100"
echo "============================================"

# Launch with torchrun for 8-GPU DDP
torchrun \
    --standalone \
    --nproc_per_node=8 \
    "${PROJECT_ROOT}/script/train.py" \
    --config_path "${CONFIG}" \
    "$@"
