#!/usr/bin/env bash
# ============================================================================
# BeetleLM Quickstart: Human-Scale Training (1 node x 8xA100)
# ============================================================================
# Trains a single 125M-param bilingual model (Dutch-English, balanced)
# on 100M tokens using 8xA100 GPUs with DDP.
#
# Usage:
#   bash script/quickstart_human_scale.sh
#   bash script/quickstart_human_scale.sh --dry_run
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG="${PROJECT_ROOT}/configs/quickstart_human_scale.yaml"

echo "============================================"
echo "BeetleLM Human-Scale Quickstart"
echo "============================================"
echo "Config : ${CONFIG}"
echo "GPUs   : 8"
echo "Model  : PicoDecoder 125M"
echo "Data   : BabyBabel Dutch + English (100M tokens)"
echo "Steps  : 10,000"
echo "============================================"

# Launch with torchrun for 8-GPU DDP
torchrun \
    --standalone \
    --nproc_per_node=8 \
    "${PROJECT_ROOT}/script/train.py" \
    --config_path "${CONFIG}" \
    "$@"
