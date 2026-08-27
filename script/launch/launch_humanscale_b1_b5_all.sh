#!/usr/bin/env bash
# ============================================================================
# Node A — 100M FineWeb B1-B5 across 13 languages + eng monolingual.
#
# Pushes to HuggingFace org: Beetle-FineWeb-100M
# Configs:                   configs/humanscale_fineweb_bilingual/
# Per-job hardware:          8xA100, single node, DDP
#
# Sequential per node (one job uses all 8 GPUs at a time).
# 13 langs x 5 tiers + 1 eng monolingual = 66 jobs.
#
# This file is a thin wrapper around launch_humanscale_b1_b5_all_safe.sh —
# it sets the full-language-set defaults and delegates all execution to the
# fully-hardened safe launcher (timeout, sentinel, kill_survivors,
# drain_gpus, crash-isolation, GPU pre-flight, etc.). See the header of
# launch_humanscale_b1_b5_all_safe.sh for the complete env contract.
#
# Usage:
#   bash script/launch/launch_humanscale_b1_b5_all.sh
#   bash script/launch/launch_humanscale_b1_b5_all.sh --dry_run
#   RESUME=1 bash script/launch/launch_humanscale_b1_b5_all.sh
#   TIERS="B1" bash script/launch/launch_humanscale_b1_b5_all.sh
#
# Env (most forwarded to _all_safe — see its header for the full list):
#   NGPUS    (default 8)
#   HF_TOKEN (required for HF push)
#   RESUME   (1 → pass --resume_checkpoint to train.py)
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Full 13-language set. Ensure all 28B datasets are on HF or Stage 0 will
# warn and training jobs for missing langs will fail at data-load.
export LANGS="${LANGS:-nld deu zho spa jpn ita rus pol tur fil hin eus arb}"
export RUN_ENG_MONO="${RUN_ENG_MONO:-1}"
export LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/humanscale}"

exec bash "${SCRIPT_DIR}/launch_humanscale_b1_b5_all_safe.sh" "$@"
