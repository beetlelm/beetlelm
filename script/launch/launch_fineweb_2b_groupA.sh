#!/usr/bin/env bash
# ============================================================================
# Node B — 2B FineWeb B1-B5, group A (9 langs + eng monolingual).
#
# Pushes to HuggingFace org: Beetle-FineWeb-2B
# Configs:                   configs/fineweb_bilingual_2b/
# Per-job hardware:          8xA100, single node, DDP
#
# Sequential per node (one job uses all 8 GPUs at a time).
# 9 langs x 5 tiers + 1 eng monolingual = 46 jobs.
#
# This file is a thin wrapper around launch_fineweb_2b_groupA_safe.sh —
# it sets the groupA language defaults and delegates all execution to the
# fully-hardened safe launcher (timeout, sentinel, kill_survivors,
# drain_gpus, crash-isolation, GPU pre-flight, etc.). See the header of
# launch_fineweb_2b_groupA_safe.sh for the complete env contract.
#
# Usage:
#   bash script/launch/launch_fineweb_2b_groupA.sh
#   bash script/launch/launch_fineweb_2b_groupA.sh --dry_run
#   RESUME=1 bash script/launch/launch_fineweb_2b_groupA.sh
#   TIERS="B1 B2" bash script/launch/launch_fineweb_2b_groupA.sh
#
# Env (most forwarded to groupA_safe — see its header for the full list):
#   NGPUS    (default 8)
#   HF_TOKEN (required for HF push)
#   RESUME   (1 → pass --resume_checkpoint to train.py)
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Group A language set (includes zho, spa, jpn — ensure their 28B datasets
# are on HF or Stage 0 will warn and training jobs will fail at data-load).
export LANGS="${LANGS:-nld deu zho spa jpn ita rus pol tur}"
export RUN_ENG_MONO="${RUN_ENG_MONO:-1}"
export LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/fineweb_2b_groupA}"

exec bash "${SCRIPT_DIR}/launch_fineweb_2b_groupA_safe.sh" "$@"
