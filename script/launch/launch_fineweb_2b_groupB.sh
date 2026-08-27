#!/usr/bin/env bash
# ============================================================================
# Node C — 2B FineWeb B1-B5, group B (4 langs + eng monolingual).
#
# Pushes to HuggingFace org: Beetle-FineWeb-2B
# Configs:                   configs/fineweb_bilingual_2b/
# Per-job hardware:          8xA100, single node, DDP
#
# Sequential per node. 4 langs x 5 tiers + 1 eng monolingual = 21 jobs.
# Group B: fil, hin, eus, arb.
#
# This file is a thin wrapper around launch_fineweb_2b_groupA_safe.sh — it
# just sets the groupB defaults (LANGS, LOG_DIR) and delegates the rest.
# All of the safe launcher's crash-resilience and disk-cleanup behavior
# (per-job timeout, sentinel-based skip, runs/<run_name>/ cleanup on
# success, GPU pre-flight, final failure summary, exit non-zero on any
# failure) applies here too — see the header of the safe launcher for the
# full env contract.
#
# NOTE: arb (Arabic) requires Beetle-Data/tokenizer-ar-en + the ar-28B and
#       en-for-ar-28B repos on HF. The first arb job acts as a canary; if
#       it fails the rest of the sweep continues (one bad job no longer
#       aborts the whole launcher).
#
# Usage:
#   bash script/launch/launch_fineweb_2b_groupB.sh
#   bash script/launch/launch_fineweb_2b_groupB.sh --dry_run
#   RESUME=1 bash script/launch/launch_fineweb_2b_groupB.sh
#   CLEANUP_AFTER_SUCCESS=0 bash script/launch/launch_fineweb_2b_groupB.sh
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Group B defaults. Caller can still override via env, e.g.
#   LANGS="arb" bash script/launch/launch_fineweb_2b_groupB.sh
export LANGS="${LANGS:-fil hin eus arb}"
export LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/fineweb_2b_groupB}"

exec bash "${SCRIPT_DIR}/launch_fineweb_2b_groupA_safe.sh" "$@"
