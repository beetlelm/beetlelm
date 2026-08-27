#!/bin/bash
# =============================================================================
# BeetleLM — Batch SLURM Submitter for Human-Scale Sweep
#
# Submits one SLURM job per config YAML in the given directory.
# Optionally filter by language or do a dry run.
#
# Usage:
#   ./script/submit_sweep.sh <CONFIG_DIR>
#   ./script/submit_sweep.sh <CONFIG_DIR> --single-lang nld
#   ./script/submit_sweep.sh <CONFIG_DIR> --dry-run
#   ./script/submit_sweep.sh <CONFIG_DIR> --single-lang nld --dry-run
# =============================================================================

set -euo pipefail

CONFIG_DIR="${1:?ERROR: provide config directory as first argument}"
shift

SINGLE_LANG=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --single-lang)
            SINGLE_LANG="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

SUBMITTED=0
SKIPPED=0

for cfg in "$CONFIG_DIR"/**/*.yaml "$CONFIG_DIR"/*.yaml; do
    [ -f "$cfg" ] || continue

    if [[ -n "$SINGLE_LANG" ]] && ! grep -q "$SINGLE_LANG" "$cfg"; then
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    if $DRY_RUN; then
        echo "[DRY RUN] Would submit: $(basename "$cfg")"
    else
        echo "Submitting: $(basename "$cfg")"
        sbatch script/slurm_train_one.sh "$cfg"
    fi
    SUBMITTED=$((SUBMITTED + 1))
done

echo ""
echo "Done. Submitted: $SUBMITTED | Skipped: $SKIPPED"
if $DRY_RUN; then
    echo "(dry run — no jobs were actually submitted)"
fi
