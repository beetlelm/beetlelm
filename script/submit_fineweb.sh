#!/bin/bash
# =============================================================================
# BeetleLM — Auto-Resubmitting Batch Submitter for FineWeb (Cambridge HPC)
#
# Submits one SLURM job per config. Skips fully trained models.
# Re-run this script periodically to resubmit incomplete models that
# ran out of wallclock time.
#
# Usage:
#   ./script/submit_fineweb.sh <CONFIG_DIR>
#   ./script/submit_fineweb.sh configs/fineweb_bilingual/core
#   ./script/submit_fineweb.sh configs/fineweb_bilingual --dry-run
# =============================================================================

set -euo pipefail

CONFIG_DIR="${1:?ERROR: provide config directory as first argument}"
shift

DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case "$1" in
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
SKIPPED_DONE=0
SKIPPED_OTHER=0

for cfg in "$CONFIG_DIR"/**/*.yaml "$CONFIG_DIR"/*.yaml; do
    [ -f "$cfg" ] || continue

    RUN_NAME=$(python3 -c "import yaml; print(yaml.safe_load(open('$cfg'))['run_name'])" 2>/dev/null) || continue
    HF_USER=$(python3 -c "import yaml; print(yaml.safe_load(open('$cfg'))['hf_user'])" 2>/dev/null) || continue
    MAX_STEPS=$(python3 -c "import yaml; print(yaml.safe_load(open('$cfg'))['training']['max_steps'])" 2>/dev/null) || continue

    # Check if fully trained on HuggingFace
    LATEST_STEP=$(python3 -c "
from huggingface_hub import list_repo_refs
import re
try:
    refs = list_repo_refs('${HF_USER}/${RUN_NAME}')
    steps = [int(m.group(1)) for r in refs.branches if (m := re.match(r'step-(\d+)', r.name))]
    print(max(steps) if steps else 0)
except:
    print(0)
" 2>/dev/null)

    if [[ "$LATEST_STEP" -ge "$MAX_STEPS" ]]; then
        echo "SKIP $RUN_NAME (fully trained at step $LATEST_STEP)"
        SKIPPED_DONE=$((SKIPPED_DONE + 1))
        continue
    fi

    if $DRY_RUN; then
        echo "[DRY RUN] Would submit: $RUN_NAME (latest step: $LATEST_STEP / $MAX_STEPS)"
    else
        echo "SUBMIT $RUN_NAME (latest step: $LATEST_STEP / $MAX_STEPS)"
        sbatch script/slurm_fineweb_one.sh "$cfg"
    fi
    SUBMITTED=$((SUBMITTED + 1))
done

echo ""
echo "Done. Submitted: $SUBMITTED | Skipped (done): $SKIPPED_DONE"
if $DRY_RUN; then
    echo "(dry run — no jobs were actually submitted)"
fi
