#!/bin/bash
# =============================================================================
# BeetleLM — FineWeb Set B: fra, deu, spa, zho, jpn
#
# Local 8xA100 node. Runs models sequentially with auto-resume from HF.
# Skips fully trained models automatically.
#
# Usage: ./script/run_fineweb_set_b.sh
# =============================================================================

set -euo pipefail

LOG_DIR="logs_fineweb_b"
mkdir -p "$LOG_DIR"

SET_B_LANGS="fra|deu|spa|zho|jpn"
CONFIG_DIR="configs/fineweb_bilingual"

for cfg in "$CONFIG_DIR"/core/*/train_*.yaml "$CONFIG_DIR"/monolingual/train_*.yaml; do
    [ -f "$cfg" ] || continue

    # Filter to Set B languages only
    basename_cfg=$(basename "$cfg" .yaml)
    if ! echo "$basename_cfg" | grep -qP "($SET_B_LANGS)"; then
        continue
    fi

    run_name=$(python3 -c "import yaml; print(yaml.safe_load(open('$cfg'))['run_name'])")
    hf_user=$(python3 -c "import yaml; print(yaml.safe_load(open('$cfg'))['hf_user'])")
    max_steps=$(python3 -c "import yaml; print(yaml.safe_load(open('$cfg'))['training']['max_steps'])")

    # Check if fully trained on HF
    latest=$(python3 -c "
from huggingface_hub import list_repo_refs
import re
try:
    refs = list_repo_refs('${hf_user}/${run_name}')
    steps = [int(m.group(1)) for r in refs.branches if (m := re.match(r'step-(\d+)', r.name))]
    print(max(steps) if steps else 0)
except:
    print(0)
" 2>/dev/null)

    if [[ "$latest" -ge "$max_steps" ]]; then
        echo "SKIP $run_name (done at step $latest)"
        continue
    fi

    RESUME=""
    if [[ "$latest" -gt 0 ]]; then
        RESUME="--resume_checkpoint"
        echo "START $run_name (resuming from step $latest)"
    else
        echo "START $run_name (from scratch)"
    fi

    torchrun --nproc_per_node=8 script/train.py \
        --config_path "$cfg" $RESUME > "$LOG_DIR/${run_name}.log" 2>&1

    echo "DONE $run_name"
done

echo ""
echo "All Set B models finished."
