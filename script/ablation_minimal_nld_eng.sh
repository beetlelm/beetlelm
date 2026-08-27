#!/usr/bin/env bash
# Minimal B1/B2/B3 ablation on nld-eng (human-scale, multi-seed).
# Runs each variant under the three a-priori-fixed seeds below so the
# best configuration within each base-condition family can be selected
# with a paired permutation test + Cohen's d_z + Holm-Bonferroni on the
# MECO alignment metric (see beetle-analyze/analyze/significance_ablation.py).
#
# Runs sequentially on a single 8xA100 node, ~1h per variant/seed.
# 8 variants x 3 seeds = 24 runs => ~24h total.
#
# Usage:
#   bash script/ablation_minimal_nld_eng.sh            # full run
#   bash script/ablation_minimal_nld_eng.sh --dry_run  # 5 steps per variant/seed
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
CFG_ROOT="${ROOT}/configs/conditions/bilingual"
TMP="${ROOT}/configs/_ablation_tmp"
LOG_DIR="${ROOT}/logs_ablation_nld_eng"
mkdir -p "$LOG_DIR" "$TMP"

LAMOL=0.2   # generative replay lambda used for all *_lamol variants

# Seeds fixed a priori to {13, 42, 97} to avoid selection bias.
SEEDS=(13 42 97)

# --- 1. Generate the non-LAMOL configs ------------------------------------
if [[ ! -f "${CFG_ROOT}/B1/train_nld_L1-eng_L2.yaml" ]]; then
    echo "[generate] base + ablation configs"
    python "${ROOT}/configs/generate_conditions_with_best.py" \
        --l1 nld --l2 eng --ablations
fi

# --- 2. Generate LAMOL overlay configs ------------------------------------
# For each LAMOL variant (B2, B4b, B5), invoke the generator with a
# lamol-only best-settings JSON, then copy the emitted yaml into a
# variant-specific dir so the canonical non-LAMOL one can be restored.
gen_lamol () {
    local base_variant="$1"        # B2 | B4b | B5
    local out_tag="${base_variant}_lamol"
    local out_dir="${CFG_ROOT}/${out_tag}"
    local json="${TMP}/best_lamol_${base_variant}.json"

    mkdir -p "$out_dir"
    cat > "$json" <<EOF
{"lamol_lambda": ${LAMOL}}
EOF
    python "${ROOT}/configs/generate_conditions_with_best.py" \
        --l1 nld --l2 eng --best-config "$json" --ablations
    cp "${CFG_ROOT}/${base_variant}/train_nld_L1-eng_L2.yaml" \
       "${out_dir}/train_nld_L1-eng_L2.yaml"
}

gen_lamol B2
# Restore canonical (non-LAMOL) B2 config after the LAMOL pass.
# (B4/B5 LAMOL overlays dropped: B4/B5 families excluded from this
# significance sweep per the approved ablation plan.)
python "${ROOT}/configs/generate_conditions_with_best.py" \
    --l1 nld --l2 eng --ablations

# --- 3. Train --------------------------------------------------------------
# Variants restricted to the B1, B2, B3 families to focus the significance
# sweep on selecting the best configuration within each base condition.
VARIANTS=(
    B1 B1b
    B2 B2_ewc B2_lamol B2_lr
    B3 B3_full
)

# Manifest of (variant, seed, checkpoint, hf_push_suffix) triples used by the
# downstream aggregation driver (beetle-analyze/analyze/significance_ablation.py).
MANIFEST="${LOG_DIR}/variants_manifest.csv"
echo "variant,seed,config_path,log_path,hf_push_suffix" > "$MANIFEST"

for s in "${SEEDS[@]}"; do
    # Exported so trainers can tag their HF-pushed checkpoint repos / local
    # dirs with a seed-specific suffix. The trainer reads HF_PUSH_SUFFIX if it
    # supports it; otherwise it is a harmless no-op.
    export HF_PUSH_SUFFIX="seed${s}"
    for v in "${VARIANTS[@]}"; do
        cfg="${CFG_ROOT}/${v}/train_nld_L1-eng_L2.yaml"
        if [[ ! -f "$cfg" ]]; then
            echo "[skip] missing config: $cfg"; continue
        fi
        log="${LOG_DIR}/${v}_seed${s}.log"
        echo "=== [$(date +%H:%M:%S)] START ${v} seed=${s} ==="
        torchrun --standalone --nproc_per_node=8 \
            "${ROOT}/script/train.py" --config_path "$cfg" --seed "$s" "$@" 2>&1 | tee "$log"
        echo "=== [$(date +%H:%M:%S)] DONE  ${v} seed=${s} ==="
        echo "${v},${s},${cfg},${log},${HF_PUSH_SUFFIX}" >> "$MANIFEST"
    done
done

N_VARIANTS=${#VARIANTS[@]}
N_SEEDS=${#SEEDS[@]}
N_RUNS=$(( N_VARIANTS * N_SEEDS ))
echo "All ${N_RUNS} runs (${N_VARIANTS} variants x ${N_SEEDS} seeds) complete."
echo "Logs in ${LOG_DIR}/; manifest at ${MANIFEST}."
echo "Next step: python -m analyze.significance_ablation --variants-file ${MANIFEST} --family {B1,B2,B3}"
