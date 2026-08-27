#!/usr/bin/env bash
# Ablation delta runner: seeds 13 and 97 only, B1/B2/B3 families on nld-eng.
#
# Companion to script/ablation_minimal_nld_eng.sh, which sweeps seeds
# {13, 42, 97}. The seed=42 pass is assumed to be covered by prior runs
# (train.py default seed is 42; see script/train.py:69), so this script
# trains ONLY the two missing seeds {13, 97} on the 8 B1/B2/B3 variants.
#
# 8 variants x 2 seeds = 16 runs on a single 8xA100 node => ~16h total.
#
# The emitted manifest pre-lists the assumed seed=42 rows so the
# significance driver (beetle-analyze/analyze/significance_ablation.py)
# sees the full {13, 42, 97} grid from one file.
#
# Usage:
#   bash script/ablation_minimal_nld_eng_seeds_13_97.sh            # full run
#   bash script/ablation_minimal_nld_eng_seeds_13_97.sh --dry_run  # 5 steps per variant/seed
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
CFG_ROOT="${ROOT}/configs/conditions/bilingual"
TMP="${ROOT}/configs/_ablation_tmp"
LOG_DIR="${ROOT}/logs_ablation_nld_eng"
mkdir -p "$LOG_DIR" "$TMP"

LAMOL=0.2   # generative replay lambda used for all *_lamol variants

# Missing seeds only (seed=42 assumed already trained).
SEEDS=(13 97)

# --- 1. Generate the non-LAMOL configs ------------------------------------
if [[ ! -f "${CFG_ROOT}/B1/train_nld_L1-eng_L2.yaml" ]]; then
    echo "[generate] base + ablation configs"
    python "${ROOT}/configs/generate_conditions_with_best.py" \
        --l1 nld --l2 eng --ablations
fi

# --- 2. Generate LAMOL overlay configs ------------------------------------
gen_lamol () {
    local base_variant="$1"        # B2 (B4/B5 excluded from this sweep)
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
python "${ROOT}/configs/generate_conditions_with_best.py" \
    --l1 nld --l2 eng --ablations

# --- 3. Variants + manifest ------------------------------------------------
VARIANTS=(
    B1 B1b
    B2 B2_ewc B2_lamol B2_lr
    B3 B3_full
)

MANIFEST="${LOG_DIR}/variants_manifest_seeds_13_97.csv"
echo "variant,seed,config_path,log_path,hf_push_suffix" > "$MANIFEST"

# Prepend assumed seed=42 rows (trained previously; log_path left empty
# because those logs may live outside this box). Downstream tooling should
# resolve seed=42 artifacts by (variant, seed) rather than log file.
for v in "${VARIANTS[@]}"; do
    cfg="${CFG_ROOT}/${v}/train_nld_L1-eng_L2.yaml"
    echo "${v},42,${cfg},,seed42" >> "$MANIFEST"
done

# --- 4. Train seeds {13, 97} ----------------------------------------------
for s in "${SEEDS[@]}"; do
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
echo "All ${N_RUNS} new runs (${N_VARIANTS} variants x ${N_SEEDS} seeds) complete."
echo "Manifest (incl. assumed seed=42 rows): ${MANIFEST}."
echo "Next step: python -m analyze.significance_ablation --variants-file ${MANIFEST} --family {B1,B2,B3}"
