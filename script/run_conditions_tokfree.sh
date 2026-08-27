#!/bin/bash
# =============================================================================
# BeetleLM Tokenizer-free Conditions Training (R1–R4)
# 4-GPU DDP per job · 2 parallel jobs per node · 8 GPUs fully utilised
# =============================================================================
#
# R1: bytes + H-Net,        balanced multilingual exposure
# R2: bytes + MambaByte,    balanced multilingual exposure   (baseline)
# R3: codepoints + H-Net,   balanced multilingual exposure
# R4: bytes + H-Net,        sequential L1→L2(→L3) curriculum
#
# Usage:  ./run_conditions_tokfree.sh <NODE_ID> [CONDITION]
#
#   NODE_ID   = 1,2,3,4
#   CONDITION = R1|R2|R3|R4|all   (optional, default: all)
# =============================================================================

set -euo pipefail

CONFIG_DIR="configs/tokfree"
LOG_DIR="logs_conditions_tokfree"
GPUS_PER_NODE=8
PARALLEL_JOBS=2
GPUS_PER_JOB=$(( GPUS_PER_NODE / PARALLEL_JOBS ))
NUM_NODES=4

mkdir -p "$LOG_DIR"

NODE_ID=${1:?ERROR: provide NODE_ID 1-${NUM_NODES}}
CONDITION=${2:-all}

if [[ "$NODE_ID" -lt 1 || "$NODE_ID" -gt "$NUM_NODES" ]]; then
    echo "ERROR: NODE_ID must be 1–${NUM_NODES}"
    exit 1
fi

GPU_SLOTS=()
for (( slot=0; slot<PARALLEL_JOBS; slot++ )); do
    start=$(( slot * GPUS_PER_JOB ))
    end=$(( start + GPUS_PER_JOB - 1 ))
    GPU_SLOTS+=( "$(seq -s, "$start" "$end")" )
done

ALL_CONFIGS=()
if [[ "$CONDITION" == "all" ]]; then
    while IFS= read -r -d '' f; do
        ALL_CONFIGS+=("$f")
    done < <(find "$CONFIG_DIR" -type f -name "*.yaml" -not -path "*/examples/*" -print0 | sort -z)
else
    COND_DIR="$CONFIG_DIR/$CONDITION"
    if [[ ! -d "$COND_DIR" ]]; then
        echo "ERROR: condition directory not found: $COND_DIR"
        echo "       Generate per-condition YAMLs first with:"
        echo "       python script/beetle_experiment.py sweep --conditions R1 R2 R3 R4 ..."
        exit 1
    fi
    while IFS= read -r -d '' f; do
        ALL_CONFIGS+=("$f")
    done < <(find "$COND_DIR" -type f -name "*.yaml" -print0 | sort -z)
fi

TOTAL=${#ALL_CONFIGS[@]}
echo "Found $TOTAL tokenizer-free configs (condition=$CONDITION)"

NODE_CONFIGS=()
for (( i=0; i<TOTAL; i++ )); do
    if (( i % NUM_NODES == NODE_ID - 1 )); then
        NODE_CONFIGS+=("${ALL_CONFIGS[$i]}")
    fi
done

echo "Node $NODE_ID assigned ${#NODE_CONFIGS[@]} configs"

run_config() {
    local cfg="$1"
    local slot="$2"
    local gpus="${GPU_SLOTS[$slot]}"
    local run_name log_file

    run_name=$(basename "$cfg" .yaml)
    log_file="${LOG_DIR}/${run_name}_node${NODE_ID}.log"

    echo "[N${NODE_ID} slot${slot} GPUs:${gpus}] START ${run_name}"
    CUDA_VISIBLE_DEVICES="$gpus" python script/train.py \
        --config_path "$cfg" > "$log_file" 2>&1

    if [[ $? -eq 0 ]]; then
        echo "[N${NODE_ID} slot${slot}] DONE  ${run_name}"
    else
        echo "[N${NODE_ID} slot${slot}] FAIL  ${run_name} (see $log_file)"
    fi
}

PIDS=()
for (( i=0; i<${#NODE_CONFIGS[@]}; i++ )); do
    slot=$(( i % PARALLEL_JOBS ))
    if (( i >= PARALLEL_JOBS )); then
        wait "${PIDS[$slot]}" || true
    fi
    run_config "${NODE_CONFIGS[$i]}" "$slot" &
    PIDS[$slot]=$!
done

for pid in "${PIDS[@]}"; do
    wait "$pid" || true
done

echo "Node $NODE_ID: all tokenizer-free configs complete"
