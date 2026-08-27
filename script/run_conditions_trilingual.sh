#!/bin/bash
# =============================================================================
# BeetleLM Trilingual Conditions Training (T1–T4)
# 4-GPU DDP per job · 2 parallel jobs per node · 8 GPUs fully utilised
# =============================================================================
#
# Usage:  ./run_conditions_trilingual.sh <NODE_ID> [CONDITION]
#
#   NODE_ID   = 1,2,3,4
#   CONDITION = T1|T2|T3|T4|all  (optional, default: all)
# =============================================================================

set -euo pipefail

CONFIG_DIR="configs/conditions/trilingual"
LOG_DIR="logs_conditions_trilingual"
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
    done < <(find "$CONFIG_DIR" -type f -name "*.yaml" -print0 | sort -z)
else
    COND_DIR="$CONFIG_DIR/$CONDITION"
    if [[ ! -d "$COND_DIR" ]]; then
        echo "ERROR: condition directory not found: $COND_DIR"
        exit 1
    fi
    while IFS= read -r -d '' f; do
        ALL_CONFIGS+=("$f")
    done < <(find "$COND_DIR" -type f -name "*.yaml" -print0 | sort -z)
fi

TOTAL=${#ALL_CONFIGS[@]}
echo "Found $TOTAL trilingual configs (condition=$CONDITION)"

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
        echo "[N${NODE_ID} slot${slot}] FAIL  ${run_name} -> ${log_file}"
    fi
}

echo "=== Node $NODE_ID starting — ${#NODE_CONFIGS[@]} configs ==="
total_cfgs=${#NODE_CONFIGS[@]}
idx=0

while (( idx < total_cfgs )); do
    pids=()
    for (( slot=0; slot<PARALLEL_JOBS; slot++ )); do
        if (( idx < total_cfgs )); then
            run_config "${NODE_CONFIGS[$idx]}" "$slot" &
            pids+=($!)
            (( idx++ ))
        fi
    done
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
done

echo "=== Node $NODE_ID finished all ${#NODE_CONFIGS[@]} configs ==="
