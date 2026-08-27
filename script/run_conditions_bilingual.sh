#!/bin/bash
# =============================================================================
# BeetleLM Bilingual Conditions Training (B1–B6)
# 4-GPU DDP per job · 2 parallel jobs per node · 8 GPUs fully utilised
# =============================================================================
#
# Usage:  ./run_conditions_bilingual.sh <NODE_ID> [CONDITION]
#
#   NODE_ID   = 1,2,3,4  (which node you are on)
#   CONDITION = B1|B2|B3|B4|B5|B6|all  (optional, default: all)
#
# Examples:
#   ./run_conditions_bilingual.sh 1        # Run all bilingual conditions on node 1
#   ./run_conditions_bilingual.sh 2 B2     # Run only B2 configs on node 2
#
# GPU layout per node:
#   Job slot 0 → CUDA_VISIBLE_DEVICES=0,1,2,3
#   Job slot 1 → CUDA_VISIBLE_DEVICES=4,5,6,7
# =============================================================================

set -euo pipefail

CONFIG_DIR="configs/conditions/bilingual"
LOG_DIR="logs_conditions_bilingual"
GPUS_PER_NODE=8
PARALLEL_JOBS=2
GPUS_PER_JOB=$(( GPUS_PER_NODE / PARALLEL_JOBS ))
NUM_NODES=4

mkdir -p "$LOG_DIR"

# ── Parse arguments ──────────────────────────────────────────────────────────
NODE_ID=${1:?ERROR: provide NODE_ID 1-${NUM_NODES} as first argument}
CONDITION=${2:-all}

if [[ "$NODE_ID" -lt 1 || "$NODE_ID" -gt "$NUM_NODES" ]]; then
    echo "ERROR: NODE_ID must be 1–${NUM_NODES}, got $NODE_ID"
    exit 1
fi

# ── Build GPU slot strings ───────────────────────────────────────────────────
GPU_SLOTS=()
for (( slot=0; slot<PARALLEL_JOBS; slot++ )); do
    start=$(( slot * GPUS_PER_JOB ))
    end=$(( start + GPUS_PER_JOB - 1 ))
    GPU_SLOTS+=( "$(seq -s, "$start" "$end")" )
done

# ── Discover configs ─────────────────────────────────────────────────────────
ALL_CONFIGS=()
if [[ "$CONDITION" == "all" ]]; then
    while IFS= read -r -d '' f; do
        ALL_CONFIGS+=("$f")
    done < <(find "$CONFIG_DIR" -type f -name "*.yaml" -print0 | sort -z)
else
    COND_DIR="$CONFIG_DIR/$CONDITION"
    if [[ ! -d "$COND_DIR" ]]; then
        echo "ERROR: condition directory not found: $COND_DIR"
        echo "Available: $(ls "$CONFIG_DIR" 2>/dev/null || echo 'none — run generate_conditions.py first')"
        exit 1
    fi
    while IFS= read -r -d '' f; do
        ALL_CONFIGS+=("$f")
    done < <(find "$COND_DIR" -type f -name "*.yaml" -print0 | sort -z)
fi

TOTAL=${#ALL_CONFIGS[@]}
echo "Found $TOTAL bilingual configs (condition=$CONDITION)"

# ── Assign configs to this node (interleaved split) ──────────────────────────
NODE_CONFIGS=()
for (( i=0; i<TOTAL; i++ )); do
    if (( i % NUM_NODES == NODE_ID - 1 )); then
        NODE_CONFIGS+=("${ALL_CONFIGS[$i]}")
    fi
done

echo "Node $NODE_ID assigned ${#NODE_CONFIGS[@]} configs"
echo "GPU slots: slot0=[${GPU_SLOTS[0]}]  slot1=[${GPU_SLOTS[1]}]"

# ── Job runner ───────────────────────────────────────────────────────────────
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

# ── Scheduler ────────────────────────────────────────────────────────────────
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
