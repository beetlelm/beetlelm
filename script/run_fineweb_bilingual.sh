#!/bin/bash
# =============================================================================
# BeetleLM FineWeb Bilingual Training
# 4-GPU DDP per job · 2 parallel jobs per node · 8 GPUs fully utilised
# =============================================================================
#
# Usage:  ./run_fineweb_bilingual.sh <NODE_ID> [TIER] [CONDITION]
#
#   NODE_ID   = 1,2,3,4  (which node you are on)
#   TIER      = core|extension|low_resource|all  (optional, default: core)
#   CONDITION = balanced|simultaneous|sequential|all  (optional, default: all)
#
# Examples:
#   ./run_fineweb_bilingual.sh 1                  # All core conditions on node 1
#   ./run_fineweb_bilingual.sh 2 core balanced    # Only core balanced on node 2
#   ./run_fineweb_bilingual.sh 3 extension        # Extension langs on node 3
# =============================================================================

set -euo pipefail

CONFIG_DIR="configs/fineweb_bilingual"
LOG_DIR="logs_fineweb_bilingual"
GPUS_PER_NODE=8
PARALLEL_JOBS=2
GPUS_PER_JOB=$(( GPUS_PER_NODE / PARALLEL_JOBS ))
NUM_NODES=4

mkdir -p "$LOG_DIR"

# ── Parse arguments ──────────────────────────────────────────────────────────
NODE_ID=${1:?ERROR: provide NODE_ID 1-${NUM_NODES} as first argument}
TIER=${2:-core}
CONDITION=${3:-all}

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
if [[ "$TIER" == "all" ]]; then
    SEARCH_DIR="$CONFIG_DIR"
else
    SEARCH_DIR="$CONFIG_DIR/$TIER"
fi

if [[ "$CONDITION" == "all" ]]; then
    while IFS= read -r -d '' f; do
        ALL_CONFIGS+=("$f")
    done < <(find "$SEARCH_DIR" -type f -name "*.yaml" -print0 | sort -z)
else
    while IFS= read -r -d '' f; do
        ALL_CONFIGS+=("$f")
    done < <(find "$SEARCH_DIR/$CONDITION" -type f -name "*.yaml" -print0 2>/dev/null | sort -z)
fi

TOTAL=${#ALL_CONFIGS[@]}
if [[ "$TOTAL" -eq 0 ]]; then
    echo "ERROR: No configs found in $SEARCH_DIR"
    echo "Run generate_fineweb_bilingual.py first."
    exit 1
fi
echo "Found $TOTAL FineWeb bilingual configs (tier=$TIER, condition=$CONDITION)"

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
