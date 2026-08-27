#!/bin/bash
# =============================================================================
# Experiment 2: FineWeb Bilingual (B1/B2/B3, L2=English, bidirectional)
#
# Wilkes3: Ampere partition, 4xA100-SXM-80GB per node
# Layout : 2 parallel jobs x 4 GPUs per node, 6h wallclock
# Target : 24B tokens, 1 epoch (91,552 steps, effective batch=512).
# Each job runs on 4 GPUs; configs must have per_gpu_batch=128 for 4-GPU
# effective batch = 512 (not yet generated — 2-parallel-jobs x 4-GPU path
# is parked; use script/slurm_fineweb_one.sh on the hpc64 tree instead).
#
# Language tiers:
#   Tier 1: zho, nld, deu (MECO overlap)
#   Tier 2: spa, rus, ita, por, tur
#
# Usage:
#   # Generate configs first
#   python configs/generate_fineweb_bilingual.py --bidirectional --hpc
#   python configs/generate_fineweb_bilingual.py --bidirectional --hpc --best-config results/best_sweep_settings.json
#
#   # Submit (4-node array)
#   sbatch --array=1-4 script/slurm_exp2_fineweb.sh [CONDITION]
#
#   CONDITION = B1|B2|B3|all  (optional, default: all)
# =============================================================================

#SBATCH -J exp2-fineweb
#SBATCH -A BUTTERY-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=36:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH -p ampere

set -euo pipefail

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

export OMP_NUM_THREADS=1
export LOGLEVEL=INFO

CONDITION=${1:-all}
NUM_NODES=4
# Wilkes3 Ampere has 4 GPUs/node (not 8). See note in slurm_exp1_sweep.sh.
GPUS_PER_NODE=4
PARALLEL_JOBS=${BEETLE_PARALLEL_JOBS:-1}
GPUS_PER_JOB=$(( GPUS_PER_NODE / PARALLEL_JOBS ))

NODE_ID=${SLURM_ARRAY_TASK_ID:-1}
LOG_DIR="logs_exp2_fineweb"
mkdir -p "$LOG_DIR"

cd "$SLURM_SUBMIT_DIR/.."
source .venv/bin/activate

# Discover FineWeb configs
CONFIG_DIR="configs/fineweb_bilingual/core"
ALL_CONFIGS=()

if [[ "$CONDITION" == "all" ]]; then
    while IFS= read -r -d '' f; do
        ALL_CONFIGS+=("$f")
    done < <(find "$CONFIG_DIR" -type f -name "*.yaml" -print0 | sort -z)
else
    cond_dir="$CONFIG_DIR/$CONDITION"
    if [[ ! -d "$cond_dir" ]]; then
        echo "ERROR: condition directory not found: $cond_dir"
        exit 1
    fi
    while IFS= read -r -d '' f; do
        ALL_CONFIGS+=("$f")
    done < <(find "$cond_dir" -type f -name "*.yaml" -print0 | sort -z)
fi

# Also include monolingual baselines
MONO_DIR="configs/fineweb_bilingual/monolingual"
if [[ -d "$MONO_DIR" ]]; then
    while IFS= read -r -d '' f; do
        ALL_CONFIGS+=("$f")
    done < <(find "$MONO_DIR" -type f -name "*.yaml" -print0 | sort -z)
fi

TOTAL=${#ALL_CONFIGS[@]}
echo "[Exp2] Condition=$CONDITION, Node=$NODE_ID/$NUM_NODES, Total configs=$TOTAL"

# Assign configs to this node (round-robin)
NODE_CONFIGS=()
for (( i=0; i<TOTAL; i++ )); do
    if (( i % NUM_NODES == NODE_ID - 1 )); then
        NODE_CONFIGS+=("${ALL_CONFIGS[$i]}")
    fi
done

echo "[Exp2] Node $NODE_ID assigned ${#NODE_CONFIGS[@]} configs"

# Build GPU slot strings
GPU_SLOTS=()
for (( slot=0; slot<PARALLEL_JOBS; slot++ )); do
    start=$(( slot * GPUS_PER_JOB ))
    end=$(( start + GPUS_PER_JOB - 1 ))
    GPU_SLOTS+=( "$(seq -s, "$start" "$end")" )
done

run_config() {
    local cfg="$1"
    local slot="$2"
    local gpus="${GPU_SLOTS[$slot]}"
    local run_name log_file

    run_name=$(basename "$cfg" .yaml)
    log_file="${LOG_DIR}/${run_name}_node${NODE_ID}.log"

    echo "[N${NODE_ID} slot${slot} GPUs:${gpus}] START ${run_name}"
    CUDA_VISIBLE_DEVICES="$gpus" torchrun \
        --nproc_per_node="$GPUS_PER_JOB" \
        script/train.py --config_path "$cfg" > "$log_file" 2>&1

    if [[ $? -eq 0 ]]; then
        echo "[N${NODE_ID} slot${slot}] DONE  ${run_name}"
    else
        echo "[N${NODE_ID} slot${slot}] FAIL  ${run_name} -> ${log_file}"
    fi
}

# Run configs: 2 parallel at a time
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

echo "[Exp2] Node $NODE_ID finished all ${#NODE_CONFIGS[@]} configs"
