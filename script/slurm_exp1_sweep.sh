#!/bin/bash
# =============================================================================
# Experiment 1: Human-Scale Bilingual Sweep (B1/B2/B3 x EWC x MAML x LR)
#
# Wilkes3: Ampere partition, 4xA100-SXM-80GB per node
# Layout : 2 parallel jobs x 4 GPUs per node, 6h wallclock
#
# Phase 1: Pilot sweep on MECO pairs (nld-eng, zho-eng, deu-eng)
# Phase 2: Best settings applied to remaining language pairs
#
# Usage:
#   # Generate pilot configs first
#   python configs/generate_human_scale_bilingual.py --pilot
#
#   # Submit pilot sweep (array across 4 nodes)
#   sbatch --array=1-4 script/slurm_exp1_sweep.sh pilot
#
#   # After finding best settings, generate remaining configs
#   python configs/generate_human_scale_bilingual.py --best-config results/best_sweep_settings.json
#
#   # Submit best configs
#   sbatch --array=1-4 script/slurm_exp1_sweep.sh best
# =============================================================================

#SBATCH -J exp1-sweep
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

PHASE=${1:-pilot}
NUM_NODES=4
# Wilkes3 Ampere has 4 GPUs/node (not 8). 2 parallel jobs × 2 GPUs each is
# the only viable split; if you want all 4 GPUs on one config use
# PARALLEL_JOBS=1.
GPUS_PER_NODE=4
PARALLEL_JOBS=${BEETLE_PARALLEL_JOBS:-1}
GPUS_PER_JOB=$(( GPUS_PER_NODE / PARALLEL_JOBS ))

NODE_ID=${SLURM_ARRAY_TASK_ID:-1}
LOG_DIR="logs_exp1_sweep_${PHASE}"
mkdir -p "$LOG_DIR"

cd "$SLURM_SUBMIT_DIR/.."
source .venv/bin/activate

# Discover sweep configs
CONFIG_DIR="configs/human_scale_bilingual/shared"
ALL_CONFIGS=()

if [[ "$PHASE" == "pilot" ]]; then
    for cond in B1 B2 B3; do
        sweep_dir="$CONFIG_DIR/sweep_${cond}"
        if [[ -d "$sweep_dir" ]]; then
            while IFS= read -r -d '' f; do
                ALL_CONFIGS+=("$f")
            done < <(find "$sweep_dir" -type f -name "*.yaml" -print0 | sort -z)
        fi
    done
elif [[ "$PHASE" == "best" ]]; then
    best_dir="$CONFIG_DIR/best"
    if [[ -d "$best_dir" ]]; then
        while IFS= read -r -d '' f; do
            ALL_CONFIGS+=("$f")
        done < <(find "$best_dir" -type f -name "*.yaml" -print0 | sort -z)
    fi
else
    echo "ERROR: unknown phase '$PHASE'. Use 'pilot' or 'best'."
    exit 1
fi

TOTAL=${#ALL_CONFIGS[@]}
echo "[Exp1] Phase=$PHASE, Node=$NODE_ID/$NUM_NODES, Total configs=$TOTAL"

# Assign configs to this node (round-robin)
NODE_CONFIGS=()
for (( i=0; i<TOTAL; i++ )); do
    if (( i % NUM_NODES == NODE_ID - 1 )); then
        NODE_CONFIGS+=("${ALL_CONFIGS[$i]}")
    fi
done

echo "[Exp1] Node $NODE_ID assigned ${#NODE_CONFIGS[@]} configs"

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
    CUDA_VISIBLE_DEVICES="$gpus" python script/train.py \
        --config_path "$cfg" > "$log_file" 2>&1

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

echo "[Exp1] Node $NODE_ID finished all ${#NODE_CONFIGS[@]} configs"
