#!/bin/bash
# =============================================================================
# BeetleLM — 4-node (16 GPU) FineWeb training on Wilkes3
#
# Why this exists alongside slurm_fineweb_one.sh:
#   slurm_fineweb_one.sh uses all 64 GPUs of the SL2 cap on ONE config — a
#   sweep then runs strictly sequentially. This template uses 16 GPUs/job so
#   FOUR configs run concurrently under the same SL2 cap, finishing a 24-job
#   sweep ~4x faster in wall-clock.
#
# Target: 24B tokens, target_effective_batch=512 (=> per_gpu_batch must be 32
#   on a 16-GPU allocation). Configs under configs/fineweb_bilingual_hpc16/
#   should set training.fabric.devices=4 and training.batch_size=32.
#
# Pass the YAML as $1. The submitter writes per-job log paths into $LOG_DIR.
#
# Usage:
#   sbatch script/slurm_fineweb_16gpu.sh <YAML>
# =============================================================================

#SBATCH -J beetle-fw16
#SBATCH -A BUTTERY-SL2-GPU
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=36:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH -p ampere
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load python/3.11.0-icl

# ── Env from setup_wilkes3.sh ────────────────────────────────────────────────
SCRATCH_ROOT="${BEETLE_SCRATCH:-/rds/user/$USER/hpc-work}"
# shellcheck disable=SC1091
source "$SCRATCH_ROOT/beetle.env"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

mkdir -p logs/slurm

CONFIG_PATH="${1:?ERROR: provide config YAML as first argument}"

# ── Rendezvous ───────────────────────────────────────────────────────────────
nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
head_node="${nodes[0]}"
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
port_number=$(shuf -i 29510-49510 -n 1)

# ── Auto-resume from HF ──────────────────────────────────────────────────────
RUN_NAME=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH'))['run_name'])")
HF_USER=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH'))['hf_user'])")
MAX_STEPS=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH'))['training']['max_steps'])")

RESUME_FLAG=""
LATEST_STEP=$(python3 -c "
from huggingface_hub import list_repo_refs
import re
try:
    refs = list_repo_refs('${HF_USER}/${RUN_NAME}')
    steps = [int(m.group(1)) for r in refs.branches if (m := re.match(r'step-(\d+)', r.name))]
    print(max(steps) if steps else 0)
except Exception:
    print(0)
" 2>/dev/null)

if [[ "$LATEST_STEP" -ge "$MAX_STEPS" ]]; then
    echo "$RUN_NAME already at $LATEST_STEP/$MAX_STEPS — exiting."
    exit 0
elif [[ "$LATEST_STEP" -gt 0 ]]; then
    RESUME_FLAG="--resume_checkpoint"
fi

echo "JobID:    $SLURM_JOB_ID"
echo "Nodes:    ${nodes[*]}"
echo "Head IP:  $head_node_ip:$port_number"
echo "Config:   $CONFIG_PATH"
echo "Run:      $RUN_NAME  (step $LATEST_STEP / $MAX_STEPS)"

# ── Watchdog (kill the job if no heartbeat for 1h) ───────────────────────────
HEARTBEAT="$RUNS_DIR/$RUN_NAME/.heartbeat"
mkdir -p "$(dirname "$HEARTBEAT")"
(
    while sleep 300; do
        if [[ -f "$HEARTBEAT" ]]; then
            age=$(( $(date +%s) - $(stat -c %Y "$HEARTBEAT") ))
            if (( age > 3600 )); then
                echo "[watchdog] heartbeat stale (${age}s) — cancelling $SLURM_JOB_ID"
                scancel "$SLURM_JOB_ID"
                exit 0
            fi
        fi
    done
) &
WATCHDOG_PID=$!
trap "kill $WATCHDOG_PID 2>/dev/null || true" EXIT

# ── Launch ───────────────────────────────────────────────────────────────────
srun torchrun \
    --nnodes "$SLURM_JOB_NUM_NODES" \
    --nproc-per-node 4 \
    --rdzv-id "$SLURM_JOB_ID" \
    --rdzv-backend c10d \
    --rdzv-endpoint "$head_node_ip:$port_number" \
    script/train.py --config_path "$CONFIG_PATH" $RESUME_FLAG
