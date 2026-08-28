#!/bin/bash
# =============================================================================
# BeetleLM — FineWeb Single Model Training on Wilkes3 (Cambridge HPC)
#
# Max SL2 allocation: 16 nodes x 4 A100-SXM-80GB = 64 GPUs total
# Target: 24B tokens, 1 epoch (91,552 steps, effective batch=512)
# Expected wall-clock: ~45 min on 64 GPUs; 4h slot gives headroom.
# Auto-resume from HuggingFace checkpoints.
#
# IMPORTANT: pass an HPC-64 config (per_gpu_batch=8, devices/node=4):
#   configs/fineweb_bilingual_hpc64/core/{B1,B2,B3}/train_*.yaml
# Passing an 8-GPU config here will run per_gpu_batch=64 on 64 GPUs
# → effective batch 4096 instead of 512 → wrong training dynamics.
#
# Usage: sbatch script/slurm_fineweb_one.sh <HPC64_CONFIG_YAML>
# =============================================================================

#SBATCH -J beetle-fw
#SBATCH -A BUTTERY-SL2-GPU
#SBATCH --nodes=16
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=36:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH -p ampere
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load python/3.11.0-icl
mkdir -p logs/slurm

SCRATCH_ROOT="${BEETLE_SCRATCH:-$HOME/beetle-scratch}"
[[ -f "$SCRATCH_ROOT/beetle.env" ]] && source "$SCRATCH_ROOT/beetle.env"

numnodes=$SLURM_JOB_NUM_NODES
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
port_number=$(shuf -i 29510-49510 -n 1)

CONFIG_PATH="$1"
if [[ -z "$CONFIG_PATH" ]]; then
    echo "ERROR: provide config YAML path as first argument"
    exit 1
fi

RUN_NAME=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH'))['run_name'])")
HF_USER=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH'))['hf_user'])")
MAX_STEPS=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH'))['training']['max_steps'])")

# ── Auto-resume: find latest checkpoint on HuggingFace ──
RESUME_FLAG=""
LATEST_STEP=$(python3 -c "
from huggingface_hub import list_repo_refs
import re
try:
    refs = list_repo_refs('${HF_USER}/${RUN_NAME}')
    steps = [int(m.group(1)) for r in refs.branches if (m := re.match(r'step-(\d+)', r.name))]
    print(max(steps) if steps else 0)
except:
    print(0)
" 2>/dev/null)

if [[ "$LATEST_STEP" -ge "$MAX_STEPS" ]]; then
    echo "Model $RUN_NAME already fully trained (step $LATEST_STEP >= $MAX_STEPS). Skipping."
    exit 0
elif [[ "$LATEST_STEP" -gt 0 ]]; then
    echo "Resuming $RUN_NAME from step $LATEST_STEP"
    RESUME_FLAG="--resume_checkpoint"
fi

echo -e "JobID: $SLURM_JOB_ID\n======"
echo "Time: $(date)"
echo "Running on master node: $(hostname)"
echo "Node IP: $head_node_ip"
echo "Num nodes: $numnodes"
echo "Config: $CONFIG_PATH"
echo "Run name: $RUN_NAME"
echo "Latest step: $LATEST_STEP / $MAX_STEPS"
echo "Current directory: $(pwd)"

cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source "${VENV_DIR:-$SCRATCH_ROOT/pytorch-ampere-env}/bin/activate"

srun torchrun \
    --nnodes $numnodes \
    --nproc-per-node 4 \
    --rdzv-id $SLURM_JOB_ID \
    --rdzv-backend c10d \
    --rdzv-endpoint $head_node_ip:$port_number \
    script/train.py --config_path "$CONFIG_PATH" $RESUME_FLAG
