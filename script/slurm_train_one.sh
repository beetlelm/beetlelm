#!/bin/bash
# =============================================================================
# BeetleLM — Single Model Training on Wilkes3 (Cambridge HPC)
#
# Ampere partition: 4xA100-SXM-80GB per node, 128 CPU cores, 1TB RAM
# SL2 limits: 64 GPUs concurrent, 36h max wallclock
#
# Usage: sbatch script/slurm_train_one.sh <CONFIG_YAML>
# =============================================================================

#SBATCH -J beetle-train
#SBATCH -A BUTTERY-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
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

# Sourced env (NCCL/InfiniBand, HF cache on /rds scratch, venv path)
SCRATCH_ROOT="${BEETLE_SCRATCH:-$HOME/beetle-scratch}"
[[ -f "$SCRATCH_ROOT/beetle.env" ]] && source "$SCRATCH_ROOT/beetle.env"

JOBID=$SLURM_JOB_ID
numnodes=$SLURM_JOB_NUM_NODES
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e 's/^\([0-9][0-9]*\).*$/\1/')

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
port_number=$(shuf -i 29510-49510 -n 1)

CONFIG_PATH="$1"
if [[ -z "$CONFIG_PATH" ]]; then
    echo "ERROR: provide config YAML path as first argument"
    exit 1
fi

echo -e "JobID: $JOBID\n======"
echo "Time: $(date)"
echo "Running on master node: $(hostname)"
echo "Node IP: $head_node_ip"
echo "Config: $CONFIG_PATH"
echo "Current directory: $(pwd)"

cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source "${VENV_DIR:-$SCRATCH_ROOT/pytorch-ampere-env}/bin/activate"

srun torchrun \
    --nnodes 1 \
    --nproc-per-node 4 \
    script/train.py --config_path "$CONFIG_PATH"
