#!/bin/bash
# =============================================================================
# BeetleLM — Wilkes3 (CSD3 Ampere) environment setup
#
# Run ONCE on a Wilkes3 login node before submitting any beetle jobs.
# Creates a venv on /rds scratch (NOT $HOME — 40 GB quota), installs
# CUDA-matched PyTorch, then beetlelm in editable mode.
#
# NOTE on Python version:
#   The Wilkes3 docs recommend `python/3.8.11/gcc-9.4.0-yb6rzr6` for
#   pytorch-ampere, but beetlelm's pyproject.toml requires python ^3.10,<3.13
#   (lightning >=2.4, transformers >=4.45). We use `python/3.11.0-icl`, which
#   is the standard newer module on Wilkes3. If it's not present on your
#   cluster, run `module avail python` and substitute the closest 3.10/3.11.
#
# Usage:
#   cd /rds/user/$USER/hpc-work
#   git clone <beetlelm-remote> beetlelm && cd beetlelm
#   bash script/setup_wilkes3.sh
# =============================================================================

set -euo pipefail

# ── 0. Sanity ────────────────────────────────────────────────────────────────
if [[ "$HOSTNAME" != *login* && -z "${BEETLE_FORCE_SETUP:-}" ]]; then
    echo "ERROR: run this on a Wilkes3 login node (or set BEETLE_FORCE_SETUP=1)."
    exit 1
fi

SCRATCH_ROOT="${BEETLE_SCRATCH:-/rds/user/$USER/hpc-work}"
VENV_DIR="${SCRATCH_ROOT}/pytorch-ampere-env"
HF_CACHE="${SCRATCH_ROOT}/hf_cache"
PRETOK_CACHE="${SCRATCH_ROOT}/pretok_cache"
RUNS_DIR="${SCRATCH_ROOT}/runs"

mkdir -p "$SCRATCH_ROOT" "$HF_CACHE" "$PRETOK_CACHE" "$RUNS_DIR"

# ── 1. Cluster modules ───────────────────────────────────────────────────────
. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load python/3.11.0-icl    # was python/3.8.11/gcc-9.4.0-yb6rzr6 — beetlelm needs >=3.10

python --version
which python

# ── 2. Venv (on /rds scratch, NOT $HOME) ─────────────────────────────────────
if [[ ! -d "$VENV_DIR" ]]; then
    python -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel setuptools

# ── 3. PyTorch (CUDA 12.1 build for A100 driver on Wilkes3) ──────────────────
# pyproject pins torch >=2.5.1,<2.7.0 with index pytorch-cu121 — match that.
pip install --index-url https://download.pytorch.org/whl/cu121 \
    "torch>=2.5.1,<2.7.0" "torchvision" "torchaudio"

# ── 4. beetlelm + deps (editable so YAML edits land without reinstall) ───────
# poetry is heavy on HPC; install via pip from pyproject's runtime deps.
pip install \
    "lightning>=2.4.0,<3.0.0" \
    "click>=8.1.7" \
    "wandb>=0.18.1" \
    "huggingface-hub[cli]>=0.25.1" \
    "datasets>=4.0.0,<5.0.0" \
    "transformers>=4.45.2" \
    "evaluate>=0.4.3" \
    "rich>=13.9.4" \
    "omegaconf>=2.3.0" \
    "deepspeed>=0.16.2" \
    "hf_transfer" \
    "PyYAML"

pip install -e .

# ── 5. Pin env so SLURM jobs see the same paths ──────────────────────────────
cat > "$SCRATCH_ROOT/beetle.env" <<EOF
# sourced by every SLURM job — keep paths absolute
export BEETLE_SCRATCH="$SCRATCH_ROOT"
export VENV_DIR="$VENV_DIR"
export HF_HOME="$HF_CACHE"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_ETAG_TIMEOUT=60
export PRETOK_CACHE="$PRETOK_CACHE"
export RUNS_DIR="$RUNS_DIR"
export OMP_NUM_THREADS=1
export LOGLEVEL=INFO
# NCCL on Wilkes3 InfiniBand
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_HCA=mlx5
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
# PyTorch memory hygiene (carried over from beetle_pipeline 2026-05-13 fix)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHRUN_RESTARTS=1
EOF

echo ""
echo "=============================================================="
echo "Setup complete."
echo "  Venv:     $VENV_DIR"
echo "  Env file: $SCRATCH_ROOT/beetle.env  (sourced by SLURM scripts)"
echo "  Scratch:  $SCRATCH_ROOT"
echo "Next: huggingface-cli login   (token cached under \$HF_HOME)"
echo "      bash script/submit_sweep_hpc.sh <CONFIG_DIR> --dry-run"
echo "=============================================================="
