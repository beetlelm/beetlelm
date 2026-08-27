#!/usr/bin/env bash
# train_with_resume.sh — outer supervisor that relaunches `torchrun` on crash.
#
# Usage:
#     script/train_with_resume.sh configs/quickstart_human_scale.yaml [extra args…]
#
# Inner layer: torchrun itself restarts workers up to TORCHRUN_RESTARTS times
#              within a single launch (handles transient worker-level failures).
# Outer layer: this loop relaunches torchrun up to MAX_OUTER_RETRIES times
#              (handles failures that kill the whole launch — NCCL timeout,
#              OOM, node preemption, network partition, etc.).
#
# Resume is driven by `checkpointing.auto_resume: true` in the training YAML
# plus the HF Hub fallback in `_resume_from_hf` — every relaunch will pick up
# from the newest local checkpoint, or if local state is gone, from the Hub.
#
# Environment overrides:
#     NPROC_PER_NODE       number of GPUs per node  (default: 8)
#     TORCHRUN_RESTARTS    torchrun --max-restarts  (default: 3)
#     MAX_OUTER_RETRIES    outer relaunches         (default: 20)
#     BACKOFF_SECONDS      pause between relaunches (default: 30)
#     PYTHON_BIN           python / torchrun binary (default: torchrun)

set -u

if [[ "$#" -lt 1 ]]; then
    echo "Usage: $0 <config_path> [extra train.py args…]" >&2
    exit 2
fi

CONFIG="$1"
shift

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
# torchrun --max-restarts: prior default was 3, which compounded multiplicatively
# with the outer retry loop (worst case ~80 launches per yaml). Dropped to 1
# so a transient worker-level fault gets one fresh-rank retry; the outer layer
# decides whether to continue. Override TORCHRUN_RESTARTS to restore the old
# behaviour if a specific failure mode benefits from more inner restarts.
TORCHRUN_RESTARTS="${TORCHRUN_RESTARTS:-1}"
MAX_OUTER_RETRIES="${MAX_OUTER_RETRIES:-20}"
BACKOFF_SECONDS="${BACKOFF_SECONDS:-30}"
PYTHON_BIN="${PYTHON_BIN:-torchrun}"

# PyTorch allocator tuning + per-process memory fraction. Operators may
# override either env var. expandable_segments cuts fragmentation OOMs on
# long runs; BP_CUDA_MEM_FRACTION makes the trainer hit a Python-catchable
# CUDA OOM at 92% per rank, before the kernel OOM-killer wedges the node.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export BP_CUDA_MEM_FRACTION="${BP_CUDA_MEM_FRACTION:-0.92}"

# Watchdog-kill sentinel: if a previous attempt was SIGKILLed by gpu_watchdog,
# the outer orchestrator drops a sentinel that we read here to abort the
# resume loop. This prevents the inner retry layer from re-launching torchrun
# on a still-degrading GPU and inflating the 80x worst-case retry compounding.
WATCHDOG_KILL_SENTINEL="${WATCHDOG_KILL_SENTINEL:-${PWD}/.gpu_watchdog_kill}"

if [[ ! -f "$CONFIG" ]]; then
    echo "[train_with_resume] ERROR: config file not found: $CONFIG" >&2
    exit 2
fi

echo "[train_with_resume] config=$CONFIG"
echo "[train_with_resume] nproc_per_node=$NPROC_PER_NODE"
echo "[train_with_resume] torchrun max-restarts=$TORCHRUN_RESTARTS"
echo "[train_with_resume] outer retries=$MAX_OUTER_RETRIES, backoff=${BACKOFF_SECONDS}s"

attempt=0
while (( attempt < MAX_OUTER_RETRIES )); do
    attempt=$(( attempt + 1 ))
    echo "[train_with_resume] ── Attempt $attempt / $MAX_OUTER_RETRIES ──"

    "$PYTHON_BIN" \
        --standalone \
        --nproc_per_node="$NPROC_PER_NODE" \
        --max-restarts="$TORCHRUN_RESTARTS" \
        script/train.py \
        --config_path "$CONFIG" \
        "$@"
    rc=$?

    if [[ "$rc" -eq 0 ]]; then
        echo "[train_with_resume] Training completed successfully (attempt $attempt)."
        exit 0
    fi

    echo "[train_with_resume] Exit code $rc on attempt $attempt." >&2

    # Abort the inner retry loop if the outer watchdog killed us. Restarting
    # torchrun on a GPU the watchdog flagged just inflates retry compounding
    # and lets driver state keep degrading. The outer orchestrator's
    # family-block / cooldown will decide whether to retry at all.
    if [[ -f "$WATCHDOG_KILL_SENTINEL" ]]; then
        echo "[train_with_resume] watchdog-kill sentinel present at $WATCHDOG_KILL_SENTINEL — aborting resume loop." >&2
        exit 137
    fi

    if (( attempt < MAX_OUTER_RETRIES )); then
        echo "[train_with_resume] Sleeping ${BACKOFF_SECONDS}s before relaunch." >&2
        sleep "$BACKOFF_SECONDS"
    fi
done

echo "[train_with_resume] Exhausted $MAX_OUTER_RETRIES relaunches. Giving up." >&2
exit 1
