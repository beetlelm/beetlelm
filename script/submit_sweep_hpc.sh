#!/bin/bash
# =============================================================================
# BeetleLM — unified Wilkes3 sweep submitter
#
# Submits one SLURM job per YAML found under <CONFIG_DIR>. Picks the right
# SLURM template based on each YAML's model size + fabric.devices, respects
# the SL2 64-GPU concurrent cap, and skips configs already fully trained on
# the HuggingFace Hub.
#
# Templates dispatched to:
#   1B / 500M / 400M models    -> slurm_fineweb_16gpu.sh   (16 GPUs/job; FSDP)
#   142M (default hpc64 confs) -> slurm_fineweb_one.sh     (64 GPUs/job)
#   single-node configs        -> slurm_train_one.sh       (4 GPUs/job)
#
# Usage:
#   # 1. Discover what would be submitted
#   bash script/submit_sweep_hpc.sh configs/fineweb_bilingual_hpc64 --dry-run
#
#   # 2. Real submission, capped at 4 concurrent jobs (= 64 GPUs at 16/job):
#   bash script/submit_sweep_hpc.sh configs/fineweb_bilingual_hpc64 \
#        --max-concurrent-jobs 4
#
#   # 3. Multiple model sizes in one go (round-robin across sizes):
#   bash script/submit_sweep_hpc.sh \
#        configs/fineweb_bilingual_hpc64 \
#        configs/fineweb_bilingual_500m/core \
#        configs/fineweb_bilingual_1b/core
#
# Re-running is safe: configs already at max_steps on HF are skipped; in-flight
# jobs in your squeue are not re-submitted (matched by --job-name).
# =============================================================================

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
DRY_RUN=false
MAX_CONCURRENT_JOBS=4         # 4 jobs × 16 GPUs = 64 = SL2 cap
ACCOUNT="${BEETLE_ACCOUNT:-BUTTERY-SL2-GPU}"
PARTITION="ampere"
MAIL_USER="${BEETLE_MAIL_USER:-salhananusha@gmail.com}"
WALLTIME="${BEETLE_WALLTIME:-36:00:00}"
FILTER_REGEX=""
FORCE_TEMPLATE=""             # override auto-dispatch
CONFIG_DIRS=()

# ── Argument parse ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)              DRY_RUN=true;            shift;;
        --max-concurrent-jobs)  MAX_CONCURRENT_JOBS="$2"; shift 2;;
        --account)              ACCOUNT="$2";            shift 2;;
        --partition)            PARTITION="$2";          shift 2;;
        --time)                 WALLTIME="$2";           shift 2;;
        --mail-user)            MAIL_USER="$2";          shift 2;;
        --filter)               FILTER_REGEX="$2";       shift 2;;
        --template)             FORCE_TEMPLATE="$2";     shift 2;;
        -h|--help)              sed -n '2,30p' "$0"; exit 0;;
        -*)                     echo "Unknown flag: $1"; exit 1;;
        *)                      CONFIG_DIRS+=("$1");     shift;;
    esac
done

if [[ ${#CONFIG_DIRS[@]} -eq 0 ]]; then
    echo "ERROR: provide at least one config directory."
    echo "Try: bash $0 configs/fineweb_bilingual_hpc64 --dry-run"
    exit 1
fi

# ── Helpers ──────────────────────────────────────────────────────────────────
yaml_field() {
    # $1 = yaml path, $2 = dotted key (e.g. training.max_steps)
    python3 - "$1" "$2" <<'PY'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1]))
for k in sys.argv[2].split('.'):
    d = d.get(k, {}) if isinstance(d, dict) else None
print(d if d is not None else "")
PY
}

hf_latest_step() {
    # $1 = "<hf_user>/<run_name>". Empty -> 0
    python3 - "$1" <<'PY' 2>/dev/null || echo 0
import sys, re
from huggingface_hub import list_repo_refs
try:
    refs = list_repo_refs(sys.argv[1])
    steps = [int(m.group(1)) for r in refs.branches
             if (m := re.match(r'step-(\d+)', r.name))]
    print(max(steps) if steps else 0)
except Exception:
    print(0)
PY
}

pick_template() {
    # $1 = yaml. Heuristic: param count from n_layers × d_model^2.
    local cfg="$1"
    if [[ -n "$FORCE_TEMPLATE" ]]; then echo "$FORCE_TEMPLATE"; return; fi
    local nl dm
    nl=$(yaml_field "$cfg" "model.params.n_layers")
    dm=$(yaml_field "$cfg" "model.params.d_model")
    nl=${nl:-0}; dm=${dm:-0}
    # ~12 × n_layers × d_model^2 ≈ param count (no embeddings)
    local pc=$(( 12 * nl * dm * dm ))
    if   (( pc > 700000000 )); then echo "script/slurm_fineweb_16gpu.sh"  # ≥1B  (16 GPU, FSDP)
    elif (( pc > 250000000 )); then echo "script/slurm_fineweb_16gpu.sh"  # 400M-500M
    elif (( pc > 50000000  )); then echo "script/slurm_fineweb_one.sh"    # 142M-ish
    else                            echo "script/slurm_train_one.sh"      # tiny / single-node
    fi
}

job_gpu_footprint() {
    # rough mapping template -> GPUs consumed
    case "$1" in
        */slurm_fineweb_one.sh)   echo 64;;
        */slurm_fineweb_16gpu.sh) echo 16;;
        */slurm_train_one.sh)     echo  4;;
        *)                        echo  4;;
    esac
}

already_in_queue() {
    # $1 = job name. Returns 0 if a job with this name is queued/running.
    squeue -u "$USER" -h -o "%j" | grep -Fxq "$1"
}

count_my_running_or_pending() {
    squeue -u "$USER" -h -o "%j" | wc -l | tr -d ' '
}

# ── Discover configs ─────────────────────────────────────────────────────────
ALL_CFGS=()
for d in "${CONFIG_DIRS[@]}"; do
    if [[ ! -d "$d" ]]; then
        echo "WARN: not a directory: $d" >&2
        continue
    fi
    while IFS= read -r -d '' f; do
        ALL_CFGS+=("$f")
    done < <(find "$d" -type f -name "*.yaml" -print0 | sort -z)
done

if [[ ${#ALL_CFGS[@]} -eq 0 ]]; then
    echo "No YAML configs found under: ${CONFIG_DIRS[*]}"
    exit 0
fi

echo "Discovered ${#ALL_CFGS[@]} configs across ${#CONFIG_DIRS[@]} dirs."

# ── Submission loop ──────────────────────────────────────────────────────────
SUBMITTED=0
SKIPPED_DONE=0
SKIPPED_INFLIGHT=0
SKIPPED_FILTER=0
WAITED=0

for cfg in "${ALL_CFGS[@]}"; do
    if [[ -n "$FILTER_REGEX" && ! "$cfg" =~ $FILTER_REGEX ]]; then
        SKIPPED_FILTER=$((SKIPPED_FILTER+1))
        continue
    fi

    RUN_NAME=$(yaml_field "$cfg" "run_name")
    HF_USER=$(yaml_field  "$cfg" "hf_user")
    MAX_STEPS=$(yaml_field "$cfg" "training.max_steps")
    [[ -z "$RUN_NAME" || -z "$HF_USER" || -z "$MAX_STEPS" ]] && {
        echo "[skip-malformed] $cfg"; continue;
    }

    JOB_NAME="beetle-${RUN_NAME}"
    TEMPLATE=$(pick_template "$cfg")
    JOB_GPUS=$(job_gpu_footprint "$TEMPLATE")

    # Already in squeue?
    if already_in_queue "$JOB_NAME"; then
        echo "[in-flight] $RUN_NAME  ($TEMPLATE)"
        SKIPPED_INFLIGHT=$((SKIPPED_INFLIGHT+1))
        continue
    fi

    # Already done on HF?
    LATEST=$(hf_latest_step "${HF_USER}/${RUN_NAME}")
    if (( LATEST >= MAX_STEPS )); then
        echo "[done]      $RUN_NAME  ($LATEST/$MAX_STEPS)"
        SKIPPED_DONE=$((SKIPPED_DONE+1))
        continue
    fi

    # SL2 concurrency throttle — block until we have a slot.
    while ! $DRY_RUN; do
        CUR=$(count_my_running_or_pending)
        (( CUR < MAX_CONCURRENT_JOBS )) && break
        if (( WAITED == 0 )); then
            echo "[throttle] $CUR jobs queued, waiting for slot under cap=$MAX_CONCURRENT_JOBS..."
        fi
        WAITED=$((WAITED+1))
        sleep 60
    done

    if $DRY_RUN; then
        printf '[dry] sbatch -A %s -t %s -p %s --mail-user=%s -J %s %s %s   (gpus=%d, %d/%d on HF)\n' \
            "$ACCOUNT" "$WALLTIME" "$PARTITION" "$MAIL_USER" \
            "$JOB_NAME" "$TEMPLATE" "$cfg" "$JOB_GPUS" "$LATEST" "$MAX_STEPS"
    else
        sbatch \
            -A "$ACCOUNT" \
            -t "$WALLTIME" \
            -p "$PARTITION" \
            --mail-user="$MAIL_USER" \
            --mail-type=FAIL,END \
            -J "$JOB_NAME" \
            "$TEMPLATE" "$cfg"
    fi
    SUBMITTED=$((SUBMITTED+1))
done

echo ""
echo "─────────────────────────────────────────"
echo "Submitted:        $SUBMITTED"
echo "Skipped (done):   $SKIPPED_DONE"
echo "Skipped (queued): $SKIPPED_INFLIGHT"
echo "Skipped (filter): $SKIPPED_FILTER"
echo "Cap:              $MAX_CONCURRENT_JOBS concurrent jobs"
$DRY_RUN && echo "(dry run — nothing was actually sbatched)"
