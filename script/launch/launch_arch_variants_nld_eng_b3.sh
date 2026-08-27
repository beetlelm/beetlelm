#!/usr/bin/env bash
# ============================================================================
# Architecture-variant sweep for nld_L1-eng_L2 at the B3 condition.
#
# Targets configs/b3_condition_sweep_2b/nld_L1-eng_L2/arch/train_arch-{VARIANT}.yaml
# Variants supported: bgpt, geo, moe, ssm. Default sweep is "bgpt moe ssm" —
# geo is excluded because it requires staged monolingual checkpoints and a
# Procrustes rotation matrix; opt back in via VARIANTS="bgpt geo moe ssm".
#
# Per-job hardware: 8x{A,H}100, single node, DDP, bf16-mixed.
# Push target:      Beetle-nld-eng-Variants (set in each YAML's hf_user).
#
# Crash-resilience contract (mirrors launch_fineweb_2b_groupA_safe.sh):
#   - One job's failure (GPU crash, OOM, NCCL hang, timeout) does NOT abort
#     the rest of the sweep.
#   - JOB_TIMEOUT (default 14h, ~12h training + buffer) bounds each job.
#   - After every job, kill_survivors() + drain_gpus() ensure orphaned
#     torchrun workers are reaped and GPU memory is freed before the next job.
#     If GPUs won't drain, the sweep aborts to protect the cluster.
#   - Pre-flight: nvidia-smi reports >=NGPUS, YAML's fabric.devices == NGPUS,
#     YAML's fabric.precision == bf16-mixed. Mismatches abort the *job*, not
#     the sweep.
#   - Per-variant pre-flight:
#       ssm: requires `import mamba_ssm`
#       moe: warns if any GPU has <70 GB total memory (still proceeds)
#       geo: requires ./tokenizers/geo_rotation_nld-eng.npy and the two HF
#            mono checkpoints to resolve.
#   - Sentinel <log>.done makes re-runs idempotent.
#   - On success, rm -rf runs/<run_name>/ to reclaim disk.
#
# Env (all optional):
#   VARIANTS               default "bgpt moe ssm"
#   NGPUS                  default 8
#   RESUME                 1 -> pass --resume_checkpoint to train.py
#   JOB_TIMEOUT            default 14h (any timeout(1) duration)
#   COOLDOWN_BETWEEN_JOBS  default 60
#   COOLDOWN_AFTER_FAIL    default 120
#   SKIP_DONE              default 1
#   CLEANUP_AFTER_SUCCESS  default 1
#   HF_TOKEN               required for HF push (warned if unset)
#
# Examples:
#   bash script/launch/launch_arch_variants_nld_eng_b3.sh
#   VARIANTS="bgpt"       bash script/launch/launch_arch_variants_nld_eng_b3.sh --dry_run
#   VARIANTS="bgpt geo moe ssm" bash script/launch/launch_arch_variants_nld_eng_b3.sh
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck source=_lib.sh
source "${SCRIPT_DIR}/_lib.sh"

NGPUS="${NGPUS:-8}"
RESUME="${RESUME:-0}"
JOB_TIMEOUT="${JOB_TIMEOUT:-14h}"
COOLDOWN_BETWEEN_JOBS="${COOLDOWN_BETWEEN_JOBS:-60}"
COOLDOWN_AFTER_FAIL="${COOLDOWN_AFTER_FAIL:-120}"
SKIP_DONE="${SKIP_DONE:-1}"
CLEANUP_AFTER_SUCCESS="${CLEANUP_AFTER_SUCCESS:-1}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/arch_variants_nld_eng_b3}"
mkdir -p "${LOG_DIR}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARN: HF_TOKEN is not set — HF push will fail." >&2
fi

locate_timeout_bin   # sets TIMEOUT_BIN

read -ra VARIANTS <<<"${VARIANTS:-bgpt moe ssm}"

EXTRA_ARGS=()
if [[ "${RESUME}" == "1" ]]; then
  EXTRA_ARGS+=(--resume_checkpoint)
fi
EXTRA_ARGS+=("$@")

SUCCEEDED=()
FAILED=()
SKIPPED=()

ARCH_DIR="${PROJECT_ROOT}/configs/b3_condition_sweep_2b/nld_L1-eng_L2/arch"

# ─── Variant-specific pre-flight checks ─────────────────────────────────────
variant_preflight_ssm() {
  python3 -c "import mamba_ssm" >/dev/null 2>&1
}

variant_preflight_moe_warn() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then return 0; fi
  local min_mb
  min_mb="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | sort -n | head -n1)"
  if [[ -n "${min_mb}" && "${min_mb}" -lt 70000 ]]; then
    echo "[preflight] WARN(moe): smallest GPU has ${min_mb} MiB total (<70 GB). If OOM, drop moe_capacity_factor to 1.0 or batch_size to 32." >&2
  fi
}

variant_preflight_geo() {
  local rot="${PROJECT_ROOT}/tokenizers/geo_rotation_nld-eng.npy"
  if [[ ! -f "${rot}" ]]; then
    echo "[preflight] geo: rotation matrix missing at ${rot}" >&2
    return 1
  fi
  python3 - <<'PY' >/dev/null 2>&1
from huggingface_hub import HfApi
api = HfApi()
for repo in (
    "Beetle-nld-eng-Variants/beetle-monolingual-fineweb-2b-nld-mono",
    "Beetle-nld-eng-Variants/beetle-monolingual-fineweb-2b-eng-mono",
):
    api.repo_info(repo)
PY
}

# ─── Per-job runner ──────────────────────────────────────────────────────────
run_job() {
  local variant="$1"
  local cfg="${ARCH_DIR}/train_arch-${variant}.yaml"
  local log="${LOG_DIR}/arch-${variant}.log"
  local label="arch-${variant}"
  local sentinel="${log%.log}.done"

  if [[ ! -f "${cfg}" ]]; then
    echo "[$(date -u +%FT%TZ)] FAIL (missing config): ${cfg}" >&2
    FAILED+=("missing-config:${label}")
    return 0
  fi

  if [[ "${SKIP_DONE}" == "1" && -f "${sentinel}" ]]; then
    echo "[$(date -u +%FT%TZ)] SKIP (sentinel present): ${label}"
    SKIPPED+=("already-done:${label}")
    return 0
  fi

  # Variant-specific pre-flight (before GPU checks — bail early on missing deps)
  case "${variant}" in
    ssm)
      if ! variant_preflight_ssm; then
        echo "[$(date -u +%FT%TZ)] FAIL (mamba_ssm not importable): ${label}" | tee -a "${log}" >&2
        FAILED+=("missing-mamba-ssm:${label}")
        return 0
      fi
      ;;
    moe)
      variant_preflight_moe_warn
      ;;
    geo)
      if ! variant_preflight_geo; then
        echo "[$(date -u +%FT%TZ)] SKIP (geo prerequisites missing): ${label}" | tee -a "${log}"
        SKIPPED+=("geo-prereq-missing:${label}")
        return 0
      fi
      ;;
    bgpt) : ;;
    *)
      echo "[$(date -u +%FT%TZ)] FAIL (unknown variant): ${variant}" >&2
      FAILED+=("unknown-variant:${label}")
      return 0
      ;;
  esac

  if ! gpu_preflight "${NGPUS}"; then
    echo "[$(date -u +%FT%TZ)] FAIL (gpu unhealthy before): ${label}" | tee -a "${log}"
    FAILED+=("gpu-unhealthy:${label}")
    sleep "${COOLDOWN_AFTER_FAIL}"
    return 0
  fi

  if ! config_gpu_compliance "${cfg}" "${NGPUS}" 2>>"${log}"; then
    echo "[$(date -u +%FT%TZ)] FAIL (gpu-config-mismatch): ${label}" | tee -a "${log}" >&2
    FAILED+=("gpu-config-mismatch:${label}")
    return 0
  fi

  echo "============================================================"
  echo "[$(date -u +%FT%TZ)] launching: ${cfg}"
  echo "  label  : ${label}"
  echo "  log    : ${log}"
  echo "  timeout: ${JOB_TIMEOUT}"
  echo "  ngpus  : ${NGPUS}"
  echo "============================================================"

  # ── Reliable exit-code capture via temp file ──────────────────────────────
  local rc_file
  rc_file="$(mktemp)"
  printf '255' > "${rc_file}"

  if [[ -n "${TIMEOUT_BIN}" ]]; then
    {
      "${TIMEOUT_BIN}" --foreground --kill-after=2m "${JOB_TIMEOUT}" \
        torchrun \
          --standalone \
          --nproc_per_node="${NGPUS}" \
          "${PROJECT_ROOT}/script/train.py" \
          --config_path "${cfg}" \
          "${EXTRA_ARGS[@]}"
      printf '%s' "$?" > "${rc_file}"
    } 2>&1 | tee "${log}" &
  else
    {
      torchrun \
        --standalone \
        --nproc_per_node="${NGPUS}" \
        "${PROJECT_ROOT}/script/train.py" \
        --config_path "${cfg}" \
        "${EXTRA_ARGS[@]}"
      printf '%s' "$?" > "${rc_file}"
    } 2>&1 | tee "${log}" &
  fi
  local pipe_pid=$!
  wait "${pipe_pid}" || true

  local rc
  rc="$(cat "${rc_file}")"
  rm -f "${rc_file}"

  # ── Post-job cleanup: kill orphans, wait for GPU drain ────────────────────
  kill_survivors
  if ! drain_gpus 180; then
    kill_survivors
    if ! drain_gpus 60; then
      echo "[$(date -u +%FT%TZ)] FATAL: GPUs did not drain after two cleanup attempts." >&2
      echo "[$(date -u +%FT%TZ)] Aborting sweep to protect the cluster." >&2
      exit 1
    fi
  fi

  if [[ "${rc}" -ne 0 ]]; then
    case "${rc}" in
      124|137) reason="timeout(rc=${rc})" ;;
      *)       reason="rc=${rc}"          ;;
    esac
    echo "[$(date -u +%FT%TZ)] FAIL (${reason}): ${label}" | tee -a "${log}"
    FAILED+=("${reason}:${label}")
    sleep "${COOLDOWN_AFTER_FAIL}"
    return 0
  fi

  : > "${sentinel}"
  SUCCEEDED+=("${label}")
  echo "[$(date -u +%FT%TZ)] OK: ${label}"

  if [[ "${CLEANUP_AFTER_SUCCESS}" == "1" ]]; then
    local run_name=""
    if run_name="$(python3 - "${cfg}" <<'PY' 2>/dev/null
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["run_name"])
PY
)"; then
      run_name="${run_name//$'\n'/}"
      cleanup_runs_if_pushed "${run_name}" "${log}"
    else
      echo "[$(date -u +%FT%TZ)] WARN: could not parse run_name from ${cfg}; skipping cleanup" | tee -a "${log}"
    fi
  fi

  rotate_log "${log}"
  sleep "${COOLDOWN_BETWEEN_JOBS}"
}

# ─── Sweep ───────────────────────────────────────────────────────────────────
echo "[$(date -u +%FT%TZ)] arch-variant sweep start: VARIANTS=(${VARIANTS[*]}) NGPUS=${NGPUS}"

if ! gpu_preflight "${NGPUS}"; then
  echo "[$(date -u +%FT%TZ)] WARN: pre-sweep GPU preflight failed, continuing anyway"
fi

if ! disk_preflight "${MIN_FREE_GB:-200}"; then
  echo "[$(date -u +%FT%TZ)] FATAL: insufficient disk space for sweep; aborting." >&2
  exit 1
fi

for VARIANT in "${VARIANTS[@]}"; do
  run_job "${VARIANT}"
done

# ─── Summary ─────────────────────────────────────────────────────────────────
echo "============================================================"
echo "[$(date -u +%FT%TZ)] launch_arch_variants_nld_eng_b3.sh: DONE"
echo "  succeeded: ${#SUCCEEDED[@]}"
echo "  skipped  : ${#SKIPPED[@]}"
echo "  failed   : ${#FAILED[@]}"
if [[ "${#FAILED[@]}" -gt 0 ]]; then
  echo "  failures :"
  for f in "${FAILED[@]}"; do
    echo "    - ${f}"
  done
fi
echo "============================================================"

if [[ "${#FAILED[@]}" -gt 0 ]]; then
  exit 1
fi
exit 0
