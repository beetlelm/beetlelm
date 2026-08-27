#!/usr/bin/env bash
# ============================================================================
# Standalone launcher: FineWeb 24B B1-B3 sweep (10 langs).
# Mirrors launch_fineweb_2b_b1_b5_remaining.sh's crash-resilience contract.
#
# Excludes nld, deu, zho per the 24B-specific scope decision.
# Most YAMLs do NOT exist on disk yet — defaults to SKIP_MISSING=1 so missing
# configs do not abort the sweep.
#
# Default LANGS (10): arb eus fil hin ita jpn pol rus spa tur
# Default TIERS (3):  B1 B2 B3
#
# Env (all optional):
#   LANGS                  default the 10-lang set above
#   TIERS                  default "B1 B2 B3"
#   SKIP_MISSING           default 1 (24B configs largely not yet generated)
#   SKIP_DONE              default 1
#   STAGE0                 default 0 (24B uses raw 28B Beetle-Data shards)
#   NGPUS                  default 8
#   JOB_TIMEOUT            default 96h
#   COOLDOWN_BETWEEN_JOBS  default 60
#   COOLDOWN_AFTER_FAIL    default 120
#   CLEANUP_AFTER_SUCCESS  default 1
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck source=_lib.sh
source "${SCRIPT_DIR}/_lib.sh"

NGPUS="${NGPUS:-8}"
RESUME="${RESUME:-0}"
SKIP_MISSING="${SKIP_MISSING:-1}"
SKIP_DONE="${SKIP_DONE:-1}"
JOB_TIMEOUT="${JOB_TIMEOUT:-96h}"
COOLDOWN_BETWEEN_JOBS="${COOLDOWN_BETWEEN_JOBS:-60}"
COOLDOWN_AFTER_FAIL="${COOLDOWN_AFTER_FAIL:-120}"
CLEANUP_AFTER_SUCCESS="${CLEANUP_AFTER_SUCCESS:-1}"
STAGE0="${STAGE0:-0}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/fineweb_24b_b1_b3}"
CONFIG_ROOT="${CONFIG_ROOT:-${PROJECT_ROOT}/configs/fineweb_bilingual_24b/core}"
mkdir -p "${LOG_DIR}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARN: HF_TOKEN is not set — HF push will fail." >&2
fi

locate_timeout_bin

read -ra LANGS <<<"${LANGS:-arb eus fil hin ita jpn pol rus spa tur}"
read -ra TIERS <<<"${TIERS:-B1 B2 B3}"

EXTRA_ARGS=()
if [[ "${RESUME}" == "1" ]]; then
  EXTRA_ARGS+=(--resume_checkpoint)
fi
EXTRA_ARGS+=("$@")

SUCCEEDED=()
FAILED=()
SKIPPED=()

run_job() {
  local cfg="$1" log="$2" label="$3"
  local sentinel="${log%.log}.done"

  if [[ ! -f "${cfg}" ]]; then
    if [[ "${SKIP_MISSING}" == "1" ]]; then
      echo "[$(date -u +%FT%TZ)] SKIP (missing config): ${cfg}"
      SKIPPED+=("missing-config:${label}")
      return 0
    fi
    echo "[$(date -u +%FT%TZ)] FAIL (missing config): ${cfg}" >&2
    FAILED+=("missing-config:${label}")
    return 0
  fi

  if [[ "${SKIP_DONE}" == "1" && -f "${sentinel}" ]]; then
    echo "[$(date -u +%FT%TZ)] SKIP (sentinel present): ${label}"
    SKIPPED+=("already-done:${label}")
    return 0
  fi

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

  echo "[$(date -u +%FT%TZ)] launching ${label}: ${cfg} (timeout=${JOB_TIMEOUT})"

  local rc_file
  rc_file="$(mktemp)"
  printf '255' > "${rc_file}"

  if [[ -n "${TIMEOUT_BIN}" ]]; then
    {
      "${TIMEOUT_BIN}" --foreground --kill-after=2m "${JOB_TIMEOUT}" \
        torchrun --standalone --nproc_per_node="${NGPUS}" \
          "${PROJECT_ROOT}/script/train.py" --config_path "${cfg}" "${EXTRA_ARGS[@]}"
      printf '%s' "$?" > "${rc_file}"
    } 2>&1 | tee "${log}" &
  else
    {
      torchrun --standalone --nproc_per_node="${NGPUS}" \
        "${PROJECT_ROOT}/script/train.py" --config_path "${cfg}" "${EXTRA_ARGS[@]}"
      printf '%s' "$?" > "${rc_file}"
    } 2>&1 | tee "${log}" &
  fi
  local pipe_pid=$!
  wait "${pipe_pid}" || true

  local rc
  rc="$(cat "${rc_file}")"
  rm -f "${rc_file}"

  kill_survivors
  if ! drain_gpus 180; then
    kill_survivors
    if ! drain_gpus 60; then
      echo "[$(date -u +%FT%TZ)] FATAL: GPUs did not drain." >&2
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

  if [[ "${CLEANUP_AFTER_SUCCESS}" == "1" ]]; then
    local run_name=""
    if run_name="$(python3 - "${cfg}" <<'PY' 2>/dev/null
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["run_name"])
PY
)"; then
      run_name="${run_name//$'\n'/}"
      cleanup_runs_if_pushed "${run_name}" "${log}"
    fi
  fi

  rotate_log "${log}"
  sleep "${COOLDOWN_BETWEEN_JOBS}"
}

echo "[$(date -u +%FT%TZ)] 24B sweep start: LANGS=(${LANGS[*]}) TIERS=(${TIERS[*]}) NGPUS=${NGPUS}"
gpu_preflight "${NGPUS}" || echo "[$(date -u +%FT%TZ)] WARN: pre-sweep gpu preflight failed"
disk_preflight "${MIN_FREE_GB:-200}" || { echo "FATAL: insufficient disk" >&2; exit 1; }

for LANG in "${LANGS[@]}"; do
  for TIER in "${TIERS[@]}"; do
    CFG="${CONFIG_ROOT}/${TIER}/train_${LANG}_L1-eng_L2.yaml"
    LOG="${LOG_DIR}/${TIER}_${LANG}_L1-eng_L2.log"
    LABEL="${TIER}_${LANG}_L1-eng_L2"
    run_job "${CFG}" "${LOG}" "${LABEL}"
  done
done

echo "============================================================"
echo "[$(date -u +%FT%TZ)] launch_fineweb_24b_b1_b3_sweep.sh: DONE"
echo "  succeeded: ${#SUCCEEDED[@]}"
echo "  skipped  : ${#SKIPPED[@]}"
echo "  failed   : ${#FAILED[@]}"
if [[ "${#FAILED[@]}" -gt 0 ]]; then
  echo "  failures:"
  for f in "${FAILED[@]}"; do echo "    - ${f}"; done
fi
echo "============================================================"

[[ "${#FAILED[@]}" -gt 0 ]] && exit 1
exit 0
