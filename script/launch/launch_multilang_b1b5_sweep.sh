#!/usr/bin/env bash
# ============================================================================
# Multilang sweep — deu/zho/spa/ita/rus/hin across 2B + 24B regimes.
#
# Mirrors the contract of launch_fineweb_24b_zho_b2_b5.sh:
#   - sequential, one job at a time, full 8xA100 DDP per job
#   - kill_survivors + drain_gpus between jobs (sweep-safe handoff)
#   - sentinel <log>.done makes re-runs idempotent
#   - cleanup_runs_if_pushed reclaims disk only after a confirmed HF push
#
# Job order (per the May 2026 sweep request):
#   1.  deu B3 24B
#   2.  zho B1..B5 2B   (5 jobs)
#   3.  spa B1..B3 2B + B1..B3 24B   (6 jobs, 2B first then 24B)
#   4.  ita B1..B3 2B   (3 jobs)
#   5.  ita B1..B3 24B  (3 jobs)
#   6.  rus B1..B3 2B   (3 jobs)
#   7.  hin B1..B3 2B   (3 jobs)
# Total: 24 jobs (7 at 24B, 17 at 2B). Estimated wall ~30-32h on 8xA100.
#
# Usage:
#   bash script/launch/launch_multilang_b1b5_sweep.sh
#   RESUME=1 bash script/launch/launch_multilang_b1b5_sweep.sh
#
# Env (all optional, defaults match the zho sweep):
#   NGPUS                  default 8
#   HF_TOKEN               required for HF push (warned if unset)
#   RESUME                 1 -> pass --resume_checkpoint to train.py
#   JOB_TIMEOUT            default 48h
#   COOLDOWN_BETWEEN_JOBS  default 120
#   COOLDOWN_AFTER_FAIL    default 180
#   SKIP_DONE              default 1
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
JOB_TIMEOUT="${JOB_TIMEOUT:-48h}"
COOLDOWN_BETWEEN_JOBS="${COOLDOWN_BETWEEN_JOBS:-120}"
COOLDOWN_AFTER_FAIL="${COOLDOWN_AFTER_FAIL:-180}"
SKIP_DONE="${SKIP_DONE:-1}"
CLEANUP_AFTER_SUCCESS="${CLEANUP_AFTER_SUCCESS:-1}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/multilang_b1b5_sweep}"
mkdir -p "${LOG_DIR}"

export PATH="/opt/conda/bin:${PATH}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARN: HF_TOKEN is not set - HF push will fail." >&2
fi

locate_timeout_bin   # sets TIMEOUT_BIN

# Manifest — entries are "cfg_path|label". Order is load-bearing: it matches
# the request verbatim and is what re-runs replay against the .done sentinels.
JOBS=(
  # 1. deu B3 24B
  "configs/fineweb_bilingual/core/B3/train_deu_L1-eng_L2.yaml|24B_B3_deu_L1-eng_L2"

  # 2. zho B1..B5 2B
  "configs/fineweb_bilingual_2b/core/B1/train_zho_L1-eng_L2.yaml|2B_B1_zho_L1-eng_L2"
  "configs/fineweb_bilingual_2b/core/B2/train_zho_L1-eng_L2.yaml|2B_B2_zho_L1-eng_L2"
  "configs/fineweb_bilingual_2b/core/B3/train_zho_L1-eng_L2.yaml|2B_B3_zho_L1-eng_L2"
  "configs/fineweb_bilingual_2b/core/B4/train_zho_L1-eng_L2.yaml|2B_B4_zho_L1-eng_L2"
  "configs/fineweb_bilingual_2b/core/B5/train_zho_L1-eng_L2.yaml|2B_B5_zho_L1-eng_L2"

  # 3. spa B1..B3 2B then B1..B3 24B
  "configs/fineweb_bilingual_2b/core/B1/train_spa_L1-eng_L2.yaml|2B_B1_spa_L1-eng_L2"
  "configs/fineweb_bilingual_2b/core/B2/train_spa_L1-eng_L2.yaml|2B_B2_spa_L1-eng_L2"
  "configs/fineweb_bilingual_2b/core/B3/train_spa_L1-eng_L2.yaml|2B_B3_spa_L1-eng_L2"
  "configs/fineweb_bilingual/core/B1/train_spa_L1-eng_L2.yaml|24B_B1_spa_L1-eng_L2"
  "configs/fineweb_bilingual/core/B2/train_spa_L1-eng_L2.yaml|24B_B2_spa_L1-eng_L2"
  "configs/fineweb_bilingual/core/B3/train_spa_L1-eng_L2.yaml|24B_B3_spa_L1-eng_L2"

  # 4. ita B1..B3 2B
  "configs/fineweb_bilingual_2b/core/B1/train_ita_L1-eng_L2.yaml|2B_B1_ita_L1-eng_L2"
  "configs/fineweb_bilingual_2b/core/B2/train_ita_L1-eng_L2.yaml|2B_B2_ita_L1-eng_L2"
  "configs/fineweb_bilingual_2b/core/B3/train_ita_L1-eng_L2.yaml|2B_B3_ita_L1-eng_L2"

  # 5. ita B1..B3 24B
  "configs/fineweb_bilingual/core/B1/train_ita_L1-eng_L2.yaml|24B_B1_ita_L1-eng_L2"
  "configs/fineweb_bilingual/core/B2/train_ita_L1-eng_L2.yaml|24B_B2_ita_L1-eng_L2"
  "configs/fineweb_bilingual/core/B3/train_ita_L1-eng_L2.yaml|24B_B3_ita_L1-eng_L2"

  # 6. rus B1..B3 2B
  "configs/fineweb_bilingual_2b/core/B1/train_rus_L1-eng_L2.yaml|2B_B1_rus_L1-eng_L2"
  "configs/fineweb_bilingual_2b/core/B2/train_rus_L1-eng_L2.yaml|2B_B2_rus_L1-eng_L2"
  "configs/fineweb_bilingual_2b/core/B3/train_rus_L1-eng_L2.yaml|2B_B3_rus_L1-eng_L2"

  # 7. hin B1..B3 2B
  "configs/fineweb_bilingual_2b/core/B1/train_hin_L1-eng_L2.yaml|2B_B1_hin_L1-eng_L2"
  "configs/fineweb_bilingual_2b/core/B2/train_hin_L1-eng_L2.yaml|2B_B2_hin_L1-eng_L2"
  "configs/fineweb_bilingual_2b/core/B3/train_hin_L1-eng_L2.yaml|2B_B3_hin_L1-eng_L2"
)

EXTRA_ARGS=()
if [[ "${RESUME}" == "1" ]]; then
  EXTRA_ARGS+=(--resume_checkpoint)
fi
EXTRA_ARGS+=("$@")

SUCCEEDED=()
FAILED=()
SKIPPED=()

# --- Per-job runner ---------------------------------------------------------
run_job() {
  local cfg="$1"
  local log="$2"
  local label="$3"
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

  # Post-job cleanup: kill orphans, wait for GPU drain
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

# --- Sweep ------------------------------------------------------------------
echo "[$(date -u +%FT%TZ)] multilang_b1b5_sweep start: ${#JOBS[@]} jobs, NGPUS=${NGPUS}"

if ! gpu_preflight "${NGPUS}"; then
  echo "[$(date -u +%FT%TZ)] FATAL: pre-sweep GPU preflight failed, aborting" >&2
  exit 1
fi

if ! disk_preflight "${MIN_FREE_GB:-200}"; then
  echo "[$(date -u +%FT%TZ)] FATAL: insufficient disk space for sweep; aborting." >&2
  exit 1
fi

JOB_INDEX=0
for entry in "${JOBS[@]}"; do
  JOB_INDEX=$((JOB_INDEX + 1))
  CFG="${PROJECT_ROOT}/${entry%%|*}"
  LABEL="${entry##*|}"
  LOG="${LOG_DIR}/$(printf '%02d' "${JOB_INDEX}")_${LABEL}.log"
  echo
  echo "############################################################"
  echo "## job ${JOB_INDEX}/${#JOBS[@]}: ${LABEL}"
  echo "############################################################"
  run_job "${CFG}" "${LOG}" "${LABEL}"
done

# --- Summary ----------------------------------------------------------------
echo "============================================================"
echo "[$(date -u +%FT%TZ)] launch_multilang_b1b5_sweep.sh: DONE"
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
