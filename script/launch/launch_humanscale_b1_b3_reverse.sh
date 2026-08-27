#!/usr/bin/env bash
# ============================================================================
# Standalone launcher: HumanScale 100M reverse-direction (L1=eng → L2=non-eng) sweep.
# Iterates configs/humanscale_fineweb_bilingual/core/${TIER}/train_eng_L1-${LANG}_L2.yaml
# over 13 langs × B1-B3.
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck source=_lib.sh
source "${SCRIPT_DIR}/_lib.sh"

NGPUS="${NGPUS:-8}"
RESUME="${RESUME:-0}"
SKIP_MISSING="${SKIP_MISSING:-0}"
SKIP_DONE="${SKIP_DONE:-1}"
JOB_TIMEOUT="${JOB_TIMEOUT:-3h}"
COOLDOWN_BETWEEN_JOBS="${COOLDOWN_BETWEEN_JOBS:-60}"
COOLDOWN_AFTER_FAIL="${COOLDOWN_AFTER_FAIL:-120}"
CLEANUP_AFTER_SUCCESS="${CLEANUP_AFTER_SUCCESS:-1}"
STAGE0="${STAGE0:-1}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/humanscale_reverse}"
mkdir -p "${LOG_DIR}"

[[ -z "${HF_TOKEN:-}" ]] && echo "WARN: HF_TOKEN not set." >&2
locate_timeout_bin

read -ra LANGS <<<"${LANGS:-arb eus fil hin ita jpn pol rus spa tur zho deu nld}"
read -ra TIERS <<<"${TIERS:-B1 B2 B3}"

EXTRA_ARGS=()
[[ "${RESUME}" == "1" ]] && EXTRA_ARGS+=(--resume_checkpoint)
EXTRA_ARGS+=("$@")

SUCCEEDED=()
FAILED=()
SKIPPED=()

if [[ "${STAGE0}" == "1" ]]; then
  TARGET_TOKENS_100M=100000000; SEQ_LEN_100M=512; SAFETY_100M=2.0
  echo "[$(date -u +%FT%TZ)] Stage 0: ensure 100M-pretok shards"
  for LANG in "${LANGS[@]}"; do
    TWOL="$(python3 -c "from beetlelm.config._fineweb_defaults import ISO3_TO_BD; print(ISO3_TO_BD.get('${LANG}',''))" 2>/dev/null || true)"
    [[ -z "${TWOL}" ]] && { echo "[stage0] WARN no 2-letter code for ${LANG}" >&2; continue; }
    python3 "${PROJECT_ROOT}/script/data/stage_pretok_subset.py" \
        --src-repo "Beetle-Data/${TWOL}-28B" --dst-repo "Beetle-Data/${TWOL}-100M-pretok" \
        --target-tokens "${TARGET_TOKENS_100M}" --seq-len "${SEQ_LEN_100M}" --safety "${SAFETY_100M}" \
        --skip-if-exists || echo "[stage0] WARN failed for ${LANG} (L1)" >&2
    python3 "${PROJECT_ROOT}/script/data/stage_pretok_subset.py" \
        --src-repo "Beetle-Data/en-for-${TWOL}-28B" --dst-repo "Beetle-Data/en-for-${TWOL}-100M-pretok" \
        --target-tokens "${TARGET_TOKENS_100M}" --seq-len "${SEQ_LEN_100M}" --safety "${SAFETY_100M}" \
        --skip-if-exists || echo "[stage0] WARN failed for ${LANG} (en-for-)" >&2
  done
fi

run_job() {
  local cfg="$1" log="$2" label="$3"
  local sentinel="${log%.log}.done"
  if [[ ! -f "${cfg}" ]]; then
    if [[ "${SKIP_MISSING}" == "1" ]]; then
      echo "[$(date -u +%FT%TZ)] SKIP (missing config): ${cfg}"
      SKIPPED+=("missing-config:${label}"); return 0
    fi
    echo "[$(date -u +%FT%TZ)] FAIL (missing config): ${cfg}" >&2
    FAILED+=("missing-config:${label}"); return 0
  fi
  if [[ "${SKIP_DONE}" == "1" && -f "${sentinel}" ]]; then
    echo "[$(date -u +%FT%TZ)] SKIP (sentinel): ${label}"
    SKIPPED+=("already-done:${label}"); return 0
  fi
  if ! gpu_preflight "${NGPUS}"; then
    echo "[$(date -u +%FT%TZ)] FAIL (gpu unhealthy): ${label}" | tee -a "${log}"
    FAILED+=("gpu-unhealthy:${label}"); sleep "${COOLDOWN_AFTER_FAIL}"; return 0
  fi
  if ! config_gpu_compliance "${cfg}" "${NGPUS}" 2>>"${log}"; then
    echo "[$(date -u +%FT%TZ)] FAIL (gpu-config-mismatch): ${label}" | tee -a "${log}" >&2
    FAILED+=("gpu-config-mismatch:${label}"); return 0
  fi

  echo "[$(date -u +%FT%TZ)] launching ${label}: ${cfg} (timeout=${JOB_TIMEOUT})"
  local rc_file; rc_file="$(mktemp)"; printf '255' > "${rc_file}"
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
  local pipe_pid=$!; wait "${pipe_pid}" || true
  local rc; rc="$(cat "${rc_file}")"; rm -f "${rc_file}"

  kill_survivors
  if ! drain_gpus 180; then
    kill_survivors
    drain_gpus 60 || { echo "FATAL: GPUs did not drain" >&2; exit 1; }
  fi

  if [[ "${rc}" -ne 0 ]]; then
    case "${rc}" in 124|137) reason="timeout(rc=${rc})" ;; *) reason="rc=${rc}" ;; esac
    echo "[$(date -u +%FT%TZ)] FAIL (${reason}): ${label}" | tee -a "${log}"
    FAILED+=("${reason}:${label}"); sleep "${COOLDOWN_AFTER_FAIL}"; return 0
  fi
  : > "${sentinel}"
  SUCCEEDED+=("${label}")

  if [[ "${CLEANUP_AFTER_SUCCESS}" == "1" ]]; then
    local run_name=""
    if run_name="$(python3 - "${cfg}" <<'PY' 2>/dev/null
import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["run_name"])
PY
)"; then
      run_name="${run_name//$'\n'/}"
      cleanup_runs_if_pushed "${run_name}" "${log}"
    fi
  fi
  rotate_log "${log}"; sleep "${COOLDOWN_BETWEEN_JOBS}"
}

echo "[$(date -u +%FT%TZ)] 100M-reverse sweep: LANGS=(${LANGS[*]}) TIERS=(${TIERS[*]})"
gpu_preflight "${NGPUS}" || echo "[$(date -u +%FT%TZ)] WARN: pre-sweep gpu preflight failed"
disk_preflight "${MIN_FREE_GB:-200}" || { echo "FATAL: disk" >&2; exit 1; }

for LANG in "${LANGS[@]}"; do
  for TIER in "${TIERS[@]}"; do
    CFG="${PROJECT_ROOT}/configs/humanscale_fineweb_bilingual/core/${TIER}/train_eng_L1-${LANG}_L2.yaml"
    LOG="${LOG_DIR}/${TIER}_eng_L1-${LANG}_L2.log"
    LABEL="${TIER}_eng_L1-${LANG}_L2"
    run_job "${CFG}" "${LOG}" "${LABEL}"
  done
done

echo "[$(date -u +%FT%TZ)] launch_humanscale_b1_b3_reverse.sh: DONE"
echo "  succeeded=${#SUCCEEDED[@]}  skipped=${#SKIPPED[@]}  failed=${#FAILED[@]}"
[[ "${#FAILED[@]}" -gt 0 ]] && { for f in "${FAILED[@]}"; do echo "  - ${f}"; done; exit 1; }
exit 0
