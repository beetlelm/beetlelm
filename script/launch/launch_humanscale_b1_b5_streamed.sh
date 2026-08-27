#!/usr/bin/env bash
# ============================================================================
# launch_humanscale_b1_b5_streamed.sh
#
# Variant of launch_humanscale_b1_b5_all_safe.sh restricted to the languages
# whose 100M-pretok subset shards have ALREADY been streamed/uploaded to
# HuggingFace as of the prior staging run. Specifically, staging completed
# through `eus` (Beetle-Data/eu-100M-pretok + Beetle-Data/en-for-eu-100M-pretok)
# and failed at `arb` because Beetle-Data/ar-28B is not on the Hub.
#
# Streamed langs (in order from the previous Stage 0 run):
#   nld deu ita rus pol tur fil hin eus
#
# Not yet streamed (skipped here):
#   arb (Beetle-Data/ar-28B missing on HF)
#   zho spa jpn (already excluded by the parent _safe launcher)
#
# Stage 0 is kept but is a no-op for these langs because
# stage_pretok_subset.py is invoked with --skip-if-exists. Set
# SKIP_STAGE0=1 to bypass it entirely and go straight to training.
#
# Crash-resilience: mirrors launch_humanscale_b1_b5_all_safe.sh — fully
# hardened with timeout, sentinel, kill_survivors, drain_gpus, crash-isolation,
# and safe Stage 0 error handling.
#
# Env (all optional):
#   LANGS         override default streamed list (space-separated ISO3)
#   TIERS         space-separated, default "B1 B2 B3 B4 B5"
#   RUN_ENG_MONO  1 to append eng monolingual after the loop, default 1
#   SKIP_MISSING  1 to skip configs whose YAML is absent, default 0
#   SKIP_DONE     1 to skip jobs whose sentinel exists, default 1
#   SKIP_STAGE0   1 to skip the Stage 0 idempotency check, default 0
#   NGPUS         default 8
#   RESUME        1 → pass --resume_checkpoint to train.py
#   JOB_TIMEOUT   per-job wall-clock cap, default 12h
#   COOLDOWN_BETWEEN_JOBS  default 60
#   COOLDOWN_AFTER_FAIL    default 120
#   CLEANUP_AFTER_SUCCESS  1 → rm -rf runs/<run_name> on success, default 1
#   HF_TOKEN      required for HF push (warned if unset)
#
# Examples:
#   bash script/launch/launch_humanscale_b1_b5_streamed.sh
#   SKIP_STAGE0=1 TIERS="B1" RUN_ENG_MONO=0 \
#       bash script/launch/launch_humanscale_b1_b5_streamed.sh
#   LANGS="nld deu ita" \
#       bash script/launch/launch_humanscale_b1_b5_streamed.sh --dry_run
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck source=_lib.sh
source "${SCRIPT_DIR}/_lib.sh"

NGPUS="${NGPUS:-8}"
RESUME="${RESUME:-0}"
RUN_ENG_MONO="${RUN_ENG_MONO:-1}"
SKIP_MISSING="${SKIP_MISSING:-0}"
SKIP_DONE="${SKIP_DONE:-1}"
SKIP_STAGE0="${SKIP_STAGE0:-0}"
JOB_TIMEOUT="${JOB_TIMEOUT:-12h}"
COOLDOWN_BETWEEN_JOBS="${COOLDOWN_BETWEEN_JOBS:-60}"
COOLDOWN_AFTER_FAIL="${COOLDOWN_AFTER_FAIL:-120}"
CLEANUP_AFTER_SUCCESS="${CLEANUP_AFTER_SUCCESS:-1}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/humanscale_streamed}"
mkdir -p "${LOG_DIR}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARN: HF_TOKEN is not set — HF push will fail." >&2
fi

locate_timeout_bin   # sets TIMEOUT_BIN

# Default to the langs whose 100M-pretok shards are already on HF.
read -ra LANGS <<<"${LANGS:-nld deu ita rus pol tur fil hin eus}"
read -ra TIERS <<<"${TIERS:-B1 B2 B3 B4 B5}"

EXTRA_ARGS=()
if [[ "${RESUME}" == "1" ]]; then
  EXTRA_ARGS+=(--resume_checkpoint)
fi
EXTRA_ARGS+=("$@")

SUCCEEDED=()
FAILED=()
SKIPPED=()

# ─── Stage 0: idempotency check on already-streamed shards ────────────────────
# || true guards: a missing src repo warns but does NOT abort the script.
TARGET_TOKENS_HUMAN=100000000
SEQ_LEN_HUMAN=512
SAFETY_HUMAN=2.0
if [[ "${SKIP_STAGE0}" == "1" ]]; then
  echo "[$(date -u +%FT%TZ)] SKIP_STAGE0=1 → skipping Stage 0 idempotency check"
else
  echo "============================================================"
  echo "[$(date -u +%FT%TZ)] Stage 0: verify 100M-pretok subset shards on HF"
  echo "============================================================"
  for LANG in "${LANGS[@]}"; do
    TWOL="$(python3 -c "from beetlelm.config._fineweb_defaults import ISO3_TO_BD; print(ISO3_TO_BD['${LANG}'])" 2>/dev/null || true)"
    if [[ -z "${TWOL}" ]]; then
      echo "[stage0] WARN: ISO3_TO_BD has no entry for '${LANG}'; jobs for this lang will FAIL at data-load." >&2
      continue
    fi
    python3 "${PROJECT_ROOT}/script/data/stage_pretok_subset.py" \
        --src-repo  "Beetle-Data/${TWOL}-28B" \
        --dst-repo  "Beetle-Data/${TWOL}-100M-pretok" \
        --target-tokens "${TARGET_TOKENS_HUMAN}" \
        --seq-len "${SEQ_LEN_HUMAN}" \
        --safety "${SAFETY_HUMAN}" \
        --skip-if-exists \
      || echo "[stage0] WARN: subset stage failed for ${LANG} (L1)" >&2
    python3 "${PROJECT_ROOT}/script/data/stage_pretok_subset.py" \
        --src-repo  "Beetle-Data/en-for-${TWOL}-28B" \
        --dst-repo  "Beetle-Data/en-for-${TWOL}-100M-pretok" \
        --target-tokens "${TARGET_TOKENS_HUMAN}" \
        --seq-len "${SEQ_LEN_HUMAN}" \
        --safety "${SAFETY_HUMAN}" \
        --skip-if-exists \
      || echo "[stage0] WARN: subset stage failed for ${LANG} (en-for-)" >&2
  done
fi

# ─── Per-job runner ──────────────────────────────────────────────────────────
run_job() {
  local cfg="$1"
  local log="$2"
  local label="$3"
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
echo "[$(date -u +%FT%TZ)] sweep start: LANGS=(${LANGS[*]}) TIERS=(${TIERS[*]}) NGPUS=${NGPUS}"

if ! gpu_preflight "${NGPUS}"; then
  echo "[$(date -u +%FT%TZ)] WARN: pre-sweep GPU preflight failed, continuing anyway"
fi

if ! disk_preflight "${MIN_FREE_GB:-200}"; then
  echo "[$(date -u +%FT%TZ)] FATAL: insufficient disk space for sweep; aborting." >&2
  exit 1
fi

for LANG in "${LANGS[@]}"; do
  for TIER in "${TIERS[@]}"; do
    CFG="${PROJECT_ROOT}/configs/humanscale_fineweb_bilingual/core/${TIER}/train_${LANG}_L1-eng_L2.yaml"
    LOG="${LOG_DIR}/${TIER}_${LANG}_L1-eng_L2.log"
    LABEL="${TIER}_${LANG}_L1-eng_L2"
    run_job "${CFG}" "${LOG}" "${LABEL}"
  done
done

if [[ "${RUN_ENG_MONO}" == "1" ]]; then
  ENG_CFG="${PROJECT_ROOT}/configs/humanscale_fineweb_bilingual/monolingual/train_eng.yaml"
  ENG_LOG="${LOG_DIR}/monolingual_eng.log"
  run_job "${ENG_CFG}" "${ENG_LOG}" "monolingual_eng"
fi

# ─── Summary ─────────────────────────────────────────────────────────────────
echo "============================================================"
echo "[$(date -u +%FT%TZ)] launch_humanscale_b1_b5_streamed.sh: DONE"
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
