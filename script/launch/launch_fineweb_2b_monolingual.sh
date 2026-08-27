#!/usr/bin/env bash
# ============================================================================
# Standalone launcher: 2B monolingual sweep — counterparts to the bilingual
# B1-B3 sweep. Iterates configs/fineweb_bilingual_2b/monolingual/train_${lang}.yaml.
#
# Default LANGS (14): arb eus fil hin ita jpn pol rus spa tur zho deu nld eng
#
# Crash-resilience contract mirrors launch_fineweb_2b_b1_b5_remaining.sh:
#   sentinel <log>.done idempotency, kill_survivors+drain_gpus between jobs,
#   stage-0 staging gate, optional READINESS_GATE.
#
# Env (all optional):
#   LANGS                  default the 14-lang set above
#   STAGE0                 1 -> stage Beetle-Data/{2L}-2B-pretok subsets, default 1
#   READINESS_GATE         1 -> two-pass scheduling, default 0
#   BEETLE_PIPELINE_ROOT   path to beetle_pipeline/ — when set, skip-list is
#                          sourced from $BEETLE_PIPELINE_ROOT/lib/skip_list.sh
#   SKIP_DONE              default 1
#   SKIP_MISSING           default 0
#   NGPUS                  default 8
#   JOB_TIMEOUT            default 12h
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
SKIP_MISSING="${SKIP_MISSING:-0}"
SKIP_DONE="${SKIP_DONE:-1}"
JOB_TIMEOUT="${JOB_TIMEOUT:-12h}"
COOLDOWN_BETWEEN_JOBS="${COOLDOWN_BETWEEN_JOBS:-60}"
COOLDOWN_AFTER_FAIL="${COOLDOWN_AFTER_FAIL:-120}"
CLEANUP_AFTER_SUCCESS="${CLEANUP_AFTER_SUCCESS:-1}"
STAGE0="${STAGE0:-1}"
READINESS_GATE="${READINESS_GATE:-0}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/fineweb_2b_monolingual}"
mkdir -p "${LOG_DIR}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARN: HF_TOKEN is not set — HF push will fail." >&2
fi

locate_timeout_bin

read -ra LANGS <<<"${LANGS:-arb eus fil hin ita jpn pol rus spa tur zho deu nld eng}"

EXTRA_ARGS=()
if [[ "${RESUME}" == "1" ]]; then
  EXTRA_ARGS+=(--resume_checkpoint)
fi
EXTRA_ARGS+=("$@")

SUCCEEDED=()
FAILED=()
SKIPPED=()

# ─── Stage 0: pretok subset staging ─────────────────────────────────────────
if [[ "${STAGE0}" == "1" ]]; then
  TARGET_TOKENS_2B=2000000000
  SEQ_LEN_2B=512
  SAFETY_2B=2.0
  echo "============================================================"
  echo "[$(date -u +%FT%TZ)] Stage 0: ensure 2B-pretok shards on HF"
  echo "============================================================"
  for LANG in "${LANGS[@]}"; do
    TWOL="$(python3 -c "from beetlelm.config._fineweb_defaults import ISO3_TO_BD; print(ISO3_TO_BD.get('${LANG}',''))" 2>/dev/null || true)"
    if [[ -z "${TWOL}" ]]; then
      echo "[stage0] WARN: no Beetle-Data 2-letter code for ${LANG}; skipping" >&2
      continue
    fi
    python3 "${PROJECT_ROOT}/script/data/stage_pretok_subset.py" \
        --src-repo "Beetle-Data/${TWOL}-28B" \
        --dst-repo "Beetle-Data/${TWOL}-2B-pretok" \
        --target-tokens "${TARGET_TOKENS_2B}" \
        --seq-len "${SEQ_LEN_2B}" \
        --safety "${SAFETY_2B}" \
        --skip-if-exists \
      || echo "[stage0] WARN: subset stage failed for ${LANG}" >&2
  done
fi

# ─── Per-job runner (mirrors bilingual launcher) ─────────────────────────────
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

  kill_survivors
  if ! drain_gpus 180; then
    kill_survivors
    if ! drain_gpus 60; then
      echo "[$(date -u +%FT%TZ)] FATAL: GPUs did not drain after two cleanup attempts." >&2
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
    fi
  fi

  rotate_log "${log}"
  sleep "${COOLDOWN_BETWEEN_JOBS}"
}

# ─── Sweep ───────────────────────────────────────────────────────────────────
echo "[$(date -u +%FT%TZ)] mono-sweep start: LANGS=(${LANGS[*]}) NGPUS=${NGPUS}"

if ! gpu_preflight "${NGPUS}"; then
  echo "[$(date -u +%FT%TZ)] WARN: pre-sweep GPU preflight failed, continuing anyway"
fi

if ! disk_preflight "${MIN_FREE_GB:-200}"; then
  echo "[$(date -u +%FT%TZ)] FATAL: insufficient disk space; aborting." >&2
  exit 1
fi

# Optional readiness gate (mono only needs {2L}-2B-pretok + tokenizer-{2L}-en).
hf_repo_exists() {
  local kind="$1" repo="$2" url code
  case "${kind}" in
    dataset) url="https://huggingface.co/api/datasets/${repo}" ;;
    model)   url="https://huggingface.co/api/models/${repo}" ;;
    *) return 1 ;;
  esac
  code="$(curl -fsS -o /dev/null -w '%{http_code}' --max-time 10 -L "${url}" 2>/dev/null || echo 000)"
  [[ "${code}" == "200" ]]
}
mono_cell_ready() {
  local lang="$1" twol
  twol="$(python3 -c "from beetlelm.config._fineweb_defaults import ISO3_TO_BD; print(ISO3_TO_BD.get('${lang}',''))" 2>/dev/null || true)"
  [[ -z "${twol}" ]] && return 1
  hf_repo_exists dataset "Beetle-Data/${twol}-2B-pretok"    || return 1
  hf_repo_exists model   "Beetle-Data/tokenizer-${twol}-en" || return 1
  return 0
}

if [[ "${READINESS_GATE}" == "1" ]]; then
  READY=()
  DEFERRED=()
  for LANG in "${LANGS[@]}"; do
    if mono_cell_ready "${LANG}"; then READY+=("${LANG}"); else DEFERRED+=("${LANG}"); fi
  done
  echo "[$(date -u +%FT%TZ)] classified: READY=${#READY[@]} DEFERRED=${#DEFERRED[@]}"
  ORDERED_LANGS=( "${READY[@]}" "${DEFERRED[@]}" )
else
  ORDERED_LANGS=( "${LANGS[@]}" )
fi

for LANG in "${ORDERED_LANGS[@]}"; do
  CFG="${PROJECT_ROOT}/configs/fineweb_bilingual_2b/monolingual/train_${LANG}.yaml"
  LOG="${LOG_DIR}/mono_${LANG}.log"
  LABEL="mono_${LANG}"
  run_job "${CFG}" "${LOG}" "${LABEL}"
done

# ─── Summary ─────────────────────────────────────────────────────────────────
echo "============================================================"
echo "[$(date -u +%FT%TZ)] launch_fineweb_2b_monolingual.sh: DONE"
echo "  succeeded: ${#SUCCEEDED[@]}"
echo "  skipped  : ${#SKIPPED[@]}"
echo "  failed   : ${#FAILED[@]}"
if [[ "${#FAILED[@]}" -gt 0 ]]; then
  echo "  failures :"
  for f in "${FAILED[@]}"; do echo "    - ${f}"; done
fi
echo "============================================================"

if [[ "${#FAILED[@]}" -gt 0 ]]; then exit 1; fi
exit 0
