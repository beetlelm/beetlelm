#!/usr/bin/env bash
# ============================================================================
# Sister of launch_fineweb_2b_groupA.sh — same body, parametrizable, hardened.
#
# Use when some Beetle-Data/{lang}-28B datasets aren't on HF yet, or when
# you want to drive subsets of the groupA sweep without editing the
# original launcher.
#
# Pushes to HuggingFace org: Beetle-FineWeb-2B
# Configs:                   configs/fineweb_bilingual_2b/
# Per-job hardware:          8x{A,H}100, single node, DDP
#
# Crash-resilience contract:
#   - One job's failure (GPU crash, OOM, NCCL hang, timeout) does NOT abort
#     the rest of the sweep; the launcher records the failure and continues.
#   - A timeout (env JOB_TIMEOUT, default 12h) bounds each job so a hung
#     all-reduce can't sit forever.
#   - After every job (success or failure), kill_survivors() terminates any
#     stray torchrun/train.py processes, then drain_gpus() waits for all
#     GPUs to return to < 500 MiB used before starting the next job.
#     If GPUs refuse to drain, the sweep aborts to protect the cluster.
#   - Between jobs, the launcher sleeps COOLDOWN_BETWEEN_JOBS seconds
#     (default 60). On failure an additional COOLDOWN_AFTER_FAIL seconds
#     (default 120) is added.
#   - Per-job sentinel files (<log>.done) make the launcher idempotent:
#     re-running skips already-completed jobs unless SKIP_DONE=0.
#
# Disk hygiene:
#   - On successful job exit, runs/<run_name>/ is removed
#     (CLEANUP_AFTER_SUCCESS, default 1). The trainer's final HF push runs
#     before train.py returns 0, so deletion is safe.
#   - pretok_cache/ is shared across tiers and never deleted by this script.
#
# Env (all optional):
#   LANGS                  space-separated, default "nld deu ita rus pol tur"
#   TIERS                  space-separated, default "B1 B2 B3 B4 B5"
#   RUN_ENG_MONO           1 to append eng monolingual after the loop, default 1
#   SKIP_MISSING           1 to skip configs whose YAML is absent, default 0
#   SKIP_DONE              1 to skip jobs whose <log>.done sentinel exists, default 1
#   NGPUS                  default 8
#   RESUME                 1 → pass --resume_checkpoint to train.py
#   JOB_TIMEOUT            per-job wall-clock cap, default 12h
#   COOLDOWN_BETWEEN_JOBS  seconds between successful jobs, default 60
#   COOLDOWN_AFTER_FAIL    seconds after a failed job before the next, default 120
#   CLEANUP_AFTER_SUCCESS  1 → rm -rf runs/<run_name> on success, default 1
#   STAGE0                 1 → run Stage 0 pretok-subset staging, default 1
#   HF_TOKEN               required for HF push (warned if unset)
#
# Examples:
#   bash script/launch/launch_fineweb_2b_groupA_safe.sh
#   LANGS="zho spa jpn" RUN_ENG_MONO=0 \
#       bash script/launch/launch_fineweb_2b_groupA_safe.sh
#   TIERS="B1" RUN_ENG_MONO=0 \
#       bash script/launch/launch_fineweb_2b_groupA_safe.sh --dry_run
#   CLEANUP_AFTER_SUCCESS=0 SKIP_DONE=0 \
#       bash script/launch/launch_fineweb_2b_groupA_safe.sh
# ============================================================================
# NOTE: errexit is intentionally NOT set so that a single job crash does not
# abort the sweep. pipefail+nounset are kept for safety.
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
JOB_TIMEOUT="${JOB_TIMEOUT:-12h}"
COOLDOWN_BETWEEN_JOBS="${COOLDOWN_BETWEEN_JOBS:-60}"
COOLDOWN_AFTER_FAIL="${COOLDOWN_AFTER_FAIL:-120}"
CLEANUP_AFTER_SUCCESS="${CLEANUP_AFTER_SUCCESS:-1}"
STAGE0="${STAGE0:-1}"
# LOG_DIR may be overridden by callers (e.g. the groupB wrapper) so the same
# hardened body can drive different lang sets without colliding on disk.
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/fineweb_2b_groupA_safe}"
mkdir -p "${LOG_DIR}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARN: HF_TOKEN is not set — HF push will fail." >&2
fi

locate_timeout_bin   # sets TIMEOUT_BIN

read -ra LANGS <<<"${LANGS:-nld deu ita rus pol tur}"
read -ra TIERS <<<"${TIERS:-B1 B2 B3 B4 B5}"

EXTRA_ARGS=()
if [[ "${RESUME}" == "1" ]]; then
  EXTRA_ARGS+=(--resume_checkpoint)
fi
EXTRA_ARGS+=("$@")

SUCCEEDED=()
FAILED=()
SKIPPED=()

# ─── Stage 0: ensure 2B-pretok subset shards exist on HF ─────────────────────
if [[ "${STAGE0}" == "1" ]]; then
  TARGET_TOKENS_2B=2000000000
  SEQ_LEN_2B=512
  SAFETY_2B=2.0
  echo "============================================================"
  echo "[$(date -u +%FT%TZ)] Stage 0: ensure 2B-pretok subset shards on HF"
  echo "============================================================"
  for LANG in "${LANGS[@]}"; do
    TWOL="$(python3 -c "from beetlelm.config._fineweb_defaults import ISO3_TO_BD; print(ISO3_TO_BD['${LANG}'])" 2>/dev/null || true)"
    if [[ -z "${TWOL}" ]]; then
      echo "[stage0] WARN: ISO3_TO_BD has no entry for '${LANG}'; jobs for this lang will FAIL at data-load." >&2
      continue
    fi
    python3 "${PROJECT_ROOT}/script/data/stage_pretok_subset.py" \
        --src-repo  "Beetle-Data/${TWOL}-28B" \
        --dst-repo  "Beetle-Data/${TWOL}-2B-pretok" \
        --target-tokens "${TARGET_TOKENS_2B}" \
        --seq-len "${SEQ_LEN_2B}" \
        --safety "${SAFETY_2B}" \
        --skip-if-exists \
      || echo "[stage0] WARN: subset stage failed for ${LANG} (L1)" >&2
    python3 "${PROJECT_ROOT}/script/data/stage_pretok_subset.py" \
        --src-repo  "Beetle-Data/en-for-${TWOL}-28B" \
        --dst-repo  "Beetle-Data/en-for-${TWOL}-2B-pretok" \
        --target-tokens "${TARGET_TOKENS_2B}" \
        --seq-len "${SEQ_LEN_2B}" \
        --safety "${SAFETY_2B}" \
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
    echo "[$(date -u +%FT%TZ)] SKIP (sentinel ${sentinel##*/} present): ${label}"
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
  # PIPESTATUS[0] through a pipe to tee can return tee's code instead of
  # torchrun's when the process is killed by signal. Writing the code to a
  # file inside the subshell is signal-safe.
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
    # A second kill attempt before giving up
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
      # cleanup_runs_if_pushed is sentinel-aware: it only removes
      # runs/<name>/ when train.py wrote .hf_upload_done. If the HF push
      # failed (or timed out), the directory is preserved so a subsequent
      # RESUME=1 job can pick up the local checkpoint.
      cleanup_runs_if_pushed "${run_name}" "${log}"
    else
      echo "[$(date -u +%FT%TZ)] WARN: could not parse run_name from ${cfg}; skipping cleanup" | tee -a "${log}"
    fi
  fi

  rotate_log "${log}"
  sleep "${COOLDOWN_BETWEEN_JOBS}"
}

# ─── Sweep ───────────────────────────────────────────────────────────────────
echo "[$(date -u +%FT%TZ)] sweep start: LANGS=(${LANGS[*]}) TIERS=(${TIERS[*]})"

if ! gpu_preflight "${NGPUS}"; then
  echo "[$(date -u +%FT%TZ)] WARN: pre-sweep GPU preflight failed, continuing anyway"
fi

# Disk preflight: a 60-job sweep needs ~200 GiB headroom for HF cache
# downloads + per-step checkpoints. Abort early rather than blow up
# mid-sweep with "no space left on device" while uploads are in flight.
if ! disk_preflight "${MIN_FREE_GB:-200}"; then
  echo "[$(date -u +%FT%TZ)] FATAL: disk preflight failed — aborting sweep." >&2
  exit 1
fi

for LANG in "${LANGS[@]}"; do
  for TIER in "${TIERS[@]}"; do
    CFG="${PROJECT_ROOT}/configs/fineweb_bilingual_2b/core/${TIER}/train_${LANG}_L1-eng_L2.yaml"
    LOG="${LOG_DIR}/${TIER}_${LANG}_L1-eng_L2.log"
    LABEL="${TIER}_${LANG}_L1-eng_L2"
    run_job "${CFG}" "${LOG}" "${LABEL}"
  done
done

if [[ "${RUN_ENG_MONO}" == "1" ]]; then
  ENG_CFG="${PROJECT_ROOT}/configs/fineweb_bilingual_2b/monolingual/train_eng.yaml"
  ENG_LOG="${LOG_DIR}/monolingual_eng.log"
  run_job "${ENG_CFG}" "${ENG_LOG}" "monolingual_eng"
fi

# ─── Summary ─────────────────────────────────────────────────────────────────
echo "============================================================"
echo "[$(date -u +%FT%TZ)] launch_fineweb_2b_groupA_safe.sh: DONE"
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
