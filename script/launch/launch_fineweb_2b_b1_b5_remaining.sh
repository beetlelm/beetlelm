#!/usr/bin/env bash
# ============================================================================
# B1-B3 pico sweep for the bilingual languages NOT yet on Beetle-FineWeb-2B.
#
# Targets: configs/fineweb_bilingual_2b/core/{B1..B3}/train_{LANG}_L1-eng_L2.yaml
# Direction: L1=non-English -> L2=English only (matches existing HF naming).
# Default LANGS (13): arb eus fil hin ita jpn pol rus spa tur zho deu nld
# Default TIERS (3):  B1 B2 B3
#
# Already-published cells are screened by the skip-list at
# beetle_pipeline/state/already_published.tsv (sourced via lib/skip_list.sh
# when BEETLE_PIPELINE_ROOT is set; otherwise an inlined fallback array is
# used so the launcher remains standalone). Precedence within run_job:
#   already-on-hf  >  data-not-ready (deferred)  >  missing-config
#   >  sentinel <log>.done
#
# Per-job hardware: 8x{A,H}100, single node, DDP, bf16-mixed.
# Push target:      Beetle-FineWeb-2B (set in each YAML's hf_user).
#
# Crash-resilience contract (mirrors launch_fineweb_2b_groupA_safe.sh):
#   - One job's failure does NOT abort the sweep.
#   - JOB_TIMEOUT (default 12h) bounds each job.
#   - Pre-flight: nvidia-smi reports >=NGPUS, YAML's fabric.devices == NGPUS,
#     YAML's fabric.precision == bf16-mixed. Mismatches abort the *job*, not
#     the sweep, so torchrun never launches with an inconsistent world-size.
#   - After every job, kill_survivors() + drain_gpus() ensure orphaned
#     torchrun workers are reaped and GPU memory is freed before the next job.
#     If GPUs won't drain, the sweep aborts to protect the cluster.
#   - Sentinel <log>.done makes re-runs idempotent.
#   - On success, rm -rf runs/<run_name>/ to reclaim disk between jobs.
#   - Stage 0 idempotently builds Beetle-Data/{2L}-2B-pretok and
#     Beetle-Data/en-for-{2L}-2B-pretok subsets from the 28B sources. If a
#     28B source is missing on HF, Stage 0 logs WARN and downstream jobs for
#     that lang FAIL fast at data-load (clean error, not a launch error).
#
# Env (all optional):
#   LANGS                  default "arb eus fil hin ita jpn pol rus spa tur zho deu nld"
#   TIERS                  default "B1 B2 B3"
#   BEETLE_PIPELINE_ROOT   path to beetle_pipeline/ — when set, skip-list is
#                          sourced from $BEETLE_PIPELINE_ROOT/lib/skip_list.sh
#   READINESS_GATE         1 -> reorder loop into READY-first / NEEDS_STAGE0-last
#                          two-pass scheduling, default 0 (legacy single-pass)
#   RUN_ENG_MONO           1 to append eng monolingual after the loop, default 0
#   STAGE0                 1 -> run Stage 0 pretok-subset staging, default 1
#   SKIP_MISSING           1 to skip configs whose YAML is absent, default 0
#   SKIP_DONE              default 1
#   NGPUS                  default 8
#   RESUME                 1 -> pass --resume_checkpoint to train.py
#   JOB_TIMEOUT            default 12h
#   COOLDOWN_BETWEEN_JOBS  default 60
#   COOLDOWN_AFTER_FAIL    default 120
#   CLEANUP_AFTER_SUCCESS  default 1
#   HF_TOKEN               required for HF push (warned if unset)
#
# Examples:
#   bash script/launch/launch_fineweb_2b_b1_b5_remaining.sh
#   LANGS="ita rus" TIERS="B1" STAGE0=0 \
#       bash script/launch/launch_fineweb_2b_b1_b5_remaining.sh --dry_run
#   LANGS="zho" CLEANUP_AFTER_SUCCESS=0 SKIP_DONE=0 \
#       bash script/launch/launch_fineweb_2b_b1_b5_remaining.sh
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck source=_lib.sh
source "${SCRIPT_DIR}/_lib.sh"

NGPUS="${NGPUS:-8}"
RESUME="${RESUME:-0}"
RUN_ENG_MONO="${RUN_ENG_MONO:-0}"
SKIP_MISSING="${SKIP_MISSING:-0}"
SKIP_DONE="${SKIP_DONE:-1}"
JOB_TIMEOUT="${JOB_TIMEOUT:-12h}"
COOLDOWN_BETWEEN_JOBS="${COOLDOWN_BETWEEN_JOBS:-60}"
COOLDOWN_AFTER_FAIL="${COOLDOWN_AFTER_FAIL:-120}"
CLEANUP_AFTER_SUCCESS="${CLEANUP_AFTER_SUCCESS:-1}"
STAGE0="${STAGE0:-1}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/fineweb_2b_b1_b5_remaining}"
mkdir -p "${LOG_DIR}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARN: HF_TOKEN is not set — HF push will fail." >&2
fi

locate_timeout_bin   # sets TIMEOUT_BIN

# Default LANGS = full 13-lang in-scope set; skip-list (not LANGS list)
# governs HF exclusions. Default TIERS = B1 B2 B3 (B4/B5 out of scope for
# the current experiment set).
read -ra LANGS <<<"${LANGS:-arb eus fil hin ita jpn pol rus spa tur zho deu nld}"
read -ra TIERS <<<"${TIERS:-B1 B2 B3}"

# ─── Skip-list integration ──────────────────────────────────────────────────
# Prefer beetle_pipeline/lib/skip_list.sh as source-of-truth when available.
# Falls back to an inlined 13-row 2B array (B1-B3) so this launcher stays
# self-contained when run outside the pipeline tree.
READINESS_GATE="${READINESS_GATE:-0}"

if [[ -n "${BEETLE_PIPELINE_ROOT:-}" && -f "${BEETLE_PIPELINE_ROOT}/lib/skip_list.sh" ]]; then
  BP_ROOT="${BEETLE_PIPELINE_ROOT}"
  # shellcheck disable=SC1090,SC1091
  source "${BEETLE_PIPELINE_ROOT}/lib/skip_list.sh"
  echo "[skip-list] sourced ${BEETLE_PIPELINE_ROOT}/lib/skip_list.sh (TSV=${SKIP_LIST_TSV})"
else
  # Inlined fallback — must stay in sync with beetle_pipeline/state/already_published.tsv
  # (only 2B B1-B3 cells; B4/B5 inert under default TIERS).
  declare -A FW2B_PUBLISHED=(
    ["B1:nld"]=1 ["B2:nld"]=1 ["B3:nld"]=1
    ["B1:eus"]=1 ["B2:eus"]=1 ["B3:eus"]=1
    ["B1:deu"]=1
    ["B2:hin"]=1 ["B3:hin"]=1
  )
  is_already_published() {
    local _scale="$1" _tier="$2" _lang="$3" _dir="$4"
    [[ "${_scale}" == "2b" && "${_dir}" == "L1-eng" ]] || return 1
    [[ -n "${FW2B_PUBLISHED[${_tier}:${_lang}]:-}" ]]
  }
fi

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
      echo "[stage0] WARN: ISO3_TO_BD has no entry for '${LANG}'; jobs for this lang will likely FAIL at data-load." >&2
      continue
    fi
    python3 "${PROJECT_ROOT}/script/data/stage_pretok_subset.py" \
        --src-repo "Beetle-Data/${TWOL}-28B" \
        --dst-repo "Beetle-Data/${TWOL}-2B-pretok" \
        --target-tokens "${TARGET_TOKENS_2B}" \
        --seq-len "${SEQ_LEN_2B}" \
        --safety "${SAFETY_2B}" \
        --skip-if-exists \
      || echo "[stage0] WARN: subset stage failed for ${LANG} (L1)" >&2
    python3 "${PROJECT_ROOT}/script/data/stage_pretok_subset.py" \
        --src-repo "Beetle-Data/en-for-${TWOL}-28B" \
        --dst-repo "Beetle-Data/en-for-${TWOL}-2B-pretok" \
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
  # label format from main loop is "${TIER}_${LANG}_L1-eng_L2"
  local _tier="${label%%_*}"
  local _rest="${label#*_}"
  local _lang="${_rest%%_*}"

  # Precedence: already-on-hf → missing-config → sentinel → run.
  if is_already_published "2b" "${_tier}" "${_lang}" "L1-eng"; then
    echo "[$(date -u +%FT%TZ)] SKIP (already-on-hf): ${label}"
    SKIPPED+=("already-on-hf:${label}")
    return 0
  fi

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

# ─── Optional two-pass readiness gate ────────────────────────────────────────
# When READINESS_GATE=1, classify each cell as READY (Beetle-Data assets exist
# on HF) or NEEDS_STAGE0 (one or more pretok/tokenizer repos missing), and
# dispatch READY first, NEEDS_STAGE0 last. Stage 0 has already executed at the
# top of the script, so by the time we hit pass 2 most missing shards should
# be in place; the gate primarily reorders so that the cheapest, most-likely-
# to-succeed jobs dispatch first.
hf_repo_exists() {
  # _hf_repo_exists <kind> <repo>; kind = dataset|model. 0=exists.
  local kind="$1" repo="$2" url code
  case "${kind}" in
    dataset) url="https://huggingface.co/api/datasets/${repo}" ;;
    model)   url="https://huggingface.co/api/models/${repo}" ;;
    *) return 1 ;;
  esac
  code="$(curl -fsS -o /dev/null -w '%{http_code}' --max-time 10 -L "${url}" 2>/dev/null || echo 000)"
  [[ "${code}" == "200" ]]
}

cell_is_ready() {
  local lang="$1" twol
  twol="$(python3 -c "from beetlelm.config._fineweb_defaults import ISO3_TO_BD; print(ISO3_TO_BD.get('${lang}',''))" 2>/dev/null || true)"
  [[ -z "${twol}" ]] && return 1
  hf_repo_exists dataset "Beetle-Data/${twol}-2B-pretok"        || return 1
  hf_repo_exists dataset "Beetle-Data/en-for-${twol}-2B-pretok" || return 1
  hf_repo_exists model   "Beetle-Data/tokenizer-${twol}-en"     || return 1
  return 0
}

if [[ "${READINESS_GATE}" == "1" ]]; then
  echo "[$(date -u +%FT%TZ)] READINESS_GATE=1 → classifying cells into READY / NEEDS_STAGE0"
  READY_CELLS=()
  DEFERRED_CELLS=()
  for LANG in "${LANGS[@]}"; do
    if cell_is_ready "${LANG}"; then
      for TIER in "${TIERS[@]}"; do READY_CELLS+=( "${LANG}:${TIER}" ); done
    else
      for TIER in "${TIERS[@]}"; do DEFERRED_CELLS+=( "${LANG}:${TIER}" ); done
    fi
  done
  echo "[$(date -u +%FT%TZ)] classified: READY=${#READY_CELLS[@]} DEFERRED=${#DEFERRED_CELLS[@]}"

  for cell in "${READY_CELLS[@]}" "${DEFERRED_CELLS[@]}"; do
    LANG="${cell%%:*}"; TIER="${cell##*:}"
    CFG="${PROJECT_ROOT}/configs/fineweb_bilingual_2b/core/${TIER}/train_${LANG}_L1-eng_L2.yaml"
    LOG="${LOG_DIR}/${TIER}_${LANG}_L1-eng_L2.log"
    LABEL="${TIER}_${LANG}_L1-eng_L2"
    run_job "${CFG}" "${LOG}" "${LABEL}"
  done
else
  for LANG in "${LANGS[@]}"; do
    for TIER in "${TIERS[@]}"; do
      CFG="${PROJECT_ROOT}/configs/fineweb_bilingual_2b/core/${TIER}/train_${LANG}_L1-eng_L2.yaml"
      LOG="${LOG_DIR}/${TIER}_${LANG}_L1-eng_L2.log"
      LABEL="${TIER}_${LANG}_L1-eng_L2"
      run_job "${CFG}" "${LOG}" "${LABEL}"
    done
  done
fi

if [[ "${RUN_ENG_MONO}" == "1" ]]; then
  ENG_CFG="${PROJECT_ROOT}/configs/fineweb_bilingual_2b/monolingual/train_eng.yaml"
  ENG_LOG="${LOG_DIR}/monolingual_eng.log"
  run_job "${ENG_CFG}" "${ENG_LOG}" "monolingual_eng"
fi

# ─── Summary ─────────────────────────────────────────────────────────────────
echo "============================================================"
echo "[$(date -u +%FT%TZ)] launch_fineweb_2b_b1_b5_remaining.sh: DONE"
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
