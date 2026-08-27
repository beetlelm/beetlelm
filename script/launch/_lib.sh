#!/usr/bin/env bash
# =============================================================================
# script/launch/_lib.sh
#
# Shared helper library sourced by all Beetle launch scripts.
# Provides crash-resilient GPU cluster management utilities:
#
#   kill_survivors           — SIGTERM then SIGKILL any stray torchrun/train.py
#                              processes that outlived the previous job
#   drain_gpus [max_secs]    — Poll nvidia-smi until all GPUs report < 500 MiB
#                              used (or abort after max_secs, default 180)
#   gpu_preflight [ngpus]    — Assert nvidia-smi sees >= ngpus GPUs
#   config_gpu_compliance    — Assert YAML fabric.devices==ngpus &&
#                              fabric.precision==bf16-mixed (+ schema checks)
#   disk_preflight [min_gb]  — Abort if free space on PROJECT_ROOT mount is
#                              below min_gb (default 200)
#   cleanup_runs_if_pushed   — rm -rf runs/<run_name>/ ONLY when the trainer
#                              wrote .hf_upload_done; otherwise preserve for
#                              safe RESUME
#   rotate_log               — gzip a per-job log file once it crosses 50 MB
#   locate_timeout_bin       — Populate TIMEOUT_BIN (timeout / gtimeout / "")
#
# Sourcing this file ALSO exports a small set of process-global env vars that
# every launcher relies on for cluster reliability:
#
#   HF_HOME / HUGGINGFACE_HUB_CACHE / HF_DATASETS_CACHE
#       → redirected to "${PROJECT_ROOT}/.hf_cache" so 60+ job sweeps cannot
#         fill ~/.cache/huggingface/ and trip a "no space left" mid-sweep.
#   NCCL_TIMEOUT (5h) / TORCH_NCCL_BLOCKING_WAIT
#       → bound NCCL collective timeouts above the worst-case rank-0
#         dataset download, so cold-cache languages don't trigger the
#         watchdog SIGABRT mid-loop.
#   HF_HUB_DOWNLOAD_TIMEOUT / HF_HUB_ETAG_TIMEOUT
#       → bound every HF Hub HTTP call so DNS or TLS hangs on rank 0 cannot
#         wedge the AsyncCheckpointer thread.
#
# Callers may override any of these by exporting them before sourcing _lib.sh.
#
# Usage (at top of each launcher, after PROJECT_ROOT is set):
#   source "${SCRIPT_DIR}/_lib.sh"
#
# All functions print diagnostics to stderr so they don't pollute stdout logs.
# =============================================================================

# ---------------------------------------------------------------------------
# Process-global env exports
# ---------------------------------------------------------------------------
# PROJECT_ROOT must be set by the caller (every launcher does this two lines
# before sourcing us). Fall back to PWD only as a last resort.
: "${PROJECT_ROOT:=${PWD}}"

# HF cache redirect — keeps datasets/hub snapshots off the home partition.
# If the caller already exported these, leave them untouched (operator may
# point them at a scratch volume).
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${HF_DATASETS_CACHE}" 2>/dev/null || true

# NCCL: a 5h watchdog bound is generous enough for rank-0 cold-cache HF
# fetches but still small enough to shake out genuine deadlocks before
# JOB_TIMEOUT (typically 12-24h) fires.
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-18000}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"

# HF Hub HTTP-level timeouts. Mirror what beetle_trainer.py sets at import
# time, but exporting here means stage_pretok_subset.py and other helper
# scripts also benefit.
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"

# PyTorch allocator: expandable segments reduces fragmentation OOMs on long
# runs by 30-60% on A100 (PyTorch >= 2.1). Operators may override.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Shared GPU lock path — same file as beetle_pipeline/lib/common.sh uses for
# BP_GPU_LOCK. A standalone launcher invoked while the pipeline is also
# running on the node will block here instead of colliding on the GPUs.
# Operators on multi-tenant hosts may point this elsewhere (e.g. a shared
# filesystem path) to coordinate across nodes — fine, because flock works
# transparently over NFS v4+.
: "${BEETLE_PIPELINE_STATE_DIR:=/Users/suchirsalhan/Documents/PHD/PHD/beetle_pipeline/state}"
: "${BEETLE_GPU_LOCK:=${BEETLE_PIPELINE_STATE_DIR}/gpu.lock}"

# Per-process CUDA memory fraction read by the trainer (beetle_trainer.py).
# 0.92 leaves ~5 GiB headroom on an 80 GiB A100 so a Python-catchable OOM
# fires before the kernel OOM-killer wedges the node.
export BP_CUDA_MEM_FRACTION="${BP_CUDA_MEM_FRACTION:-0.92}"

# ---------------------------------------------------------------------------
# kill_survivors
#
# Finds any surviving torchrun master or python train.py worker processes that
# are NOT the current shell ($$) and terminates them with SIGTERM, waits 10 s,
# then SIGKILL for anything still alive. Safe to call even when no strays
# exist (no-op).
# ---------------------------------------------------------------------------
kill_survivors() {
  local stray
  stray="$(pgrep -f 'torchrun|python[0-9.]* .*train\.py' 2>/dev/null \
           | grep -v "^${$}$" || true)"
  if [[ -z "${stray}" ]]; then return 0; fi

  echo "[cleanup] Stray training procs detected — sending SIGTERM:" >&2
  echo "          $(echo "${stray}" | tr '\n' ' ')" >&2
  # shellcheck disable=SC2086
  kill -TERM ${stray} 2>/dev/null || true
  sleep 10

  stray="$(pgrep -f 'torchrun|python[0-9.]* .*train\.py' 2>/dev/null \
           | grep -v "^${$}$" || true)"
  if [[ -z "${stray}" ]]; then
    echo "[cleanup] All stray procs gone after SIGTERM." >&2
    return 0
  fi

  echo "[cleanup] Still alive — sending SIGKILL:" >&2
  echo "          $(echo "${stray}" | tr '\n' ' ')" >&2
  # shellcheck disable=SC2086
  kill -KILL ${stray} 2>/dev/null || true
  sleep 5

  # Final check — warn loudly if something is truly unkillable (D-state / hung CUDA)
  stray="$(pgrep -f 'torchrun|python[0-9.]* .*train\.py' 2>/dev/null \
           | grep -v "^${$}$" || true)"
  if [[ -n "${stray}" ]]; then
    echo "[cleanup] WARN: processes ${stray} survived SIGKILL (likely in D-state)." >&2
    echo "[cleanup] GPU driver reset or node reboot may be required." >&2
  fi
}

# ---------------------------------------------------------------------------
# drain_gpus [max_secs]
#
# Polls nvidia-smi every 10 s until every GPU reports < 500 MiB used, which
# indicates CUDA contexts have been released. Returns 0 on success, 1 on
# timeout. Callers should treat a non-zero return as a cluster health hazard
# and abort the sweep.
#
# Falls back to a 30-second sleep when nvidia-smi is not available (e.g.
# on a login node running the sweep script remotely over SSH — rare).
# ---------------------------------------------------------------------------
drain_gpus() {
  local max_secs="${1:-180}"
  local poll=10
  local elapsed=0

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[drain] nvidia-smi not found — sleeping ${max_secs}s as conservative wait." >&2
    sleep "${max_secs}"
    return 0
  fi

  echo "[drain] Waiting for all GPUs to idle (threshold < 500 MiB used, max ${max_secs}s)…" >&2
  while [[ ${elapsed} -lt ${max_secs} ]]; do
    local top
    top="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
          | awk 'BEGIN{m=0} {v=$1+0; if(v>m) m=v} END{print m}')"
    if [[ -n "${top}" && "${top}" -lt 500 ]]; then
      echo "[drain] GPUs idle — max used: ${top} MiB." >&2
      return 0
    fi
    echo "[drain]   ${elapsed}s / ${max_secs}s — max used: ${top:-?} MiB" >&2
    sleep "${poll}"
    elapsed=$(( elapsed + poll ))
  done

  echo "[drain] WARN: GPUs still show > 500 MiB after ${max_secs}s." >&2
  return 1
}

# ---------------------------------------------------------------------------
# gpu_preflight [ngpus]
#
# Returns 0 if nvidia-smi is reachable and exposes >= ngpus GPUs.
# Returns 1 with a diagnostic on stderr otherwise.
# ---------------------------------------------------------------------------
gpu_preflight() {
  local ngpus="${1:-8}"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[preflight] WARN: nvidia-smi missing — skipping GPU health check." >&2
    return 0
  fi
  local visible
  if ! visible="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null \
                  | wc -l | tr -d ' ')"; then
    echo "[preflight] ERROR: nvidia-smi query failed." >&2
    return 1
  fi
  if [[ "${visible}" -lt "${ngpus}" ]]; then
    echo "[preflight] ERROR: only ${visible} GPU(s) visible, need ${ngpus}." >&2
    return 1
  fi
  return 0
}

# ---------------------------------------------------------------------------
# config_gpu_compliance <cfg_path> <ngpus>
#
# Validates that the YAML's training.fabric.devices == ngpus and
# training.fabric.precision == "bf16-mixed".
# Writes diagnostic to stderr; returns 1 on mismatch.
# ---------------------------------------------------------------------------
config_gpu_compliance() {
  local cfg="$1"
  local ngpus="$2"
  python3 - "${cfg}" "${ngpus}" <<'PY'
import sys, yaml
cfg_path, ngpus = sys.argv[1], int(sys.argv[2])
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)
fabric = (cfg.get("training") or {}).get("fabric") or {}
devices   = fabric.get("devices")
precision = fabric.get("precision")
errs = []
if devices != ngpus:
    errs.append(f"fabric.devices={devices} != NGPUS={ngpus}")
if precision != "bf16-mixed":
    errs.append(f"fabric.precision={precision!r} != 'bf16-mixed'")

# ── Schema checks beyond GPU-count parity ──────────────────────────────────
# These mismatches manifest only after data load + first forward pass,
# which on an 8xA100 burns ~30 min of cluster time per failed job.
training = cfg.get("training") or {}
model    = cfg.get("model") or {}
params   = (model.get("params") or {}) if isinstance(model, dict) else {}

# 1) max_seq_len must divide evenly across attention heads (rotary embedding
#    requires head_dim = d_model / n_heads to be an integer).
d_model  = params.get("d_model") or model.get("d_model")
n_heads  = (params.get("attention_n_heads")
            or model.get("attention_n_heads")
            or params.get("n_heads")
            or model.get("n_heads"))
if d_model and n_heads and (int(d_model) % int(n_heads) != 0):
    errs.append(
        f"model.d_model={d_model} not divisible by attention_n_heads={n_heads}"
    )

# 2) global batch parity. Skip silently when the relevant keys aren't all
#    set (older/legacy configs may not declare global_batch_size).
data_cfg = cfg.get("data") or {}
gbs   = training.get("global_batch_size")
mbs   = data_cfg.get("batch_size") or training.get("micro_batch_size")
gacc  = (training.get("optimization") or {}).get("gradient_accumulation_steps") \
        or training.get("grad_accum_steps")
if gbs and mbs and gacc and devices:
    expected = int(mbs) * int(gacc) * int(devices)
    if expected != int(gbs):
        errs.append(
            f"global_batch={gbs} != micro({mbs}) * grad_accum({gacc}) * "
            f"devices({devices}) = {expected}"
        )

if errs:
    sys.stderr.write("; ".join(errs) + "\n")
    sys.exit(1)
sys.exit(0)
PY
}

# ---------------------------------------------------------------------------
# disk_preflight [min_gb]
#
# Aborts the sweep if the filesystem holding PROJECT_ROOT has less than
# min_gb GiB free (default 200). 60-job sweeps download tens of GB of
# pretokenized shards and write tens of GB of checkpoints, so starting a
# sweep on a near-full disk is a guaranteed mid-sweep failure.
#
# Returns 0 on healthy disk, 1 otherwise. df -BG is GNU-specific; on macOS
# we fall back to df -k and convert. If df is unusable we WARN and pass.
# ---------------------------------------------------------------------------
disk_preflight() {
  local min_gb="${1:-200}"
  local target="${PROJECT_ROOT:-${PWD}}"
  local avail_gb=""
  if df -BG "${target}" >/dev/null 2>&1; then
    avail_gb="$(df -BG --output=avail "${target}" 2>/dev/null \
                | tail -n1 | tr -dc '0-9')"
  elif df -k "${target}" >/dev/null 2>&1; then
    local kb
    kb="$(df -k "${target}" 2>/dev/null | awk 'NR==2 {print $4}')"
    if [[ -n "${kb}" ]]; then
      avail_gb=$(( kb / 1024 / 1024 ))
    fi
  fi
  if [[ -z "${avail_gb}" ]]; then
    echo "[disk_preflight] WARN: could not determine free space at ${target}; continuing." >&2
    return 0
  fi
  if [[ "${avail_gb}" -lt "${min_gb}" ]]; then
    echo "[disk_preflight] ERROR: only ${avail_gb} GiB free at ${target}, need >= ${min_gb} GiB." >&2
    echo "[disk_preflight]        Aborting sweep before any GPU work begins." >&2
    return 1
  fi
  echo "[disk_preflight] ${avail_gb} GiB free at ${target} (>= ${min_gb} GiB threshold)." >&2
  return 0
}

# ---------------------------------------------------------------------------
# cleanup_runs_if_pushed <run_name> [log_path]
#
# Remove runs/<run_name>/ ONLY when the trainer wrote
# runs/<run_name>/.hf_upload_done. Otherwise (sentinel absent OR
# .hf_upload_failed present) preserve the directory so a subsequent
# RESUME=1 job can pick up the local checkpoint without depending on a
# possibly-incomplete HF Hub mirror.
#
# Used by every safe launcher in place of a bare `rm -rf runs/<name>/`.
# ---------------------------------------------------------------------------
cleanup_runs_if_pushed() {
  local run_name="$1"
  local log="${2:-/dev/null}"
  if [[ -z "${run_name}" ]]; then
    echo "[cleanup_runs_if_pushed] WARN: empty run_name — skipping." | tee -a "${log}" >&2
    return 0
  fi
  local run_dir="${PROJECT_ROOT}/runs/${run_name}"
  if [[ ! -d "${run_dir}" ]]; then
    return 0  # nothing to do
  fi
  if [[ -f "${run_dir}/.hf_upload_failed" ]]; then
    echo "[$(date -u +%FT%TZ)] cleanup: KEEPING ${run_dir} (.hf_upload_failed present)" | tee -a "${log}"
    return 0
  fi
  if [[ ! -f "${run_dir}/.hf_upload_done" ]]; then
    echo "[$(date -u +%FT%TZ)] cleanup: KEEPING ${run_dir} (.hf_upload_done missing — HF push not confirmed)" | tee -a "${log}"
    return 0
  fi
  echo "[$(date -u +%FT%TZ)] cleanup: rm -rf runs/${run_name}" | tee -a "${log}"
  rm -rf "${run_dir}"
}

# ---------------------------------------------------------------------------
# rotate_log <log_path> [size_mib]
#
# If the log exceeds size_mib (default 50), gzip it in place to free disk.
# Idempotent and quiet on small logs. Used by safe launchers post-job.
# ---------------------------------------------------------------------------
rotate_log() {
  local log="$1"
  local thresh="${2:-50}"
  [[ -f "${log}" ]] || return 0
  local size_mib
  if size_mib="$(du -m "${log}" 2>/dev/null | awk '{print $1}')"; then
    if [[ -n "${size_mib}" && "${size_mib}" -gt "${thresh}" ]]; then
      gzip -f "${log}" 2>/dev/null \
        && echo "[rotate_log] compressed ${log} (${size_mib} MiB)" >&2 \
        || true
    fi
  fi
}

# ---------------------------------------------------------------------------
# locate_timeout_bin
#
# Sets the global TIMEOUT_BIN variable to the available timeout binary:
#   "timeout"   — GNU coreutils (Linux)
#   "gtimeout"  — GNU coreutils via Homebrew (macOS)
#   ""          — neither found; callers must run without a timeout
#
# Safe to call multiple times (no-op if TIMEOUT_BIN already set).
# ---------------------------------------------------------------------------
locate_timeout_bin() {
  if [[ -n "${TIMEOUT_BIN+x}" ]]; then return 0; fi   # already set
  if command -v timeout >/dev/null 2>&1; then
    TIMEOUT_BIN="timeout"
  elif command -v gtimeout >/dev/null 2>&1; then
    TIMEOUT_BIN="gtimeout"
  else
    echo "WARN: neither timeout nor gtimeout found — JOB_TIMEOUT will be ignored." >&2
    TIMEOUT_BIN=""
  fi
}

# ---------------------------------------------------------------------------
# gpu_acquire_lock / gpu_release_lock
#
# Coordinate with beetle_pipeline/lib/common.sh on the same lock file so a
# standalone launcher can't collide with an in-flight pipeline run. Uses
# flock(1) when present (Linux); falls back to a PID-stamped mkdir mutex on
# macOS dev hosts that reclaims itself if the previous holder died.
# ---------------------------------------------------------------------------
gpu_acquire_lock() {
  mkdir -p "$(dirname "$BEETLE_GPU_LOCK")" 2>/dev/null || true
  if command -v flock >/dev/null 2>&1; then
    exec 202>"$BEETLE_GPU_LOCK"
    flock -x 202
    BEETLE_GPU_LOCK_FD=202
  else
    local mtx="${BEETLE_GPU_LOCK}.d"
    while true; do
      if mkdir "$mtx" 2>/dev/null; then
        echo "$$" > "$mtx/pid" 2>/dev/null || true
        break
      fi
      local holder=""
      [[ -f "$mtx/pid" ]] && holder="$(cat "$mtx/pid" 2>/dev/null)"
      if [[ -z "$holder" ]] || ! kill -0 "$holder" 2>/dev/null; then
        echo "[gpu_lock] reclaiming stale lock (holder='${holder}' dead)" >&2
        rm -rf "$mtx" 2>/dev/null
        continue
      fi
      sleep 1
    done
    BEETLE_GPU_LOCK_FD="mkdir:$mtx"
  fi
}

gpu_release_lock() {
  if [[ -n "${BEETLE_GPU_LOCK_FD:-}" ]]; then
    if [[ "${BEETLE_GPU_LOCK_FD}" == mkdir:* ]]; then
      local mtx="${BEETLE_GPU_LOCK_FD#mkdir:}"
      rm -f "$mtx/pid" 2>/dev/null
      rmdir "$mtx" 2>/dev/null || true
    else
      flock -u 202 2>/dev/null || true
      exec 202>&- 2>/dev/null || true
    fi
    unset BEETLE_GPU_LOCK_FD
  fi
}

# ---------------------------------------------------------------------------
# gpu_watchdog <target_pgid> <watchdog_log>
#
# Background safety monitor for an in-flight GPU job. Polls nvidia-smi every
# GPU_WATCHDOG_POLL_S seconds and SIGKILLs the entire process group
# <target_pgid> the moment ANY of these are detected:
#
#   - any GPU memory.used / memory.total >= GPU_MEM_KILL_PCT for
#     GPU_WATCHDOG_STRIKES consecutive polls (default 95% for 2 polls ≈ 10s)
#   - any GPU's uncorrectable-ECC counter increased since job start
#   - host MemAvailable < HOST_RAM_MIN_MIB MiB (Linux only)
#
# The kill is non-interactive: `kill -KILL -<pgid>` then kill_survivors.
# Rationale: on 8xA100 nodes, every second past memory pressure makes a
# wedged-node outcome more likely. Asking the operator for permission costs
# us >1 week of experiment time when the kernel OOM-killer wins the race.
#
# Watchdog exits cleanly when its target PGID no longer has live members.
# Designed to be launched with `&` and disowned/waited.
# ---------------------------------------------------------------------------
: "${GPU_MEM_KILL_PCT:=95}"
: "${GPU_WATCHDOG_POLL_S:=5}"
: "${GPU_WATCHDOG_STRIKES:=2}"
: "${HOST_RAM_MIN_MIB:=8192}"
: "${DISK_KILL_PCT:=97}"
: "${HEARTBEAT_STALL_S:=600}"
: "${HEARTBEAT_FILE:=}"

gpu_watchdog() {
  local target_pgid="$1" wlog="$2"
  [[ -z "${target_pgid}" || -z "${wlog}" ]] && return 0
  mkdir -p "$(dirname "$wlog")"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[$(date -u +%FT%TZ)] [WATCHDOG] nvidia-smi missing — inactive" >>"$wlog"
    return 0
  fi
  local ecc_baseline
  ecc_baseline="$(nvidia-smi --query-gpu=index,ecc.errors.uncorrected.aggregate.total --format=csv,noheader,nounits 2>/dev/null || true)"
  local nvlink_baseline
  nvlink_baseline="$(nvidia-smi nvlink --errorcounters 2>/dev/null | grep -E 'Replay|Recovery|CRC|Flit' | tr -d ' ' || true)"
  local _wd_start_ts
  _wd_start_ts="$(date -u +%s)"
  local _power_info
  _power_info="$(nvidia-smi --query-gpu=index,power.limit,power.draw --format=csv,noheader,nounits 2>/dev/null | tr '\n' '|' || true)"
  echo "[$(date -u +%FT%TZ)] [WATCHDOG] start pgid=${target_pgid} poll=${GPU_WATCHDOG_POLL_S}s mem_kill=${GPU_MEM_KILL_PCT}% strikes=${GPU_WATCHDOG_STRIKES} ram_min=${HOST_RAM_MIN_MIB}MiB xid_scan=on nvlink_scan=on thermal_scan=on disk_kill=${DISK_KILL_PCT}% heartbeat_stall=${HEARTBEAT_STALL_S}s" >>"$wlog"
  [[ -n "$_power_info" ]] && echo "[$(date -u +%FT%TZ)] [WATCHDOG] power_state: ${_power_info}" >>"$wlog"

  local strike_count=0
  while true; do
    if ! kill -0 -- -"${target_pgid}" 2>/dev/null; then
      echo "[$(date -u +%FT%TZ)] [WATCHDOG] target pgid=${target_pgid} gone; exiting" >>"$wlog"
      return 0
    fi

    # GPU memory pressure
    local breach=0 worst_idx="" worst_pct=0 idx mu mt pct
    while IFS=',' read -r idx mu mt; do
      idx="${idx// /}"; mu="${mu// /}"; mt="${mt// /}"
      [[ -z "$mu" || -z "$mt" || "$mt" == "0" ]] && continue
      [[ "$mu" =~ ^[0-9]+$ ]] || continue
      [[ "$mt" =~ ^[0-9]+$ ]] || continue
      pct=$(( mu * 100 / mt ))
      if (( pct >= GPU_MEM_KILL_PCT )); then
        breach=1
        if (( pct > worst_pct )); then worst_pct=$pct; worst_idx=$idx; fi
      fi
    done < <(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null)

    if (( breach == 1 )); then
      strike_count=$((strike_count + 1))
      echo "[$(date -u +%FT%TZ)] [WATCHDOG] mem-pressure strike ${strike_count}/${GPU_WATCHDOG_STRIKES} (GPU${worst_idx}=${worst_pct}%)" >>"$wlog"
      if (( strike_count >= GPU_WATCHDOG_STRIKES )); then
        echo "[$(date -u +%FT%TZ)] [WATCHDOG] KILL pgid=${target_pgid}: GPU${worst_idx} at ${worst_pct}% memory (>= ${GPU_MEM_KILL_PCT}%)" >>"$wlog"
        kill -KILL -- -"${target_pgid}" 2>/dev/null || true
        sleep 2
        kill_survivors
        return 0
      fi
    else
      strike_count=0
    fi

    # ECC growth
    local ecc_now
    ecc_now="$(nvidia-smi --query-gpu=index,ecc.errors.uncorrected.aggregate.total --format=csv,noheader,nounits 2>/dev/null || true)"
    if [[ -n "$ecc_now" && -n "$ecc_baseline" && "$ecc_now" != "$ecc_baseline" ]]; then
      local kill_for_ecc=0 row cur_v base_v
      while IFS= read -r row; do
        idx="$(echo "$row" | awk -F',' '{gsub(/ /,"",$1); print $1}')"
        cur_v="$(echo "$ecc_now"     | awk -F',' -v k="$idx" '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($1==k) print $2}')"
        base_v="$(echo "$ecc_baseline" | awk -F',' -v k="$idx" '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($1==k) print $2}')"
        if [[ "$cur_v" =~ ^[0-9]+$ && "$base_v" =~ ^[0-9]+$ ]] && (( cur_v > base_v )); then
          echo "[$(date -u +%FT%TZ)] [WATCHDOG] ECC uncorrectable increase on GPU${idx}: ${base_v} → ${cur_v}" >>"$wlog"
          kill_for_ecc=1
        fi
      done <<< "$ecc_now"
      if (( kill_for_ecc == 1 )); then
        echo "[$(date -u +%FT%TZ)] [WATCHDOG] KILL pgid=${target_pgid}: uncorrectable ECC growth" >>"$wlog"
        kill -KILL -- -"${target_pgid}" 2>/dev/null || true
        sleep 2
        kill_survivors
        return 0
      fi
    fi

    # Host RAM exhaustion
    if [[ -r /proc/meminfo ]]; then
      local mem_avail_kb mem_avail_mib
      mem_avail_kb="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null)"
      if [[ "$mem_avail_kb" =~ ^[0-9]+$ ]]; then
        mem_avail_mib=$(( mem_avail_kb / 1024 ))
        if (( mem_avail_mib < HOST_RAM_MIN_MIB )); then
          echo "[$(date -u +%FT%TZ)] [WATCHDOG] KILL pgid=${target_pgid}: host RAM ${mem_avail_mib} MiB < ${HOST_RAM_MIN_MIB} MiB" >>"$wlog"
          kill -KILL -- -"${target_pgid}" 2>/dev/null || true
          sleep 2
          kill_survivors
          return 0
        fi
      fi
    fi

    # Critical NVRM XID events (driver faults). See beetle_pipeline/lib/common.sh
    # for the full XID rationale; same list of kill-worthy IDs.
    local xid_pat='NVRM: Xid \(.*\): (13|31|43|63|64|69|74|79|80|95|119|120)[,)]'
    local xid_hits=""
    if command -v journalctl >/dev/null 2>&1; then
      xid_hits="$(journalctl -k --since "@${_wd_start_ts}" --no-pager -q 2>/dev/null \
                  | grep -E "$xid_pat" | head -n3)"
    fi
    if [[ -n "$xid_hits" ]]; then
      echo "[$(date -u +%FT%TZ)] [WATCHDOG] KILL pgid=${target_pgid}: critical NVRM XID event(s):" >>"$wlog"
      echo "$xid_hits" >>"$wlog"
      kill -KILL -- -"${target_pgid}" 2>/dev/null || true
      sleep 2
      kill_survivors
      return 0
    fi

    # NVLink error growth
    if [[ -n "$nvlink_baseline" ]]; then
      local nvlink_now
      nvlink_now="$(nvidia-smi nvlink --errorcounters 2>/dev/null | grep -E 'Replay|Recovery|CRC|Flit' | tr -d ' ')"
      if [[ -n "$nvlink_now" && "$nvlink_now" != "$nvlink_baseline" ]]; then
        echo "[$(date -u +%FT%TZ)] [WATCHDOG] KILL pgid=${target_pgid}: NVLink error counters grew since job start" >>"$wlog"
        kill -KILL -- -"${target_pgid}" 2>/dev/null || true
        sleep 2
        kill_survivors
        return 0
      fi
    fi

    # Thermal throttling (sustained)
    local thermal_hits=0 _idx _mask
    while IFS=, read -r _idx _mask; do
      _idx="${_idx// /}"; _mask="${_mask// /}"
      [[ "$_mask" =~ ^0x[0-9A-Fa-f]+$ ]] || continue
      if (( (16#${_mask#0x}) & 0x40 )); then
        thermal_hits=$((thermal_hits+1))
        echo "[$(date -u +%FT%TZ)] [WATCHDOG] GPU${_idx} HW_THERMAL_SLOWDOWN active (mask=${_mask})" >>"$wlog"
      fi
    done < <(nvidia-smi --query-gpu=index,clocks_throttle_reasons.active --format=csv,noheader,nounits 2>/dev/null)
    if (( thermal_hits > 0 )); then
      : "${_thermal_strikes:=0}"
      _thermal_strikes=$((_thermal_strikes + 1))
      if (( _thermal_strikes >= GPU_WATCHDOG_STRIKES )); then
        echo "[$(date -u +%FT%TZ)] [WATCHDOG] KILL pgid=${target_pgid}: HW_THERMAL_SLOWDOWN sustained for ${_thermal_strikes} strikes" >>"$wlog"
        kill -KILL -- -"${target_pgid}" 2>/dev/null || true
        sleep 2
        kill_survivors
        return 0
      fi
    else
      _thermal_strikes=0
    fi

    # Disk fill
    local _disk_pct
    _disk_pct="$(df -P "${PROJECT_ROOT:-/}" 2>/dev/null | awk 'NR==2 {gsub("%",""); print $5}')"
    if [[ "$_disk_pct" =~ ^[0-9]+$ ]] && (( _disk_pct >= DISK_KILL_PCT )); then
      echo "[$(date -u +%FT%TZ)] [WATCHDOG] KILL pgid=${target_pgid}: disk ${_disk_pct}% full (>= ${DISK_KILL_PCT}%)" >>"$wlog"
      kill -KILL -- -"${target_pgid}" 2>/dev/null || true
      sleep 2
      kill_survivors
      return 0
    fi

    # Heartbeat stall (catches NCCL hang)
    if (( HEARTBEAT_STALL_S > 0 )); then
      local hb_file="$HEARTBEAT_FILE"
      if [[ -z "$hb_file" && -d "${PROJECT_ROOT:-}/runs" ]]; then
        hb_file="$(ls -1t "${PROJECT_ROOT}/runs"/*/.heartbeat 2>/dev/null | head -n1)"
      fi
      if [[ -n "$hb_file" && -f "$hb_file" ]]; then
        local hb_mtime now_ts hb_age
        hb_mtime="$(stat -c %Y "$hb_file" 2>/dev/null || stat -f %m "$hb_file" 2>/dev/null)"
        now_ts="$(date -u +%s)"
        if [[ "$hb_mtime" =~ ^[0-9]+$ ]]; then
          hb_age=$(( now_ts - hb_mtime ))
          if (( hb_age >= HEARTBEAT_STALL_S )); then
            echo "[$(date -u +%FT%TZ)] [WATCHDOG] KILL pgid=${target_pgid}: heartbeat ${hb_file} stale for ${hb_age}s (>= ${HEARTBEAT_STALL_S}s)" >>"$wlog"
            kill -KILL -- -"${target_pgid}" 2>/dev/null || true
            sleep 2
            kill_survivors
            return 0
          fi
        fi
      fi
    fi

    sleep "${GPU_WATCHDOG_POLL_S}"
  done
}

# ---------------------------------------------------------------------------
# wrap_with_cgroup — same contract as the beetle_pipeline helper. Echoes a
# command-prefix that runs the wrapped command in a systemd scope with a
# MemoryMax cap (90% of host RAM by default). Empty prefix on hosts where
# systemd-run is unavailable / already inside a scope / non-Linux.
# Operators may set BEETLE_DISABLE_CGROUP=1 to opt out.
# ---------------------------------------------------------------------------
: "${BEETLE_CGROUP_MEM_PCT:=90}"
wrap_with_cgroup() {
  [[ "${BEETLE_DISABLE_CGROUP:-0}" == "1" ]] && return 0
  command -v systemd-run >/dev/null 2>&1 || return 0
  [[ -n "${INVOCATION_ID:-}" || -n "${SLURM_JOB_ID:-}" ]] && return 0
  [[ -r /proc/meminfo ]] || return 0
  local mem_total_kb mem_max_kb
  mem_total_kb="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo 2>/dev/null)"
  [[ "$mem_total_kb" =~ ^[0-9]+$ ]] || return 0
  mem_max_kb=$(( mem_total_kb * BEETLE_CGROUP_MEM_PCT / 100 ))
  echo "systemd-run --user --scope --quiet --collect -p MemoryMax=${mem_max_kb}K -p MemorySwapMax=0 --"
}
