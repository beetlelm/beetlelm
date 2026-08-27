#!/usr/bin/env bash
# eval_final_ckpts.sh — run BLiMP / MECO / BLiSS on the FINAL checkpoint of
# every 100M nld-eng B3 variant on a single GPU subprocess at a time
# (mirrors beetle-analyze/eval-final.sh's one-model-per-subprocess pattern,
# which exists because earlier parallel-rank evals OOM'd on shared 80GB
# A100s).
#
# Per variant: ~5 min BLiMP + ~10 min MECO + ~3 min BLiSS = ~20 min.
# 7 variants × 20 min = 2.5 h total.
#
# Skip-on-cache logic in each evaluator (eval_ledger.py) makes re-runs cheap.
#
# Env:
#   HF_TOKEN                 required
#   CUDA_VISIBLE_DEVICES     default 0   (pinning to one GPU to avoid OOM)
#   VARIANTS                 default ALL (space-sep subset of manifest)
#
# Outputs (under /root/beetle-analyze/results/100m-variants/):
#   blimp/<variant>/...
#   meco/<variant>/...
#   bliss/<variant>/...
set -uo pipefail

REPO_ROOT="/root/beetle-analyze"
cd "${REPO_ROOT}"

export HF_TOKEN="${HF_TOKEN:-}"
export HUGGINGFACE_TOKEN="${HF_TOKEN}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export PATH="/opt/conda/bin:${PATH}"
export PYTHONPATH="/root/beetlelm:${PYTHONPATH:-}"

OUT_ROOT="${REPO_ROOT}/results/100m-variants"
mkdir -p "${OUT_ROOT}"/{blimp,meco,bliss}

# Manifest is the source of truth (script/eval_100m_variants/manifest.py).
read -ra ALL_VARIANTS <<<"$(/opt/conda/bin/python3 -c '
import sys
sys.path.insert(0, "/root/beetlelm")
from script.eval_100m_variants.manifest import all_variants
print(" ".join(v.name for v in all_variants()))')"
read -ra VARIANTS <<<"${VARIANTS:-${ALL_VARIANTS[*]}}"

# helper: clear CUDA between evaluators
clear_cuda() {
    /opt/conda/bin/python3 - <<'PY'
import gc, torch
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache(); torch.cuda.ipc_collect()
PY
}

# helper: resolve a variant short-name → HF repo via the manifest
repo_for() {
    /opt/conda/bin/python3 -c '
import sys
sys.path.insert(0, "/root/beetlelm")
from script.eval_100m_variants.manifest import all_variants
target = sys.argv[1]
for v in all_variants():
    if v.name == target:
        print(v.hf_repo); break
' "$1"
}

run_eval() {
    local stage="$1" variant="$2" ; shift 2
    local repo
    repo="$(repo_for "${variant}")"
    if [[ -z "${repo}" ]]; then
        echo "[skip] ${variant}: not in manifest"; return 0
    fi
    echo "--- [${stage}] ${variant} (${repo}) ---"
    if ! "$@" --model "${repo}"; then
        echo "    !! [${stage}] FAILED on ${variant} (continuing)"
    fi
    clear_cuda
}

for variant in "${VARIANTS[@]}"; do
    echo "============================================================"
    echo "Evaluating variant: ${variant}"
    echo "============================================================"

    # BLiMP — eng + multilingual eng/nld benchmarks
    run_eval "BLiMP"  "${variant}" \
        /opt/conda/bin/python3 evals/eval_blimp.py \
        --output "${OUT_ROOT}/blimp" \
        --benchmarks blimp_eng

    # MECO — L2 surprisal under the trained model
    run_eval "MECO"   "${variant}" \
        /opt/conda/bin/python3 evals/eval_meco.py \
        --output "${OUT_ROOT}/meco" \
        --stimuli "${REPO_ROOT}/data/meco_l2_stims.tsv"

    # BLiSS — RP@0, RP@τ, NGS, CPS on ALTACambridge/BLiSS train split
    run_eval "BLiSS"  "${variant}" \
        /opt/conda/bin/python3 evals/eval_bliss.py \
        --output "${OUT_ROOT}/bliss" \
        --split train \
        --batch-size 32
done

echo "============================================================"
echo "All variants evaluated. Outputs under ${OUT_ROOT}/."
echo "============================================================"
