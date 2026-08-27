#!/usr/bin/env bash
# Wrapper executed inside the tmux 'beetle100m' session.
# Exports the env the sweep needs and chains to launch_nld_eng_b3_100m_safe.sh.
set -uo pipefail

export PATH="/opt/conda/bin:${PATH}"
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGINGFACE_TOKEN="${HF_TOKEN}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

# Quieter HF logs and a cache off the home partition (set by _lib.sh too).
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"

# expandable_segments=True: avoid the rank-0 fragmentation pattern that nearly
# OOM'd the cluster on 2026-05-06 (POLCA fp32 Lanczos walked rank 0 to 79 GiB
# / 80 GiB and never released; AsyncCheckpointer's later fp32 copies couldn't
# fit). Even with POLCA disabled, this gives the allocator room to recover
# from spiky alloc/free patterns.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Sequential sweep: full default JOBS list (fineweb baseline + 6 variants + babybabel).
# To override: tmux send-keys -t beetle100m:train "JOBS='fineweb ewc' bash …"
cd /root/beetlelm
# Sweep covers FineWeb baseline + 6 variants. Excluded:
#   * babybabel — user opted out 2026-05-07
#   * maml + layer-growth — trainer-level bugs (HF state_dict mismatch and
#     Lightning fabric.setup-twice respectively); leave for a future patch
JOBS="${JOBS:-fineweb ewc lamol maml pre-pretrain alibi}"

exec bash script/launch/launch_nld_eng_b3_100m_safe.sh "$@"
