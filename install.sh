#!/usr/bin/env bash
# Convenience installer.
#
#   ./install.sh          # full install: creates .venv, installs beetlelm[dev]
#   ./install.sh --toy    # light install: only the download-free CPU quickstart deps
#
# The full install pulls the complete training stack (Lightning, datasets,
# wandb, HF Hub, ...). The --toy install stays small and runs on CPU.
set -euo pipefail

PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
MODE="${1:-full}"

echo "[install] creating virtualenv in $VENV_DIR"
"$PYTHON" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip

if [ "$MODE" = "--toy" ]; then
  echo "[install] light (toy) install"
  pip install "torch>=2.5.1" "transformers>=4.45.2" "numpy>=1.24" "pyyaml>=6.0" "click>=8.1.7" pytest
  pip install -e . --no-deps
  echo "[install] running a 3-step toy smoke train"
  beetlelm-toy --config_path configs/smoke.yaml
else
  echo "[install] full install (this pulls the complete training stack)"
  pip install -e ".[dev]"
  echo "[install] running a 3-step toy smoke train to verify"
  beetlelm-toy --config_path configs/smoke.yaml
fi

echo "[install] done. Activate with: source $VENV_DIR/bin/activate"
