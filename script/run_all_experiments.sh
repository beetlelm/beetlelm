#!/bin/bash
# =============================================================================
# run_all_experiments.sh — Complete BeetleLM experiment pipeline.
#
# Runs all experiments in the correct order:
#   1. Generate all config files
#   2. Train tokenizers
#   3. Human-scale EWC/MAML/LR sweep (balanced + simultaneous)
#   4. Human-scale full conditions (all 6 mixes)
#   5. Human-scale trilingual + MAML
#   6. FineWeb bilingual training (core, extension, low-resource)
#
# Usage:
#   # Full pipeline on a specific node
#   bash script/run_all_experiments.sh <NODE_ID>
#
#   # Specific phase only
#   bash script/run_all_experiments.sh <NODE_ID> --phase 1   # configs only
#   bash script/run_all_experiments.sh <NODE_ID> --phase 2   # tokenizers
#   bash script/run_all_experiments.sh <NODE_ID> --phase 3   # human-scale sweep
#   bash script/run_all_experiments.sh <NODE_ID> --phase 4   # human-scale conditions
#   bash script/run_all_experiments.sh <NODE_ID> --phase 5   # trilingual + MAML
#   bash script/run_all_experiments.sh <NODE_ID> --phase 6   # FineWeb bilingual
#
#   # Human-scale only (phases 1-5)
#   bash script/run_all_experiments.sh <NODE_ID> --human-scale-only
#
#   # FineWeb only (phase 6, assumes configs already generated)
#   bash script/run_all_experiments.sh <NODE_ID> --fineweb-only
# =============================================================================

set -euo pipefail

NODE_ID=${1:?ERROR: provide NODE_ID (1-4) as first argument}
shift

PHASE="all"
HUMAN_SCALE_ONLY=false
FINEWEB_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --phase) PHASE="$2"; shift 2 ;;
        --human-scale-only) HUMAN_SCALE_ONLY=true; shift ;;
        --fineweb-only) FINEWEB_ONLY=true; shift ;;
        *) shift ;;
    esac
done

echo "============================================================"
echo "BeetleLM: Run All Experiments"
echo "  Node ID:  $NODE_ID"
echo "  Phase:    $PHASE"
echo "============================================================"

# ── Phase 1: Generate all config files ───────────────────────────────────────
if [[ "$PHASE" == "all" || "$PHASE" == "1" ]] && [[ "$FINEWEB_ONLY" == "false" ]]; then
    echo ""
    echo "── Phase 1: Generating config files ──"

    echo "[configs] Human-scale bilingual (all mixes + sweep)..."
    python3 configs/generate_human_scale_bilingual.py

    echo "[configs] Human-scale trilingual..."
    python3 configs/generate_human_scale_trilingual.py

    echo "[configs] Human-scale MAML sweep..."
    python3 configs/generate_human_scale_maml.py

    echo "[configs] FineWeb bilingual..."
    python3 configs/generate_fineweb_bilingual.py

    echo "[configs] All configs generated."
fi

# ── Phase 2: Train tokenizers ────────────────────────────────────────────────
if [[ "$PHASE" == "all" || "$PHASE" == "2" ]] && [[ "$FINEWEB_ONLY" == "false" ]]; then
    echo ""
    echo "── Phase 2: Training tokenizers ──"
    bash script/run_tokenizers.sh
fi

# ── Phase 3: Human-scale EWC/MAML/LR sweep ──────────────────────────────────
if [[ "$PHASE" == "all" || "$PHASE" == "3" ]] && [[ "$FINEWEB_ONLY" == "false" ]]; then
    echo ""
    echo "── Phase 3: Human-scale EWC/MAML/LR sweep (balanced + simultaneous) ──"
    bash script/run_human_scale_sweep.sh "$NODE_ID"
fi

# ── Phase 4: Human-scale full conditions ─────────────────────────────────────
if [[ "$PHASE" == "all" || "$PHASE" == "4" ]] && [[ "$FINEWEB_ONLY" == "false" ]]; then
    echo ""
    echo "── Phase 4: Human-scale bilingual conditions ──"
    bash script/run_conditions_bilingual.sh "$NODE_ID"
fi

# ── Phase 5: Trilingual + MAML ──────────────────────────────────────────────
if [[ "$PHASE" == "all" || "$PHASE" == "5" ]] && [[ "$FINEWEB_ONLY" == "false" ]]; then
    echo ""
    echo "── Phase 5: Human-scale trilingual conditions ──"
    bash script/run_conditions_trilingual.sh "$NODE_ID"
fi

if [[ "$HUMAN_SCALE_ONLY" == "true" ]]; then
    echo ""
    echo "Human-scale experiments complete (--human-scale-only)."
    exit 0
fi

# ── Phase 6: FineWeb bilingual ───────────────────────────────────────────────
if [[ "$PHASE" == "all" || "$PHASE" == "6" ]]; then
    echo ""
    echo "── Phase 6: FineWeb bilingual training ──"

    # NOTE: Before running, update BEST_EWC_LAMBDA, BEST_MAML_RATIO, BEST_LR_BOOST
    # in configs/generate_fineweb_bilingual.py based on human-scale sweep results.
    # Then re-generate: python3 configs/generate_fineweb_bilingual.py

    echo "[fineweb] Core languages (20 langs x 3 conditions)..."
    bash script/run_fineweb_bilingual.sh "$NODE_ID" core

    echo "[fineweb] Extension languages (8 langs x 3 conditions)..."
    bash script/run_fineweb_bilingual.sh "$NODE_ID" extension

    echo "[fineweb] Low-resource languages (9 langs x 3 conditions)..."
    bash script/run_fineweb_bilingual.sh "$NODE_ID" low_resource
fi

echo ""
echo "============================================================"
echo "BeetleLM: All experiments complete on node $NODE_ID."
echo "  Checkpoints: runs/"
echo "  Models: HuggingFace Hub"
echo "============================================================"
