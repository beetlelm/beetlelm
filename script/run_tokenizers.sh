#!/bin/bash
# =============================================================================
# BeetleLM Tokenizer Training
#
# Trains all required tokenizers and pushes to HF Hub.
#
# Phase 1: Human-Scale Monolingual  → Beetle-HumanScale/bpe-humanscale-{lang}
# Phase 2: Human-Scale Bilingual    → Beetle-HumanScale/bpe-humanscale-{l1}-{l2}
# Phase 3: FineWeb Bilingual        → Beetle-Data/tokenizer-{code}-en
#
# Usage:
#   bash script/run_tokenizers.sh              # Human-scale only (phases 1+2)
#   bash script/run_tokenizers.sh --fineweb    # All phases including FineWeb
#   bash script/run_tokenizers.sh --no-push    # Train locally, skip HF push
# =============================================================================

set -euo pipefail

# ── Config ───────────────────────────────────────────────────────────────────

LOG_DIR="logs_tokenizers"
VOCAB_SIZE=50000

# Human-scale: 100M tokens budget, BabyBabel data
HS_TARGET_TOKENS=100000000
HS_ORG="Beetle-HumanScale"
HS_DATA_ORG="Beetle-Data"

# FineWeb: 1B tokens budget
FW_TARGET_TOKENS=1000000000
FW_DATA_ORG="Beetle-Data"

# Human-scale languages
HS_LANGS="zho fas eng nld bul fra ind deu ukr"

# Human-scale bilingual pairs (sorted alphabetically per pair)
# Matches TOKENIZER_PAIRS in src/config/_human_scale_defaults.py
HS_PAIRS="
bul:deu bul:fra bul:ind bul:ukr
deu:ukr
eng:bul eng:deu eng:fra eng:ind eng:nld eng:ukr
fas:bul fas:deu fas:eng fas:fra fas:ind fas:nld fas:ukr
fra:deu fra:ind fra:ukr
ind:deu ind:ukr
nld:bul nld:deu nld:fra nld:ind nld:ukr
zho:bul zho:deu zho:eng zho:fas zho:fra zho:ind zho:nld zho:ukr
"

# FineWeb EXP2 languages (paired with English)
FW_LANGS="nld ita rus pol tur fra deu spa zho jpn"

# ── Parse args ───────────────────────────────────────────────────────────────

RUN_FINEWEB=false
PUSH_FLAG=""

for arg in "$@"; do
    case "$arg" in
        --fineweb)  RUN_FINEWEB=true ;;
        --no-push)  PUSH_FLAG="--no-push" ;;
        *)          echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# ── ISO3 → 2-letter code (FineWeb dataset naming) ───────────────────────────

iso3_to_bd() {
    case "$1" in
        nld) echo "nl" ;; ita) echo "it" ;; rus) echo "ru" ;;
        pol) echo "pl" ;; tur) echo "tr" ;; fra) echo "fr" ;;
        deu) echo "de" ;; spa) echo "es" ;; zho) echo "zh" ;;
        jpn) echo "ja" ;; eng) echo "en" ;; eus) echo "eu" ;;
        ind) echo "id" ;; tgl) echo "tl" ;; fas) echo "fa" ;;
        hin) echo "hi" ;; tam) echo "ta" ;; swe) echo "sv" ;;
        cat) echo "ca" ;; arb) echo "ar" ;; *) echo "$1" ;;
    esac
}

# ── Helpers ──────────────────────────────────────────────────────────────────

TRAINED=0
FAILED=0

train_tokenizer() {
    local name="$1"
    shift
    local hf_repo_args=""

    # Build hf_repo flag unless --no-push is set
    if [ -z "$PUSH_FLAG" ]; then
        hf_repo_args="--hf_repo $HF_REPO"
    fi

    echo "  Training: $name ..."
    if python script/train_tokenizer.py "$@" $hf_repo_args \
        > "$LOG_DIR/tok_${name}.log" 2>&1; then
        echo "    OK: $name"
        TRAINED=$((TRAINED + 1))
    else
        echo "    FAIL: $name (see $LOG_DIR/tok_${name}.log)"
        FAILED=$((FAILED + 1))
    fi
}

# ── Setup ────────────────────────────────────────────────────────────────────

mkdir -p "$LOG_DIR"
echo "============================================="
echo "BeetleLM Tokenizer Training"
echo "  Vocab size    : $VOCAB_SIZE"
echo "  FineWeb       : $RUN_FINEWEB"
echo "  Push to Hub   : $([ -z "$PUSH_FLAG" ] && echo yes || echo no)"
echo "============================================="

# ═════════════════════════════════════════════════════════════════════════════
# Phase 1: Human-Scale Monolingual
# ═════════════════════════════════════════════════════════════════════════════

echo ""
echo "=== Phase 1: Human-Scale Monolingual Tokenizers ==="

for lang in $HS_LANGS; do
    HF_REPO="${HS_ORG}/bpe-humanscale-${lang}"
    train_tokenizer "hs_mono_${lang}" \
        --datasets "${lang}:${HS_DATA_ORG}/BabyBabel-${lang}-50M" \
        --vocab_size "$VOCAB_SIZE" \
        --target_tokens "$HS_TARGET_TOKENS" \
        --output_dir "./tokenizers/humanscale_mono"
done

echo "Phase 1 complete."

# ═════════════════════════════════════════════════════════════════════════════
# Phase 2: Human-Scale Bilingual
# ═════════════════════════════════════════════════════════════════════════════

echo ""
echo "=== Phase 2: Human-Scale Bilingual Tokenizers ==="

for pair in $HS_PAIRS; do
    l1="${pair%%:*}"
    l2="${pair##*:}"
    HF_REPO="${HS_ORG}/bpe-humanscale-${l1}-${l2}"
    train_tokenizer "hs_bi_${l1}-${l2}" \
        --datasets "${l1}:${HS_DATA_ORG}/BabyBabel-${l1}-50M" \
                   "${l2}:${HS_DATA_ORG}/BabyBabel-${l2}-50M" \
        --vocab_size "$VOCAB_SIZE" \
        --target_tokens "$HS_TARGET_TOKENS" \
        --output_dir "./tokenizers/humanscale_bilingual"
done

echo "Phase 2 complete."

# ═════════════════════════════════════════════════════════════════════════════
# Phase 3: FineWeb Bilingual (optional)
# ═════════════════════════════════════════════════════════════════════════════

if [ "$RUN_FINEWEB" = true ]; then
    echo ""
    echo "=== Phase 3: FineWeb Bilingual Tokenizers ==="

    for lang in $FW_LANGS; do
        code=$(iso3_to_bd "$lang")
        HF_REPO="${FW_DATA_ORG}/tokenizer-${code}-en"
        train_tokenizer "fw_bi_${lang}-eng" \
            --datasets "${lang}:${FW_DATA_ORG}/${code}-28B" \
                       "eng:${FW_DATA_ORG}/en-for-${code}-28B" \
            --vocab_size "$VOCAB_SIZE" \
            --target_tokens "$FW_TARGET_TOKENS" \
            --output_dir "./tokenizers/fineweb_bilingual"
    done

    echo "Phase 3 complete."
fi

# ═════════════════════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════════════════════

echo ""
echo "============================================="
echo "Tokenizer training finished."
echo "  Trained : $TRAINED"
echo "  Failed  : $FAILED"
echo "  Logs    : $LOG_DIR/"
echo "============================================="

if [ "$FAILED" -gt 0 ]; then
    exit 1
fi
