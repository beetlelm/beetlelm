"""
src/config/_humanscale_fineweb_defaults.py

Hardcoded constants for the 100M FineWeb regime.

FineWeb data sources + tokenizers (data path matches the 24B FineWeb runs) at
a Human-Scale token budget (100M total per pair) and compute footprint
(max_steps=10_000, effective batch=64, 8 GPUs per node).

Models push to the dedicated ``Beetle-FineWeb-100M`` org so this experimental
track is cleanly separated from the BabyBabelLM-data ``Beetle-HumanScale`` runs.

Token budget : 100M total per pair (curriculum sets the L1/L2 split).
Steps        : 10,000 (= 100M / (64 effective batch * 512 seq_len) ≈ 3,052;
                rounded up to 10_000 to match the existing Human-Scale step
                counts. Data is reused via the FineWeb pretokenized cache,
                not exhaustively consumed).

Distinct from BabyBabelLM Human-Scale runs:
  - Data sources: Beetle-Data/{lang}-28B (FineWeb) — not BabyBabel
  - Tokenizers : Beetle-Data/tokenizer-{lang}-en — not Beetle-HumanScale BPE
  - dataset_source: "fineweb" (not "babybabel")
  - HF org     : Beetle-FineWeb-100M (not Beetle-HumanScale)
"""

from beetlelm.config._fineweb_defaults import (
    HF_ORG_DATA,
    SEQ_LEN,
    VOCAB_SIZE,
    TOKENIZER_TYPE,
    MODEL_BASE,
    ISO3_TO_BD,
    fineweb_l1_dataset,
    fineweb_en_dataset,
    fineweb_tokenizer_repo,
)

# ── HuggingFace orgs ──────────────────────────────────────────────────────────

# Push to Beetle-FineWeb-100M (dedicated org for the 100M FineWeb regime),
# keeping this experimental track cleanly separated from the BabyBabelLM-data
# Beetle-HumanScale runs.
HF_USER_MODELS    = "Beetle-FineWeb-100M"
DATASET_SOURCE    = "fineweb"                         # tokenizer dataset_source
DATASET_SCALE     = "fineweb-100m"                    # leads with "fineweb-100m-" in run_name

# ── Scale ─────────────────────────────────────────────────────────────────────

TOKENS_TOTAL      = 100_000_000                       # 100M total per pair
TOKENIZER_TOKENS  = 100_000_000                       # tokenizer training budget

# Compute footprint: matches existing human_scale_bilingual configs.
EFFECTIVE_BATCH   = 64
BATCH_SIZE        = 64                                # per-GPU; matches devices=8 below
TOKENS_PER_STEP   = EFFECTIVE_BATCH * SEQ_LEN         # 32,768
MAX_STEPS         = 10_000

MAX_STEPS_BILINGUAL   = MAX_STEPS
MAX_STEPS_MONOLINGUAL = MAX_STEPS

LR_WARMUP_STEPS   = 500                               # matches existing HS runs

# ── Training defaults ─────────────────────────────────────────────────────────

TRAINING_DEFAULTS = {
    "max_steps": MAX_STEPS,
    "batch_size": BATCH_SIZE,
    "seq_len": SEQ_LEN,
    "target_effective_batch": EFFECTIVE_BATCH,
    "fabric": {
        "accelerator": "cuda",
        "devices": 8,
        "strategy": "ddp",
        "precision": "bf16-mixed",
    },
    "optimization": {
        "optimizer": "adamw",
        "lr": 5e-4,
        "lr_warmup_steps": LR_WARMUP_STEPS,
        "lr_scheduler": "linear_with_warmup",
    },
}

# ── Checkpointing defaults ────────────────────────────────────────────────────

CHECKPOINTING_DEFAULTS = {
    "runs_dir": "./runs",
    "auto_resume": True,
    "save_to_hf": True,
    "export_final_only": False,
    "keep_local_checkpoints": 1,
    # No periodic (linear) checkpoint keys: checkpoint writes are gated solely
    # by the log₂ schedule + curriculum phase boundaries.
}

# ── Languages ────────────────────────────────────────────────────────────────

# All 13 FineWeb-2 L1 languages paired with English (eng L2).
EXP_LANGS = [
    "nld", "deu", "zho",
    "spa", "jpn", "ita", "rus", "pol", "tur",
    "fil", "hin", "eus", "arb",
]

# ── Condition codes ──────────────────────────────────────────────────────────

CONDITIONS = ["B1", "B2", "B3", "B4", "B5"]
"""
B1 - Balanced:      50:50 L1/L2 throughout
B2 - Simultaneous:  100% L1 until midpoint, then 50:50
B3 - Sequential:    100% L1 until midpoint, then 33:67 L1:L2
B4 - Classroom:     80:20 L1:L2 with episodic L2 bursts (L2 introduced at 50%)
B5 - Late:          L2 introduced at 80% of training, target 50:50
"""

__all__ = [
    "HF_ORG_DATA",
    "HF_USER_MODELS",
    "DATASET_SOURCE",
    "DATASET_SCALE",
    "SEQ_LEN",
    "VOCAB_SIZE",
    "TOKENIZER_TYPE",
    "TOKENIZER_TOKENS",
    "EFFECTIVE_BATCH",
    "BATCH_SIZE",
    "TOKENS_TOTAL",
    "TOKENS_PER_STEP",
    "MAX_STEPS",
    "MAX_STEPS_BILINGUAL",
    "MAX_STEPS_MONOLINGUAL",
    "LR_WARMUP_STEPS",
    "TRAINING_DEFAULTS",
    "CHECKPOINTING_DEFAULTS",
    "MODEL_BASE",
    "ISO3_TO_BD",
    "EXP_LANGS",
    "CONDITIONS",
    "fineweb_l1_dataset",
    "fineweb_en_dataset",
    "fineweb_tokenizer_repo",
]
