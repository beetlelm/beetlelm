"""
src/config/_fineweb_2b_defaults.py

Hardcoded constants for FineWeb-scale pretraining at the 2B-token horizon.

Same data pipeline, model architecture, and effective-batch invariants as the
24B FineWeb runs (see ``_fineweb_defaults``); only ``TOKENS_TOTAL``,
``MAX_STEPS``, warmup, checkpoint cadence, and the HF model-org / dataset slot
differ. Dataset and tokenizer helpers (``fineweb_l1_dataset``,
``fineweb_en_dataset``, ``fineweb_tokenizer_repo``) are re-exported from
``_fineweb_defaults`` so the FineWeb data layer stays single-sourced.

Token budget: 2B total per pair (curriculum sets the L1/L2 split).
Steps       : 7,629 (= 2B / (512 effective batch * 512 seq_len)).
"""

from beetlelm.config._fineweb_defaults import (
    HF_ORG_DATA,
    SEQ_LEN,
    VOCAB_SIZE,
    TOKENIZER_TYPE,
    TOKENIZER_TOKENS,
    EFFECTIVE_BATCH,
    MODEL_BASE,
    ISO3_TO_BD,
    fineweb_l1_dataset,
    fineweb_en_dataset,
    fineweb_tokenizer_repo,
    fineweb_per_gpu_batch,
)

# ── HuggingFace orgs ──────────────────────────────────────────────────────────

HF_USER_MODELS    = "Beetle-FineWeb-2B"              # trained models org
DATASET_SOURCE    = "fineweb"                         # tokenizer dataset_source
DATASET_SCALE     = "fineweb-2b"                      # token used in run_name

# ── Scale ─────────────────────────────────────────────────────────────────────

TOKENS_TOTAL      = 2_000_000_000                     # 2B total per pair
TOKENS_PER_STEP   = EFFECTIVE_BATCH * SEQ_LEN         # 262,144
MAX_STEPS         = TOKENS_TOTAL // TOKENS_PER_STEP   # 7,629

MAX_STEPS_BILINGUAL   = MAX_STEPS
MAX_STEPS_MONOLINGUAL = MAX_STEPS

# Reference per-GPU batch for the canonical 8-GPU single-node setup.
BATCH_SIZE        = EFFECTIVE_BATCH // 8              # 64

# ── Training defaults ─────────────────────────────────────────────────────────

# warmup ≈ 5% of max_steps (mirrors 625 / 91552 ≈ 0.7% scaled up for shorter run)
LR_WARMUP_STEPS   = max(100, MAX_STEPS // 20)         # 381

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
    "auto_resume": False,
    "save_to_hf": True,
    "export_final_only": False,
    "keep_local_checkpoints": 1,
    # No periodic (linear) checkpoint keys: writes are gated solely by the log₂
    # schedule + curriculum phase boundaries.
}

# ── Languages ────────────────────────────────────────────────────────────────

# Experiment 2B languages: 13 FineWeb-2 L1s paired with English (eng L2).
# Tamil (tam) was removed from the launch matrix; zho/jpn/fil/hin added.
EXP_LANGS = [
    "nld", "deu", "zho",
    "spa", "jpn", "ita", "rus", "pol", "tur",
    "fil", "hin", "eus", "arb",
]

# ── Condition codes (Experiment 2B) ──────────────────────────────────────────

FINEWEB_CONDITIONS = ["B1", "B2", "B3", "B4", "B5"]
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
    "TOKENS_TOTAL",
    "TOKENS_PER_STEP",
    "MAX_STEPS",
    "MAX_STEPS_BILINGUAL",
    "MAX_STEPS_MONOLINGUAL",
    "BATCH_SIZE",
    "LR_WARMUP_STEPS",
    "TRAINING_DEFAULTS",
    "CHECKPOINTING_DEFAULTS",
    "MODEL_BASE",
    "ISO3_TO_BD",
    "EXP_LANGS",
    "FINEWEB_CONDITIONS",
    "fineweb_l1_dataset",
    "fineweb_en_dataset",
    "fineweb_tokenizer_repo",
    "fineweb_per_gpu_batch",
]
