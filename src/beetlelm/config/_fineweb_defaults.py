"""
src/config/_fineweb_defaults.py

Hardcoded constants for FineWeb-scale pretraining — 24B tokens, 1 epoch.

These values define the FineWeb experimental setup and must be shared
identically across all FineWeb config generators.

Token budgets (all target exactly 1 epoch over the pretokenized data):
  - Monolingual: 24B tokens
  - Bilingual: 12B per language (50:50 L1:EN split, 24B total)
  - Trilingual: 8B per language (24B total)

Batching invariants:
  - Effective global batch: 512 samples = 262,144 tokens per optimizer step.
  - max_steps = 91,552 (constant across device counts).
  - Per-GPU batch scales inversely with device count:
      8  GPUs → per_gpu=64, 16 GPUs → per_gpu=32,
      32 GPUs → per_gpu=16, 64 GPUs → per_gpu=8.
    Use fineweb_per_gpu_batch(devices) to compute it.

Distinct from BabyBabel/Human-Scale training:
  - FineWeb tokenizers: {hf_user}/fineweb_bpe_{langs}
  - BabyBabel tokenizers: Beetle-HumanScale/bpe-humanscale-{sorted_langs}
"""

# ── HuggingFace orgs ──────────────────────────────────────────────────────────

HF_ORG_DATA       = "Beetle-Data"                   # pretokenized data + tokenizers (same org)
HF_USER_MODELS    = "Beetle-FineWeb2-24B"             # trained models org

DATASET_SOURCE    = "fineweb"                        # prefix for tokenizer repo names
DATASET_SCALE     = "fineweb"                        # token used in model repo names

# ── Scale ─────────────────────────────────────────────────────────────────────

TOKENS_MONOLINGUAL = 24_000_000_000    # 24B per language for monolingual baselines
TOKENS_BILINGUAL   = 12_000_000_000    # 12B per language for bilingual (50:50 split)
TOKENS_TRILINGUAL  = 8_000_000_000     # 8B per language for trilingual
TOKENIZER_TOKENS   = 1_000_000_000     # token budget for training the tokenizer

# ── Training ──────────────────────────────────────────────────────────────────

SEQ_LEN           = 512
VOCAB_SIZE        = 50_000
TOKENIZER_TYPE    = "bpe"

# Token budget & batching invariants (1 epoch over 24B unique tokens).
# Effective batch is fixed across device counts; per-GPU batch = EFFECTIVE_BATCH // devices.
TOKENS_TOTAL     = 24_000_000_000
EFFECTIVE_BATCH  = 512                              # samples/optimizer step (global)
TOKENS_PER_STEP  = EFFECTIVE_BATCH * SEQ_LEN        # 262,144
MAX_STEPS        = TOKENS_TOTAL // TOKENS_PER_STEP  # 91,552 ≈ 1 epoch

# Historical per-scale aliases (all equal now that 1 epoch = 24B tokens everywhere).
MAX_STEPS_BILINGUAL   = MAX_STEPS
MAX_STEPS_MONOLINGUAL = MAX_STEPS
MAX_STEPS_TRILINGUAL  = MAX_STEPS

# Reference per-GPU batch for the canonical 8-GPU single-node setup.
# Other device counts should derive per-GPU batch via fineweb_per_gpu_batch(devices).
BATCH_SIZE        = EFFECTIVE_BATCH // 8            # 64


def fineweb_per_gpu_batch(devices: int) -> int:
    """Per-GPU micro-batch so effective batch stays at EFFECTIVE_BATCH (grad_accum=1)."""
    return max(1, EFFECTIVE_BATCH // devices)

# ── Model ─────────────────────────────────────────────────────────────────────

MODEL_BASE = {
    "architecture": "pico",
    "params": {
        "n_layers": 14,
        "d_model": 768,
        "vocab_size": VOCAB_SIZE,
        "attention_n_heads": 12,
        "attention_n_kv_heads": 1,
        "max_seq_len": SEQ_LEN,
        "batch_size": BATCH_SIZE,
        "position_emb_theta": 10000.0,
        "activation_hidden_dim": 3072,
        "norm_eps": 1e-5,
        "dropout": 0.1,
    },
}

# ── Training defaults ─────────────────────────────────────────────────────────

TRAINING_DEFAULTS = {
    "max_steps": MAX_STEPS,
    "batch_size": BATCH_SIZE,                # 64 (matches 8-GPU default)
    "seq_len": SEQ_LEN,
    "target_effective_batch": EFFECTIVE_BATCH,  # 512 — honest global batch
    "fabric": {
        "accelerator": "cuda",
        "devices": 8,                        # 8xA100 single-node default
        "strategy": "ddp",
        "precision": "bf16-mixed",
    },
    "optimization": {
        "optimizer": "adamw",
        "lr": 5e-4,
        "lr_warmup_steps": 625,              # 5000 / 8 (rescaled with max_steps)
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
    # schedule (fineweb_checkpoint_schedule for >100k steps) + curriculum phase
    # boundaries. checkpoint_steps is computed dynamically by
    # CurriculumCheckpointScheduler from data_mix.step_boundaries(); generators
    # may inject an explicit list to override.
}

# ── Languages ────────────────────────────────────────────────────────────────

# Core FineWeb languages (original 19 + Arabic = 20 non-English)
CORE_LANGS = [
    "pol", "nld", "spa", "ell", "jpn", "fra", "zho", "deu",
    "ita", "eus", "tur", "ind", "tgl", "fas", "hin", "tam",
    "swe", "rus", "cat", "arb", "eng",
]

# Extension languages (lower priority, separate training runs)
EXTENSION_LANGS = [
    "urd", "ben", "ces", "guj", "tha", "vie", "kor", "dan",
]

# Low-resource languages (lowest priority, may not reach full token counts)
LOW_RESOURCE_LANGS = [
    "hun", "bul", "hrv", "ukr", "slv",   # European
    "som", "amh", "yor", "wol",           # African
]

ALL_LANGS = CORE_LANGS + EXTENSION_LANGS + LOW_RESOURCE_LANGS

# ── ISO 639-3 → beetle-data 2-letter code mapping ───────────────────────────
# beetle-data uses 2-letter codes for HF dataset names (e.g., Beetle-Data/pl-28B)
# while BeetleLM uses 3-letter ISO 639-3 codes internally.

ISO3_TO_BD = {
    "pol": "pl", "nld": "nl", "spa": "es", "ell": "el", "jpn": "ja",
    "fra": "fr", "zho": "zh", "deu": "de", "ita": "it", "eus": "eu",
    "tur": "tr", "ind": "id", "tgl": "tl", "fil": "tl", "fas": "fa",
    "hin": "hi", "tam": "ta", "swe": "sv", "rus": "ru", "cat": "ca",
    "arb": "ar", "eng": "en", "por": "pt",
    # extension
    "urd": "ur", "ben": "bn", "ces": "cs", "guj": "gu",
    "tha": "th", "vie": "vi", "kor": "ko", "dan": "da",
    # low-resource
    "hun": "hu", "bul": "bg", "hrv": "hr", "ukr": "uk", "slv": "sl",
    "som": "so", "amh": "am", "yor": "yo", "wol": "wo",
}


def fineweb_l1_dataset(iso3: str) -> str:
    """HF dataset name for pretokenized L1 Arrow data."""
    return f"{HF_ORG_DATA}/{ISO3_TO_BD[iso3]}-28B"


def fineweb_en_dataset(iso3: str) -> str:
    """HF dataset name for pretokenized English Arrow data paired with L1."""
    return f"{HF_ORG_DATA}/en-for-{ISO3_TO_BD[iso3]}-28B"


def fineweb_tokenizer_repo(iso3: str) -> str:
    """HF tokenizer repo for a bilingual {lang}-en pair."""
    return f"{HF_ORG_DATA}/tokenizer-{ISO3_TO_BD[iso3]}-en"

# ── Language tiers (Experiment 2) ────────────────────────────────────────────
# L2 is always English for FineWeb experiments.

FINEWEB_TIER_1 = ["zho", "nld", "deu"]               # MECO overlap — top priority
FINEWEB_TIER_2 = ["spa", "rus", "ita", "por", "tur"]  # high-resource, broad typology

# ── Condition codes (Experiment 2) ──────────────────────────────────────────

FINEWEB_CONDITIONS = ["B1", "B2", "B3", "B4", "B5"]
"""
B1 - Balanced:      50:50 L1/L2 throughout
B2 - Simultaneous:  100% L1 until midpoint, then 50:50
B3 - Sequential:    100% L1 until midpoint, then 33:67 L1:L2
B4 - Classroom:     80:20 L1:L2 with episodic L2 bursts (L2 introduced at 50%)
B5 - Late:          L2 introduced at 80% of training, target 50:50
"""
