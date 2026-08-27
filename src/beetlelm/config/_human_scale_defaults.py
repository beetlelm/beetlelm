"""
src/config/_human_scale_defaults.py

Hardcoded constants for BabyBabel / Human-Scale Pretraining (~100M words total).

These values are intentionally not configurable at runtime — they define the
human-scale experimental setup and must be shared identically across:
  - All config generators (configs/generate_human_scale_*.py)
  - Updated existing generators (generate_tier1.py, generate_babybabel_polyglot.py, etc.)

What IS configurable per experiment:
  - Amount of data per language (50M bilingual / 33M trilingual)
  - Learning rate and scheduler type
  - EWC lambda, MAML hybrid_ratio
  - Polyglot language set (which languages to combine)
  - vocab_sharing: "shared" | "per_language"

Distinct from FineWeb training:
  - FineWeb tokenizers: {hf_user}/fineweb_bpe_{langs}
  - BabyBabel tokenizers: Beetle-HumanScale/bpe-humanscale-{sorted_langs}  ← this setup
"""

# ── HuggingFace orgs ──────────────────────────────────────────────────────────

HF_ORG            = "Beetle-HumanScale"               # data + tokenizer + model org
HF_ORG_DATA       = HF_ORG                            # backward compat alias
HF_USER_MODELS    = HF_ORG                            # backward compat alias

DATASET_SCALE     = "humanscale"                      # token used in model repo names

# ── Scale ─────────────────────────────────────────────────────────────────────

TOKENS_MONOLINGUAL = 100_000_000   # token budget for monolingual baselines
TOKENS_BILINGUAL   = 50_000_000    # per-language token budget for bilingual models
TOKENS_TRILINGUAL  = 33_000_000    # per-language token budget for trilingual models
TOKENS_CLASSROOM_L1 = 80_000_000   # L1 token budget for B4 classroom
TOKENS_CLASSROOM_L2 = 20_000_000   # L2 token budget for B4 classroom
TOKENIZER_TOKENS   = 100_000_000   # token budget for training the tokenizer itself

# ── Training ──────────────────────────────────────────────────────────────────

MAX_STEPS         = 10_000
SEQ_LEN           = 512
BATCH_SIZE        = 64
VOCAB_SIZE        = 50_000
TOKENIZER_TYPE    = "bpe"

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
    "batch_size": BATCH_SIZE,
    "seq_len": SEQ_LEN,
    "target_effective_batch": 64,
    "fabric": {
        "accelerator": "cuda",
        "devices": 4,
        "strategy": "ddp",
        "precision": "bf16-mixed",
    },
    "optimization": {
        "optimizer": "adamw",
        "lr": 5e-4,
        "lr_warmup_steps": 500,
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
    # checkpoint_steps: computed dynamically by CurriculumCheckpointScheduler
    # from data_mix.step_boundaries(). Config generators inject explicit lists;
    # quickstart configs use dynamic computation as fallback.
    #
    # Learning dynamics: mirrors pico's LD extractor so beetle repos are
    # consumable by pico-analyze. Enabled by default so every generated
    # config emits LD commits on `main` — the trainer gates extraction on
    # `learning_dynamics.enabled` (src/training/beetle_trainer.py:268-269),
    # and without this block it defaults to False.
    "learning_dynamics": {
        "enabled": True,
        "log_every_n_steps": 1,
        "batch_size": 64,
        "sequence_idx": -1,
        "layer_suffixes": ["attention", "swiglu", "norm"],
        "compute_gradients": False,
        "learning_dynamics_dir": "learning_dynamics",
        "save_training_batch": True,
        "cleanup_local": True,
        "keep_n_local": 1,
    },
}

# ── Languages available at human scale ───────────────────────────────────────

LANGS = ["zho", "fas", "eng", "nld", "bul", "fra", "ind", "deu", "ukr"]

# MECO languages (top priority — have MECO reading-time benchmarks)
MECO_LANGS = ["zho", "eng", "nld", "deu"]

# Non-MECO languages (lower priority)
NON_MECO_LANGS = ["fas", "bul", "ukr", "ind", "fra"]

# Minimal sweep set for L1 (paired with eng as L2)
PILOT_L1_LANGS = ["nld", "zho", "deu"]

# ── Condition codes ──────────────────────────────────────────────────────────

BILINGUAL_CONDITIONS = ["B1", "B2", "B3", "B4", "B5", "B6"]
TRILINGUAL_CONDITIONS = ["T1", "T2", "T3", "T4"]


# ── Dataset repo naming ─────────────────────────────────────────────────────

def babybabel_mono_dataset(lang: str, target: str = "100M") -> str:
    """HF repo ID for monolingual pretokenized data.

    >>> babybabel_mono_dataset("nld")
    'Beetle-HumanScale/nld-100M'
    """
    return f"{HF_ORG}/{lang}-{target}"


def babybabel_l1_dataset(l1: str, l2: str, target: str = "50M") -> str:
    """HF repo ID for L1 side of a bilingual pair.

    >>> babybabel_l1_dataset("nld", "eng")
    'Beetle-HumanScale/nld-50M-eng-nld'
    """
    tok_pair = "-".join(sorted([l1, l2]))
    return f"{HF_ORG}/{l1}-{target}-{tok_pair}"


def babybabel_l2_dataset(l1: str, l2: str, target: str = "50M") -> str:
    """HF repo ID for L2 side of a bilingual pair.

    >>> babybabel_l2_dataset("nld", "eng")
    'Beetle-HumanScale/eng-50M-eng-nld'
    """
    tok_pair = "-".join(sorted([l1, l2]))
    return f"{HF_ORG}/{l2}-{target}-{tok_pair}"


def babybabel_tri_dataset(lang: str, l1: str, l2: str, l3: str, target: str = "33M") -> str:
    """HF repo ID for one language side of a trilingual triple.

    >>> babybabel_tri_dataset("nld", "eng", "nld", "zho")
    'Beetle-HumanScale/nld-33M-eng-nld-zho'
    """
    tok_triple = "-".join(sorted([l1, l2, l3]))
    return f"{HF_ORG}/{lang}-{target}-{tok_triple}"


# ── Tokenizer repo naming ───────────────────────────────────────────────────

def _lang_order(langs):
    """Sort languages by LANGS priority order (unknown langs fall back to
    alphabetical, sorted after known ones). Used for tokenizer repo names,
    which on HF follow the LANGS priority — not alphabetical."""
    return sorted(langs, key=lambda l: (LANGS.index(l) if l in LANGS else (len(LANGS), l)))


def babybabel_mono_tokenizer(lang: str) -> str:
    """HF repo ID for monolingual tokenizer.

    >>> babybabel_mono_tokenizer("nld")
    'Beetle-HumanScale/bpe-humanscale-nld'
    """
    return f"{HF_ORG}/bpe-humanscale-{lang}"


def babybabel_bilingual_tokenizer(l1: str, l2: str) -> str:
    """HF repo ID for bilingual tokenizer (LANGS-priority order).

    >>> babybabel_bilingual_tokenizer("eng", "zho")
    'Beetle-HumanScale/bpe-humanscale-zho-eng'
    >>> babybabel_bilingual_tokenizer("nld", "eng")
    'Beetle-HumanScale/bpe-humanscale-eng-nld'
    """
    pair = "-".join(_lang_order([l1, l2]))
    return f"{HF_ORG}/bpe-humanscale-{pair}"


def babybabel_trilingual_tokenizer(l1: str, l2: str, l3: str) -> str:
    """HF repo ID for trilingual tokenizer (LANGS-priority order).

    >>> babybabel_trilingual_tokenizer("eng", "nld", "zho")
    'Beetle-HumanScale/bpe-humanscale-zho-eng-nld'
    """
    triple = "-".join(_lang_order([l1, l2, l3]))
    return f"{HF_ORG}/bpe-humanscale-{triple}"
