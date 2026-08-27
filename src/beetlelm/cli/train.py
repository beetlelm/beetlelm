"""
beetlelm.cli.train
──────────────────
Console-script entry for ``beetlelm-train``.

Installed by ``pip install beetlelm`` via the ``[tool.poetry.scripts]`` table
in pyproject.toml. The pre-flight ``--validate_only`` schema check is
imported from ``script/train.py`` when run from a repo clone, and falls back
to a minimal in-package implementation when installed standalone.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import click  # noqa: E402

from beetlelm.training.beetle_trainer import Trainer  # noqa: E402


# ────────────────────────────────────────────────────────────────────────────
# Validation helper
# ────────────────────────────────────────────────────────────────────────────

def _run_config_validation(config_path: Path) -> int:
    """Cheap, GPU-free YAML schema check used by sweep launchers before torchrun.

    Mirrors the repo-side ``script/train.py::_run_config_validation`` so the
    pip-installed CLI has the same pre-flight behaviour. Returns 0 if OK, 1
    on first detected error. Tolerant of missing keys — only fails on
    detected mismatches.
    """
    import yaml

    errs: list[str] = []
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[validate] failed to parse YAML: {e}")
        return 1

    model = cfg.get("model", {}).get("params", {})
    training = cfg.get("training", {})
    tokenizer = cfg.get("tokenizer", {})

    # Head divisibility.
    d = model.get("d_model")
    h = model.get("attention_n_heads")
    if d and h and d % h != 0:
        errs.append(f"d_model={d} not divisible by attention_n_heads={h}")

    # Sequence-length parity between model and training blocks.
    msl = model.get("max_seq_len")
    tsl = training.get("seq_len")
    if msl and tsl and msl != tsl:
        errs.append(
            f"model.params.max_seq_len ({msl}) != training.seq_len ({tsl})"
        )

    # Vocab parity (best-effort).
    mv = model.get("vocab_size")
    tv = tokenizer.get("vocab_size")
    if mv and tv and mv != tv:
        errs.append(
            f"model.params.vocab_size ({mv}) != tokenizer.vocab_size ({tv})"
        )

    if errs:
        for e in errs:
            print(f"[validate][FAIL] {e}")
        return 1

    print(f"[validate][OK] {config_path}")
    return 0


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

@click.command()
@click.option(
    "--config_path",
    "config_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Path to the training configuration file.",
    required=True,
)
@click.option(
    "--dry_run",
    is_flag=True, default=False,
    help="Run a few steps to validate config + data without full training.",
)
@click.option(
    "--resume_checkpoint",
    type=str, default=None,
    help="Override default checkpoint auto-resume path.",
)
@click.option(
    "--tokenizer_only",
    is_flag=True, default=False,
    help="Only train or load the bilingual tokenizer and exit.",
)
@click.option(
    "--seed",
    "seed", type=int, default=42, show_default=True,
    help="Random seed for fabric.seed_everything and reproducibility.",
)
@click.option(
    "--validate_only",
    is_flag=True, default=False,
    help=(
        "Parse the config and run schema/consistency checks (vocab parity, "
        "head_dim divisibility, seq-len parity). Exit non-zero on the first "
        "error without launching torchrun. Cheap pre-flight for sweeps."
    ),
)
def main(
    config_path: Path,
    dry_run: bool,
    resume_checkpoint: str,
    tokenizer_only: bool,
    seed: int,
    validate_only: bool,
) -> None:
    """Train a BeetleLM model from a YAML configuration."""
    if validate_only:
        sys.exit(_run_config_validation(config_path))

    trainer = Trainer(config_path=str(config_path), seed=seed)

    if resume_checkpoint:
        trainer.configs["checkpointing"]["resume_from_path"] = resume_checkpoint

    if tokenizer_only:
        trainer._setup_tokenizer(trainer.configs["tokenizer"])
        print("[Trainer] Tokenizer setup complete.")
        return

    if dry_run:
        print("[Trainer] Running dry run: performing a few steps only.")
        original_max_steps = trainer.configs["training"]["max_steps"]
        trainer.configs["training"]["max_steps"] = min(5, original_max_steps)
        trainer.train()
        trainer.configs["training"]["max_steps"] = original_max_steps
        print("[Trainer] Dry run complete.")
    else:
        trainer.train()


if __name__ == "__main__":
    main()
