"""
beetlelm.toy.cli
────────────────
Console entry point ``beetlelm-toy`` for the download-free CPU quickstart.

Example
-------
    beetlelm-toy --config_path configs/tiny_cpu.yaml
    beetlelm-toy --config_path configs/tiny_cpu.yaml --seed 7 --max_steps 100
    beetlelm-toy --config_path configs/tiny_cpu.yaml --validate_only
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from beetlelm.toy.config import load_config
from beetlelm.toy.trainer import train


def _run_config_validation(config_path: Path) -> int:
    """Cheap, GPU-free schema check: head divisibility and seq-len parity."""
    try:
        cfg = load_config(config_path)
    except Exception as e:  # noqa: BLE001
        print(f"[validate][FAIL] could not parse config: {e}")
        return 1

    errs = []
    d, h = cfg.model.d_model, cfg.model.attention_n_heads
    if d % h != 0:
        errs.append(f"d_model={d} not divisible by attention_n_heads={h}")
    if h % cfg.model.attention_n_kv_heads != 0:
        errs.append(
            f"attention_n_heads={h} not divisible by "
            f"attention_n_kv_heads={cfg.model.attention_n_kv_heads}"
        )
    if cfg.model.max_seq_len < cfg.training.seq_len:
        errs.append(
            f"model.max_seq_len ({cfg.model.max_seq_len}) < "
            f"training.seq_len ({cfg.training.seq_len})"
        )

    if errs:
        for e in errs:
            print(f"[validate][FAIL] {e}")
        return 1
    print(f"[validate][OK] {config_path}")
    return 0


@click.command()
@click.option(
    "--config_path",
    "config_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the training configuration YAML.",
)
@click.option("--seed", type=int, default=None, help="Override training.seed.")
@click.option("--max_steps", type=int, default=None, help="Override training.max_steps.")
@click.option("--d_model", type=int, default=None, help="Override model.d_model.")
@click.option("--output_dir", type=str, default=None, help="Override training.output_dir.")
@click.option(
    "--validate_only",
    is_flag=True,
    default=False,
    help="Run schema/consistency checks and exit without training.",
)
def main(config_path, seed, max_steps, d_model, output_dir, validate_only):
    """Train a tiny BeetleLM model on CPU from a YAML configuration."""
    if validate_only:
        sys.exit(_run_config_validation(config_path))

    overrides: dict = {"model": {}, "training": {}}
    if seed is not None:
        overrides["training"]["seed"] = seed
    if max_steps is not None:
        overrides["training"]["max_steps"] = max_steps
    if output_dir is not None:
        overrides["training"]["output_dir"] = output_dir
    if d_model is not None:
        overrides["model"]["d_model"] = d_model

    results = train(str(config_path), overrides=overrides)
    print(
        f"[beetlelm-toy] done: final_loss={results['final_loss']} "
        f"val_loss={results['final_val_loss']} checkpoint={results['checkpoint_dir']}"
    )


if __name__ == "__main__":
    main()
