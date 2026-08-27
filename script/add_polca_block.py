"""
add_polca_block.py — One-shot migration that appends a default ``polca:`` block
to every training YAML in the FineWeb / b3-condition / curriculum sweeps.

Why:  POLCA (Phase-wise Loss Contribution Analysis; Chen et al., ICLR 2026,
arXiv 2510.05064) needs per-checkpoint gradient SVD bases and Hessian top-k
eigenvectors.  Wiring the live training callback in ``beetlelm`` only helps
*future* runs if the YAMLs already declare the ``polca:`` block, so this
script ensures every training config in the canonical sweep directories has
that block present (defaults conservative, GPU-safe — see ``DEFAULT_POLCA``
below).

Idempotent: skips any YAML that already contains a top-level ``polca:`` key.
Validates that each touched file remains parseable by ``yaml.safe_load``
after modification.

Usage
-----
    python script/add_polca_block.py                       # apply, all dirs
    python script/add_polca_block.py --dry-run             # report only
    python script/add_polca_block.py --dirs configs/b3_condition_sweep_2b
    python script/add_polca_block.py --disable-for tokenizer  # set enabled:false
                                                              # for files whose
                                                              # path contains
                                                              # 'tokenizer'

Design notes
------------
We deliberately use *textual append* rather than a YAML round-trip
(``ruamel.yaml``) because (a) ``ruamel`` is not in the project's runtime
deps, (b) PyYAML cannot preserve comments / list styling, and (c) ``polca:``
is a new top-level key whose ordering relative to existing top-level keys
is not load-bearing — the trainer reads it via ``self.configs.get("polca",
{})`` (cf. how ``ewc`` / ``maml`` / ``generative_replay`` are read in
``src/training/beetle_trainer.py:243-256``).

The appended text is a fixed multi-line string with two-space indentation,
matching the style of every existing block in these YAMLs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import yaml


# ---------------------------------------------------------------------------
# Defaults — conservative, GPU-safe (8×A100 80GB; 0.2B model).  Numbers chosen
# so that the live callback's holdout pass adds <5 % to step time and never
# exceeds the existing per-step memory peak.  See plan
# (.claude/plans/binary-bouncing-quokka.md) for the rationale.
# ---------------------------------------------------------------------------

DEFAULT_POLCA_BLOCK = """\
polca:
  enabled: true
  every_n_steps: 500
  k_eigenvectors: 32
  basis_rank: 32
  holdout_batch_size: 8
  holdout_seq_len: 512
  artifact_dir: polca
  push_to_hf: true
  precision: fp32
"""

DISABLED_POLCA_BLOCK = """\
polca:
  enabled: false
"""

# Default sweep directories that contain training YAMLs which should be
# POLCA-compatible by default.  Paths are relative to the repo root
# (``beetlelm/``).
DEFAULT_DIRS = [
    "configs/b3_condition_sweep_2b",
    "configs/humanscale_fineweb_bilingual",
    "configs/fineweb_bilingual_2b",
    "configs/fineweb_bilingual",
    "configs/fineweb_bilingual_hpc64",
    "configs/fineweb_curriculum",
]


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _has_polca_key(text: str) -> bool:
    """True iff the YAML text already declares a top-level ``polca:`` key.

    A "top-level" key is one whose ``polca:`` token starts at column 0.
    Lines beginning with ``#`` (comments) are ignored.
    """
    for line in text.splitlines():
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        if line.startswith("polca:") or line.rstrip() == "polca:":
            return True
    return False


def _append_polca_block(text: str, block: str) -> str:
    """Append ``block`` to ``text`` ensuring exactly one trailing newline
    between them and a final newline at EOF."""
    if not text.endswith("\n"):
        text = text + "\n"
    return text + block


def _validate_yaml(path: Path) -> None:
    """Raise on any YAML parse failure for the (possibly modified) file."""
    with path.open("r", encoding="utf-8") as f:
        yaml.safe_load(f)


def _path_matches_disable(path: Path, patterns: Iterable[str]) -> bool:
    s = str(path)
    return any(p in s for p in patterns)


def _iter_yaml_files(roots: Iterable[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            print(f"  [warn] root not found: {root}", file=sys.stderr)
            continue
        for p in sorted(root.rglob("*.yaml")):
            if p in seen:
                continue
            seen.add(p)
            yield p


def migrate(
    roots: list[Path],
    *,
    disable_patterns: list[str],
    dry_run: bool,
) -> tuple[int, int, int]:
    """Apply the migration.

    Returns
    -------
    touched, skipped_present, skipped_invalid
    """
    touched = 0
    skipped_present = 0
    skipped_invalid = 0

    for path in _iter_yaml_files(roots):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"  [error] read {path}: {exc}", file=sys.stderr)
            skipped_invalid += 1
            continue

        # Idempotency check.
        if _has_polca_key(text):
            skipped_present += 1
            continue

        # Choose enabled vs disabled block.
        block = (
            DISABLED_POLCA_BLOCK
            if _path_matches_disable(path, disable_patterns)
            else DEFAULT_POLCA_BLOCK
        )

        new_text = _append_polca_block(text, block)

        if dry_run:
            print(f"  [dry-run] would update {path}")
            touched += 1
            continue

        # Atomic-ish write: write to .tmp then replace.
        tmp = path.with_suffix(path.suffix + ".polca-tmp")
        try:
            tmp.write_text(new_text, encoding="utf-8")
            # Validate the *new* file before swapping in.
            _validate_yaml(tmp)
            tmp.replace(path)
            touched += 1
            print(f"  [ok]      updated   {path}")
        except yaml.YAMLError as exc:
            print(f"  [error] post-edit YAML invalid for {path}: {exc}",
                  file=sys.stderr)
            try:
                tmp.unlink()
            except OSError:
                pass
            skipped_invalid += 1

    return touched, skipped_present, skipped_invalid


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Add a default polca: block to every training YAML.",
    )
    p.add_argument(
        "--dirs",
        nargs="+",
        default=None,
        help=(
            "Sweep directories to migrate (relative to repo root). "
            "Defaults to all canonical sweep dirs."
        ),
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repository root (default: parent of this script's directory).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Report which files would be updated without writing.",
    )
    p.add_argument(
        "--disable-for",
        nargs="+",
        default=[],
        help=(
            "Path-substring patterns; any YAML whose path contains one of "
            "these gets polca.enabled: false instead of the full block."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    selected = args.dirs or DEFAULT_DIRS
    roots = [args.repo_root / d for d in selected]

    print(f"add_polca_block.py — repo_root={args.repo_root}")
    print(f"  dirs           : {selected}")
    print(f"  dry_run        : {args.dry_run}")
    print(f"  disable_for    : {args.disable_for}")

    touched, skipped_present, skipped_invalid = migrate(
        roots,
        disable_patterns=args.disable_for,
        dry_run=args.dry_run,
    )

    print()
    print(f"  files updated  : {touched}")
    print(f"  already had key: {skipped_present}")
    print(f"  skipped (error): {skipped_invalid}")

    return 0 if skipped_invalid == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
