#!/usr/bin/env python3
"""
script/train.py
───────────────
Repo-local launcher that defers to the installed CLI.

After ``pip install -e .`` (or any install of this package), the canonical
entry point is ``beetlelm-train`` — a console script wired to
``beetlelm.cli.train:main`` via ``[project.scripts]`` in pyproject.toml.

This file remains so existing shell scripts and Slurm jobs that call
``python script/train.py --config_path ...`` keep working without changes.
It adds the repo root to ``sys.path`` so the import works even from a fresh
checkout that hasn't been installed yet.
"""

import sys
from pathlib import Path

# Add repo's src/ to sys.path so the package resolves without `pip install`.
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from beetlelm.cli.train import main  # noqa: E402

if __name__ == "__main__":
    main()
