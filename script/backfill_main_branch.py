#!/usr/bin/env python3
"""
Back-fill `main` branch for legacy beetlelm HF repos

Existing beetlelm HF repos were pushed with a `step-{N}` branch per checkpoint
and an empty `main`. This script walks those branches in step order and
replays pico-train-compatible commits onto `main` so pico-analyze can scan
the commit history.

Per-step commit sequence on `main` (matches the live push in
`beetle_trainer.py :: _push_pico_compatible_to_main`):

    [first step only]
        Saving Tokenizer -- Step N
        Saving Training Config -- Step N          (if training_config.yaml present)
        Saving Model Class -- Step N              (if pico_decoder.py present)

    [every step]
        Saving HF Model -- Step N                 (config.json + model.safetensors)
        Saving Fabric Checkpoint -- Step N        (fabric_checkpoint/)
        Saving Learning Dynamics Data (train) -- Step N   (if learning_dynamics/ present)

Usage:
    python script/backfill_main_branch.py --repo_id BeetleLM/beetlelm_eng-nld_late
    python script/backfill_main_branch.py --repo_id ... --dry_run
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path

import click
from huggingface_hub import HfApi, create_repo, snapshot_download, upload_file, upload_folder


STEP_BRANCH_RE = re.compile(r"^step-(\d+)$")

TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
]


def _ordered_step_branches(api: HfApi, repo_id: str) -> list[tuple[int, str]]:
    """Return [(step, branch_name), ...] sorted ascending by step."""
    refs = api.list_repo_refs(repo_id)
    out: list[tuple[int, str]] = []
    for b in refs.branches:
        m = STEP_BRANCH_RE.match(b.name)
        if m:
            out.append((int(m.group(1)), b.name))
    return sorted(out, key=lambda x: x[0])


def _ensure_main(api: HfApi, repo_id: str) -> None:
    existing = {b.name for b in api.list_repo_refs(repo_id).branches}
    if "main" not in existing:
        api.create_branch(repo_id=repo_id, branch="main")


def _replay_step(
    repo_id: str,
    step: int,
    branch_name: str,
    is_first_step: bool,
    dry_run: bool,
) -> None:
    """Download step-{N} branch contents and replay pico-style commits onto main."""
    if dry_run:
        print(f"[backfill] step {step} ({branch_name}):")
        if is_first_step:
            print(f"  would commit 'Saving Tokenizer -- Step {step}'")
            print(f"  would commit 'Saving Training Config -- Step {step}'")
            print(f"  would commit 'Saving Model Class -- Step {step}'")
        print(f"  would commit 'Saving HF Model -- Step {step}'")
        print(f"  would commit 'Saving Fabric Checkpoint -- Step {step}'")
        print(f"  would commit 'Saving Learning Dynamics Data (train) -- Step {step}' (if present)")
        return

    with tempfile.TemporaryDirectory(prefix=f"beetle_backfill_step{step}_") as tmp:
        local = snapshot_download(repo_id=repo_id, revision=branch_name, local_dir=tmp)
        local = Path(local)

        # ── First-step one-off commits ──
        if is_first_step:
            for fname in TOKENIZER_FILES:
                src = local / fname
                if src.exists():
                    upload_file(
                        path_or_fileobj=str(src),
                        path_in_repo=fname,
                        repo_id=repo_id,
                        revision="main",
                        commit_message=f"Saving Tokenizer -- Step {step}",
                    )

            cfg_yaml = local / "training_config.yaml"
            if cfg_yaml.exists():
                upload_file(
                    path_or_fileobj=str(cfg_yaml),
                    path_in_repo="training_config.yaml",
                    repo_id=repo_id,
                    revision="main",
                    commit_message=f"Saving Training Config -- Step {step}",
                )

            model_class = local / "pico_decoder.py"
            if model_class.exists():
                upload_file(
                    path_or_fileobj=str(model_class),
                    path_in_repo="pico_decoder.py",
                    repo_id=repo_id,
                    revision="main",
                    commit_message=f"Saving Model Class -- Step {step}",
                )

        # ── Per-step: HF model files ──
        for fname in ("config.json", "model.safetensors"):
            src = local / fname
            if src.exists():
                upload_file(
                    path_or_fileobj=str(src),
                    path_in_repo=fname,
                    repo_id=repo_id,
                    revision="main",
                    commit_message=f"Saving HF Model -- Step {step}",
                )

        # ── Per-step: fabric checkpoint ──
        fabric_dir = local / "fabric_checkpoint"
        if fabric_dir.exists() and any(fabric_dir.iterdir()):
            upload_folder(
                folder_path=str(fabric_dir),
                path_in_repo="fabric_checkpoint",
                repo_id=repo_id,
                revision="main",
                commit_message=f"Saving Fabric Checkpoint -- Step {step}",
            )

        # ── Per-step: learning dynamics (only if present) ──
        ld_dir = local / "learning_dynamics"
        if ld_dir.exists() and any(ld_dir.iterdir()):
            upload_folder(
                folder_path=str(ld_dir),
                path_in_repo="learning_dynamics",
                repo_id=repo_id,
                revision="main",
                commit_message=f"Saving Learning Dynamics Data (train) -- Step {step}",
            )


@click.command()
@click.option("--repo_id", required=True, help="HF repo id, e.g. BeetleLM/beetlelm_eng-nld_late")
@click.option("--dry_run", is_flag=True, default=False, help="Print planned commit sequence without pushing")
@click.option("--hf_token", default=None, help="HF token. Falls back to HF_TOKEN env var if unset.")
def main(repo_id: str, dry_run: bool, hf_token: str | None) -> None:
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    elif not os.getenv("HF_TOKEN") and not dry_run:
        print(
            "[backfill] HF_TOKEN is not set and --dry_run was not given. "
            "Push calls will fail without a valid token.",
            file=sys.stderr,
        )

    api = HfApi()
    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)

    step_branches = _ordered_step_branches(api, repo_id)
    if not step_branches:
        print(f"[backfill] No step-{{N}} branches found on {repo_id}; nothing to do.")
        return

    print(f"[backfill] {repo_id}: found {len(step_branches)} step branches")
    for step, branch in step_branches:
        print(f"  step-{step} ({branch})")

    if not dry_run:
        _ensure_main(api, repo_id)

    for idx, (step, branch_name) in enumerate(step_branches):
        _replay_step(
            repo_id=repo_id,
            step=step,
            branch_name=branch_name,
            is_first_step=(idx == 0),
            dry_run=dry_run,
        )

    print(f"[backfill] Done{' (dry run)' if dry_run else ''}.")


if __name__ == "__main__":
    main()
