#!/usr/bin/env python3
"""
Read-only audit of beetle FineWeb HF repos.

For every repo under the given org(s), print:
  - inferred condition (B1..B5) from run_name
  - expected number of step-{N} branches per the deterministic schedule in
    configs/ckpt_schedule.fineweb_checkpoint_schedule
  - actual number of step-{N} branches
  - the highest step actually pushed
  - whether `main` carries config.json / model.safetensors / pico_decoder.py
    (the three files instruction_tune_beetle.py needs to load with
    revision="main")

This script does NOT push, create, or delete anything. It only reads. Use it
to decide which repos need backfill_main_branch.py and which (if any) need
training to be resumed.

Usage:
    python script/audit_hub_branches.py \
        --org Beetle-FineWeb-2B \
        --org Beetle-FineWeb2-24B

Exits 0 on success.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass

import click
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

# Reuse the same schedule helper that the trainer uses, so any future change
# to the schedule (e.g. subsample_log2) is reflected automatically.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
from configs.ckpt_schedule import fineweb_checkpoint_schedule  # noqa: E402

STEP_BRANCH_RE = re.compile(r"^step-(\d+)$")
CONDITION_RE = re.compile(r"-(b[1-5])-", re.IGNORECASE)

# Token budgets per org. Mirrors src/config/_fineweb_2b_defaults.py:42 and
# src/config/_fineweb_defaults.py:50-53. 24B token budget covers both the
# legacy `Beetle-FineWeb` org and the newer `Beetle-FineWeb2-24B` org.
ORG_MAX_STEPS = {
    "Beetle-FineWeb-2B":   7_629,    # 2B tokens
    "Beetle-FineWeb":      91_552,   # 24B tokens (legacy)
    "Beetle-FineWeb2-24B": 91_552,   # 24B tokens (catch-up org)
}

# Required files on `main` for instruction_tune_beetle.py to load with
# trust_remote_code=True. tokenizer.json checked separately because it's
# always pushed at step 0 and is independent of model completeness.
MAIN_REQUIRED_FILES = ("config.json", "model.safetensors", "pico_decoder.py")


def _phase_starts(condition: str, max_steps: int) -> list[int]:
    """Mirror configs/generate_fineweb_2b.py:_phase_starts and
    configs/generate_fineweb_bilingual.py:_PHASE_STARTS. They use the same
    rule, so we centralize it here."""
    cond = condition.upper()
    if cond == "B1":
        return [0]
    if cond in ("B2", "B3", "B4"):
        return [0, max_steps // 2]
    if cond == "B5":
        return [0, int(max_steps * 0.8)]
    # Unknown condition: assume single-phase.
    return [0]


def _expected_ckpt_count(condition: str, max_steps: int) -> int:
    return len(
        fineweb_checkpoint_schedule(
            _phase_starts(condition, max_steps),
            max_steps=max_steps,
            extra_offsets=[2, 4, 6, 8],
        )
    )


def _infer_condition(run_name: str) -> str | None:
    m = CONDITION_RE.search(run_name)
    return m.group(1).upper() if m else None


@dataclass
class RepoAudit:
    repo_id: str
    condition: str | None
    expected: int | None
    actual: int
    max_step: int | None
    main_has_config: bool
    main_has_safetensors: bool
    main_has_pico: bool

    @property
    def status(self) -> str:
        if self.expected is None:
            return "UNKNOWN_COND"
        if self.actual < self.expected:
            return "SHORT"
        if not all((self.main_has_config, self.main_has_safetensors, self.main_has_pico)):
            return "MAIN_INCOMPLETE"
        return "OK"


def _audit_repo(api: HfApi, repo_id: str, max_steps: int) -> RepoAudit:
    refs = api.list_repo_refs(repo_id)
    step_branches: list[int] = []
    for b in refs.branches:
        m = STEP_BRANCH_RE.match(b.name)
        if m:
            step_branches.append(int(m.group(1)))

    condition = _infer_condition(repo_id.split("/", 1)[-1])
    expected = _expected_ckpt_count(condition, max_steps) if condition else None

    # Check `main` for the three files needed by AutoModelForCausalLM with
    # trust_remote_code=True. list_repo_files is metadata-only — no download.
    has_config = has_safetensors = has_pico = False
    try:
        main_files = set(api.list_repo_files(repo_id, revision="main"))
        has_config       = "config.json"      in main_files
        has_safetensors  = "model.safetensors" in main_files
        has_pico         = "pico_decoder.py"   in main_files
    except HfHubHTTPError:
        # No main branch, or branch is empty.
        pass

    return RepoAudit(
        repo_id=repo_id,
        condition=condition,
        expected=expected,
        actual=len(step_branches),
        max_step=max(step_branches) if step_branches else None,
        main_has_config=has_config,
        main_has_safetensors=has_safetensors,
        main_has_pico=has_pico,
    )


@click.command()
@click.option(
    "--org", "orgs", multiple=True, required=True,
    help="HF org(s) to audit. Repeatable. Each org must be a key of ORG_MAX_STEPS.",
)
@click.option(
    "--name-filter", default=None,
    help="Optional substring filter on repo names (e.g. 'deu-eng').",
)
def main(orgs: tuple[str, ...], name_filter: str | None) -> None:
    api = HfApi()
    bad_orgs = [o for o in orgs if o not in ORG_MAX_STEPS]
    if bad_orgs:
        raise click.UsageError(
            f"Unknown org(s) {bad_orgs}. Known: {sorted(ORG_MAX_STEPS)}"
        )

    rows: list[RepoAudit] = []
    for org in orgs:
        max_steps = ORG_MAX_STEPS[org]
        models = api.list_models(author=org)
        for m in models:
            if name_filter and name_filter not in m.id:
                continue
            try:
                rows.append(_audit_repo(api, m.id, max_steps))
            except Exception as e:
                print(f"[audit] {m.id}: error {e}", file=sys.stderr)

    # Print a compact, fixed-width table.
    header = (
        f"{'STATUS':<16}{'COND':<5}{'EXP':>5}{'GOT':>5}{'MAX':>8}"
        f"{'CFG':>5}{'SAFE':>6}{'PICO':>6}  REPO"
    )
    print(header)
    print("-" * len(header))
    for r in sorted(rows, key=lambda x: (x.status, x.repo_id)):
        print(
            f"{r.status:<16}"
            f"{(r.condition or '?'):<5}"
            f"{(r.expected if r.expected is not None else '-'):>5}"
            f"{r.actual:>5}"
            f"{(r.max_step if r.max_step is not None else '-'):>8}"
            f"{('Y' if r.main_has_config else 'N'):>5}"
            f"{('Y' if r.main_has_safetensors else 'N'):>6}"
            f"{('Y' if r.main_has_pico else 'N'):>6}"
            f"  {r.repo_id}"
        )

    # Summary by status.
    print()
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.status] = counts.get(r.status, 0) + 1
    for status, n in sorted(counts.items()):
        print(f"  {status}: {n}")


if __name__ == "__main__":
    main()
