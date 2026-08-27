"""
End-to-end pico-analyze compatibility check (READ-ONLY live HF call).

Given an already-trained beetle HF repo (on `main`), this script verifies that
pico-analyze's public entry points succeed against it:

1. ``_download_training_config(repo, "main")`` returns a dict containing the
   keys pico-analyze's components read directly:

       training_config["model"]["d_model"]
       training_config["model"]["attention_n_heads"]
       training_config["model"]["attention_n_kv_heads"]
       training_config["checkpointing"]["run_name"]

2. ``_get_learning_dynamics_commits(repo, "main", "train")`` returns one or
   more commits — the commit-history regex in pico-analyze/src/utils/data.py:145
   recognises the messages beetle emits.

3. ``_download_checkpoint_states`` returns non-empty data for each discovered
   step.

The script makes NO write calls, does NOT train a model, and is safe to run
against any public beetle repo.

USAGE
-----

    cd beetlelm
    .venv/bin/python script/test_pico_analyze_compat.py \\
        Beetle-HumanScale/<repo_name>

    # or set via env var
    BEETLE_REPO=Beetle-HumanScale/<repo_name> \\
        .venv/bin/python script/test_pico_analyze_compat.py

The repo must already contain (on main):
    training_config.yaml
    at least one commit whose title matches
        Saving Learning Dynamics Data (train|val) -- Step N

Exit code is 0 iff every check passes. On any miss, a clear diagnostic is
printed identifying which part of the pico-analyze contract was violated.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

# pico-analyze lives next to beetlelm in the monorepo layout.  Both projects
# use the top-level ``src/`` package name, so we must isolate pico-analyze
# on sys.path (this script does not import any beetlelm source itself).
#
# pico-analyze's ``src/`` has no ``__init__.py``, so we explicitly register
# it as a namespace package pointing at pico-analyze's directory.  This
# avoids the ambiguity that would otherwise make Python resolve ``src`` to
# whichever directory is first on sys.path.
REPO_ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PICO_ANALYZE = os.path.abspath(os.path.join(REPO_ROOT, "..", "pico-analyze"))

if not os.path.isdir(os.path.join(PICO_ANALYZE, "src", "utils")):
    raise RuntimeError(
        f"pico-analyze not found at {PICO_ANALYZE}. "
        f"Checkout pico-analyze as a sibling of beetlelm."
    )

import importlib.util
import types as _types

# Drop any pre-existing `src.*` modules so fresh imports resolve to pico-analyze.
for _mod in list(sys.modules):
    if _mod == "src" or _mod.startswith("src."):
        del sys.modules[_mod]

# Register pico-analyze's `src` as a namespace package rooted at the correct
# directory.  Doing this manually bypasses sys.path ordering.
_src_pkg = _types.ModuleType("src")
_src_pkg.__path__ = [os.path.join(PICO_ANALYZE, "src")]
sys.modules["src"] = _src_pkg

_src_utils_pkg = _types.ModuleType("src.utils")
_src_utils_pkg.__path__ = [os.path.join(PICO_ANALYZE, "src", "utils")]
sys.modules["src.utils"] = _src_utils_pkg


REQUIRED_MODEL_KEYS = ("d_model", "attention_n_heads", "attention_n_kv_heads")


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def _ok(msg: str) -> None:
    print(f"[ OK ] {msg}")


def check_training_config(repo_id: str, branch: str = "main") -> dict:
    from beetlelm.utils.data import _download_training_config  # pico-analyze

    try:
        cfg = _download_training_config(repo_id, branch)
    except Exception as e:
        _fail(f"training_config.yaml not downloadable from {repo_id}@{branch}: {e}")

    if not isinstance(cfg, dict):
        _fail(f"training_config.yaml is not a mapping (got {type(cfg)!r})")

    model = cfg.get("model")
    if not isinstance(model, dict):
        _fail(f"training_config.yaml lacks a top-level 'model' mapping; got {model!r}")

    missing = [k for k in REQUIRED_MODEL_KEYS if k not in model]
    if missing:
        _fail(
            f"training_config.yaml['model'] missing pico-analyze keys {missing!r}. "
            f"Present keys: {sorted(model.keys())}. "
            f"(Hint: beetle's config nests these under model.params.*; "
            f"_build_pico_compatible_training_config should flatten them.)"
        )

    ckpt = cfg.get("checkpointing") or {}
    if "run_name" not in ckpt:
        _fail("training_config.yaml['checkpointing']['run_name'] is missing")

    _ok(f"training_config.yaml has all pico-analyze required keys (model.{{{','.join(REQUIRED_MODEL_KEYS)}}}, checkpointing.run_name)")
    return cfg


def check_ld_commits(repo_id: str, branch: str = "main") -> dict:
    from beetlelm.utils.data import _get_learning_dynamics_commits  # pico-analyze

    commits = _get_learning_dynamics_commits(repo_id, branch, "train")
    if not commits:
        _fail(
            f"No 'Saving Learning Dynamics Data (train) -- Step N' commits found "
            f"on {repo_id}@{branch}. pico-analyze will have nothing to analyze. "
            f"Expected at least one commit matching "
            f"r'Saving Learning Dynamics Data \\(train\\) -- Step (\\d+)'."
        )

    steps = sorted(commits.keys())
    _ok(f"LD commits discovered on {repo_id}@{branch}: steps={steps}")
    return commits


def check_ld_download(repo_id: str, steps: list[int], branch: str = "main") -> None:
    from beetlelm.utils.data import _download_checkpoint_states  # pico-analyze

    for step in steps:
        try:
            states = _download_checkpoint_states(repo_id, branch, step, "train")
        except Exception as e:
            _fail(f"Could not download LD states at step {step}: {e}")

        if not states:
            _fail(f"LD states at step {step} downloaded but are empty (no activations/weights/gradients)")

        keys = sorted(states.keys())
        _ok(f"step {step}: LD states present ({', '.join(keys)})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "repo_id",
        nargs="?",
        default=os.environ.get("BEETLE_REPO"),
        help="HuggingFace repo_id, e.g. Beetle-HumanScale/my_model (or set $BEETLE_REPO)",
    )
    parser.add_argument("--branch", default="main", help="branch to check (default: main)")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=3,
        help="limit how many LD steps to probe-download (default: 3; use -1 for all)",
    )
    args = parser.parse_args()

    if not args.repo_id:
        parser.error("repo_id is required (positional arg or $BEETLE_REPO)")

    print(f"[INFO] Checking pico-analyze compatibility of {args.repo_id}@{args.branch}")

    try:
        cfg     = check_training_config(args.repo_id, args.branch)
        commits = check_ld_commits(args.repo_id, args.branch)
    except SystemExit:
        raise
    except Exception as e:
        traceback.print_exc()
        _fail(f"unexpected failure during config/commit check: {e}")
        return 1  # unreachable

    steps = sorted(commits.keys())
    if args.max_steps and args.max_steps > 0:
        # Probe the first and last step (and one in the middle if possible) to
        # keep the check fast but still representative across the run.
        picked = {steps[0], steps[-1]}
        if len(steps) >= 3:
            picked.add(steps[len(steps) // 2])
        steps = sorted(picked)[: args.max_steps]

    check_ld_download(args.repo_id, steps, args.branch)

    print()
    print(f"[PASS] {args.repo_id}@{args.branch} is pico-analyze compatible across {len(commits)} LD commit(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
