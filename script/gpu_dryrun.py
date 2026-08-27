#!/usr/bin/env python3
"""
BeetleLM GPU Dry-Run

Standalone dry-run that trains a tiny PicoDecoder on GPU (8x CUDA DDP,
bf16-mixed) for a handful of steps and (optionally) pushes checkpoints to
HF Hub so that both `beetle-analyze` (step-{N} branches) and `pico-analyze`
(commit history on `main`) can consume the resulting repo.

Invariant: this script does NOT touch beetle's dynamic phase-aware
checkpoint schedule. It only forces the fabric block onto GPU. Checkpoint
steps are whatever `CurriculumCheckpointScheduler` emits for the config's
curriculum + max_steps. For the default `configs/dryrun_gpu.yaml`
(balanced, max_steps=6), the natural schedule is [0, 1, 2, 4, 6].

Usage:
    python script/gpu_dryrun.py \\
        --config_path configs/dryrun_gpu.yaml

    # skip HF push AND run analyzers in local mode
    python script/gpu_dryrun.py --config_path configs/dryrun_gpu.yaml --skip_hf

    # skip analysis subprocesses entirely
    python script/gpu_dryrun.py --config_path configs/dryrun_gpu.yaml --skip_analysis
"""

from __future__ import annotations

import os
import sys
import subprocess
import tempfile
from pathlib import Path

# ensure project root is on the path so `from beetlelm.*` works
os.environ["TOKENIZERS_PARALLELISM"] = "false"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import click  # noqa: E402
import yaml  # noqa: E402

from beetlelm.training.beetle_trainer import Trainer  # noqa: E402


GPU_FABRIC_OVERRIDES = {
    "accelerator": "cuda",
    "devices": 8,
    "strategy": "ddp",
    "precision": "bf16-mixed",
    "num_nodes": 1,
}


def _apply_gpu_fabric_overrides(configs: dict) -> None:
    """Force the fabric block to GPU. Touches NOTHING else."""
    training = configs.setdefault("training", {})
    fabric = training.setdefault("fabric", {})
    fabric.update(GPU_FABRIC_OVERRIDES)


def _materialise_gpu_config(src_config_path: Path, skip_hf: bool) -> Path:
    """Load YAML, apply GPU fabric overrides (+ optional skip_hf), write to a
    temp file, and return the temp path. Leaves the original config untouched."""
    with open(src_config_path, "r") as f:
        cfg = yaml.safe_load(f)

    _apply_gpu_fabric_overrides(cfg)
    if skip_hf:
        cfg.setdefault("checkpointing", {})["save_to_hf"] = False

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix="_gpu_dryrun.yaml",
        delete=False,
        dir=str(src_config_path.parent),
    )
    yaml.safe_dump(cfg, tmp, sort_keys=False)
    tmp.close()
    return Path(tmp.name)


def _run_beetle_analyze(hf_user: str, run_name: str, skip_hf: bool) -> int:
    """Invoke beetle-analyze as a subprocess in either HF or local mode.

    beetle-analyze.py reads MODE / HF_USER / HF_MODELS / LOCAL_CHECKPOINT_DIR as
    module-level globals. We load the module via importlib (which runs its top
    level and sets the defaults), then overwrite those globals and call main().
    """
    beetle_analyze = PROJECT_ROOT.parent / "beetle-analyze" / "beetle-analyze.py"
    if not beetle_analyze.exists():
        print(f"[gpu_dryrun] beetle-analyze.py not found at {beetle_analyze} — skipping.")
        return 0

    mode_value = "local" if skip_hf else "hf"
    local_dir = str((PROJECT_ROOT / "runs" / run_name).resolve())
    launcher_lines = [
        "import importlib.util, pathlib, sys",
        f"sys.path.insert(0, {str(beetle_analyze.parent)!r})",
        f"spec = importlib.util.spec_from_file_location('beetle_analyze', {str(beetle_analyze)!r})",
        "mod = importlib.util.module_from_spec(spec)",
        "spec.loader.exec_module(mod)",
        f"mod.MODE = {mode_value!r}",
        f"mod.HF_USER = {hf_user!r}",
        f"mod.HF_MODELS = [{run_name!r}]",
        f"mod.LOCAL_CHECKPOINT_DIR = pathlib.Path({local_dir!r})",
        "mod.LOCAL_GLOB = 'step_*'",
        "mod.main()",
    ]
    launcher = "\n".join(launcher_lines)
    env = os.environ.copy()
    print(f"[gpu_dryrun] Running beetle-analyze (mode={mode_value})…")
    return subprocess.call([sys.executable, "-c", launcher], env=env)


def _run_pico_analyze(
    hf_user: str, run_name: str, skip_hf: bool, analysis_config: Path
) -> int:
    """Invoke pico-analyze as a subprocess in either HF or local mode."""
    pico_analyze_root = PROJECT_ROOT.parent / "pico-analyze"
    analyze_script = pico_analyze_root / "scripts" / "analyze.py"
    if not analyze_script.exists():
        print(f"[gpu_dryrun] pico-analyze not found at {analyze_script} — skipping.")
        return 0
    if not analysis_config.exists():
        print(f"[gpu_dryrun] analysis config not found at {analysis_config} — skipping.")
        return 0

    cmd = [sys.executable, str(analyze_script), "--config_path", str(analysis_config)]
    if skip_hf:
        run_path = (PROJECT_ROOT / "runs" / run_name).resolve()
        cmd.extend(["--run_path", str(run_path), "--branch", "main"])
    else:
        cmd.extend(["--repo_id", f"{hf_user}/{run_name}"])

    print(f"[gpu_dryrun] Running pico-analyze: {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(pico_analyze_root))


@click.command()
@click.option(
    "--config_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=Path("configs/dryrun_gpu.yaml"),
    show_default=True,
    help="Path to the dry-run training configuration.",
)
@click.option(
    "--skip_hf",
    is_flag=True,
    default=False,
    help="Force save_to_hf=false and run analyzers in local mode.",
)
@click.option(
    "--skip_analysis",
    is_flag=True,
    default=False,
    help="Skip the beetle-analyze + pico-analyze subprocess calls.",
)
@click.option(
    "--pico_analysis_config",
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to pico-analyze metrics config. Defaults to "
         "../pico-analyze/configs/beetle_dryrun.yaml.",
)
def main(
    config_path: Path,
    skip_hf: bool,
    skip_analysis: bool,
    pico_analysis_config: Path | None,
) -> None:
    # Peek at the config to learn run_name / hf_user for the analyzer subprocesses.
    with open(config_path, "r") as f:
        raw_cfg = yaml.safe_load(f)
    run_name = raw_cfg.get("run_name", "beetlelm_gpu_dryrun")
    hf_user = raw_cfg.get("hf_user", "Beetle-HumanScale")

    print(f"[gpu_dryrun] run_name={run_name} hf_user={hf_user}")
    print(f"[gpu_dryrun] skip_hf={skip_hf} skip_analysis={skip_analysis}")

    # Sanity-check the invariant: no explicit checkpoint_steps override in the
    # YAML on disk. Beetle's dynamic schedule must drive the run.
    if raw_cfg.get("checkpointing", {}).get("checkpoint_steps"):
        print(
            "[gpu_dryrun] WARNING: checkpoint_steps is set in the config. The GPU "
            "dry-run is meant to exercise beetle's DYNAMIC phase-aware schedule; "
            "consider removing the explicit list.",
            file=sys.stderr,
        )

    # Materialise a GPU-forced copy of the config to a temp YAML. Trainer reads
    # fabric settings during __init__, so we must patch before construction.
    gpu_cfg_path = _materialise_gpu_config(config_path, skip_hf=skip_hf)
    print(f"[gpu_dryrun] Materialised GPU config → {gpu_cfg_path}")

    try:
        trainer = Trainer(config_path=str(gpu_cfg_path))
        # Train (real training loop — NOT the --dry_run shortcut).
        trainer.train()
    finally:
        try:
            gpu_cfg_path.unlink()
        except OSError:
            pass

    if skip_analysis:
        print("[gpu_dryrun] --skip_analysis set; done.")
        return

    # Resolve pico-analyze config default.
    if pico_analysis_config is None:
        pico_analysis_config = (
            PROJECT_ROOT.parent / "pico-analyze" / "configs" / "beetle_dryrun.yaml"
        )

    # Run analyzers. Each returns 0 on success; we surface a non-zero rc
    # without aborting the whole script.
    rc_beetle = _run_beetle_analyze(hf_user=hf_user, run_name=run_name, skip_hf=skip_hf)
    rc_pico = _run_pico_analyze(
        hf_user=hf_user,
        run_name=run_name,
        skip_hf=skip_hf,
        analysis_config=pico_analysis_config,
    )

    print("\n[gpu_dryrun] === Summary ===")
    print(f"  Local run dir : {PROJECT_ROOT}/runs/{run_name}")
    if not skip_hf:
        print(f"  HF repo       : https://huggingface.co/{hf_user}/{run_name}")
        print(f"    step-{{N}} branches     → beetle-analyze")
        print(f"    `main` commit history → pico-analyze")
    print(f"  beetle-analyze rc : {rc_beetle}")
    print(f"  pico-analyze   rc : {rc_pico}")


if __name__ == "__main__":
    main()
