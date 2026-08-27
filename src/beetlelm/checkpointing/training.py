#src/checkpointing/training.py

"""
Canonical checkpointing module for all trainers in the repo.

Both the pico trainer (dataclass configs) and the beetle trainer (dict configs)
go through this module. src/training/utils/checkpointing.py is a deprecated shim
that re-exports from here.

Checkpoint directory structure
-------------------------------
{runs_dir}/
    └── {run_name}/
        └── training_config.yaml
        └── checkpoints/
            ├── step_{N}/
            │   ├── config.json
            │   ├── model.safetensors
            │   ├── tokenizer.json
            │   ├── tokenizer_config.json
            │   └── fabric_checkpoint/
            │       └── checkpoint.pt
            └── latest -> step_{N}/
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import yaml
from huggingface_hub import upload_file, upload_folder
from lightning.fabric import Fabric
from lightning.fabric.strategies import DeepSpeedStrategy
from lightning.fabric.utilities.seed import _collect_rng_states, _set_rng_states
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import PreTrainedTokenizerBase

from beetlelm.training.utils.io import use_backoff

if TYPE_CHECKING:
    from beetlelm.config import CheckpointingConfig

try:
    from beetlelm.config import CheckpointingConfig  # type: ignore[assignment]
except ImportError:
    CheckpointingConfig = None  # type: ignore[assignment,misc]


# ─────────────────────────────────────────────────────────────────────────────
# Config normalisation
# ─────────────────────────────────────────────────────────────────────────────

class _CkptCfg:
    """Uniform attribute access over a CheckpointingConfig dataclass or a plain dict."""

    _DEFAULTS: Dict[str, Any] = {
        "runs_dir":                   "runs",
        "run_name":                   "run",
        "checkpoints_dir":            "checkpoints",
        "fabric_checkpoint_dir":      "fabric_checkpoint",
        "fabric_checkpoint_filename": "checkpoint.pt",
        "logs_dir":                   "logs",
        "save_to_hf":                 False,
        "keep_local_checkpoints":     1,
    }

    def __init__(self, raw: Any) -> None:
        self._d = raw
        self._is_dict = isinstance(raw, dict)

    def __getattr__(self, name: str) -> Any:
        if self._is_dict:
            if name in self._d:
                return self._d[name]
            if name in self._DEFAULTS:
                return self._DEFAULTS[name]
            raise AttributeError(f"CheckpointingConfig has no field '{name}'")
        return getattr(self._d, name)


def _resolve(configs_or_cfg: Any) -> _CkptCfg:
    """Accept a dataclass, a checkpointing sub-dict, or a top-level configs dict."""
    if isinstance(configs_or_cfg, dict):
        if "checkpointing" in configs_or_cfg:
            return _CkptCfg(configs_or_cfg["checkpointing"])
        return _CkptCfg(configs_or_cfg)
    return _CkptCfg(configs_or_cfg)



def delete_checkpoint(
    configs: Any,
    checkpoint_step: int,
    fabric: Optional[Fabric] = None,
) -> None:
    """
    Delete a single step checkpoint directory from disk.

    Used by CurriculumCheckpointScheduler after a successful HF Hub upload
    to free local disk space.

    Args:
        configs:          Top-level configs dict or CheckpointingConfig dataclass.
        checkpoint_step:  The step number whose directory should be removed.
        fabric:           Fabric instance for rank-gating (optional).
    """
    if fabric is not None and fabric.global_rank != 0:
        return

    cfg = _resolve(configs)
    root_ckpt = os.path.join(cfg.runs_dir, cfg.run_name, cfg.checkpoints_dir)
    ckpt_path = Path(root_ckpt) / f"step_{checkpoint_step}"

    if not ckpt_path.exists():
        return

    try:
        shutil.rmtree(ckpt_path)
        if fabric is not None:
            fabric.print(f"[checkpointing] Deleted local checkpoint {ckpt_path}")
        else:
            print(f"[checkpointing] Deleted local checkpoint {ckpt_path}")
    except OSError as exc:
        print(f"[checkpointing] Could not delete {ckpt_path}: {exc}")

    # Update the latest symlink if it was pointing at the deleted step
    latest = Path(root_ckpt) / "latest"
    if latest.is_symlink() and latest.readlink() == Path(f"step_{checkpoint_step}"):
        latest.unlink()

# ─────────────────────────────────────────────────────────────────────────────
# Local checkpoint pruning
# ─────────────────────────────────────────────────────────────────────────────

def prune_local_checkpoints(
    ckpt_dir: str,
    keep_last_n: int = 1,
    current_step: Optional[int] = None,
) -> None:
    """Keep only the most recent `keep_last_n` step checkpoints on disk.

    When ``current_step`` is supplied (the step we just finished saving),
    any on-disk ``step_N`` directory with ``N > current_step`` is treated
    as a leftover from a prior run that overshot this one — it is left
    alone *and* excluded from the keep_last_n quota.  Without this guard,
    a stale ``step_4/`` would rank higher than the freshly-written
    ``step_0/``, get kept by the prune, and the just-written step_0 would
    be deleted before the smoke test could load it back.
    """
    if keep_last_n <= 0:
        return

    ckpt_path = Path(ckpt_dir)
    if not ckpt_path.exists():
        return

    entries: list[tuple[int, Path]] = []
    for entry in ckpt_path.iterdir():
        if entry.is_symlink():
            continue
        m = re.match(r"^step_(\d+)(\.pt)?$", entry.name)
        if m:
            step_num = int(m.group(1))
            if current_step is not None and step_num > current_step:
                continue
            entries.append((step_num, entry))

    entries.sort(key=lambda x: x[0])
    to_delete = entries[:-keep_last_n] if len(entries) > keep_last_n else []

    for step_num, entry in to_delete:
        try:
            shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
        except OSError as exc:
            print(f"[prune_local_checkpoints] Could not delete {entry}: {exc}")

        tok_dir = ckpt_path / f"tokenizer_step_{step_num}"
        if tok_dir.exists():
            shutil.rmtree(tok_dir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# Save checkpoint
# ─────────────────────────────────────────────────────────────────────────────

@use_backoff()
def save_checkpoint(
    configs: Any,
    checkpoint_step: int,
    fabric: Fabric,
    model: nn.Module,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    pipeline_state: Any = None,
    upload_logs: bool = False,
) -> None:
    """Save model + optimiser + scheduler + optional pipeline state to disk."""
    cfg = _resolve(configs)

    run_path  = os.path.join(cfg.runs_dir, cfg.run_name)
    root_ckpt = os.path.join(run_path, cfg.checkpoints_dir)
    ckpt_path = os.path.join(root_ckpt, f"step_{checkpoint_step}")
    os.makedirs(ckpt_path, exist_ok=True)

    # HF model (pico trainer only)
    hf_model = None
    if fabric.global_rank == 0 and hasattr(model, "convert_to_hf_model"):
        hf_model = model.convert_to_hf_model()
        hf_model.save_pretrained(ckpt_path)

    # Tokenizer
    if tokenizer is not None and fabric.global_rank == 0:
        tokenizer.save_pretrained(ckpt_path)

    # Fabric state dict
    fabric_dir = os.path.join(ckpt_path, cfg.fabric_checkpoint_dir)
    os.makedirs(fabric_dir, exist_ok=True)

    state: Dict[str, Any] = {
        "_model":           model,
        "_optimizer":       optimizer,
        "_lr_scheduler":    lr_scheduler,
        "_checkpoint_step": checkpoint_step,
    }

    if pipeline_state is not None:
        state["_pipeline_state"] = (
            pipeline_state.state_dict()
            if hasattr(pipeline_state, "state_dict")
            else pipeline_state
        )

    if not isinstance(fabric.strategy, DeepSpeedStrategy):
        state["_rng_states"] = _collect_rng_states()
        save_file = os.path.join(fabric_dir, cfg.fabric_checkpoint_filename)
    else:
        save_file = fabric_dir

    fabric.save(save_file, state)

    if fabric.global_rank == 0:
        # training_config.yaml (written once)
        config_path = os.path.join(run_path, "training_config.yaml")
        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                yaml.dump(configs if isinstance(configs, dict) else {}, f)

        # ── Atomic-completion sentinel ────────────────────────────────────
        # All payload writes (HF model, tokenizer, fabric state) finished
        # above. Drop a `.complete` marker file LAST and fsync the directory
        # so a SIGKILL / power-cut / disk-full between this point and the
        # next save leaves the checkpoint either fully visible or marked
        # incomplete — never half-loaded by a future resume. load_checkpoint
        # filters out any step_N/ without this sentinel.
        try:
            complete_path = os.path.join(ckpt_path, ".complete")
            with open(complete_path, "w") as f:
                f.write(f"step={checkpoint_step}\n")
                f.flush()
                os.fsync(f.fileno())
            # Fsync the parent dir so the metadata entry is durable.
            try:
                dir_fd = os.open(ckpt_path, os.O_RDONLY)
                os.fsync(dir_fd)
                os.close(dir_fd)
            except OSError:
                pass
        except Exception as _e:
            # If we cannot write the sentinel, the checkpoint is unsafe.
            # Surface a loud warning — but do not raise, because tearing
            # down an in-flight 8-rank training mid-save would be worse
            # than a missing sentinel (next resume will simply pick an
            # earlier complete step).
            try:
                fabric.print(
                    f"[checkpointing][WARN] failed to write .complete sentinel "
                    f"at {ckpt_path}: {_e!r}. This checkpoint will be "
                    f"treated as incomplete on next resume."
                )
            except Exception:
                pass

        # latest symlink — point at the just-completed step
        latest = os.path.join(root_ckpt, "latest")
        if os.path.lexists(latest):
            os.remove(latest)
        os.symlink(f"step_{checkpoint_step}", latest, target_is_directory=True)

        # HF Hub push (pico trainer only)
        if cfg.save_to_hf:
            repo_id: Optional[str] = None
            try:
                repo_id = cfg._d.hf_checkpoint.repo_id if not cfg._is_dict else None
            except AttributeError:
                pass

            if repo_id:
                if hf_model is not None:
                    hf_model.push_to_hub(
                        repo_id=repo_id,
                        commit_message=f"Saving HF Model -- Step {checkpoint_step}",
                        revision=cfg.run_name,
                        token=os.getenv("HF_TOKEN"),
                    )
                if checkpoint_step == 0:
                    if tokenizer is not None:
                        tokenizer.push_to_hub(
                            repo_id=repo_id,
                            commit_message=f"Saving Tokenizer -- Step {checkpoint_step}",
                            revision=cfg.run_name,
                            token=os.getenv("HF_TOKEN"),
                        )
                    upload_file(
                        path_or_fileobj=config_path,
                        path_in_repo="training_config.yaml",
                        repo_id=repo_id,
                        commit_message=f"Saving Training Config -- Step {checkpoint_step}",
                        revision=cfg.run_name,
                        token=os.getenv("HF_TOKEN"),
                    )
                upload_folder(
                    folder_path=fabric_dir,
                    path_in_repo=cfg.fabric_checkpoint_dir,
                    repo_id=repo_id,
                    commit_message=f"Saving Fabric Checkpoint -- Step {checkpoint_step}",
                    revision=cfg.run_name,
                    token=os.getenv("HF_TOKEN"),
                )
                if upload_logs:
                    upload_folder(
                        folder_path=os.path.join(run_path, cfg.logs_dir),
                        path_in_repo=cfg.logs_dir,
                        repo_id=repo_id,
                        commit_message=f"Saving Logs -- Step {checkpoint_step}",
                        revision=cfg.run_name,
                        token=os.getenv("HF_TOKEN"),
                    )

        prune_local_checkpoints(
            root_ckpt,
            cfg.keep_local_checkpoints,
            current_step=checkpoint_step,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Load checkpoint
# ─────────────────────────────────────────────────────────────────────────────

@use_backoff()
def load_checkpoint(
    checkpointing_config: Any,
    checkpoint_step: Union[str, int],
    fabric: Fabric,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    lr_scheduler: Optional[LRScheduler] = None,
    pipeline_state: Any = None,
) -> Optional[Tuple]:
    """
    Load checkpoint from disk.

    Returns (model, optimizer, lr_scheduler, extra_state_dict) or None if not found.
    """
    cfg = _resolve(checkpointing_config)

    if isinstance(checkpoint_step, int):
        checkpoint_step = f"step_{checkpoint_step}"

    root = os.path.join(cfg.runs_dir, cfg.run_name, cfg.checkpoints_dir)

    # Filter step dirs to only those carrying the atomic-completion sentinel.
    # A step_N/ without .complete was interrupted mid-save (SIGKILL,
    # disk-full, watchdog kill) and may contain torn fabric_checkpoint files.
    # Loading it can crash silently or — worse — succeed with garbage state.
    def _is_complete(_step_dir: str) -> bool:
        return os.path.isfile(os.path.join(_step_dir, ".complete"))

    if checkpoint_step == "latest":
        latest = os.path.join(root, "latest")
        chosen = None
        if os.path.islink(latest):
            candidate = os.readlink(latest)
            if _is_complete(os.path.join(root, candidate)):
                chosen = candidate
            else:
                fabric.print(
                    f"[checkpointing] latest -> {candidate} is incomplete "
                    f"(no .complete sentinel); scanning for fallback."
                )
        if chosen is None:
            steps = [
                int(m.group(1))
                for d in (os.listdir(root) if os.path.exists(root) else [])
                if (m := re.match(r"^step_(\d+)$", d))
                and _is_complete(os.path.join(root, d))
            ]
            if not steps:
                fabric.print(
                    "[checkpointing] No COMPLETE checkpoints found, starting fresh."
                )
                return None
            chosen = f"step_{max(steps)}"
        checkpoint_step = chosen

    ckpt_path = os.path.join(root, checkpoint_step)
    if not os.path.exists(ckpt_path):
        fabric.print(f"[checkpointing] {ckpt_path} not found, starting fresh.")
        return None
    if not _is_complete(ckpt_path):
        fabric.print(
            f"[checkpointing] {ckpt_path} is missing .complete sentinel — "
            f"refusing to load possibly-torn state. Starting fresh."
        )
        return None

    fabric_dir = os.path.join(ckpt_path, cfg.fabric_checkpoint_dir)
    load_file = (
        os.path.join(fabric_dir, cfg.fabric_checkpoint_filename)
        if not isinstance(fabric.strategy, DeepSpeedStrategy)
        else fabric_dir
    )

    state: Dict[str, Any] = {
        "_model":        model,
        "_optimizer":    optimizer,
        "_lr_scheduler": lr_scheduler,
    }
    extra = fabric.load(load_file, state=state)

    if pipeline_state is not None and "_pipeline_state" in extra:
        if hasattr(pipeline_state, "load_state_dict"):
            pipeline_state.load_state_dict(extra["_pipeline_state"])
        fabric.print(
            f"[checkpointing] Restored pipeline state "
            f"(step {getattr(pipeline_state, 'step_counter', '?')})"
        )

    if "_rng_states" in extra:
        _set_rng_states(extra["_rng_states"])

    fabric.print(f"[checkpointing] Loaded checkpoint from {ckpt_path}")
    return model, optimizer, lr_scheduler, extra


# ─────────────────────────────────────────────────────────────────────────────
# Learning dynamics  (pico trainer — no-op stubs for beetle compatibility)
# ─────────────────────────────────────────────────────────────────────────────

def compute_learning_dynamics_states(
    configs: Any = None,
    checkpointing_config: Any = None,
    checkpoint_step: Optional[int] = None,
    fabric: Optional[Fabric] = None,
    model: Optional[nn.Module] = None,
    dataset: Any = None,
    compute_gradients: bool = False,
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    """
    Compute learning-dynamics states (activations, weights, gradients).

    Delegates to the real ``CheckpointStateExtractor`` in
    ``src/checkpointing/learning_dynamics`` when enabled.
    """
    raw = configs if configs is not None else checkpointing_config
    if raw is None:
        return None

    cfg = _resolve(raw)
    ld = getattr(cfg._d, "learning_dynamics", None)
    if ld is None:
        ld_dict = cfg._d.get("learning_dynamics", {}) if cfg._is_dict else None
        if not ld_dict or not ld_dict.get("enabled", False):
            return None
    elif not getattr(ld, "enabled", False):
        return None

    if fabric is None or model is None or dataset is None:
        return None

    # Build LD config from dict or dataclass
    from beetlelm.config.checkpointing_config import LearningDynamicsCheckpointingConfig

    if isinstance(ld, LearningDynamicsCheckpointingConfig):
        ld_config = ld
    elif isinstance(ld, dict):
        ld_config = LearningDynamicsCheckpointingConfig(**{
            k: v for k, v in ld.items()
            if k in LearningDynamicsCheckpointingConfig.__dataclass_fields__
        })
    else:
        # dict-based config path
        ld_dict = cfg._d.get("learning_dynamics", {}) if cfg._is_dict else {}
        ld_config = LearningDynamicsCheckpointingConfig(**{
            k: v for k, v in ld_dict.items()
            if k in LearningDynamicsCheckpointingConfig.__dataclass_fields__
        })

    from beetlelm.checkpointing.learning_dynamics import (
        compute_learning_dynamics_states as _real_compute,
    )

    # Build a minimal checkpointing config object for the real function
    _ckpt_cfg = type("_Cfg", (), {
        "learning_dynamics": ld_config,
        "runs_dir": cfg.runs_dir,
        "run_name": cfg.run_name,
        "checkpoints_dir": cfg.checkpoints_dir,
    })()

    fabric.print(
        f"[learning_dynamics] Extracting states at step {checkpoint_step}…"
    )
    return _real_compute(
        checkpointing_config=_ckpt_cfg,
        fabric=fabric,
        model=model,
        dataset=dataset,
        compute_gradients=compute_gradients,
    )


def save_learning_dynamics_states(
    configs: Any = None,
    checkpointing_config: Any = None,
    checkpoint_step: Optional[int] = None,
    prefix: str = "train",
    fabric: Optional[Fabric] = None,
    learning_dynamics_states: Optional[Dict[str, Any]] = None,
    learning_dynamics_dataset: Any = None,
    tokenizer: Any = None,
    **kwargs: Any,
) -> None:
    """
    Persist learning-dynamics states to disk (and optionally HF Hub).

    Delegates to the real ``save_learning_dynamics_states`` in
    ``src/checkpointing/learning_dynamics`` when enabled.
    """
    raw = configs if configs is not None else checkpointing_config
    if raw is None:
        return

    cfg = _resolve(raw)
    ld = getattr(cfg._d, "learning_dynamics", None)
    if ld is None:
        ld_dict = cfg._d.get("learning_dynamics", {}) if cfg._is_dict else None
        if not ld_dict or not ld_dict.get("enabled", False):
            return
    elif not getattr(ld, "enabled", False):
        return

    if not learning_dynamics_states:
        return

    from beetlelm.config.checkpointing_config import LearningDynamicsCheckpointingConfig

    if isinstance(ld, LearningDynamicsCheckpointingConfig):
        ld_config = ld
    elif isinstance(ld, dict):
        ld_config = LearningDynamicsCheckpointingConfig(**{
            k: v for k, v in ld.items()
            if k in LearningDynamicsCheckpointingConfig.__dataclass_fields__
        })
    else:
        ld_dict = cfg._d.get("learning_dynamics", {}) if cfg._is_dict else {}
        ld_config = LearningDynamicsCheckpointingConfig(**{
            k: v for k, v in ld_dict.items()
            if k in LearningDynamicsCheckpointingConfig.__dataclass_fields__
        })

    from beetlelm.checkpointing.learning_dynamics import (
        save_learning_dynamics_states as _real_save,
    )

    # Build a minimal checkpointing config for the real function
    _ckpt_cfg = type("_Cfg", (), {
        "learning_dynamics": ld_config,
        "learning_dynamics_dir": ld_config.learning_dynamics_dir,
        "runs_dir": cfg.runs_dir,
        "run_name": cfg.run_name,
        "checkpoints_dir": cfg.checkpoints_dir,
        "save_to_hf": ld_config.push_to_hf,
        "hf_checkpoint": getattr(cfg._d, "hf_checkpoint", None)
            if not cfg._is_dict
            else type("_HF", (), {"repo_id": None})(),
    })()

    fabric.print(
        f"[learning_dynamics] Saving states at step {checkpoint_step} "
        f"prefix={prefix}…"
    )
    _real_save(
        checkpointing_config=_ckpt_cfg,
        checkpoint_step=checkpoint_step,
        prefix=prefix,
        fabric=fabric,
        learning_dynamics_states=learning_dynamics_states,
        learning_dynamics_dataset=learning_dynamics_dataset,
        tokenizer=tokenizer,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation results  (pico trainer — no-op stub for beetle compatibility)
# ─────────────────────────────────────────────────────────────────────────────

def save_evaluation_results(
    configs: Any = None,
    checkpointing_config: Any = None,
    checkpoint_step: Optional[int] = None,
    fabric: Optional[Fabric] = None,
    evaluation_results: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> None:
    """
    Persist evaluation results alongside the checkpoint.

    Writes a JSON file at:
        {runs_dir}/{run_name}/checkpoints/step_{N}/evaluation_results.json

    Beetle trainer: no-op if evaluation_results is None or empty.
    """
    if not evaluation_results:
        return

    raw = configs if configs is not None else checkpointing_config
    if raw is None:
        return

    cfg = _resolve(raw)

    if fabric is not None and fabric.global_rank != 0:
        return

    ckpt_path = os.path.join(
        cfg.runs_dir, cfg.run_name, cfg.checkpoints_dir,
        f"step_{checkpoint_step}",
    )
    os.makedirs(ckpt_path, exist_ok=True)

    out_path = os.path.join(ckpt_path, "evaluation_results.json")
    with open(out_path, "w") as f:
        json.dump(evaluation_results, f, indent=2)

    if fabric is not None:
        fabric.print(f"[checkpointing] Saved evaluation results to {out_path}")
