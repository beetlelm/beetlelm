"""
Utilities for initializing components of the training process.

Here, we initialize all of the components that are part of the learning process. From logging,
and checkpointing to the optimizer to the dataset and the dataloader, this file contains the
logic for setting up the classes and functions that are used in the training loop.

As always, this code is meant to be basic. We hard-code the obvious defaults, and leave the
more experimental stuff to you.
"""

import logging
import os
import warnings
from dataclasses import fields, is_dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional, Union
from pathlib import Path
import lightning as L
import torch
import wandb
import yaml
from datasets import Dataset, DownloadConfig, load_dataset
from datasets import config as datasets_config
from huggingface_hub import add_collection_item, create_branch, create_repo
from lightning.fabric.loggers import Logger as FabricLogger
from lightning.fabric.utilities.rank_zero import rank_zero_only
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from wandb.integration.lightning.fabric import WandbLogger
from beetlelm.checkpointing.curriculum_checkpoint import CheckpointingConfig
from beetlelm.config.monitoring_config import MonitoringConfig
from beetlelm.config.training_config import TrainingConfig
from beetlelm.training.utils.logging import initialize_logging
from beetlelm.pico_decoder.pico_decoder import PicoDecoder
from beetlelm.training.utils.io import use_backoff
from beetlelm.checkpointing import CheckpointingConfig

import yaml
from dataclasses import fields, is_dataclass, replace
from typing import Dict, Union, Optional
from beetlelm.config.training_config import TrainingConfig
from beetlelm.config.monitoring_config import MonitoringConfig
from beetlelm.checkpointing.curriculum_checkpoint import CheckpointingConfig

warnings.filterwarnings(
    "ignore",
    message=".*This integration is tested and supported for lightning Fabric.*",
)
warnings.filterwarnings(
    "ignore",
    message=".*Please report any issues to.*",
)




def _apply_config_overrides(dataclass_obj, overrides: dict):
    """
    Update a dataclass instance with a dictionary of overrides,
    filling only the fields that exist in the dataclass.
    Missing fields remain as their default values.
    """
    if not is_dataclass(dataclass_obj):
        # Not a dataclass, just return the dictionary itself
        return overrides

    dataclass_fields = {f.name for f in fields(dataclass_obj)}
    valid_overrides = {k: v for k, v in overrides.items() if k in dataclass_fields}

    # Replace only the valid fields
    return replace(dataclass_obj, **valid_overrides)


def initialize_configuration(config_path: str = None):
    # 1. Initialize Default Dataclass Instances
    # These must be assigned to the variable names used in the final dict
    training_config = TrainingConfig()
    monitoring_config = MonitoringConfig()
    checkpointing_config = CheckpointingConfig()
    
    overrides = {}
    if config_path:
        with open(config_path, "r") as f:
            overrides = yaml.safe_load(f) or {}

    # 2. Extract top-level metadata
    run_name = overrides.get("run_name")
    hf_user = overrides.get("hf_user", "BeetleLM")

    # 3. Apply overrides and RE-ASSIGN to the variables
    # This ensures monitoring_config and checkpointing_config are defined
    training_config = _apply_config_overrides(training_config, overrides.get("training", {}))
    monitoring_config = _apply_config_overrides(monitoring_config, overrides.get("monitoring", {}))
    checkpointing_config = _apply_config_overrides(checkpointing_config, overrides.get("checkpointing", {}))

    # 4. Construct the final dictionary.
    # Preserve every top-level YAML key (including manipulation blocks like
    # `tilt`, `layer_growth`, `ewc`, `working_memory`, `sas_regularizer`,
    # `lang_id`, `alignment_loss`, `lora`, `kl_distillation`, `pre_pretraining`,
    # `gaussian_noise`, `polca`, `curriculum`, `generative_replay`,
    # `similarity_replay`, `maml`); previously these were silently dropped by a
    # hard-coded allowlist, so every manipulation builder saw `cfg={}` and
    # returned None — see beetlelm/src/continual/<name>.py builders that gate
    # on `cfg.get("enabled", False)`.
    configs = dict(overrides)
    configs.update({
        "run_name": run_name,
        "hf_user": hf_user,
        "training": training_config,
        "monitoring": monitoring_config,
        "checkpointing": checkpointing_config,
        "tokenizer": overrides.get("tokenizer", {}),
        "model": overrides.get("model", {}),
        "data": overrides.get("data", {}),
    })

    return configs

# -------------------------------
# Initialize run directory safely
# -------------------------------
def initialize_run_dir(checkpointing_config: CheckpointingConfig) -> str:
    """
    Initialize a directory for the current run.

    Args:
        checkpointing_config (CheckpointingConfig): Dataclass containing checkpointing info.

    Returns:
        str: Path to the run directory.
    """
    if checkpointing_config is None:
        raise ValueError("checkpointing_config cannot be None")

    # Ensure run_name exists
    if not checkpointing_config.run_name:
        checkpointing_config.run_name = "beetlelm_run"

    # Use the runs_dir from the dataclass
    runs_dir = Path(checkpointing_config.runs_dir)
    os.makedirs(runs_dir, exist_ok=True)

    # Construct run directory path
    run_dir = runs_dir / checkpointing_config.run_name
    os.makedirs(run_dir, exist_ok=True)

    # Optional: make timestamped folder if multiple runs needed
    counter = 1
    base_run_dir = run_dir
    while run_dir.exists() and any(run_dir.iterdir()):
        run_dir = Path(f"{base_run_dir}_{counter}")
        counter += 1
        os.makedirs(run_dir, exist_ok=True)

    print(f"[RunDir] Using run directory: {run_dir}")
    return str(run_dir)



def initialize_fabric(training_config, wandb_logger=None):
    """
    Initialize Lightning Fabric. 
    Handles both Dataclass and Dictionary inputs for training_config.
    """
    # 1. Safely extract the fabric configuration block
    if isinstance(training_config, dict):
        fabric_cfg = training_config.get("fabric", {})
    else:
        fabric_cfg = training_config.fabric

    # 2. Extract parameters regardless of whether fabric_cfg is a dict or object
    if isinstance(fabric_cfg, dict):
        accelerator = fabric_cfg.get("accelerator", "cpu")
        precision   = fabric_cfg.get("precision", 32)
        # Support both YAML key "devices" and dataclass key "num_devices"
        num_devices = fabric_cfg.get("devices", fabric_cfg.get("num_devices", 1))
        num_nodes   = fabric_cfg.get("num_nodes", 1)
        cfg_strategy = fabric_cfg.get("strategy", None)
        timeout_hours = fabric_cfg.get("timeout_hours", 4)
    else:
        accelerator = fabric_cfg.accelerator
        precision   = fabric_cfg.precision
        num_devices = fabric_cfg.num_devices
        num_nodes   = fabric_cfg.num_nodes
        cfg_strategy = getattr(fabric_cfg, "strategy", None)
        timeout_hours = getattr(fabric_cfg, "timeout_hours", 4)

    # 3. Override from torchrun/elastic env vars if present, so the same config
    # works for any --nproc_per_node without manual edits.
    if "WORLD_SIZE" in os.environ:
        try:
            ws = int(os.environ["WORLD_SIZE"])
            lws = int(os.environ.get("LOCAL_WORLD_SIZE", ws))
            detected_devices = lws
            detected_nodes = max(1, ws // lws)
            if detected_devices != num_devices or detected_nodes != num_nodes:
                logging.info(
                    "Overriding fabric.devices=%s num_nodes=%s with torchrun "
                    "WORLD_SIZE=%s LOCAL_WORLD_SIZE=%s -> devices=%s num_nodes=%s",
                    num_devices, num_nodes, ws, lws, detected_devices, detected_nodes,
                )
            num_devices = detected_devices
            num_nodes = detected_nodes
        except (ValueError, ZeroDivisionError):
            pass  # fall back to config values

    # 4. Determine strategy
    if cfg_strategy:
        strategy = cfg_strategy
    else:
        total_devices = num_devices * num_nodes
        strategy = "ddp" if total_devices > 1 else "auto"

    # 4b. Build an explicit DDPStrategy with a long NCCL watchdog timeout so
    # rank-0-only work (e.g. large pretokenized dataset downloads) doesn't
    # trip the default 30-minute ALLREDUCE timeout on the other ranks while
    # they sit at `fabric.barrier()`.
    if isinstance(strategy, str) and strategy == "ddp":
        try:
            from lightning.fabric.strategies import DDPStrategy
            strategy = DDPStrategy(timeout=timedelta(hours=timeout_hours))
            logging.info(
                "Using DDPStrategy with NCCL timeout=%dh (override via "
                "training.fabric.timeout_hours in YAML).",
                timeout_hours,
            )
        except Exception as _e:  # pragma: no cover — keep string fallback
            logging.warning(
                "Could not construct DDPStrategy with custom timeout (%s); "
                "falling back to default 'ddp' string strategy.", _e,
            )

    # 5. Initialize and launch
    fabric = L.Fabric(
        accelerator=accelerator,
        precision=precision,
        devices=num_devices,
        num_nodes=num_nodes,
        loggers=[wandb_logger] if wandb_logger is not None else None,
        strategy=strategy,
    )
    fabric.launch()
    return fabric

def initialize_model(model_config):
    """Initialize the model for training."""
    if model_config.model_type == "pico_decoder":
        return PicoDecoder(model_config)
    else:
        raise ValueError(f"Invalid model type: {model_config.model_type}")


def initialize_optimizer(training_config: Union[TrainingConfig, dict], model: torch.nn.Module):
    # Safe extraction of optimization sub-config
    if isinstance(training_config, dict):
        opt_cfg = training_config.get("optimization", {})
    else:
        opt_cfg = training_config.optimization

    def _get(field: str, default):
        if isinstance(opt_cfg, dict):
            return opt_cfg.get(field, default)
        return getattr(opt_cfg, field, default)

    lr = _get("lr", 1e-4)
    opt_type = _get("optimizer", "adamw")

    if opt_type == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr)
    elif opt_type == "schedule_free_adamw":
        # Schedule-Free AdamW (Defazio et al. NeurIPS 2024). The trainer's
        # step loop must call optimizer.train()/.eval() around training and
        # eval/checkpoint phases — see src/training/beetle_trainer.py.
        # Warmup is handled INSIDE this optimizer via warmup_steps; the
        # external LR scheduler is bypassed when this optimizer is selected.
        from beetlelm.optim import AdamWScheduleFree
        betas = tuple(_get("betas", (0.9, 0.999)))
        return AdamWScheduleFree(
            model.parameters(),
            lr=lr,
            betas=betas,
            eps=float(_get("eps", 1e-8)),
            weight_decay=float(_get("weight_decay", 0.0)),
            warmup_steps=int(_get("warmup_steps", 0)),
            r=float(_get("r", 0.0)),
            weight_lr_power=float(_get("weight_lr_power", 2.0)),
        )
    else:
        raise ValueError(f"Invalid optimizer: {opt_type}")


class _NoOpLRScheduler:
    """LR-scheduler stub for optimizers that handle warmup/decay internally.

    Used by Schedule-Free AdamW (Defazio et al., NeurIPS 2024) where warmup
    is set via the optimizer's ``warmup_steps`` parameter and no external
    scheduler should run. Supplies the minimal surface area exercised by the
    trainer / checkpointing code: ``step()``, ``state_dict()``,
    ``load_state_dict()``, ``get_last_lr()``.
    """

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self, *args, **kwargs):
        return None

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        return None

    def get_last_lr(self):
        return [g.get("lr", 0.0) for g in self.optimizer.param_groups]


def initialize_lr_scheduler(
    training_config: Union[TrainingConfig, dict],
    optimizer: torch.optim.Optimizer
):
    """
    Initialize a learning rate scheduler with warmup and decay.
    Supports both dict and Dataclass for training_config and optimization cfg.
    """

    # Schedule-Free AdamW manages warmup internally; any external scheduler
    # would double-modulate the LR. Return a no-op so the existing trainer
    # and checkpointing code paths remain untouched.
    from beetlelm.optim import AdamWScheduleFree
    if isinstance(optimizer, AdamWScheduleFree):
        return _NoOpLRScheduler(optimizer)

    # 1. Safely extract optimization config and max_steps from the top-level config
    if isinstance(training_config, dict):
        opt_cfg = training_config.get("optimization", {})
        max_steps = training_config.get("max_steps", 1000)
    else:
        opt_cfg = training_config.optimization
        max_steps = training_config.max_steps

    # 2. Safely extract scheduler parameters from the optimization config
    if isinstance(opt_cfg, dict):
        lr_scheduler_type = opt_cfg.get("lr_scheduler")
        warmup_steps = opt_cfg.get("lr_warmup_steps", 0)
    else:
        lr_scheduler_type = opt_cfg.lr_scheduler
        warmup_steps = opt_cfg.lr_warmup_steps

    # 3. Initialize the scheduler
    if lr_scheduler_type == "exposure_based":
        from beetlelm.continual.exposure_lr import build_exposure_lr
        scheduler = build_exposure_lr(
            cfg=opt_cfg,
            optimizer=optimizer,
            training_config=training_config,
        )
        if scheduler is None:
            raise ValueError(
                "lr_scheduler='exposure_based' but build_exposure_lr returned None. "
                "Add an 'exposure_lr:' block under 'optimization:' in your YAML config."
            )
        return scheduler

    elif lr_scheduler_type == "linear_with_warmup":

        def _lr_lambda(curr_step, num_warmup_steps, total_steps):
            # Linear warmup
            if curr_step < num_warmup_steps:
                return float(curr_step) / float(max(1, num_warmup_steps))
            # Linear decay
            return max(
                0.0,
                float(total_steps - curr_step)
                / float(max(1, total_steps - num_warmup_steps)),
            )

        # Create the lambda closure
        lr_lambda = lambda step: _lr_lambda(
            step,
            warmup_steps,
            max_steps,
        )
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
    else:
        raise ValueError(
            f"Invalid learning rate scheduler: {lr_scheduler_type!r}. "
            f"Supported: 'linear_with_warmup', 'exposure_based'."
        )

@use_backoff()
def initialize_wandb(
    monitoring_config: MonitoringConfig, checkpointing_config: CheckpointingConfig
):
    """Initialize Weights and Biases.

    On a multi-job sweep, a transient outage at the W&B service must not
    block training. We:
      - default WANDB_MODE=offline if no API key is configured, so init
        does not block on a network call;
      - swallow any init exception and return None, letting the trainer
        proceed without a wandb logger rather than wedging the run.
    """
    import os
    if not os.environ.get("WANDB_API_KEY"):
        os.environ.setdefault("WANDB_MODE", "offline")

    assert (
        monitoring_config.wandb.project is not None
        and monitoring_config.wandb.project != ""
    ), "Wandb project must be provided if wandb is to be used."
    assert (
        monitoring_config.wandb.entity is not None
        and monitoring_config.wandb.entity != ""
    ), "Wandb entity must be provided if wandb is to be used."

    _run_id = None
    if checkpointing_config.training.auto_resume:
        try:
            previous_runs = wandb.Api().runs(
                path=f"{monitoring_config.wandb.entity}/{monitoring_config.wandb.project}",
                filters={"display_name": checkpointing_config.run_name},
            )
            if len(previous_runs) == 1:
                _run_id = previous_runs[0].id
        except (ValueError, Exception):  # noqa: BLE001 — must not block init
            _run_id = None

    try:
        wandb_logger = WandbLogger(
            project=monitoring_config.wandb.project,
            entity=monitoring_config.wandb.entity,
            id=_run_id,
            name=checkpointing_config.run_name,
        )
        return wandb_logger
    except Exception as exc:  # noqa: BLE001 — sweep robustness
        import logging
        logging.getLogger(__name__).warning(
            "WandbLogger init failed (%s: %s); proceeding without W&B.",
            type(exc).__name__, exc,
        )
        return None


@rank_zero_only
@use_backoff()
def initialize_hf_checkpointing(
    checkpointing_config: CheckpointingConfig, fabric: L.Fabric
):
    """Initialize HuggingFace Checkpointing."""

    huggingface_repo_id = checkpointing_config.hf_checkpoint.repo_id
    assert (
        huggingface_repo_id is not None and huggingface_repo_id != ""
    ), "hf_checkpoint.repo_id must be provided."

    repo = create_repo(huggingface_repo_id, exist_ok=True)
    checkpointing_config.hf_checkpoint.repo_id = repo.repo_id
    huggingface_repo_id = repo.repo_id

    if checkpointing_config.hf_checkpoint.collection_slug:
        add_collection_item(
            checkpointing_config.hf_checkpoint.collection_slug,
            huggingface_repo_id,
            repo.repo_type,
            exists_ok=True,
        )

    create_branch(
        repo_id=huggingface_repo_id,
        branch=checkpointing_config.run_name,
        exist_ok=True,
    )
