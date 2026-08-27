"""
src/pre_pretraining/stage.py
────────────────────────────
Stage-1 pre-pretraining loop.

Runs a short optimisation phase on a formal language (Shuffle-Dyck) BEFORE the
main natural-language training loop begins.  Reuses the existing
``Trainer.model``, ``Trainer.optimizer``, ``Trainer.fabric``, and
``Trainer._compute_loss`` so there is no parallel training machinery.

After stage 1 finishes we optionally (default: true) reset the optimizer and
LR scheduler so that stage 2 starts from a clean warmup — mirroring the
two-command recipe in ``michahu/pre-pretraining`` where stage 2 is a fresh
``train.py`` run loading the stage-1 checkpoint.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch

from beetlelm.pre_pretraining.formal_lang import ShuffleDyckIterableDataset

logger = logging.getLogger(__name__)


# Default vocab offset for Shuffle-Dyck symbols. Most BPE tokenizers used in
# BeetleLM reserve the first ~16 IDs for special tokens (<pad>, <bos>, <eos>,
# <unk>, <mask>, <lang:*>, …); 256 is comfortably above that span and below
# the start of any non-trivial natural-language token. Configs that need a
# tighter packing can override per-tokenizer.
_DEFAULT_VOCAB_OFFSET = 256


@dataclass
class PrePretrainingConfig:
    enabled: bool = False
    formal_lang: str = "shuffle_dyck"
    steps: int = 500
    num_symbols: int = 20
    max_depth: int = 8
    p_open: float = 0.5
    vocab_offset: int = _DEFAULT_VOCAB_OFFSET
    reset_optimizer: bool = True
    save_stage1_checkpoint: bool = True

    @classmethod
    def from_dict(cls, cfg: dict) -> "PrePretrainingConfig":
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            formal_lang=str(cfg.get("formal_lang", "shuffle_dyck")),
            steps=int(cfg.get("steps", 500)),
            num_symbols=int(cfg.get("num_symbols", 20)),
            max_depth=int(cfg.get("max_depth", 8)),
            p_open=float(cfg.get("p_open", 0.5)),
            vocab_offset=int(cfg.get("vocab_offset", _DEFAULT_VOCAB_OFFSET)),
            reset_optimizer=bool(cfg.get("reset_optimizer", True)),
            save_stage1_checkpoint=bool(cfg.get("save_stage1_checkpoint", True)),
        )


def build_pre_pretraining(cfg: dict) -> Optional[PrePretrainingConfig]:
    """
    Build a ``PrePretrainingConfig`` from the ``pre_pretraining:`` YAML block.
    Returns ``None`` if the block is absent or ``enabled: false``.

    ``vocab_offset`` defaults to 256 so Shuffle-Dyck symbols do NOT collide
    with the tokenizer's special-token reserved span (<pad>, <bos>, <eos>,
    <unk>, <mask>, <lang:*>). If the YAML explicitly sets a value below 16, a
    warning is logged so the experimenter is aware that stage-1 will be
    re-using special-token embedding rows as Dyck symbols.
    """
    if not cfg or not cfg.get("enabled", False):
        return None
    pp = PrePretrainingConfig.from_dict(cfg)
    if pp.formal_lang != "shuffle_dyck":
        raise NotImplementedError(
            f"[pre-pretraining] formal_lang={pp.formal_lang!r} is not yet supported. "
            "Only 'shuffle_dyck' is currently available."
        )
    if pp.vocab_offset < 16:
        logger.warning(
            "[pre-pretraining] vocab_offset=%d is below the typical "
            "special-token reserved span (≤16). Shuffle-Dyck symbols "
            "%d..%d will overlap with <pad>/<bos>/<eos>/etc. and re-use "
            "those embedding rows during stage 1. Set vocab_offset>=16 "
            "(default 256) unless you have explicitly verified your "
            "tokenizer's reserved span.",
            pp.vocab_offset,
            pp.vocab_offset, pp.vocab_offset + 2 * pp.num_symbols - 1,
        )
    logger.info(
        "[pre-pretraining] Enabled | formal_lang=%s | steps=%d | "
        "num_symbols=%d max_depth=%d | vocab_offset=%d | reset_optimizer=%s",
        pp.formal_lang, pp.steps, pp.num_symbols, pp.max_depth,
        pp.vocab_offset, pp.reset_optimizer,
    )
    return pp


def run_pre_pretraining_stage(trainer) -> None:
    """
    Run the formal-language pre-pretraining stage on ``trainer``.

    Mutates ``trainer.model`` in place.  On exit the main training loop can
    proceed as usual; if ``reset_optimizer`` is true, ``trainer.optimizer`` and
    ``trainer.lr_scheduler`` are rebuilt so stage 2 starts from a clean warmup.
    """
    pp: Optional[PrePretrainingConfig] = getattr(trainer, "_pre_pretraining", None)
    if pp is None:
        return

    raw = getattr(trainer.model, "module", trainer.model)
    mc = raw.model_config
    vocab_size = int(mc.vocab_size)
    seq_len = int(mc.max_seq_len)
    batch_size = int(mc.batch_size)

    dataset = ShuffleDyckIterableDataset(
        num_symbols=pp.num_symbols,
        seq_len=seq_len,
        batch_size=batch_size,
        max_depth=pp.max_depth,
        p_open=pp.p_open,
        vocab_size=vocab_size,
        vocab_offset=pp.vocab_offset,
        seed=trainer.seed,
    )
    data_iter = iter(dataset)

    trainer.log(
        f"[pre-pretraining] Starting stage 1 | {pp.formal_lang} "
        f"| steps={pp.steps} | batch={batch_size}x{seq_len} | "
        f"num_symbols={pp.num_symbols} max_depth={pp.max_depth}"
    )

    trainer.model.train()
    optimizer = trainer.optimizer
    lr_scheduler = trainer.lr_scheduler

    for step in range(1, pp.steps + 1):
        batch = next(data_iter).to(trainer.fabric.device)

        optimizer.zero_grad()
        loss = trainer._compute_loss(batch, step=step)
        trainer.fabric.backward(loss)
        optimizer.step()
        # Advance the LR scheduler each step. Without this, the warmup
        # phase (lr_warmup_steps in YAML) never moves and lr stays at 0,
        # so optimizer.step() leaves weights unchanged for the entire
        # stage-1 run — which is exactly the bug that left pre-pretrain
        # bit-identical to baseline at every checkpoint.
        if lr_scheduler is not None:
            lr_scheduler.step()

        if step == 1 or step % max(1, pp.steps // 10) == 0 or step == pp.steps:
            current_lr = (
                optimizer.param_groups[0].get("lr", 0.0)
                if hasattr(optimizer, "param_groups") else 0.0
            )
            trainer.log(
                f"[pre-pretraining] step {step}/{pp.steps} "
                f"| loss={loss.item():.4f} | lr={current_lr:.2e}"
            )

    trainer.log("[pre-pretraining] Stage 1 complete.")

    # Optional stage-1 checkpoint (local only — not pushed to HF).
    if pp.save_stage1_checkpoint:
        try:
            runs_dir = (trainer.checkpoint_cfg or {}).get("runs_dir", "./runs")
            out_dir = os.path.join(runs_dir, trainer.run_name, "pre_pretraining")
            os.makedirs(out_dir, exist_ok=True)
            ckpt_path = os.path.join(out_dir, "stage1_model.pt")
            state = {k: v.detach().cpu() for k, v in raw.state_dict().items()}
            torch.save(state, ckpt_path)
            trainer.log(f"[pre-pretraining] Saved stage-1 weights → {ckpt_path}")
        except Exception as exc:  # pragma: no cover  (best-effort only)
            logger.warning("[pre-pretraining] Failed to save stage-1 checkpoint: %s", exc)

    # Reset optimizer + LR scheduler so stage 2 starts from a clean warmup.
    if pp.reset_optimizer:
        from beetlelm.training.utils.initialization import (
            initialize_optimizer,
            initialize_lr_scheduler,
        )
        new_opt = initialize_optimizer(
            training_config=trainer.training_cfg, model=raw
        )
        trainer.lr_scheduler = initialize_lr_scheduler(
            training_config=trainer.training_cfg, optimizer=new_opt
        )
        # Re-wrap optimiser with fabric so gradient sync stays intact.
        _, trainer.optimizer = trainer.fabric.setup(raw, new_opt)
        trainer.log("[pre-pretraining] Optimizer + LR scheduler reset for stage 2.")
