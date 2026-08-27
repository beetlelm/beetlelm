"""
src/training/geo_bilingual_supervisor.py
──────────────────────────────────────────
Single-invocation supervisor for the geo_bilingual 4-stage pipeline:

  Stage 1: mono_a  — train monolingual L1 model
  Stage 2: mono_b  — train monolingual L2 model
  Stage 3: align   — compute Procrustes rotation between embedding spaces
  Stage 4: bilingual — train bilingual model initialised from alignment

Usage:
    python -m beetlelm.training.geo_bilingual_supervisor \\
        --config configs/samples/geo_bilingual/supervisor.yaml

    # Resume from a specific stage:
    python -m beetlelm.training.geo_bilingual_supervisor \\
        --config configs/samples/geo_bilingual/supervisor.yaml \\
        --start-stage align
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


STAGES = ["mono_a", "mono_b", "align", "bilingual"]


class GeoBilingualSupervisor:
    """
    Orchestrates the four-stage geo_bilingual pipeline.

    Each inner trainer call is fresh (own Fabric, own data pipeline); the
    supervisor holds only paths and metrics, not model state.

    Parameters
    ----------
    config_path : str | Path
        Path to the supervisor YAML file (top-level ``geo_bilingual:`` block).
    start_stage : str
        One of ``"mono_a"``, ``"mono_b"``, ``"align"``, ``"bilingual"``.
        Skips earlier stages (assumes their outputs already exist).
    """

    def __init__(self, config_path: str, start_stage: str = "mono_a"):
        with open(config_path, "r") as f:
            raw = yaml.safe_load(f)
        self._cfg = raw.get("geo_bilingual", raw)
        self.start_stage = start_stage
        self._metrics: Dict[str, float] = {}

    # ─── Stage dispatch ───────────────────────────────────────────────────

    def run(self) -> None:
        start_idx = STAGES.index(self.start_stage)
        for stage in STAGES[start_idx:]:
            logger.info("[GeoBilingual] ─── Stage: %s ───", stage)
            getattr(self, f"_run_{stage}")()
        logger.info("[GeoBilingual] Pipeline complete.")

    def _run_mono_a(self) -> None:
        cfg = self._cfg.get("mono_a", {})
        self._train_stage("mono_a", cfg)

    def _run_mono_b(self) -> None:
        cfg = self._cfg.get("mono_b", {})
        self._train_stage("mono_b", cfg)

    def _train_stage(self, name: str, stage_cfg: dict) -> None:
        config_path = stage_cfg.get("config_path")
        if not config_path:
            raise ValueError(f"[GeoBilingual] {name}.config_path is required.")
        from beetlelm.training.beetle_trainer import BeetleLMTrainer
        trainer = BeetleLMTrainer(config_path=config_path)
        trainer.train()
        logger.info("[GeoBilingual] Stage '%s' complete.", name)

    def _run_align(self) -> None:
        """Extract embeddings from both monolingual checkpoints and compute R."""
        align_cfg = self._cfg.get("alignment", {})
        mono_a_cfg = self._cfg.get("mono_a", {})
        mono_b_cfg = self._cfg.get("mono_b", {})

        ckpt_a = self._find_checkpoint(mono_a_cfg)
        ckpt_b = self._find_checkpoint(mono_b_cfg)
        save_rotation_path = align_cfg.get("save_rotation_path")
        method = align_cfg.get("method", "procrustes")

        from beetlelm.bilingual.geometry.alignment import linear_cka, procrustes, vec2vec_align
        from beetlelm.bilingual.geometry.extract_embeddings import (
            extract_token_embeddings, shared_token_embeddings,
        )

        logger.info("[GeoBilingual:align] Loading embeddings from checkpoints.")
        sd_a = torch.load(ckpt_a, map_location="cpu")
        sd_b = torch.load(ckpt_b, map_location="cpu")

        def _get_emb(sd):
            for k, v in sd.items():
                if "embedding_proj.weight" in k:
                    return v.float()
            raise RuntimeError("Could not find 'embedding_proj.weight' in checkpoint.")

        E_a = _get_emb(sd_a)
        E_b = _get_emb(sd_b)

        E_a_s, E_b_s = shared_token_embeddings(E_a, E_b)

        if method == "procrustes":
            R = procrustes(E_a_s, E_b_s)
        elif method == "vec2vec":
            R = vec2vec_align(E_a_s, E_b_s)
        else:
            raise ValueError(f"[GeoBilingual] Unknown alignment method {method!r}.")

        # CKA score before alignment.
        cka_before = float(linear_cka(E_a_s, E_b_s).item())
        cka_after  = float(linear_cka(E_a_s @ R, E_b_s).item())
        self._metrics.update({
            "alignment/cka_before": cka_before,
            "alignment/cka_after":  cka_after,
        })
        logger.info(
            "[GeoBilingual:align] CKA before=%.4f → after=%.4f | method=%s",
            cka_before, cka_after, method,
        )

        if save_rotation_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_rotation_path)), exist_ok=True)
            np.save(save_rotation_path, R.cpu().numpy())
            logger.info("[GeoBilingual:align] Rotation saved to %s.", save_rotation_path)

    def _run_bilingual(self) -> None:
        cfg = self._cfg.get("bilingual", {})
        self._train_stage("bilingual", cfg)

    def _find_checkpoint(self, stage_cfg: dict) -> str:
        """Return the checkpoint path for a finished mono stage."""
        ckpt = stage_cfg.get("checkpoint_path", stage_cfg.get("checkpoint_step"))
        if ckpt is None:
            raise ValueError(
                "[GeoBilingual] checkpoint_path / checkpoint_step required for "
                "align stage but not found in stage config."
            )
        return str(ckpt)

    @property
    def metrics(self) -> Dict[str, float]:
        return dict(self._metrics)


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Geo-bilingual supervisor")
    parser.add_argument("--config", required=True, help="Path to supervisor YAML.")
    parser.add_argument(
        "--start-stage",
        default="mono_a",
        choices=STAGES,
        help="Stage to start from (previous stages skipped).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    args = _parse_args()
    supervisor = GeoBilingualSupervisor(
        config_path=args.config,
        start_stage=args.start_stage,
    )
    supervisor.run()


if __name__ == "__main__":
    main()
