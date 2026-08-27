"""smoke_variants_fix.py — verify lamol/alibi/maml YAMLs initialise and (for maml) save step-0.

Run with a single GPU (CUDA_VISIBLE_DEVICES=0) — bypasses torchrun. Uses
fabric.devices=1 override so we don't need the full 8-GPU dance.

Pass: trainer init succeeds; for maml, _save_step_checkpoint_local(0) succeeds.
Fail: prints the traceback and exits 1.
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import yaml

REPO = Path("/root/beetlelm")
sys.path.insert(0, str(REPO))

VARIANT_YAMLS = {
    "lamol": REPO / "configs/humanscale_fineweb_bilingual/core/B3/variants_nld_eng_100m/train_nld_L1-eng_L2.lamol.yaml",
    "alibi": REPO / "configs/humanscale_fineweb_bilingual/core/B3/variants_nld_eng_100m/train_nld_L1-eng_L2.alibi.yaml",
    "maml":  REPO / "configs/humanscale_fineweb_bilingual/core/B3/variants_nld_eng_100m/train_nld_L1-eng_L2.maml.yaml",
}


def _override_yaml_for_smoke(src: Path, dst: Path) -> None:
    """Single-GPU, tiny-batch override of the variant YAML for fast smoke."""
    cfg = yaml.safe_load(src.read_text())
    cfg["run_name"] = cfg["run_name"] + "-smoke"
    cfg["training"]["fabric"]["devices"] = 1
    cfg["training"]["max_steps"] = 1
    cfg["training"]["global_batch_size"] = 8
    cfg["training"]["micro_batch_size"]  = 8
    cfg["data"]["batch_size"] = 8
    cfg["model"]["params"]["batch_size"] = 8
    cfg["checkpointing"]["save_to_hf"] = False
    cfg["checkpointing"]["checkpoint_steps"] = [0, 1]
    dst.write_text(yaml.safe_dump(cfg, sort_keys=False))


def _try_init(variant: str, src_yaml: Path) -> tuple[bool, str]:
    """Construct Trainer; for maml also save step-0. Returns (ok, msg)."""
    smoke = REPO / f"_smoke_{variant}.yaml"
    _override_yaml_for_smoke(src_yaml, smoke)
    try:
        # lazy import — fail late if torch/lightning not on path
        from beetlelm.training.beetle_trainer import Trainer
        trainer = Trainer(config_path=str(smoke), seed=42)
        if variant == "maml":
            # the original failure: step-0 save throws on classifier_head keys
            trainer._save_step_checkpoint_local(0)
        return True, "OK"
    except Exception as e:
        return False, traceback.format_exc()
    finally:
        smoke.unlink(missing_ok=True)


def main() -> int:
    failed = []
    for variant, yml in VARIANT_YAMLS.items():
        print(f"\n=== smoke {variant} ===")
        ok, msg = _try_init(variant, yml)
        if ok:
            print(f"  PASS {variant}")
        else:
            print(f"  FAIL {variant}:\n{msg}")
            failed.append(variant)
    print(f"\n{len(VARIANT_YAMLS) - len(failed)} / {len(VARIANT_YAMLS)} variants pass smoke")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
