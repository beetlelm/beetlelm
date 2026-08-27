"""smoke_pre_pretrain_fix.py — verify pre-pretrain stage-1 actually
updates weights and that loss decreases.

Reproduces the exact bug from the EMNLP 100M sweep:
  * stage-1 loss flat at ~ln(50000) ≈ 11.02 for all 500 steps
  * step-0 save bit-identical to baseline (no weight change)

Pass criteria:
  * loss goes down across stage-1 steps (median final < 0.95 * median initial)
  * post-stage-1 model.embedding_proj.weight differs from pre-stage-1
  * lr scheduler advances during stage-1 (lr > 0 at last step)
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import yaml

REPO = Path("/root/beetlelm")
sys.path.insert(0, str(REPO))

PRE_PRETRAIN_YAML = (
    REPO / "configs/humanscale_fineweb_bilingual/core/B3/"
    "variants_nld_eng_100m/train_nld_L1-eng_L2.pre-pretrain.yaml"
)


def _override_for_smoke(src: Path, dst: Path) -> None:
    cfg = yaml.safe_load(src.read_text())
    cfg["run_name"] = cfg["run_name"] + "-pp-smoke"
    cfg["training"]["fabric"]["devices"] = 1
    cfg["training"]["max_steps"] = 1
    cfg["training"]["global_batch_size"] = 8
    cfg["training"]["micro_batch_size"]  = 8
    cfg["data"]["batch_size"] = 8
    cfg["model"]["params"]["batch_size"] = 8
    # Tiny warmup so lr is non-zero quickly. Set max_steps so the
    # linear-with-warmup decay tail doesn't immediately drop to 0 at the
    # end of stage-1 (the test wants the lr to stay non-zero across stage-1).
    cfg["training"]["optimization"]["lr_warmup_steps"] = 5
    cfg["training"]["max_steps"] = 1000
    cfg["pre_pretraining"]["steps"] = 30  # enough to see clear loss drop
    cfg["pre_pretraining"]["save_stage1_checkpoint"] = False
    cfg["checkpointing"]["save_to_hf"] = False
    cfg["checkpointing"]["checkpoint_steps"] = [0, 1]
    dst.write_text(yaml.safe_dump(cfg, sort_keys=False))


def main() -> int:
    import torch

    smoke = REPO / "_smoke_pp.yaml"
    _override_for_smoke(PRE_PRETRAIN_YAML, smoke)

    try:
        from beetlelm.training.beetle_trainer import Trainer
        from beetlelm.pre_pretraining import run_pre_pretraining_stage

        trainer = Trainer(config_path=str(smoke), seed=42)
        raw = getattr(trainer.model, "module", trainer.model)

        # Snapshot weights before stage-1
        pre_w = raw.embedding_proj.weight.detach().cpu().clone()
        pre_lr = trainer.optimizer.param_groups[0]["lr"] \
            if hasattr(trainer.optimizer, "param_groups") else None
        print(f"  pre-stage-1 lr: {pre_lr}")

        # Run stage-1 (capture loss series + lr series via monkey-patched logger)
        losses: list[float] = []
        lrs: list[float] = []
        orig_log = trainer.log

        def capture(msg, *args, **kwargs):
            s = str(msg)
            if "[pre-pretraining] step" in s and "loss=" in s:
                try:
                    losses.append(float(s.split("loss=")[1].split()[0]))
                except (ValueError, IndexError):
                    pass
                if "lr=" in s:
                    try:
                        lrs.append(float(s.split("lr=")[1].split()[0]))
                    except (ValueError, IndexError):
                        pass
            return orig_log(msg, *args, **kwargs)

        trainer.log = capture
        run_pre_pretraining_stage(trainer)
        trainer.log = orig_log

        # Snapshot weights after
        post_w = raw.embedding_proj.weight.detach().cpu().clone()
        weight_delta = (pre_w - post_w).abs().max().item()
        max_lr_during_stage1 = max(lrs) if lrs else 0.0
        print(f"  max lr during stage-1: {max_lr_during_stage1:.2e}")
        print(f"  embedding weight max_abs_diff: {weight_delta:.6e}")
        print(f"  losses captured: {losses}")

        ok = True
        reasons = []

        if weight_delta < 1e-5:
            ok = False
            reasons.append(
                f"weights barely changed after stage-1 (max_abs_diff={weight_delta:.2e})"
            )

        if max_lr_during_stage1 <= 0:
            ok = False
            reasons.append("lr never advanced past 0 during stage-1 (scheduler not stepping)")

        if len(losses) >= 4:
            first_few = sum(losses[: max(1, len(losses) // 4)]) / max(1, len(losses) // 4)
            last_few  = sum(losses[-max(1, len(losses) // 4):]) / max(1, len(losses) // 4)
            # Expect loss to drop substantially on shuffle-dyck (uniform 11
            # at start, ~3-4 once structure is learned).
            if last_few >= first_few * 0.7:
                ok = False
                reasons.append(
                    f"loss drop weak: first-quartile mean {first_few:.3f} → "
                    f"last-quartile mean {last_few:.3f} (<30% reduction)"
                )

        print()
        if ok:
            print("PASS pre-pretrain stage-1 updates weights AND drops loss")
            return 0
        else:
            print("FAIL pre-pretrain stage-1:")
            for r in reasons:
                print(f"  - {r}")
            return 1
    except Exception:
        print("FAIL — exception:")
        print(traceback.format_exc())
        return 1
    finally:
        smoke.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
