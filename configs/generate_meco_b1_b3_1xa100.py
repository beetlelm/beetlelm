#!/usr/bin/env python3
"""
Generate 1xA100 B1-B3 training YAMLs for the MECO-target languages
(ita, spa, hin, rus, tur, eus) at both human-scale (100M) and
fineweb-2B scales, plus Spanish BGPT-arch variants.

Reads templates from configs/{humanscale_fineweb_bilingual,fineweb_bilingual_2b}/core/{B1,B2,B3}/
and emits modified copies under configs/<scale>/1xa100/<B*>/.

Modifications vs. template:
- training.fabric.devices: 8 -> 1
  (trainer auto-derives gradient_accumulation_steps from
   target_effective_batch / (per_gpu_batch * num_gpus); see
   src/training/beetle_trainer.py L1381-1390.)
- run_name suffixed with -1xa100
- For Spanish BGPT variants: model.architecture: pico -> bgpt,
  attention_n_kv_heads: 1 -> 12 (no GQA),
  norm_eps -> layer_norm_epsilon, drop position_emb_theta,
  run_name suffixed with -arch-bgpt.
"""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent
LANGS = ["ita", "spa", "hin", "rus", "tur", "eus"]
CONDITIONS = ["B1", "B2", "B3"]
SCALES = [
    ("humanscale_fineweb_bilingual", "100m"),
    ("fineweb_bilingual_2b", "2b"),
]


def patch_for_1xa100(cfg: dict) -> dict:
    cfg["training"]["fabric"]["devices"] = 1
    cfg["run_name"] = f"{cfg['run_name']}-1xa100"
    if "polca" in cfg:
        cfg["polca"] = {"enabled": False}
    # auto_resume on by default in 2B configs is fine; leave alone.
    return cfg


def patch_for_bgpt(cfg: dict) -> dict:
    params = cfg["model"]["params"]
    cfg["model"]["architecture"] = "bgpt"
    params["attention_n_kv_heads"] = 12  # no GQA in GPT-2
    if "norm_eps" in params:
        params["layer_norm_epsilon"] = params.pop("norm_eps")
    params.pop("position_emb_theta", None)  # bgpt uses learned pos emb
    cfg["run_name"] = f"{cfg['run_name']}-arch-bgpt"
    return cfg


def main() -> None:
    written = 0
    for scale_dir, _tag in SCALES:
        for cond in CONDITIONS:
            src_dir = ROOT / scale_dir / "core" / cond
            dst_dir = ROOT / scale_dir / "1xa100" / cond
            dst_dir.mkdir(parents=True, exist_ok=True)
            for lang in LANGS:
                src = src_dir / f"train_{lang}_L1-eng_L2.yaml"
                if not src.exists():
                    raise FileNotFoundError(src)
                with src.open() as f:
                    cfg = yaml.safe_load(f)
                cfg = patch_for_1xa100(cfg)
                dst = dst_dir / f"train_{lang}_L1-eng_L2.yaml"
                with dst.open("w") as f:
                    yaml.safe_dump(cfg, f, sort_keys=False)
                written += 1

                if lang == "spa":
                    with src.open() as f:
                        cfg2 = yaml.safe_load(f)
                    cfg2 = patch_for_1xa100(cfg2)
                    cfg2 = patch_for_bgpt(cfg2)
                    dst2 = dst_dir / f"train_{lang}_L1-eng_L2.bgpt.yaml"
                    with dst2.open("w") as f:
                        yaml.safe_dump(cfg2, f, sort_keys=False)
                    written += 1

    print(f"wrote {written} yaml files")


if __name__ == "__main__":
    main()
