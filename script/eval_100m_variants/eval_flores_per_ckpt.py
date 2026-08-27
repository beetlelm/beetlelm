"""
eval_flores_per_ckpt.py — score every step-N HF branch of every variant on
FLORES-200 (eng, nld) and emit one CSV per (variant, lang).

Output layout:
    /root/beetle-analyze/results/100m-variants/flores/
        <variant>/<iso>__step-<N>.csv      one row per FLORES sentence
        ppl_summary.csv                    rolled up: variant,step,iso,mean_nll

Re-runs are safe: per-(variant, step, iso) sentinel skip via the eval_ledger
inherited from beetle-analyze/evals/eval_flores_ppl.py.

Usage:
    HF_TOKEN=... CUDA_VISIBLE_DEVICES=0 \\
      python script/eval_100m_variants/eval_flores_per_ckpt.py \\
          --variants baseline ewc lamol \\
          --steps 0 4096 5000 9096 \\
          --langs eng nld
"""
from __future__ import annotations

import argparse
import gc
import math
import sys
from pathlib import Path
from typing import List

import pandas as pd

_REPO_ROOT = Path("/root/beetle-analyze")
sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import HfApi

from evals.model_loader import (  # type: ignore  # noqa: E402
    DEVICE,
    load_model,
    unload_model,
    setup_logging,
    resolve_vocab_size,
)

sys.path.insert(0, "/root/beetlelm")
from script.eval_100m_variants.manifest import all_variants, checkpoint_steps, Variant  # type: ignore  # noqa: E402

OUT_DIR = Path("/root/beetle-analyze/results/100m-variants/flores")
SUMMARY_CSV = OUT_DIR / "ppl_summary.csv"

ISO_TO_FLORES_CODE = {"eng": "eng_Latn", "nld": "nld_Latn"}


def _flores_sentences(iso: str) -> list[str]:
    """Load FLORES-200 devtest split for one ISO 639-3 code.

    Uses openlanguagedata/flores_plus (parquet) — the legacy facebook/flores
    builder script was removed by datasets >= 4.x.
    """
    import os
    code = ISO_TO_FLORES_CODE[iso]
    token = os.environ.get("HF_TOKEN")
    ds = load_dataset(
        "openlanguagedata/flores_plus",
        code,
        split="devtest",
        token=token,
    )
    return [ex["text"] for ex in ds]


@torch.no_grad()
def _score(model, tokenizer, sentences: list[str], batch_size: int = 16) -> list[float]:
    """Per-sentence mean NLL (nats per token, excluding the first BOS-conditioned token)."""
    nlls: list[float] = []
    model.eval()
    for start in range(0, len(sentences), batch_size):
        batch = sentences[start : start + batch_size]
        enc = tokenizer(
            batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(DEVICE)
        ids = enc["input_ids"]
        mask = enc["attention_mask"]
        logits = model(ids, attention_mask=mask).logits  # (B, T, V)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_targets = ids[:, 1:].contiguous()
        shift_mask = mask[:, 1:].contiguous().float()

        loss = F.cross_entropy(
            shift_logits.transpose(1, 2),  # (B, V, T-1)
            shift_targets,
            reduction="none",
        )  # (B, T-1)
        per_seq = (loss * shift_mask).sum(1) / shift_mask.sum(1).clamp_min(1.0)
        nlls.extend(per_seq.cpu().tolist())
    return nlls


def _ckpt_exists(api: HfApi, repo: str, step: int) -> bool:
    try:
        refs = api.list_repo_refs(repo)
        return any(b.name == f"step-{step}" for b in refs.branches)
    except Exception:
        return False


def _eval_one(
    api: HfApi, v: Variant, step: int, isos: List[str], batch_size: int
) -> List[dict]:
    """Eval a single (variant, step) pair on the requested languages."""
    revision = f"step-{step}"
    if not _ckpt_exists(api, v.hf_repo, step):
        print(f"  SKIP {v.name}@{revision} (branch missing)")
        return []

    out_dir = OUT_DIR / v.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Skip if all isos already done for this (variant, step).
    if all((out_dir / f"{iso}__step-{step}.csv").exists() for iso in isos):
        print(f"  SKIP {v.name}@{revision} (already cached)")
        return [
            {
                "variant": v.name,
                "step": step,
                "iso": iso,
                "mean_nll": float(
                    pd.read_csv(out_dir / f"{iso}__step-{step}.csv")["nll"].mean()
                ),
            }
            for iso in isos
        ]

    print(f"  load {v.hf_repo}@{revision}")
    bundle = load_model(v.hf_repo, DEVICE, revision=revision)
    if bundle is None:
        print(f"  SKIP {v.name}@{revision} (load failed)")
        return []
    model, tokenizer = bundle

    summaries = []
    for iso in isos:
        cache = out_dir / f"{iso}__step-{step}.csv"
        if cache.exists():
            summaries.append(
                {
                    "variant": v.name,
                    "step": step,
                    "iso": iso,
                    "mean_nll": float(pd.read_csv(cache)["nll"].mean()),
                }
            )
            continue

        sentences = _flores_sentences(iso)
        nlls = _score(model, tokenizer, sentences, batch_size=batch_size)
        df = pd.DataFrame({"sentence_idx": range(len(nlls)), "nll": nlls})
        df.to_csv(cache, index=False)
        mean_nll = float(df["nll"].mean())
        ppl = math.exp(mean_nll)
        print(f"    {iso:5s}  mean_nll={mean_nll:.4f}  ppl={ppl:.2f}  ({len(nlls)} sent)")
        summaries.append(
            {"variant": v.name, "step": step, "iso": iso, "mean_nll": mean_nll}
        )

    unload_model(model, tokenizer)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summaries


def main() -> None:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--variants", nargs="*", default=None, help="default: all variants")
    p.add_argument("--steps", nargs="*", type=int, default=None,
                   help="default: every checkpoint step from each variant's YAML")
    p.add_argument("--langs", nargs="*", default=["eng", "nld"])
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging("flores_per_ckpt", OUT_DIR, rank=0)
    api = HfApi()

    variants = all_variants()
    if args.variants:
        wanted = set(args.variants)
        variants = [v for v in variants if v.name in wanted]
        if not variants:
            sys.exit(f"no variants matched {args.variants}")

    rows: List[dict] = []
    for v in variants:
        steps = args.steps if args.steps is not None else checkpoint_steps(v)
        print(f"=== {v.name}  ({v.hf_repo})  {len(steps)} steps ===")
        for step in steps:
            rows.extend(_eval_one(api, v, step, args.langs, args.batch_size))

    if not rows:
        print("No rows produced.")
        return

    df = pd.DataFrame(rows)
    if SUMMARY_CSV.exists():
        old = pd.read_csv(SUMMARY_CSV)
        df = pd.concat([old, df]).drop_duplicates(
            subset=["variant", "step", "iso"], keep="last"
        )
    df.sort_values(["variant", "iso", "step"]).to_csv(SUMMARY_CSV, index=False)
    print(f"\nSummary → {SUMMARY_CSV} ({len(df)} rows)")


if __name__ == "__main__":
    main()
