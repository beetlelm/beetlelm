#!/usr/bin/env python3
"""
script/train_tokenizer.py

Standalone CLI for training tokenizers via MultilingualTokenizerManager.
Supports monolingual and bilingual modes. Optionally pushes to HF Hub.

Usage:
  # Monolingual
  python script/train_tokenizer.py \
      --datasets eng:Beetle-Data/BabyBabel-eng-50M \
      --vocab_size 50000 --target_tokens 100000000 \
      --output_dir ./tokenizers/humanscale_mono \
      --hf_repo Beetle-HumanScale/bpe-humanscale-eng

  # Bilingual
  python script/train_tokenizer.py \
      --datasets eng:Beetle-Data/BabyBabel-eng-50M \
                 nld:Beetle-Data/BabyBabel-nld-50M \
      --vocab_size 50000 --target_tokens 100000000 \
      --output_dir ./tokenizers/humanscale_bilingual \
      --hf_repo Beetle-HumanScale/bpe-humanscale-eng-nld
"""

import sys
import os
import argparse
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from beetlelm.bilingual.tok.multilingual_tokenizer_manager import MultilingualTokenizerManager


def _load_stream(ds_name: str):
    """Stream text from an HF dataset, buffering into ~4KB chunks."""
    ds = load_dataset(ds_name, split="train")
    col = next(c for c, t in ds.features.items() if t.dtype == "string")
    buf = ""
    for line in ds[col]:
        text = line.strip()
        if text:
            buf += text + " "
            if len(buf) > 4096:
                yield buf
                buf = ""
    if buf:
        yield buf


def parse_dataset_arg(arg: str):
    """Parse 'LANG_KEY:HF_DATASET_NAME' into (lang_key, dataset_name)."""
    if ":" not in arg:
        raise argparse.ArgumentTypeError(
            f"Dataset must be in LANG:HF_DATASET format, got: {arg!r}"
        )
    lang_key, ds_name = arg.split(":", 1)
    return lang_key.strip(), ds_name.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Train a tokenizer via MultilingualTokenizerManager"
    )
    parser.add_argument(
        "--datasets", nargs="+", required=True,
        help="One or more LANG:HF_DATASET pairs (e.g. eng:Beetle-Data/BabyBabel-eng-50M)",
    )
    parser.add_argument("--vocab_size", type=int, default=50000)
    parser.add_argument("--target_tokens", type=int, default=100_000_000)
    parser.add_argument("--output_dir", type=str, default="./tokenizers")
    parser.add_argument("--hf_repo", type=str, default=None,
                        help="HF Hub repo to push tokenizer (e.g. Beetle-HumanScale/bpe-humanscale-eng)")
    parser.add_argument("--tokenizer_type", type=str, default="bpe",
                        choices=["bpe", "superbpe", "unigram"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # Parse dataset arguments
    datasets = {}
    for ds_arg in args.datasets:
        lang_key, ds_name = parse_dataset_arg(ds_arg)
        datasets[lang_key] = ds_name

    n_langs = len(datasets)
    mode = "monolingual" if n_langs == 1 else f"{n_langs}-lingual"
    print(f"[Tokenizer] Mode={mode} | langs={list(datasets.keys())} "
          f"| type={args.tokenizer_type} | vocab={args.vocab_size} "
          f"| target_tokens={args.target_tokens:,}")

    # Create streaming iterators
    lang_streams = {key: _load_stream(ds_name) for key, ds_name in datasets.items()}

    # Train tokenizer
    os.makedirs(args.output_dir, exist_ok=True)
    manager = MultilingualTokenizerManager(output_dir=args.output_dir)

    raw_tokenizer = manager.get_tokenizer(
        lang_streams=lang_streams,
        tokenizer_type=args.tokenizer_type,
        vocab_size=args.vocab_size,
        target_tokens=args.target_tokens,
        overwrite=args.overwrite,
        add_lang_tokens=(n_langs > 1),
    )

    print(f"[Tokenizer] Trained: vocab_size={raw_tokenizer.get_vocab_size()}")

    # Optionally push to HF Hub
    if args.hf_repo:
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=raw_tokenizer,
            unk_token="[UNK]", sep_token="[SEP]",
            pad_token="[PAD]", cls_token="[CLS]", mask_token="[MASK]",
        )
        tokenizer.save_pretrained(args.output_dir)
        tokenizer.push_to_hub(args.hf_repo, private=False)
        print(f"[Tokenizer] Pushed to HF Hub: {args.hf_repo}")


if __name__ == "__main__":
    main()
