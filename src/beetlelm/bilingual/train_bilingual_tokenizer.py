#!/usr/bin/env python3
"""
train_bilingual_tokenizer.py
Train a bilingual tokenizer on two Hugging Face datasets with dynamic curriculum.

- Uses BilingualTokenizerManager (in-memory streaming)
- Supports any HF dataset
- Supports DataMixConfig curricula (BALANCED, SEQUENTIAL, etc.)
- Integrates Fabric/Lightning-safe token counting
- Saves locally and optionally pushes to HF Hub

Usage example:
python train_bilingual_tokenizer.py \
    --l1_dataset BabyLM-community/babylm-spa \
    --l2_dataset BabyLM-community/babylm-tur \
    --tokenizer_type bpe \
    --vocab_size 32000 \
    --data_mix sequential \
    --target_tokens 1_000_000 \
    --max_batches 2000 \
    --output_dir ./tokenizers \
    --push_to_hub
"""

import os
import argparse
from datasets import load_dataset
from beetlelm.bilingual.tok.multilingual_tokenizer_manager import MultilingualTokenizerManager
from beetlelm.bilingual.data.curriculum import DataMixConfig
from beetlelm.bilingual.data.condition_presets import get_condition, CONDITION_ALIASES, ALL_CONDITION_BUILDERS

HUGGINGFACE_USER = os.environ.get("HUGGINGFACE_USER", "suchirsalhan")


# -------------------------------
# Dataset loader
# -------------------------------
def load_text_stream(dataset_name: str, split: str = "train"):
    """
    Returns a generator of text lines from HF dataset.
    Picks first string column automatically.
    """
    ds = load_dataset(dataset_name, split=split)
    text_cols = [col for col, typ in ds.features.items() if typ.dtype == "string"]
    if not text_cols:
        raise ValueError(f"No string column found in dataset {dataset_name}")
    text_col = text_cols[0]
    for item in ds[text_col]:
        if item.strip():
            yield item.strip()


# -------------------------------
# Token counter (Fabric / Lightning-safe)
# -------------------------------
class TokenCounter:
    """
    Tracks total tokens seen.
    Used for dynamic curriculum progress (DataMixConfig).
    """
    def __init__(self):
        self.tokens_seen = 0

    def add(self, n: int):
        self.tokens_seen += n

    def progress(self, total_tokens: int) -> float:
        return min(self.tokens_seen / max(total_tokens, 1), 1.0)


# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train bilingual tokenizer with dynamic curriculum")
    parser.add_argument("--l1_dataset", type=str, required=True)
    parser.add_argument("--l2_dataset", type=str, required=True)
    parser.add_argument("--tokenizer_type", type=str, default="bpe", choices=["bpe","unigram","superbpe"])
    parser.add_argument("--vocab_size", type=int, default=32768)
    parser.add_argument("--target_tokens", type=int, default=1_000_000)
    parser.add_argument("--max_batches", type=int, default=None, help="Optional: total steps for SEQUENTIAL / step-based curricula")
    _valid_names = sorted(set(list(CONDITION_ALIASES.keys()) + list(ALL_CONDITION_BUILDERS.keys())))
    parser.add_argument(
        "--data_mix",
        type=str,
        default="balanced",
        choices=_valid_names,
        help=f"Data mixing curriculum. Options: {', '.join(_valid_names)}"
    )
    parser.add_argument("--output_dir", type=str, default="./tokenizers")
    parser.add_argument("--no_push", action="store_true", help="Do NOT push tokenizer to HF Hub")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing tokenizer")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # -------------------------------
    # Resolve DataMixConfig
    # -------------------------------
    total_steps = args.max_batches if args.max_batches is not None else args.target_tokens
    data_mix_config: DataMixConfig = get_condition(
        args.data_mix, total_steps=total_steps,
    )

    # -------------------------------
    # Load datasets
    # -------------------------------
    print(f"[Datasets] Loading {args.l1_dataset} (L1) and {args.l2_dataset} (L2)")
    l1_stream = load_text_stream(args.l1_dataset)
    l2_stream = load_text_stream(args.l2_dataset)

    # -------------------------------
    # Initialize tokenizer manager
    # -------------------------------
    tokenizer_manager = MultilingualTokenizerManager(output_dir=args.output_dir)

    l1_key = args.l1_dataset.split("/")[-1]
    l2_key = args.l2_dataset.split("/")[-1]

    # -------------------------------
    # Train tokenizer
    # -------------------------------
    tokenizer = tokenizer_manager.get_tokenizer(
        lang_streams={l1_key: l1_stream, l2_key: l2_stream},
        tokenizer_type=args.tokenizer_type,
        vocab_size=args.vocab_size,
        target_tokens=args.target_tokens,
        overwrite=args.overwrite,
    )

    # -------------------------------
    # Save tokenizer
    # -------------------------------
    save_path = os.path.join(
        args.output_dir,
        f"{args.tokenizer_type}_{args.l1_dataset.split('/')[-1]}-{args.l2_dataset.split('/')[-1]}.json"
    )
    tokenizer.save(save_path)
    print(f"[Tokenizer] Saved locally at {save_path}")

    # -------------------------------
    # Optional push to HF Hub
    # -------------------------------
    if not args.no_push:
        repo_id = f"{HUGGINGFACE_USER}/{args.tokenizer_type}_{args.l1_dataset.split('/')[-1]}-{args.l2_dataset.split('/')[-1]}"
        print(f"[HF Hub] Pushing tokenizer to {repo_id}")
        tokenizer.push_to_hub(repo_id, private=False)
        print("[HF Hub] Push complete")


if __name__ == "__main__":
    main()

