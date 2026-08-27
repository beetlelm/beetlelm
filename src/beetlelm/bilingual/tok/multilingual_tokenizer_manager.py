"""
multilingual_tokenizer_manager.py

End-to-end multilingual tokenizer manager.
- Supports N languages (generalised from the original bilingual design)
- Equal-compression streaming across all languages
- BPE, Unigram, SuperBPE architectures
- Per-language control tokens (<lang:xx>)
- Vocab size heuristic scaled to number of languages
- Fully streaming: no full corpus materialisation in memory
"""

import os
import re
import math
import logging
from typing import Dict, List, Union, Iterator, Optional

from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("MultilingualTokenizerManager")

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
BASE_SPECIAL_TOKENS = ["<UNK>", "<PAD>", "<CLS>", "<SEP>", "<MASK>"]
SUPPORTED_TYPES     = {"bpe", "superbpe", "unigram"}


# ─────────────────────────────────────────────
# Equal-compression helpers
# ─────────────────────────────────────────────
def _estimate_avg_tokens(texts: List[str], sample_size: int = 1000) -> float:
    """Estimate average whitespace-token count from a random sample of lines."""
    sample = texts[:sample_size]
    if not sample:
        return 1.0
    return sum(len(t.split()) for t in sample) / len(sample) or 1.0


def equal_compression_sample(
    lang_texts: Dict[str, List[str]],
    target_tokens: int,
) -> Dict[str, List[str]]:
    """
    Downsample or repeat each language's lines so that every language
    contributes approximately target_tokens whitespace tokens.

    Args:
        lang_texts:    {lang_code: [line, ...]}
        target_tokens: target token budget per language

    Returns:
        {lang_code: balanced_lines}
    """
    import random

    balanced: Dict[str, List[str]] = {}

    for lang, texts in lang_texts.items():
        texts = texts.copy()
        random.shuffle(texts)

        avg = _estimate_avg_tokens(texts)
        n_lines = max(1, int(target_tokens / avg))

        repeated: List[str] = []
        while len(repeated) < n_lines:
            repeated.extend(texts)

        balanced[lang] = repeated[:n_lines]
        logger.info(
            "Language '%s': %d source lines → %d balanced lines (avg %.1f tok/line)",
            lang, len(texts), len(balanced[lang]), avg,
        )

    return balanced


# ─────────────────────────────────────────────
# Manager
# ─────────────────────────────────────────────
class MultilingualTokenizerManager:
    """
    Trains and caches shared tokenizers over any number of languages.

    Usage
    -----
    manager = MultilingualTokenizerManager(output_dir="tokenizers")

    tokenizer = manager.get_tokenizer(
        lang_streams={
            "en": open("en.txt"),
            "fr": open("fr.txt"),
            "de": open("de.txt"),
        },
        tokenizer_type="bpe",
        vocab_size=65536,
        target_tokens=1_000_000,
    )
    """

    def __init__(self, output_dir: str = "tokenizers"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # In-memory cache: lang_key → Tokenizer
        self._cache: Dict[str, Tokenizer] = {}

    # ── Paths ────────────────────────────────────────────────────────────

    def _save_path(self, tokenizer_type: str, langs: List[str]) -> str:
        """
        Deterministic save path based on sorted language codes.
        Uses '_' as separator to avoid clashes with hyphenated codes (zh-Hant).
        """
        tag = "_".join(sorted(langs))
        return os.path.join(self.output_dir, f"{tokenizer_type}_{tag}.json")

    # ── Pre-tokenizers ───────────────────────────────────────────────────

    @staticmethod
    def _standard_pre_tokenizer():
        return pre_tokenizers.Whitespace()

    @staticmethod
    def _superbpe_pre_tokenizer():
        """
        Isolates digits and punctuation before BPE merges begin,
        producing 'super-token' boundaries.
        """
        return pre_tokenizers.Sequence([
            pre_tokenizers.WhitespaceSplit(),
            pre_tokenizers.Punctuation(),
            pre_tokenizers.Split(
                pattern=re.compile(r"\d+"),
                behavior="isolated",
            ),
        ])

    # ── Streaming compression iterators ────────────────────────────────

    @staticmethod
    def _interleave(
        lang_iters: Dict[str, Iterator[str]],
        target_tokens: int,
    ) -> Iterator[str]:
        """
        Equal-compression round-robin: every language gets the same
        *target_tokens* budget.

        Unlike materialising all data up-front, this works with infinite or
        very large file iterators.
        """
        counts: Dict[str, int] = {lang: 0 for lang in lang_iters}
        active: Dict[str, Iterator[str]] = dict(lang_iters)

        while active:
            for lang in list(active):
                # Language has met its budget — retire it.
                if counts[lang] >= target_tokens:
                    del active[lang]
                    logger.debug("Language '%s' reached token budget (%d).", lang, target_tokens)
                    continue

                try:
                    line = next(active[lang])
                    counts[lang] += len(line.split())
                    yield line
                except StopIteration:
                    del active[lang]
                    logger.warning(
                        "Language '%s' stream exhausted at %d / %d tokens.",
                        lang, counts[lang], target_tokens,
                    )

    @staticmethod
    def _interleave_weighted(
        lang_iters: Dict[str, Iterator[str]],
        lang_weights: Dict[str, float],
        total_tokens: int,
    ) -> Iterator[str]:
        """
        Weighted round-robin: each language gets a share of *total_tokens*
        proportional to its weight.

        Parameters
        ----------
        lang_iters   : {lang: text_iterator}
        lang_weights : {lang: weight} — need not sum to 1 (normalised here).
        total_tokens : total token budget across all languages.
        """
        weight_sum = sum(lang_weights.get(lang, 1.0) for lang in lang_iters) or 1.0
        budgets = {
            lang: max(1, int(total_tokens * lang_weights.get(lang, 1.0) / weight_sum))
            for lang in lang_iters
        }
        counts: Dict[str, int] = {lang: 0 for lang in lang_iters}
        active: Dict[str, Iterator[str]] = dict(lang_iters)

        while active:
            for lang in list(active):
                if counts[lang] >= budgets[lang]:
                    del active[lang]
                    logger.debug(
                        "Language '%s' reached weighted budget (%d / %d).",
                        lang, counts[lang], budgets[lang],
                    )
                    continue

                try:
                    line = next(active[lang])
                    counts[lang] += len(line.split())
                    yield line
                except StopIteration:
                    del active[lang]
                    logger.warning(
                        "Language '%s' stream exhausted at %d / %d tokens.",
                        lang, counts[lang], budgets[lang],
                    )

    @staticmethod
    def _compute_proportional_weights(
        lang_streams: Dict[str, Union[Iterator[str], List[str]]],
        sample_size: int = 5000,
    ) -> Dict[str, float]:
        """Estimate corpus size per language from a sample, return weights.

        Only works for list-based streams.  Iterator streams fall back to
        equal weights (1.0 each).
        """
        sizes: Dict[str, float] = {}
        for lang, stream in lang_streams.items():
            if isinstance(stream, list):
                sample = stream[:sample_size]
                sizes[lang] = float(
                    sum(len(line.split()) for line in sample)
                )
            else:
                sizes[lang] = 1.0
        total = sum(sizes.values()) or 1.0
        return {lang: size / total for lang, size in sizes.items()}

    # ── Vocab scaling heuristic ──────────────────────────────────────────

    @staticmethod
    def suggested_vocab_size(n_langs: int, base: int = 32768) -> int:
        """
        Scale vocab size with √n_langs so shared capacity grows with language
        count without exploding embedding tables.

        n_langs=1 → base, n_langs=4 → 2×base, n_langs=16 → 4×base.
        Result is rounded to the nearest power of two for hardware efficiency.
        """
        raw = base * math.sqrt(n_langs)
        # Round to nearest power of two
        power = round(math.log2(raw))
        return 2 ** power

    # ── Model + trainer factory ──────────────────────────────────────────

    def _build_model_and_trainer(
        self,
        tokenizer_type: str,
        vocab_size: int,
        lang_tokens: List[str],
    ):
        special_tokens = BASE_SPECIAL_TOKENS + lang_tokens

        if tokenizer_type in {"bpe", "superbpe"}:
            tok_model = models.BPE(unk_token="<UNK>")
            trainer_kwargs = dict(
                vocab_size=vocab_size,
                special_tokens=special_tokens,
                show_progress=True,
            )
            if tokenizer_type == "superbpe":
                # '##' prefix marks continuation subwords, matching BERT-style
                # downstream decoders that expect it.
                trainer_kwargs["continuing_subword_prefix"] = "##"
            trainer = trainers.BpeTrainer(**trainer_kwargs)

        elif tokenizer_type == "unigram":
            tok_model = models.Unigram()
            trainer = trainers.UnigramTrainer(
                vocab_size=vocab_size,
                special_tokens=special_tokens,
                unk_token="<UNK>",
            )
        else:
            raise ValueError(
                f"Unknown tokenizer_type {tokenizer_type!r}. "
                f"Supported: {SUPPORTED_TYPES}"
            )

        return tok_model, trainer

    # ── Public API ───────────────────────────────────────────────────────

    def get_tokenizer(
        self,
        lang_streams: Dict[str, Union[Iterator[str], List[str]]],
        tokenizer_type: str = "bpe",
        vocab_size: Optional[int] = None,
        target_tokens: int = 1_000_000,
        overwrite: bool = False,
        add_lang_tokens: bool = True,
        shared: bool = True,
        compression: str = "equal",
        lang_weights: Optional[Dict[str, float]] = None,
    ) -> Union[Tokenizer, Dict[str, Tokenizer]]:
        """
        Return a trained (or cached) tokenizer for all provided languages.

        Parameters
        ----------
        lang_streams:
            Mapping of language code → text iterator or list of strings.
            Any number of languages ≥ 1 is accepted.
        tokenizer_type:
            One of 'bpe', 'superbpe', 'unigram'.
        vocab_size:
            Vocabulary size.  When None, ``suggested_vocab_size`` is used
            automatically based on the number of languages.
        target_tokens:
            Token budget per language for equal-compression interleaving.
        overwrite:
            Re-train even if a saved tokenizer already exists.
        add_lang_tokens:
            Prepend per-language control tokens (<lang:xx>) to special_tokens.
            Enables language-conditioned downstream models.
        shared : bool
            If True (default), train one shared tokenizer over all languages.
            If False, train one separate tokenizer per language key and return
            a dict {lang_key: Tokenizer}.  Per-language tokenizers are saved
            with a ``_perlang_{lang}`` suffix in the filename.
            Used for vocabulary-sharing investigation in human-scale experiments.
        compression : str
            Compression strategy for balancing token budgets across languages.
            ``"equal"`` (default): every language gets *target_tokens*.
            ``"proportional"``: budget proportional to raw corpus size.
            ``"weighted"``: budget set by explicit *lang_weights*.
        lang_weights : dict[str, float] | None
            Per-language weights for ``compression="weighted"``.
            Ignored for other compression modes.

        Returns
        -------
        tokenizers.Tokenizer       (shared=True)
        dict[str, Tokenizer]       (shared=False)
        """
        if not shared:
            return self._get_per_language_tokenizers(
                lang_streams=lang_streams,
                tokenizer_type=tokenizer_type,
                vocab_size=vocab_size,
                target_tokens=target_tokens,
                overwrite=overwrite,
                add_lang_tokens=add_lang_tokens,
            )
        if not lang_streams:
            raise ValueError("lang_streams must contain at least one language.")

        tokenizer_type = tokenizer_type.lower()
        if tokenizer_type not in SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported tokenizer_type {tokenizer_type!r}. "
                f"Choose from: {SUPPORTED_TYPES}"
            )

        langs     = sorted(lang_streams.keys())
        lang_key  = f"{tokenizer_type}__{'_'.join(langs)}"
        save_path = self._save_path(tokenizer_type, langs)

        # ── Cache hit ────────────────────────────────────────────────────
        if lang_key in self._cache and not overwrite:
            logger.info("Returning cached tokenizer for %s.", lang_key)
            return self._cache[lang_key]

        # ── Disk hit ─────────────────────────────────────────────────────
        if os.path.exists(save_path) and not overwrite:
            logger.info("Loading tokenizer from %s.", save_path)
            tokenizer = Tokenizer.from_file(save_path)
            self._cache[lang_key] = tokenizer
            return tokenizer

        # ── Vocab size ───────────────────────────────────────────────────
        if vocab_size is None:
            vocab_size = self.suggested_vocab_size(len(langs))
            logger.info(
                "vocab_size not set — using suggested %d for %d language(s).",
                vocab_size, len(langs),
            )

        # ── Per-language control tokens ───────────────────────────────────
        lang_tokens = [f"<lang:{l}>" for l in langs] if add_lang_tokens else []
        if lang_tokens:
            logger.info("Adding language control tokens: %s", lang_tokens)

        # ── Build model + trainer ─────────────────────────────────────────
        tok_model, trainer = self._build_model_and_trainer(
            tokenizer_type, vocab_size, lang_tokens
        )
        tokenizer = Tokenizer(tok_model)

        # ── Pre-tokenizer ─────────────────────────────────────────────────
        if tokenizer_type == "superbpe":
            tokenizer.pre_tokenizer = self._superbpe_pre_tokenizer()
        else:
            tokenizer.pre_tokenizer = self._standard_pre_tokenizer()

        # ── Train ────────────────────────────────────────────────────────
        logger.info(
            "Training %s tokenizer on %d language(s): %s | vocab=%d | target_tokens=%d",
            tokenizer_type, len(langs), langs, vocab_size, target_tokens,
        )

        lang_iters = {lang: iter(stream) for lang, stream in lang_streams.items()}

        if compression == "equal":
            text_iterator = self._interleave(lang_iters, target_tokens)
        elif compression == "proportional":
            weights = self._compute_proportional_weights(lang_streams)
            total = target_tokens * len(langs)
            text_iterator = self._interleave_weighted(lang_iters, weights, total)
            logger.info("Proportional compression weights: %s", weights)
        elif compression == "weighted":
            if not lang_weights:
                raise ValueError(
                    "compression='weighted' requires lang_weights dict "
                    "(e.g. {'en': 0.6, 'fr': 0.4})"
                )
            total = target_tokens * len(langs)
            text_iterator = self._interleave_weighted(lang_iters, lang_weights, total)
            logger.info("Weighted compression: %s", lang_weights)
        else:
            raise ValueError(
                f"Unknown compression strategy: {compression!r}. "
                f"Choose from: 'equal', 'proportional', 'weighted'."
            )

        tokenizer.train_from_iterator(text_iterator, trainer=trainer)

        # ── Save ─────────────────────────────────────────────────────────
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        tokenizer.save(save_path)
        logger.info("Tokenizer saved to %s.", save_path)

        self._cache[lang_key] = tokenizer
        return tokenizer

    # ── Per-language tokenizer training ──────────────────────────────────

    def _get_per_language_tokenizers(
        self,
        lang_streams: Dict[str, Union[Iterator[str], List[str]]],
        tokenizer_type: str,
        vocab_size: Optional[int],
        target_tokens: int,
        overwrite: bool,
        add_lang_tokens: bool,
    ) -> Dict[str, Tokenizer]:
        """Train one independent tokenizer per language key.

        Each tokenizer is saved as ``{type}_{lang}_perlang.json`` to distinguish
        from shared tokenizers.  Returns {lang_key: Tokenizer}.
        """
        result: Dict[str, Tokenizer] = {}
        for lang_key, stream in lang_streams.items():
            logger.info(
                "[Per-language] Training tokenizer for '%s' (type=%s, vocab=%s)",
                lang_key, tokenizer_type, vocab_size,
            )
            # Temporarily override save path for per-language variant
            original_save_path_fn = self._save_path

            def _perlang_save_path(tt: str, langs: List[str], _lk: str = lang_key) -> str:
                tag = "_".join(sorted(langs))
                return os.path.join(self.output_dir, f"{tt}_{tag}_perlang_{_lk}.json")

            # Monkey-patch save path for this call only
            self._save_path = _perlang_save_path  # type: ignore[method-assign]
            try:
                tok = self.get_tokenizer(
                    lang_streams={lang_key: stream},
                    tokenizer_type=tokenizer_type,
                    vocab_size=vocab_size,
                    target_tokens=target_tokens,
                    overwrite=overwrite,
                    add_lang_tokens=add_lang_tokens,
                    shared=True,  # recurse into the shared path for single-lang
                )
            finally:
                self._save_path = original_save_path_fn  # type: ignore[method-assign]
            result[lang_key] = tok
        return result

    # ── Convenience: encode / decode ─────────────────────────────────────

    def encode(self, tokenizer: Tokenizer, text: str) -> List[int]:
        return tokenizer.encode(text).ids

    def decode(self, tokenizer: Tokenizer, ids: List[int]) -> str:
        return tokenizer.decode(ids)

    def batch_encode(
        self, tokenizer: Tokenizer, texts: List[str]
    ) -> List[List[int]]:
        return [tokenizer.encode(t).ids for t in texts]

    # ── Diagnostics ──────────────────────────────────────────────────────

    def vocab_size(self, tokenizer: Tokenizer) -> int:
        return tokenizer.get_vocab_size()

    def list_cached(self) -> List[str]:
        return list(self._cache.keys())

    def evict(self, lang_key: str) -> None:
        """Remove a tokenizer from the in-memory cache (disk copy is kept)."""
        self._cache.pop(lang_key, None)
