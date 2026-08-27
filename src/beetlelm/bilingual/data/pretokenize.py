"""
src/bilingual/data/pretokenize.py

Pretokenized Arrow dataset loading for BabyBabel / human-scale pretraining.

Pretokenization itself is performed in the `beetle-data` repo
(`pipeline/pretokenize_babybabel.py`), which packs each document into
fixed-length (seq_len+1) chunks and pushes the resulting Arrow datasets to
the `Beetle-HumanScale` HuggingFace org. This module only downloads and
caches those Arrow datasets and exposes them to the trainer as an
`IterableDataset`.

Usage (from beetle_trainer._setup_data_pipeline)
-------------------------------------------------
    arrows = load_pretokenized_hf_multilang(
        lang_datasets={"l1": "Beetle-HumanScale/nld-50M-eng-nld",
                       "l2": "Beetle-HumanScale/eng-50M-eng-nld"},
        cache_dir="./pretok_cache/my_run",
    )
    dataset = PretokenizedMultilingualDataset(
        arrows=arrows,
        pipeline_state=self.pipeline_state,
        seq_len=512,
    )
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Iterator

import torch
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from torch.utils.data import IterableDataset

from beetlelm.bilingual.data.pipeline_state import DataPipelineState

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Load already-pretokenized Arrow datasets from HuggingFace Hub
# ─────────────────────────────────────────────────────────────────────────────

def load_pretokenized_hf_arrow(
    hf_dataset_name: str,
    cache_dir: str,
    lang_key: str,
    overwrite: bool = False,
) -> str:
    """Download an already-pretokenized HF Arrow dataset.

    The HF dataset must already contain an ``input_ids`` column with
    fixed-length chunks (seq_len+1 tokens), as produced by
    `beetle-data/pipeline/pretokenize_babybabel.py`.

    Parameters
    ----------
    hf_dataset_name : str
        HuggingFace dataset identifier, e.g. "Beetle-HumanScale/nld-50M-eng-nld".
    cache_dir : str
        Local directory to cache the Arrow dataset.
    lang_key : str
        Short language identifier (e.g. "l1"). Used to name the output subdir.
    overwrite : bool
        If False and the output subdir already exists, skip download.

    Returns
    -------
    str
        Absolute path to the local Arrow dataset directory.
    """
    out_path = Path(cache_dir) / lang_key
    if out_path.exists() and not overwrite:
        try:
            load_from_disk(str(out_path))
            logger.info("[LoadHFArrow] Cache hit for '%s' at %s — skipping.", lang_key, out_path)
            return str(out_path)
        except Exception:
            logger.warning("[LoadHFArrow] Invalid cache for '%s' at %s — recomputing.", lang_key, out_path)
            shutil.rmtree(out_path, ignore_errors=True)

    logger.info("[LoadHFArrow] Downloading '%s' from %s ...", lang_key, hf_dataset_name)
    out_path.mkdir(parents=True, exist_ok=True)

    try:
        ds = load_dataset(hf_dataset_name, split="train")
    except Exception as e:
        # Fallback: the installed `datasets` version can't parse the Arrow
        # schema metadata (e.g. an unknown feature `_type` pushed by a newer
        # version — raises TypeError/ValueError/etc depending on the lib
        # version). Since these shards are plain `input_ids: list<int>`,
        # bypass `Features.from_arrow_schema` by downloading the raw Arrow
        # files and loading them directly.
        logger.warning(
            "[LoadHFArrow] load_dataset failed for '%s' (%s: %s). "
            "Falling back to raw Arrow file ingestion.",
            hf_dataset_name, type(e).__name__, e,
        )
        import pyarrow as pa
        import pyarrow.ipc as ipc
        from huggingface_hub import snapshot_download

        local_repo = snapshot_download(
            repo_id=hf_dataset_name,
            repo_type="dataset",
            allow_patterns=["*.arrow", "*.parquet"],
            cache_dir=os.environ.get("HUGGINGFACE_HUB_CACHE"),
        )
        # Collect all local Arrow/Parquet files from the downloaded snapshot.
        arrow_files  = sorted(Path(local_repo).rglob("*.arrow"))
        parquet_files = sorted(Path(local_repo).rglob("*.parquet"))

        if parquet_files:
            # Prefer Parquet: load_dataset("parquet", ...) uses memory-mapped
            # Arrow under the hood — no heap copy regardless of file size.
            ds = load_dataset(
                "parquet",
                data_files=[str(p) for p in parquet_files],
                split="train",
            )
        elif arrow_files:
            # Dataset.from_file uses memory-mapped IPC access — avoids the
            # read_all() + concat_tables() RAM spike of the old approach.
            from datasets import concatenate_datasets as _cat
            parts = [Dataset.from_file(str(p)) for p in arrow_files]
            ds = _cat(parts) if len(parts) > 1 else parts[0]
        else:
            raise RuntimeError(
                f"No .arrow or .parquet files found under {local_repo} for "
                f"{hf_dataset_name}; cannot use raw-file fallback."
            ) from e

    # Validate that the dataset is already pretokenized
    if "input_ids" not in ds.column_names:
        raise ValueError(
            f"Dataset {hf_dataset_name} does not contain 'input_ids' column. "
            f"Available columns: {ds.column_names}. "
            f"Pretokenization is performed in the beetle-data repo "
            f"(pipeline/pretokenize_babybabel.py); this loader only consumes "
            f"already-pretokenized Arrow datasets."
        )

    logger.info(
        "[LoadHFArrow] '%s': %d sequences (columns: %s)",
        lang_key, len(ds), ds.column_names,
    )

    ds.save_to_disk(str(out_path))
    logger.info("[LoadHFArrow] Saved '%s' to %s", lang_key, out_path)
    return str(out_path)


def load_pretokenized_hf_multilang(
    lang_datasets: Dict[str, str],
    cache_dir: str,
    overwrite: bool = False,
) -> Dict[str, str]:
    """Download multiple already-pretokenized Arrow datasets from HuggingFace.

    Parameters
    ----------
    lang_datasets : dict[str, str]
        {lang_key: hf_dataset_name}, e.g.
        {"l1": "Beetle-HumanScale/nld-50M-eng-nld",
         "l2": "Beetle-HumanScale/eng-50M-eng-nld"}
    cache_dir : str
        Root cache directory.
    overwrite : bool

    Returns
    -------
    dict[str, str]
        {lang_key: arrow_path}
    """
    return {
        lang_key: load_pretokenized_hf_arrow(
            hf_dataset_name=hf_name,
            cache_dir=cache_dir,
            lang_key=lang_key,
            overwrite=overwrite,
        )
        for lang_key, hf_name in lang_datasets.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# IterableDataset wrapper
# ─────────────────────────────────────────────────────────────────────────────

class PretokenizedMultilingualDataset(IterableDataset):
    """IterableDataset over pretokenized Arrow datasets, mixing languages via
    the DataPipelineState curriculum.

    At each step, DataMixConfig.next_source() selects the active language, and
    this dataset samples the next chunk from that language's Arrow dataset.

    Notes
    -----
    - DataMixConfig.step() must be called by the *training loop* (not here)
      to keep state advancement in one place and avoid double-stepping.
    - Each worker in a multi-worker DataLoader gets the same pipeline_state
      reference; the DDP rank-level sharding is handled upstream by the trainer.
    """

    def __init__(
        self,
        arrows: Dict[str, str],
        pipeline_state: DataPipelineState,
        seq_len: int,
    ) -> None:
        super().__init__()
        self.pipeline_state = pipeline_state
        self.seq_len = seq_len

        # Load Arrow datasets into memory-mapped format
        self._pools: Dict[str, Dataset] = {
            lang_key: load_from_disk(path)
            for lang_key, path in arrows.items()
        }
        # Per-language cyclic index (avoids reshuffling overhead mid-epoch)
        self._indices: Dict[str, int] = {k: 0 for k in self._pools}
        # Register on the pipeline_state so our _indices are saved/restored
        # alongside its state_dict, preserving multi-seed determinism on resume.
        if hasattr(pipeline_state, "register_dataset"):
            pipeline_state.register_dataset(self)

    def state_dict(self) -> Dict[str, int]:
        # Plain dict is json/pickle-safe; copy to decouple from live state.
        return {"indices": dict(self._indices)}

    def load_state_dict(self, state: Dict[str, int]) -> None:
        if not isinstance(state, dict):
            return
        saved = state.get("indices") or {}
        for k in self._indices:
            if k in saved:
                self._indices[k] = int(saved[k])

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        while True:
            lang_key = self.pipeline_state.data_mix.next_source()
            if lang_key not in self._pools:
                raise KeyError(
                    f"Curriculum requested lang={lang_key!r} but pretokenized "
                    f"pools only contain {sorted(self._pools)}. "
                    f"Check that the curriculum condition matches the configured "
                    f"lang_sources / l2_streams."
                )
            pool = self._pools[lang_key]
            idx = self._indices[lang_key] % len(pool)
            self._indices[lang_key] += 1
            ids = pool[idx]["input_ids"]
            yield {"input_ids": torch.tensor(ids, dtype=torch.long)}
