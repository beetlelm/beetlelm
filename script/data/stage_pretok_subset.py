#!/usr/bin/env python3
"""
script/data/stage_pretok_subset.py

Stream a subset of an HF pretokenized Arrow/Parquet dataset (e.g.
``Beetle-Data/nl-28B``) and push it as a smaller HF dataset sized to the
training token budget plus headroom (e.g. ``Beetle-Data/nl-100M-pretok``).

Eliminates the structural over-download in
``beetlelm/bilingual/data/pretokenize.py:load_pretokenized_hf_arrow`` for cases
where a 28B-token source shard is consumed by a 100M or 2B training run.

Design notes
------------
* Streaming-only: uses ``datasets.load_dataset(..., streaming=True)`` so host
  RAM is bounded by ``--rows-per-shard`` (default 50k rows ≈ tens of MB),
  regardless of source-shard size or subset size.
* Idempotent: ``--skip-if-exists`` no-ops when the destination already has
  data files. Safe to call from a launcher's prep step on every run.
* Output is plain Parquet shards uploaded via ``upload_folder``; the loader
  side (``load_pretokenized_hf_arrow``) consumes ``*.arrow`` or ``*.parquet``
  via either ``load_dataset`` or its raw-Arrow fallback.

Usage
-----
  python script/data/stage_pretok_subset.py \
      --src-repo Beetle-Data/nl-28B \
      --dst-repo Beetle-Data/nl-100M-pretok \
      --target-tokens 100000000 \
      --seq-len 512 \
      --safety 2.0 \
      --skip-if-exists
"""
import argparse
import math
import os
import sys
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import HfApi, create_repo, hf_hub_download, upload_folder


def _dst_already_populated(api: HfApi, dst_repo: str) -> bool:
    """Return True iff dst_repo exists and contains parquet/arrow data files."""
    try:
        files = api.list_repo_files(dst_repo, repo_type="dataset")
    except Exception:
        return False
    return any(f.endswith((".parquet", ".arrow")) for f in files)


def _iter_rows(src_repo: str, api: HfApi, token):
    """Yield {'input_ids': [...]} rows from src_repo.

    Tries `load_dataset(..., streaming=True)` first. If the installed
    `datasets` version can't parse the Arrow schema metadata (e.g. unknown
    `List` feature `_type` written by a newer datasets release), falls back
    to enumerating the repo's `.parquet`/`.arrow` files via the HF Hub API
    and reading them directly with PyArrow. The fallback streams one file
    (and one record batch) at a time, so the caller's `break` on
    `n_rows_target` keeps download volume bounded.
    """
    try:
        ds = load_dataset(src_repo, split="train", streaming=True)
        for row in ds:
            yield row
        return
    except Exception as e:
        print(
            f"[stage] load_dataset failed ({type(e).__name__}: {e}); "
            f"falling back to raw HF file streaming.",
            file=sys.stderr,
        )

    files = [
        f for f in api.list_repo_files(src_repo, repo_type="dataset")
        if f.endswith((".arrow", ".parquet"))
    ]
    files.sort()
    if not files:
        raise RuntimeError(f"No .arrow/.parquet files in {src_repo}")

    for rel in files:
        local = hf_hub_download(
            repo_id=src_repo, filename=rel,
            repo_type="dataset", token=token,
        )
        if rel.endswith(".parquet"):
            pf = pq.ParquetFile(local)
            for batch in pf.iter_batches(batch_size=4096, columns=["input_ids"]):
                # Drop HF metadata so unknown `_type` entries don't surface
                # if anything later tries to interpret the schema.
                tbl = pa.Table.from_batches([batch]).replace_schema_metadata({})
                for v in tbl.column("input_ids").to_pylist():
                    yield {"input_ids": v}
        else:  # .arrow
            source = pa.memory_map(local, "r")
            try:
                reader = ipc.open_stream(source)
                stream_mode = True
            except pa.ArrowInvalid:
                source.seek(0)
                reader = ipc.open_file(source)
                stream_mode = False

            if stream_mode:
                while True:
                    try:
                        batch = reader.read_next_batch()
                    except StopIteration:
                        break
                    tbl = pa.Table.from_batches([batch]).replace_schema_metadata({})
                    for v in tbl.column("input_ids").to_pylist():
                        yield {"input_ids": v}
            else:
                for i in range(reader.num_record_batches):
                    batch = reader.get_batch(i)
                    tbl = pa.Table.from_batches([batch]).replace_schema_metadata({})
                    for v in tbl.column("input_ids").to_pylist():
                        yield {"input_ids": v}


def main() -> int:
    p = argparse.ArgumentParser(
        description="Stream a token-budget subset from an HF dataset and push it as a new HF dataset.",
    )
    p.add_argument("--src-repo", required=True,
                   help="Source HF dataset repo (e.g. Beetle-Data/nl-28B).")
    p.add_argument("--dst-repo", required=True,
                   help="Destination HF dataset repo (e.g. Beetle-Data/nl-100M-pretok).")
    p.add_argument("--target-tokens", type=int, required=True,
                   help="Training token budget; subset is sized to safety x this.")
    p.add_argument("--seq-len", type=int, default=512,
                   help="Sequence length used at training time. Each row holds seq_len+1 tokens.")
    p.add_argument("--safety", type=float, default=2.0,
                   help="Multiplier on target_tokens for headroom. Default 2.0.")
    p.add_argument("--rows-per-shard", type=int, default=50_000,
                   help="Rows per output parquet shard. Bounds host RAM. Default 50k.")
    p.add_argument("--skip-if-exists", action="store_true",
                   help="No-op when --dst-repo already has data files.")
    p.add_argument("--private", action="store_true",
                   help="Create destination repo as private.")
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN")
    if token is None:
        print("WARN: HF_TOKEN is not set; uploads will fail.", file=sys.stderr)

    api = HfApi(token=token)

    if args.skip_if_exists and _dst_already_populated(api, args.dst_repo):
        print(f"[stage] {args.dst_repo}: already populated -> skip.")
        return 0

    chunk_tokens = args.seq_len + 1
    n_rows_target = math.ceil(args.target_tokens * args.safety / chunk_tokens)
    print(f"[stage] {args.src_repo} -> {args.dst_repo}")
    print(f"[stage]   target_tokens={args.target_tokens:,}  safety={args.safety}  "
          f"chunk_tokens={chunk_tokens}  n_rows={n_rows_target:,}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        shard_idx = 0
        rows_collected = 0
        buf_ids: list = []

        def flush() -> None:
            nonlocal shard_idx, buf_ids
            if not buf_ids:
                return
            tbl = pa.table({
                "input_ids": pa.array(buf_ids, type=pa.list_(pa.int64())),
            })
            out = tmp_dir / f"part-{shard_idx:05d}.parquet"
            pq.write_table(tbl, out, compression="zstd")
            shard_idx += 1
            buf_ids = []

        for row in _iter_rows(args.src_repo, api, token):
            ids = row.get("input_ids")
            if ids is None:
                raise RuntimeError(
                    f"{args.src_repo}: row missing 'input_ids' column. "
                    f"Got keys: {list(row.keys())}"
                )
            buf_ids.append(ids)
            rows_collected += 1
            if len(buf_ids) >= args.rows_per_shard:
                flush()
                print(f"[stage]   shard {shard_idx-1} written; "
                      f"rows={rows_collected:,}/{n_rows_target:,}")
            if rows_collected >= n_rows_target:
                break

        flush()
        print(f"[stage]   total_rows={rows_collected:,}  shards={shard_idx}")

        if rows_collected < n_rows_target:
            print(f"[stage] WARN: source had only {rows_collected:,} rows "
                  f"(< target {n_rows_target:,}); pushing what we have.",
                  file=sys.stderr)

        create_repo(args.dst_repo, repo_type="dataset", exist_ok=True,
                    private=args.private, token=token)
        upload_folder(folder_path=str(tmp_dir), repo_id=args.dst_repo,
                      repo_type="dataset", token=token)
        print(f"[stage] uploaded -> https://huggingface.co/datasets/{args.dst_repo}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
