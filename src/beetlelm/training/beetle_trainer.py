#!/usr/bin/env python3
"""
BeetleLM Trainer  —  monolingual · bilingual · polyglot
with optional continual-learning extensions (EWC + MAML/SMLMT)

HUGGINGFACE COMPATIBILITY
--------------------------
Every checkpoint branch is immediately loadable with:

    model = AutoModelForCausalLM.from_pretrained(
        "BeetleLM/beetlelm_eng_mono",
        revision="step-1000",
        trust_remote_code=True,
    )

CONTINUAL-LEARNING EXTENSIONS
-------------------------------
    ewc:
      enabled: true
      lambda_: 20.0
      n_fisher_samples: 10
      mu: 0.0
      trigger: phase_change

    maml:
      enabled: true
      hybrid_ratio: 0.5
      n_way: 5
      k_shot: 1
      inner_steps: 3
      inner_lr: 0.1
      head:
        num_layers: 2
        hidden_dim: 256
        init_method: xavier

CHECKPOINTING
-------------
Uses beetlelm.checkpointing.training exclusively (shared with the pico trainer).
beetlelm.training.utils.checkpointing is a deprecated shim that re-exports from there.

Checkpoints are saved as step_{N}/ directories via fabric.save().
Local pruning: keep_local_checkpoints (default 1) in the checkpointing YAML block.
HF push:       each step_{N}/ directory is pushed as branch step-{N}.
Resume:        _resume_from_hf uses snapshot_download to fetch step_{N}/ from HF.
"""

import os
import re
import signal
import logging
import argparse
import threading
import time
from queue import Queue
from threading import Thread
from types import SimpleNamespace
from dataclasses import asdict, is_dataclass
from typing import Dict, Iterator, List, Optional
import json as json_mod
import tempfile
import copy
import yaml

# ── HF Hub network timeouts ─────────────────────────────────────────────────
# Bound every HF Hub HTTP call so a network hang or DNS blackhole cannot
# wedge the AsyncCheckpointer worker thread (and through it, the next
# `fabric.barrier()`, freezing the whole 8xA100 node). These env vars are
# read by `huggingface_hub.utils._http.configure_http_backend` at import
# time, so they must be set BEFORE the `huggingface_hub` import below.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")

from huggingface_hub import HfApi, create_repo, upload_file, upload_folder, snapshot_download
import torch
import torch.nn.functional as F
from datasets import load_dataset
from lightning.fabric.utilities.rank_zero import rank_zero_only
from transformers import PreTrainedTokenizerFast
from beetlelm.bilingual.data.datamodule import BilingualDataModule
from beetlelm.bilingual.tok.multilingual_tokenizer_manager import MultilingualTokenizerManager
from beetlelm.bilingual.data.condition_presets import get_condition, CONDITION_ALIASES
from beetlelm.bilingual.data.stream_state import DeterministicStreamState
from beetlelm.bilingual.data.pipeline_state import DataPipelineState
from beetlelm.checkpointing.curriculum_scheduler import CurriculumCheckpointScheduler
# ── Single canonical checkpointing module for all trainers ───────────────────
from beetlelm.checkpointing.training import (
    load_checkpoint,
    save_checkpoint,
    prune_local_checkpoints,
)
from beetlelm.training.utils.initialization import (
    initialize_configuration,
    initialize_fabric,
    initialize_optimizer,
    initialize_lr_scheduler,
)
from beetlelm.optim import AdamWScheduleFree
from beetlelm.training.utils.logging import initialize_logging
from beetlelm.training.utils.io import use_backoff
from beetlelm.continual import (
    build_ewc, build_maml, EWCRegularizer, SMLMTModule,
    ExposureLRScheduler, build_exposure_lr,
    GenerativeReplayModule, build_generative_replay,
    SimilarityReplayModule, build_similarity_replay,
    TILTModule, build_tilt,
    SASRegularizer, build_sas_regularizer,
    GaussianNoiseInjection, build_gaussian_noise,
    KLDistillationModule, build_kl_distillation,
    LangIDHead,            build_lang_id_head,
    LoRAModule,            build_lora,
    LayerGrowthModule,     build_layer_growth,
    AlignmentLossModule,   build_alignment_loss,
)
from beetlelm.working_memory import build_working_memory, WorkingMemoryScheduler
from beetlelm.pre_pretraining import (
    build_pre_pretraining, run_pre_pretraining_stage, PrePretrainingConfig,
)
from beetlelm.bilingual.data.pretokenize import (
    load_pretokenized_hf_multilang,
    PretokenizedMultilingualDataset,
)

# Fallback only — configs should always set hf_user explicitly.
HUGGINGFACE_USER = "Beetle-HumanScale"

_PICO_DECODER_SRC = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "src", "pico_decoder", "pico_decoder.py",
)

# Per-architecture modeling source files. Used when staging the checkpoint
# directory before pushing to the HF ``step-{N}`` branch so that
# ``AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`` can
# resolve the custom class. ``register_for_auto_class()`` + ``save_pretrained``
# already copies the file; this is a defensive belt-and-braces measure that
# also catches the (currently relevant) case where save_pretrained is bypassed.
# Path resolution mirrors ``_PICO_DECODER_SRC``: try the absolute layout first,
# fall back to the cwd-relative ``src/`` layout used by Hyperion launches.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_ARCH_MODELING_SRC = {
    "pico":          ("pico_decoder",  "pico_decoder/pico_decoder.py"),
    "bgpt":          ("bgpt",          "bgpt/bgpt.py"),
    "beetle_moe":    ("beetle_moe",    "beetle_moe/beetle_moe.py"),
    "beetle_ssm":    ("beetle_ssm",    "beetle_ssm/beetle_ssm.py"),
    "geo_bilingual": ("geo_bilingual", "geo_bilingual/geo_bilingual.py"),
}


def _resolve_modeling_src(arch: str) -> Optional[tuple]:
    """Return (modeling_filename, absolute_source_path) for ``arch``, or None."""
    entry = _ARCH_MODELING_SRC.get(arch)
    if entry is None:
        return None
    module_name, rel = entry
    abs_candidates = [
        os.path.join(_REPO_ROOT, "src", rel),
        os.path.join("src", rel),
    ]
    for path in abs_candidates:
        if os.path.exists(path):
            return os.path.basename(path), path
    return os.path.basename(rel), None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _short(dataset_name: str) -> str:
    base = dataset_name.split("/")[-1].lower()
    for prefix in ("babylm-", "babybabel-"):
        if prefix in base:
            base = base.split(prefix, 1)[1]
            break
    return base.split("-")[0]


class PrefetchIterator:
    """Background-thread prefetcher over an iterator.

    Lifecycle / cleanup
    -------------------
    - Use as a context manager (``with PrefetchIterator(it) as p:``) where
      possible: ``__exit__`` calls ``close()`` and joins the worker thread
      (with a 5 s timeout) so file descriptors and sockets opened by the
      underlying iterator are released before the next sweep job starts.
    - ``close()`` is also wired into ``__del__`` for the legacy non-context
      path; it is idempotent.
    - The worker is non-daemon when ``daemon=False`` is requested at
      construction time; this matters when the iterator owns a network or
      filesystem handle that must finish flushing before process exit.
    """

    def __init__(self, iterator, buffer_size: int = 64, daemon: bool = True):
        self.iterator = iterator
        self.queue    = Queue(maxsize=buffer_size)
        self._stop    = False
        self._closed  = False

        def worker():
            try:
                for item in iterator:
                    if self._stop:
                        break
                    self.queue.put(item)
            finally:
                self.queue.put(None)

        self.thread = Thread(target=worker, daemon=daemon)
        self.thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if item is None:
            raise StopIteration
        return item

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._stop = True
        # Drain the queue so worker.put() can complete and the thread
        # observes self._stop; otherwise it can block on a full queue.
        try:
            while not self.queue.empty():
                self.queue.get_nowait()
        except Exception:
            pass
        try:
            self.thread.join(timeout=5)
        except Exception:
            pass
        # Forward close() to the underlying iterator if it supports it
        # (e.g. HF datasets streaming reader holds an HTTP connection).
        inner_close = getattr(self.iterator, "close", None)
        if callable(inner_close):
            try:
                inner_close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class TokenBuffer:
    def __init__(self, tokenizer, seq_len: int, buffer_tokens: int = 200_000):
        self.tokenizer     = tokenizer
        self.seq_len       = seq_len
        self.buffer_tokens = buffer_tokens
        self.tokens: list  = []

    def add_text(self, text: str):
        self.tokens.extend(self.tokenizer.encode(text))

    def ready(self) -> bool:
        return len(self.tokens) >= self.buffer_tokens

    def next_batch(self, batch_size: int):
        needed = batch_size * (self.seq_len + 1)
        if len(self.tokens) < needed:
            return None
        batch = []
        for _ in range(batch_size):
            seq         = self.tokens[: self.seq_len + 1]
            self.tokens = self.tokens[self.seq_len + 1:]
            batch.append(seq)
        return torch.tensor(batch, dtype=torch.long)


class AsyncCheckpointer:
    """
    Serializes background save/upload operations onto a single worker thread.

    Used to move rank-0-only HuggingFace uploads off the training critical path:
    the main loop calls ``save(...)`` to dispatch, and ``join()`` before the next
    dispatch to guarantee uploads are serialized (and before process teardown to
    guarantee the final upload has flushed).

    Wall-clock guard
    ----------------
    ``join(timeout=...)`` is bounded so a wedged HF Hub upload cannot block the
    main loop forever. If the background thread does not finish within the
    timeout, ``join()`` returns and records the upload as "in flight but
    abandoned" via ``upload_failed = True`` and ``timed_out = True`` so the
    caller can preserve the local checkpoint (the launcher-level cleanup is
    gated on ``.hf_upload_done`` written only when ``upload_failed`` is False).

    The thread is non-daemon on purpose — a daemon thread would be silently
    killed if the main thread exits early, potentially losing the final
    checkpoint upload.
    """

    # Default wall-clock cap for a single async save. Long enough to cover a
    # multi-GB checkpoint upload over a slow link, short enough that a wedged
    # connection doesn't burn an entire JOB_TIMEOUT.
    DEFAULT_JOIN_TIMEOUT_SEC = 900  # 15 minutes

    def __init__(self):
        self._lock   = threading.Lock()
        self._thread = None
        # Sticky flags consumed by the trainer's end-of-run sentinel writer.
        self.upload_failed: bool = False
        self.last_error: Optional[BaseException] = None
        self.timed_out: bool = False

    def save(self, save_fn, *args, **kwargs):
        def _run():
            try:
                with self._lock:
                    save_fn(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001 — record everything
                # Record but do NOT raise: uncaught exceptions in a non-daemon
                # thread can deadlock interpreter shutdown on some Pythons.
                self.upload_failed = True
                self.last_error = exc
                logging.getLogger(__name__).error(
                    "[AsyncCheckpointer] save_fn raised: %s: %s",
                    type(exc).__name__, exc,
                )
        self._thread = Thread(target=_run, daemon=False)
        self._thread.start()

    def join(self, timeout: Optional[float] = None) -> None:
        """Block until the in-flight save finishes, or until ``timeout`` seconds.

        On timeout we mark ``timed_out=True`` and ``upload_failed=True`` and
        return — the caller (the trainer's end-of-run path) uses these flags
        to decide whether the local checkpoint is safe to delete.
        """
        if self._thread is None:
            return
        wait = timeout if timeout is not None else self.DEFAULT_JOIN_TIMEOUT_SEC
        self._thread.join(timeout=wait)
        if self._thread.is_alive():
            self.upload_failed = True
            self.timed_out = True
            logging.getLogger(__name__).error(
                "[AsyncCheckpointer] join() timed out after %.0fs — upload "
                "thread is still running. Marking upload_failed=True so the "
                "launcher will preserve runs/<run_name>/ for safe RESUME.",
                wait,
            )
            # Leave the thread alive (it's non-daemon and will eventually
            # exit when the network call finally returns or errors). We
            # detach our reference so subsequent .save() can dispatch a new
            # one without blocking on the old one.
        self._thread = None


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, config_path: str, tokenizer_only: bool = False, seed: int = 42):
        self.config_path    = config_path
        self.tokenizer_only = tokenizer_only
        # Seed plumbing for multi-seed ablations (e.g. B1/B2/B3 significance sweep).
        # Overrides the historical hardcoded 42. Logged into HF config.json for
        # reproducibility.
        self.seed           = int(seed)

        raw_configs      = initialize_configuration(config_path)
        self.configs     = self._process_raw_config(raw_configs)

        self.checkpoint_cfg = self.configs.get("checkpointing", {})
        self.training_cfg   = self.configs.get("training", {})
        self.monitoring_cfg = self.configs.get("monitoring", {})
        self.data_cfg       = self.configs.get("data", {})
        self.tokenizer_cfg  = self.configs.get("tokenizer", {})

        # Pushback queue used by ``_collect_fisher_batches`` (and similar
        # introspection passes) on the pretokenized path: any batch consumed
        # from ``self.train_iterator`` for Fisher / centroid computation is
        # appended here so the main training loop can re-yield it on the next
        # iteration.  Without this, the first ``n_fisher_samples`` batches
        # after the L1→L2 boundary would be silently lost from the L2 loss.
        from collections import deque as _deque
        self._pushback_batches: "_deque" = _deque()

        self._ewc_cfg  = self.configs.get("ewc",  {}) or {}
        self._maml_cfg = self.configs.get("maml", {}) or {}
        self._gen_replay_cfg = self.configs.get("generative_replay", {}) or {}
        self._sim_replay_cfg = self.configs.get("similarity_replay", {}) or {}
        self._tilt_cfg = self.configs.get("tilt", {}) or {}
        self._working_memory_cfg = self.configs.get("working_memory", {}) or {}
        self._pre_pretraining_cfg = self.configs.get("pre_pretraining", {}) or {}
        self._sas_cfg = self.configs.get("sas_regularizer", {}) or {}
        self._gni_cfg = self.configs.get("gaussian_noise", {}) or {}
        self._kl_distill_cfg     = self.configs.get("kl_distillation",  {}) or {}
        self._lang_id_cfg        = self.configs.get("lang_id",           {}) or {}
        self._lora_cfg           = self.configs.get("lora",              {}) or {}
        self._layer_growth_cfg   = self.configs.get("layer_growth",      {}) or {}
        self._alignment_loss_cfg = self.configs.get("alignment_loss",    {}) or {}

        # Heartbeat plumbing — see _update_heartbeat. Initialized to 0 so the
        # first call writes immediately; subsequent calls throttle to once
        # every BP_HEARTBEAT_THROTTLE_S (default 10s) to avoid disk thrash.
        self._heartbeat_last_ts = 0.0
        self._heartbeat_throttle_s = float(os.environ.get("BP_HEARTBEAT_THROTTLE_S", "10"))

        self.run_name = self.configs.get("run_name")
        if not self.run_name:
            raise ValueError(f"'run_name' must be specified in YAML config: {config_path}")
        self.hf_user = self.configs.get("hf_user", HUGGINGFACE_USER)
        self.checkpoint_cfg["run_name"] = self.run_name

        hf_cache = self.data_cfg.get("hf_cache_dir", "./hf_cache")
        os.makedirs(hf_cache, exist_ok=True)
        os.environ.setdefault("HF_DATASETS_CACHE", hf_cache)

        self.logger       = initialize_logging()
        self.wandb_logger = None

        raw_training_obj = raw_configs.get("training", self.training_cfg)
        self.fabric = initialize_fabric(
            training_config=raw_training_obj,
            wandb_logger=self.wandb_logger,
        )
        self.fabric.seed_everything(self.seed)

        # Per-process CUDA memory cap. Forces a Python-catchable
        # ``torch.cuda.OutOfMemoryError`` at the configured fraction (default
        # 0.92 = 92% of per-GPU memory) BEFORE the kernel OOM-killer fires.
        # This is the difference between "one job dies cleanly" and "the
        # whole 8xA100 node wedges and needs a reboot". See
        # beetle_pipeline/lib/common.sh for the orchestrator-side watchdog
        # that catches anything this doesn't.
        try:
            _mem_frac = float(os.environ.get("BP_CUDA_MEM_FRACTION", "0.92"))
        except (TypeError, ValueError):
            _mem_frac = 0.92
        if torch.cuda.is_available() and 0.0 < _mem_frac < 1.0:
            try:
                _dev = self.fabric.device
                # Fabric returns torch.device("cuda:N") on GPU ranks; CPU
                # ranks return torch.device("cpu") and we skip silently.
                if getattr(_dev, "type", "cpu") == "cuda":
                    torch.cuda.set_per_process_memory_fraction(_mem_frac, _dev)
                    self.logger.info(
                        "[mem_guard] torch.cuda.set_per_process_memory_fraction(%.2f) on %s",
                        _mem_frac, _dev,
                    )
            except Exception as _e:
                # Never block training if the cap can't be installed
                # (older PyTorch, exotic device). Just log and continue.
                self.logger.warning(
                    "[mem_guard] set_per_process_memory_fraction failed: %r", _e,
                )

        self.async_checkpointer = AsyncCheckpointer()

        self.tokenizer = self._setup_tokenizer(self.tokenizer_cfg)
        if self.tokenizer_only:
            self.log("Tokenizer-only mode enabled. Training will be skipped.")
            return

        self._setup_data_pipeline()
        self.model = self._setup_model()

        # ── Optional working-memory ALiBi scheduler ──────────────────────────
        # Read num_heads from the freshly-built model config so the scheduler's
        # per-head slope tensor is sized correctly.
        _raw_model = getattr(self.model, "module", self.model)
        _n_heads = int(
            getattr(_raw_model.model_config, "attention_n_heads",
                    getattr(_raw_model.model_config, "n_heads", 12))
        )
        self._working_memory: Optional[WorkingMemoryScheduler] = build_working_memory(
            self._working_memory_cfg, num_heads=_n_heads,
        )
        if self._working_memory is not None:
            if not hasattr(_raw_model, "set_working_memory"):
                raise RuntimeError(
                    "[working_memory] Model architecture does not support the "
                    "working_memory hook. Supported architectures: 'pico', 'bgpt'."
                )
            _raw_model.set_working_memory(self._working_memory)

        # ── Optional formal-language pre-pretraining stage ───────────────────
        self._pre_pretraining: Optional[PrePretrainingConfig] = build_pre_pretraining(
            self._pre_pretraining_cfg
        )

        self._ewc:  Optional[EWCRegularizer] = build_ewc(self._ewc_cfg)
        self._maml: Optional[SMLMTModule]    = self._init_maml()
        # Auto-resolve <lang:{L1}> → token id for LAMOL-style task-conditional
        # generation, so configs don't have to hand-encode tokenizer ids.
        # The factory will raise if prompt_style='lang_tag' is requested but no
        # id is resolvable — that's intentional, it surfaces tokenizer/cfg
        # mismatches loudly instead of silently degrading to unconditional.
        self._maybe_resolve_gen_replay_lang_token()
        self._gen_replay: Optional[GenerativeReplayModule] = build_generative_replay(self._gen_replay_cfg)
        self._sim_replay: Optional[SimilarityReplayModule] = build_similarity_replay(self._sim_replay_cfg)
        self._tilt: Optional[TILTModule]     = build_tilt(self._tilt_cfg)
        self._sas:  Optional[SASRegularizer]        = build_sas_regularizer(self._sas_cfg)
        self._gni:  Optional[GaussianNoiseInjection] = build_gaussian_noise(self._gni_cfg)
        self._kl_distill: Optional[KLDistillationModule] = build_kl_distillation(self._kl_distill_cfg)
        if self._sas is not None:
            self._sas.attach_probe(self.model)

        # ── Lang-ID head — built BEFORE optimizer so head params are included ─
        _lang_id_lang_keys = list(self.data_cfg.get("lang_keys", []))
        self._lang_id: Optional[LangIDHead] = build_lang_id_head(
            self._lang_id_cfg, self.model, lang_keys=_lang_id_lang_keys
        )

        # ── LoRA / Layer-growth / Alignment-loss — built but not yet applied ──
        self._lora:           Optional[LoRAModule]         = build_lora(self._lora_cfg)
        self._layer_growth:   Optional[LayerGrowthModule]  = build_layer_growth(self._layer_growth_cfg)
        self._alignment_loss: Optional[AlignmentLossModule] = build_alignment_loss(self._alignment_loss_cfg)
        if self._alignment_loss is not None:
            self._alignment_loss.attach(self.model)

        # Check beetle_ssm + working_memory incompatibility.
        _arch_name = (self.configs.get("model") or {}).get("architecture", "")
        if _arch_name == "beetle_ssm" and self._working_memory is not None:
            raise RuntimeError(
                "[Config] architecture: beetle_ssm is incompatible with "
                "working_memory.enabled: true — the SSM has no attention heads."
            )

        # Learning dynamics config
        self._ld_cfg = self.checkpoint_cfg.get("learning_dynamics", {}) or {}
        self._ld_enabled = self._ld_cfg.get("enabled", False)

        # POLCA extractor (Phase-wise Loss Contribution Analysis;
        # arXiv 2510.05064).  Reads the top-level ``polca:`` block written
        # into every config by ``script/add_polca_block.py``.  Builder
        # returns None when the block is absent or has ``enabled: false``,
        # so behaviour for legacy configs is unchanged.  The held-out
        # batch is installed lazily once data pipeline yields its first
        # batch (see _train()).
        from beetlelm.training.polca_extractor import build_polca_extractor
        self._polca_cfg = self.configs.get("polca", {}) or {}
        self._polca = build_polca_extractor(
            self._polca_cfg,
            runs_dir=self.checkpoint_cfg.get("runs_dir", "runs"),
            run_name=self.run_name,
        )
        self._polca_holdout_installed = False

        self.optimizer    = initialize_optimizer(
            training_config=self.training_cfg, model=self.model
        )
        self.lr_scheduler = initialize_lr_scheduler(
            training_config=self.training_cfg, optimizer=self.optimizer
        )
        # Detect exposure-based LR for phase-change notifications
        self._exposure_lr: Optional[ExposureLRScheduler] = (
            self.lr_scheduler
            if isinstance(self.lr_scheduler, ExposureLRScheduler)
            else None
        )

        # Manipulation-block enablement contract. Catches the failure mode
        # fixed in 75a2cd39 where every CL block was silently dropped from
        # initialize_configuration. Run AFTER all builders (incl. polca and
        # exposure_lr) have completed so every ``self._<x>`` attribute exists.
        self._assert_enabled_manipulations_built(_arch_name)

        self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)

        # Schedule-Free AdamW: enter training mode so step() is allowed and
        # p holds the y iterate. eval() will be called around checkpoint
        # saves and evaluation. For all other optimizers this is a no-op.
        self._optimizer_train_mode()

        self.curriculum_scheduler = CurriculumCheckpointScheduler(
            configs=self.configs,
            pipeline_state=self.pipeline_state,
            fabric=self.fabric,
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            tokenizer=self.tokenizer,
        )

        self.initial_batch_step = 0
        if self.checkpoint_cfg.get("auto_resume", False):
            resume = load_checkpoint(
                checkpointing_config=self.checkpoint_cfg,
                checkpoint_step="latest",
                fabric=self.fabric,
                model=self.model,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                pipeline_state=self.pipeline_state,
            )

            if not resume and self.checkpoint_cfg.get("save_to_hf", False):
                resume = self._resume_from_hf()

            if resume:
                self.model, self.optimizer, self.lr_scheduler, extra = resume
                self.initial_batch_step = self.pipeline_state.step_counter
                self._restore_continual_state(extra)
                # Checkpoints are saved with the Schedule-Free optimizer in
                # eval mode (so persisted weights are the running average x).
                # Restore train mode so step() is allowed when the loop resumes.
                self._optimizer_train_mode()
                self.log(
                    f"[Resume] Restored to step {self.initial_batch_step} "
                    f"| tokens={self.pipeline_state.token_counter:,} "
                    f"| phase: {self.pipeline_state.data_mix.current_phase_label()}"
                )
                if self.initial_batch_step > 0:
                    self.log(f"[Resume] Skipping {self.initial_batch_step} batches...")
                    for _ in range(self.initial_batch_step):
                        try:
                            next(self.train_iterator)
                        except StopIteration:
                            break
                    self.log("[Resume] Data iterator aligned.")

        self._log_continual_config()

    # ─────────────────────────────────────────────────────────────────────────
    # Continual-learning helpers
    # ─────────────────────────────────────────────────────────────────────────

    # Map: top-level YAML block → (built-attr, builder name for diagnostics).
    # Used by ``_assert_enabled_manipulations_built`` to verify every
    # ``enabled: true`` block produced a non-None component.
    _MANIPULATION_CONTRACT = (
        ("ewc",               "_ewc",            "build_ewc"),
        ("maml",              "_maml",           "build_maml"),
        ("generative_replay", "_gen_replay",     "build_generative_replay"),
        ("similarity_replay", "_sim_replay",     "build_similarity_replay"),
        ("tilt",              "_tilt",           "build_tilt"),
        ("working_memory",    "_working_memory", "build_working_memory"),
        ("pre_pretraining",   "_pre_pretraining","build_pre_pretraining"),
        ("sas_regularizer",   "_sas",            "build_sas_regularizer"),
        ("gaussian_noise",    "_gni",            "build_gaussian_noise"),
        ("kl_distillation",   "_kl_distill",     "build_kl_distillation"),
        ("lang_id",           "_lang_id",        "build_lang_id_head"),
        ("lora",              "_lora",           "build_lora"),
        ("layer_growth",      "_layer_growth",   "build_layer_growth"),
        ("alignment_loss",    "_alignment_loss", "build_alignment_loss"),
        ("polca",             "_polca",          "build_polca_extractor"),
    )

    # ──────────────────────────────────────────────────────────────────
    # Schedule-Free AdamW lifecycle helpers
    #
    # AdamWScheduleFree (Defazio et al., NeurIPS 2024) requires explicit
    # ``train()`` / ``eval()`` calls to switch ``p`` between the auxiliary
    # ``y`` iterate (training) and the running average ``x`` (inference /
    # checkpointing). For all other optimizers these helpers are a no-op,
    # so callers can invoke them unconditionally.
    #
    # NOTE: Lightning Fabric wraps the optimizer in ``_FabricOptimizer``;
    # the underlying instance is exposed via ``.optimizer``. We unwrap so
    # the isinstance check remains valid post-``fabric.setup``.
    # ──────────────────────────────────────────────────────────────────

    def _inner_optimizer(self):
        opt = getattr(self, "optimizer", None)
        if opt is None:
            return None
        return getattr(opt, "optimizer", opt)

    def _is_schedule_free(self) -> bool:
        return isinstance(self._inner_optimizer(), AdamWScheduleFree)

    def _optimizer_train_mode(self) -> None:
        inner = self._inner_optimizer()
        if isinstance(inner, AdamWScheduleFree):
            inner.train()

    def _optimizer_eval_mode(self) -> None:
        inner = self._inner_optimizer()
        if isinstance(inner, AdamWScheduleFree):
            inner.eval()

    def _assert_enabled_manipulations_built(self, arch: str) -> None:
        """Crash loudly if any ``enabled: true`` block produced a None component.

        Catches three failure modes:
            1. The block was silently dropped before reaching ``self.configs``
               (regression of the bug fixed in commit 75a2cd39).
            2. The builder rejected the cfg shape (e.g. typo in a sub-key)
               and returned None despite ``enabled: true``.
            3. A required hook is missing for the chosen architecture
               (e.g. ``working_memory`` on ``beetle_ssm``, already covered
               above; this generalises the same contract to every block).
        """
        failures: List[str] = []
        for cfg_key, attr_name, builder_name in self._MANIPULATION_CONTRACT:
            cfg = self.configs.get(cfg_key, {}) or {}
            if not isinstance(cfg, dict):
                continue
            if not cfg.get("enabled", False):
                continue
            built = getattr(self, attr_name, None)
            if built is None:
                failures.append(
                    f"  - {cfg_key!r}: enabled=true in YAML but {attr_name} is None "
                    f"(builder: {builder_name}). Either the YAML block was dropped "
                    f"before reaching the trainer, or {builder_name} rejected its cfg shape."
                )

        # LR scheduler is a special case: ``optimization.lr_scheduler:
        # exposure_based`` requires an ``exposure_lr`` block.  If the user
        # picked exposure_based, ``initialize_lr_scheduler`` already raises
        # — but assert here too for symmetry with the rest of the contract.
        opt = (self.training_cfg.get("optimization", {})
               if isinstance(self.training_cfg, dict)
               else getattr(self.training_cfg, "optimization", {})) or {}
        if isinstance(opt, dict) and opt.get("lr_scheduler") == "exposure_based":
            if getattr(self, "_exposure_lr", None) is None:
                # _exposure_lr is set later than this call, so just check the
                # block exists rather than the built component.
                exposure_cfg = opt.get("exposure_lr", {}) or {}
                if not exposure_cfg.get("enabled", True):
                    failures.append(
                        "  - 'optimization.exposure_lr': required by "
                        "lr_scheduler='exposure_based' but enabled=false."
                    )

        if failures:
            joined = "\n".join(failures)
            raise RuntimeError(
                "[Trainer] Enablement contract violated — the following "
                "manipulations were marked enabled in YAML but did NOT produce "
                "a runtime component. Training would silently fall back to a "
                "baseline LM run. Refusing to start.\n"
                f"  architecture: {arch}\n"
                f"{joined}\n"
                "If this is intentional (e.g. block kept for documentation), "
                "set enabled=false explicitly."
            )

    def _maybe_resolve_gen_replay_lang_token(self) -> None:
        """Resolve ``<lang:{L1}>`` → tokenizer id and inject it into the
        generative-replay config when ``prompt_style='lang_tag'``.

        Faithful LAMOL replay (Sun et al. 2020) conditions generation on a
        task-identifier token; in BeetleLM the natural identifier is the L1
        ``<lang:xx>`` control token added by ``MultilingualTokenizerManager``.
        This helper resolves the id end-to-end so configs can stay
        tokenizer-agnostic.  Resolution priority:

        1. ``lang_token_id`` set in YAML — used as-is.
        2. ``<lang:{L1}>`` present in tokenizer vocab — looked up and used.
        3. Neither — drop ``prompt_style`` to ``unconditional`` and emit a
           loud warning. This branch matters for the bilingual nl-en BPE
           tokenizer (and any other tokenizer that lacks lang-tag tokens).
           Failing the run loudly used to be intentional to surface the
           historic LAMOL faithfulness bug, but in practice it just blocks
           sweeps that legitimately have no lang tag — the warning here
           makes the degradation visible without crashing.
        """
        cfg = self._gen_replay_cfg
        if not cfg or not cfg.get("enabled", False):
            return
        if cfg.get("prompt_style", "unconditional") != "lang_tag":
            return
        if cfg.get("lang_token_id") is not None:
            return

        lang_keys = list(self.data_cfg.get("lang_keys", []))
        l1 = lang_keys[0] if lang_keys else None
        token_str = f"<lang:{l1}>" if l1 else None

        tok_id = None
        if token_str:
            try:
                tok_id = self.tokenizer.convert_tokens_to_ids(token_str)
            except Exception:
                tok_id = None
            unk = getattr(self.tokenizer, "unk_token_id", None)
            if tok_id is None or (unk is not None and tok_id == unk):
                tok_id = None

        if tok_id is not None:
            cfg["lang_token_id"] = int(tok_id)
            self.log(
                f"[GenReplay] Resolved prompt_style='lang_tag' → "
                f"{token_str} (token_id={tok_id}) for L1={l1}."
            )
            return

        # No lang_token_id available — fall back to unconditional with a
        # loud, actionable warning so downstream readers know LAMOL is
        # NOT faithful here (no task-identifier prefix).
        cfg["prompt_style"] = "unconditional"
        self.log(
            "[GenReplay] WARNING: prompt_style='lang_tag' requested but no "
            f"lang token found (looked up {token_str!r}). Falling back to "
            "prompt_style='unconditional'. This is a NON-faithful LAMOL "
            "approximation: replay generations will not be conditioned on "
            "a task-identifier token. Add <lang:{L1}> to the tokenizer or "
            "set lang_token_id explicitly in YAML to recover faithfulness."
        )

    def _init_maml(self) -> Optional[SMLMTModule]:
        if not self._maml_cfg or not self._maml_cfg.get("enabled", False):
            return None
        raw     = getattr(self.model, "module", self.model)
        mc      = raw.model_config
        mask_id = (
            getattr(self.tokenizer, "mask_token_id", None)
            or getattr(self.tokenizer, "pad_token_id", None)
            or 0
        )
        maml = build_maml(
            cfg=self._maml_cfg, model=raw,
            vocab_size=int(mc.vocab_size), d_model=int(mc.d_model),
            mask_token_id=mask_id,
        )
        self.log(
            f"[MAML] Enabled | hybrid_ratio={maml.hybrid_ratio} "
            f"| {maml.n_way}-way {maml.k_shot}-shot "
            f"| inner_steps={maml.inner_steps} lr={maml.inner_optimizer.param_groups[0]['lr']}"
        )
        return maml

    def _log_continual_config(self) -> None:
        if self._ewc:
            self.log(
                f"[EWC] Enabled | λ={self._ewc.lambda_} "
                f"| n_fisher_samples={self._ewc.n_fisher_samples} "
                f"| trigger={self._ewc_cfg.get('trigger', 'phase_change')}"
            )
        if self._maml:
            self.log(
                f"[MAML] Module attached | head params: "
                f"{sum(p.numel() for p in self._maml.head_params):,}"
            )
        if self._gen_replay:
            self.log(
                f"[GenReplay] Enabled | λ={self._gen_replay.lambda_replay} "
                f"| ratio={self._gen_replay.replay_ratio} "
                f"| temp={self._gen_replay.temperature} "
                f"| trigger={self._gen_replay_cfg.get('trigger', 'phase_change')}"
            )
        if self._sim_replay:
            self.log(
                f"[SimReplay] Enabled | λ={self._sim_replay.lambda_replay} "
                f"| metric={self._sim_replay.similarity_metric} "
                f"| trigger={self._sim_replay_cfg.get('trigger', 'phase_change')}"
            )
        if self._tilt:
            self.log(
                f"[TILT] Enabled | trainable_modules={self._tilt.trainable_modules} "
                f"| reset_embeddings={self._tilt.reset_embeddings} "
                f"| trigger={self._tilt_cfg.get('trigger', 'phase_change')}"
            )
        if self._ld_enabled:
            self.log(
                f"[LearningDynamics] Enabled | interval={self._ld_cfg.get('log_every_n_steps', 1000)} "
                f"| gradients={self._ld_cfg.get('compute_gradients', False)}"
            )
        if self._working_memory is not None:
            self.log(
                f"[WorkingMemory] Enabled | schedule={self._working_memory.schedule} "
                f"| initial_slope={self._working_memory.initial_slope} "
                f"| final_slope={self._working_memory.final_slope} "
                f"| num_heads={self._working_memory.num_heads}"
            )
        if self._pre_pretraining is not None:
            self.log(
                f"[PrePretraining] Enabled | formal_lang={self._pre_pretraining.formal_lang} "
                f"| steps={self._pre_pretraining.steps} "
                f"| num_symbols={self._pre_pretraining.num_symbols} "
                f"| reset_optimizer={self._pre_pretraining.reset_optimizer}"
            )
        if self._kl_distill is not None:
            self.log(
                f"[KL-Distill] Enabled | weight={self._kl_distill.weight} "
                f"| T={self._kl_distill.temperature} "
                f"| teacher_dtype={self._kl_distill.teacher_dtype} "
                f"| trigger={self._kl_distill_cfg.get('trigger', 'phase_change')}"
            )
        if self._lang_id is not None:
            self.log(
                f"[LangID] Enabled | layer={self._lang_id_cfg.get('layer', 6)} "
                f"| hidden={self._lang_id_cfg.get('hidden_dim', 256)} "
                f"| num_langs={self._lang_id.num_languages} "
                f"| weight={self._lang_id.weight}"
            )
        if self._lora is not None:
            self.log(
                f"[LoRA] Enabled | rank={self._lora.rank} | alpha={self._lora.alpha} "
                f"| target={self._lora.target_modules} "
                f"| freeze_backbone={self._lora.freeze_backbone} "
                f"| trigger={self._lora_cfg.get('trigger', 'phase_change')}"
            )
        if self._layer_growth is not None:
            self.log(
                f"[LayerGrowth] Enabled | add_n_layers={self._layer_growth.add_n_layers} "
                f"| init_mode={self._layer_growth.init_mode} "
                f"| trigger={self._layer_growth_cfg.get('trigger', 'phase_change')}"
            )
        if self._alignment_loss is not None:
            self.log(
                f"[AlignmentLoss] Enabled | mode={self._alignment_loss.mode} "
                f"| layer={self._alignment_loss.layer} "
                f"| weight={self._alignment_loss.weight} "
                f"| source={self._alignment_loss.parallel_source}"
            )

    def _restore_continual_state(self, extra: dict) -> None:
        if self._ewc and "ewc" in extra:
            self._ewc.load_state_dict(extra["ewc"])
        if self._maml and "maml" in extra:
            self._maml.load_state_dict(extra["maml"])
        if self._exposure_lr and "exposure_lr" in extra:
            self._exposure_lr.load_state_dict(extra["exposure_lr"])
        if self._gen_replay and "gen_replay" in extra:
            self._gen_replay.load_state_dict(extra["gen_replay"])
        if self._sim_replay and "sim_replay" in extra:
            self._sim_replay.load_state_dict(extra["sim_replay"])
        if self._tilt and "tilt" in extra:
            self._tilt.load_state_dict(extra["tilt"])
            if self._tilt.is_ready:
                # Re-apply requires_grad mask to the live model after resume.
                # `_resuming=True` skips embedding reset so L2 progress is kept.
                self._tilt.apply(self.model, _resuming=True)
        if self._kl_distill and "kl_distill" in extra:
            # Pass the live model so the teacher's saved state_dict can be
            # loaded onto a fresh deep-copy of the same architecture.
            self._kl_distill.load_state_dict(extra["kl_distill"], model=self.model)
        if self._lang_id and "lang_id" in extra:
            self._lang_id.load_state_dict_head(extra["lang_id"])
        if self._lora and "lora" in extra:
            self._lora.load_state_dict(extra["lora"])
        if self._layer_growth and "layer_growth" in extra:
            self._layer_growth.load_state_dict(extra["layer_growth"])
        if self._alignment_loss and "alignment_loss" in extra:
            self._alignment_loss.load_state_dict(extra["alignment_loss"])

    def _continual_extra_state(self) -> dict:
        extra: dict = {}
        if self._ewc:
            extra["ewc"]  = self._ewc.state_dict()
        if self._maml:
            extra["maml"] = self._maml.state_dict()
        if self._exposure_lr:
            extra["exposure_lr"] = self._exposure_lr.state_dict()
        if self._gen_replay:
            extra["gen_replay"] = self._gen_replay.state_dict()
        if self._sim_replay:
            extra["sim_replay"] = self._sim_replay.state_dict()
        if self._tilt:
            extra["tilt"] = self._tilt.state_dict()
        if self._kl_distill:
            extra["kl_distill"] = self._kl_distill.state_dict()
        if self._lang_id:
            extra["lang_id"] = self._lang_id.state_dict_head()
        if self._lora:
            extra["lora"] = self._lora.state_dict()
        if self._layer_growth:
            extra["layer_growth"] = self._layer_growth.state_dict()
        if self._alignment_loss:
            extra["alignment_loss"] = self._alignment_loss.state_dict()
        return extra

    # ─────────────────────────────────────────────────────────────────────────
    # Phase-change hook
    # ─────────────────────────────────────────────────────────────────────────

    def _on_phase_change(
        self, step: int, new_phase: str, prev_phase: str = "l1"
    ) -> None:
        self.log(f"[Curriculum] Phase → '{new_phase}' at step {step}")

        # Notify exposure-based LR of the phase change (boost plasticity)
        if self._exposure_lr is not None:
            tokens_seen = self.pipeline_state.token_counter
            self._exposure_lr.notify_phase_change(
                new_phase=new_phase, tokens_seen=tokens_seen
            )

        # ── EWC ──────────────────────────────────────────────────────────────
        if self._ewc:
            if self._ewc_cfg.get("trigger", "phase_change") != "phase_change":
                pass
            elif self._ewc.is_ready:
                self.log("[EWC] Fisher already computed — skipping re-snapshot.")
            else:
                self.log("[EWC] Phase change detected: snapshotting θ* and estimating Fisher…")
                self._ewc.snapshot(self.model)
                fisher_batches = self._collect_fisher_batches()
                self._ewc.compute_fisher(
                    model=self.model, batches=fisher_batches, device=self.fabric.device,
                )
                for k, v in self._ewc.diagnostics().items():
                    try:
                        self.fabric.log(k, v, step=step)
                    except Exception:
                        pass
                self.log(f"[EWC] Ready. λ={self._ewc.lambda_}")

        # ── Generative Replay ────────────────────────────────────────────────
        if self._gen_replay:
            if self._gen_replay_cfg.get("trigger", "phase_change") != "phase_change":
                pass
            elif self._gen_replay.is_ready:
                self.log("[GenReplay] Already active — skipping re-snapshot.")
            else:
                self.log("[GenReplay] Phase change: snapshotting model for replay generation…")
                self._gen_replay.snapshot(self.model)
                for k, v in self._gen_replay.diagnostics().items():
                    try:
                        self.fabric.log(k, v, step=step)
                    except Exception:
                        pass

        # ── Similarity Replay ────────────────────────────────────────────────
        if self._sim_replay:
            if self._sim_replay_cfg.get("trigger", "phase_change") != "phase_change":
                pass
            else:
                self.log("[SimReplay] Phase change: storing samples and computing centroids…")
                # Store replay samples from the finishing phase
                replay_batches = self._collect_fisher_batches()
                # prev_phase comes from the method parameter
                replay_tensors = []
                for b in replay_batches:
                    ids = b["input_ids"] if isinstance(b, dict) else b
                    replay_tensors.append(ids)
                if replay_tensors:
                    self._sim_replay.store_replay_samples(prev_phase, replay_tensors)

                # Compute centroids from all available task data
                centroid_batches = self._collect_centroid_batches()
                if centroid_batches:
                    self._sim_replay.compute_centroids(
                        self.model, centroid_batches, self.fabric.device,
                    )
                    self._sim_replay.compute_similarity_matrix()

                for k, v in self._sim_replay.diagnostics().items():
                    try:
                        self.fabric.log(k, v, step=step)
                    except Exception:
                        pass

        # ── TILT ─────────────────────────────────────────────────────────────
        if self._tilt:
            if self._tilt_cfg.get("trigger", "phase_change") != "phase_change":
                pass
            elif self._tilt.is_ready:
                self.log("[TILT] Already applied — skipping re-freeze.")
            else:
                self.log("[TILT] Phase change detected: freezing transformer blocks…")
                self._tilt.apply(self.model)
                for k, v in self._tilt.diagnostics().items():
                    try:
                        self.fabric.log(k, v, step=step)
                    except Exception:
                        pass

        # ── KL distillation ──────────────────────────────────────────────────
        if self._kl_distill:
            if self._kl_distill_cfg.get("trigger", "phase_change") != "phase_change":
                pass
            elif self._kl_distill.is_ready:
                self.log("[KL-Distill] Teacher already snapshotted — skipping.")
            else:
                self.log("[KL-Distill] Phase change detected: snapshotting frozen teacher…")
                self._kl_distill.snapshot(self.model)
                for k, v in self._kl_distill.diagnostics().items():
                    try:
                        self.fabric.log(k, v, step=step)
                    except Exception:
                        pass

        # ── LoRA adapters ────────────────────────────────────────────────────
        if self._lora:
            if self._lora_cfg.get("trigger", "phase_change") != "phase_change":
                pass
            elif self._lora.is_applied:
                self.log("[LoRA] Already applied — skipping.")
            else:
                self.log("[LoRA] Phase change detected: injecting LoRA adapters…")
                self._lora.apply(self.model)
                from beetlelm.training.utils.optimizer_rebuild import rebuild_optimizer_after_param_change
                rebuild_optimizer_after_param_change(self)
                for k, v in self._lora.diagnostics().items():
                    try:
                        self.fabric.log(k, v, step=step)
                    except Exception:
                        pass

        # ── Layer growth ──────────────────────────────────────────────────────
        if self._layer_growth:
            if self._layer_growth_cfg.get("trigger", "phase_change") != "phase_change":
                pass
            elif self._layer_growth.is_ready:
                self.log("[LayerGrowth] Already applied — skipping.")
            else:
                self.log(
                    f"[LayerGrowth] Phase change: growing model by "
                    f"{self._layer_growth.add_n_layers} layers…"
                )
                self._layer_growth.apply(self.model)
                # Re-attach working memory to the newly added layers.
                if self._working_memory is not None:
                    _raw = getattr(self.model, "module", self.model)
                    if hasattr(_raw, "set_working_memory"):
                        _raw.set_working_memory(self._working_memory)
                from beetlelm.training.utils.optimizer_rebuild import rebuild_optimizer_after_param_change
                rebuild_optimizer_after_param_change(self)
                for k, v in self._layer_growth.diagnostics().items():
                    try:
                        self.fabric.log(k, v, step=step)
                    except Exception:
                        pass

    def _collect_fisher_batches(self) -> list:
        """Collect ``n_fisher_samples`` batches for Fisher estimation.

        Streaming path (``self.dm`` is set): build a fresh iterator from the
        BilingualDataModule's dataloader so we never touch the live training
        stream. The fresh iterator has its own random offset; the batches it
        produces are valid samples from the same distribution.

        Pretokenized path (``self.dm`` is None): pull from ``self.train_iterator``
        but PUSH BACK each consumed batch onto ``self._pushback_batches`` so
        the main training loop re-yields them on the following iterations.
        Fisher computation only takes gradients (no optimizer step), so re-using
        these batches for the actual L2 loss does not double-count any update.
        Without this, the first ``n_fisher_samples`` batches after the L1→L2
        boundary would be silently dropped from the L2 token budget.
        """
        n = self._ewc.n_fisher_samples if self._ewc else 10
        batches = []
        try:
            if self.dm is not None:
                tmp_iter = iter(self.dm.train_dataloader())
                for _ in range(n):
                    batches.append(next(tmp_iter))
            else:
                # Pretokenized path: consume + push back.
                for _ in range(n):
                    b = next(self.train_iterator)
                    batches.append(b)
                    self._pushback_batches.append(b)
        except StopIteration:
            pass
        return batches

    def _collect_centroid_batches(self) -> Dict[str, list]:
        """Collect per-language batches for centroid computation."""
        n = self._sim_replay.n_centroid_samples if self._sim_replay else 100
        result: Dict[str, list] = {}
        try:
            if self.dm is not None:
                tmp_iter = iter(self.dm.train_dataloader())
            else:
                tmp_iter = self.train_iterator
            # Collect a batch per known language key
            for _ in range(n):
                batch = next(tmp_iter)
                ids = batch["input_ids"] if isinstance(batch, dict) else batch
                phase = self.pipeline_state.data_mix.current_phase_label()
                result.setdefault(phase, []).append(ids)
        except StopIteration:
            pass
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Loss
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_loss(self, input_ids: torch.Tensor, step: int) -> torch.Tensor:
        if self._maml:
            loss, stats = self._maml.episode(
                input_ids=input_ids, model=self.model, fabric=self.fabric,
            )
            self._log_meta_stats(stats, step)
        else:
            raw       = getattr(self.model, "module", self.model)
            V         = raw.model_config.vocab_size
            x         = input_ids[:, :-1].clamp(0, V - 1)
            y         = input_ids[:, 1:].clamp(0, V - 1)
            logits, _ = raw(x)
            B, T, _   = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), y.contiguous().view(-1))

        if self._ewc and self._ewc.is_ready:
            ewc_term = self._ewc.penalty(self.model)
            try:
                self.fabric.log("ewc/penalty", ewc_term.item(), step=step)
            except Exception:
                pass
            loss = loss + ewc_term

        # KL distillation from frozen L1 teacher
        if self._kl_distill and self._kl_distill.is_ready:
            kl_term = self._kl_distill.kl_term(self.model, input_ids)
            try:
                self.fabric.log("kl_distill/term", kl_term.item(), step=step)
            except Exception:
                pass
            loss = loss + kl_term

        # SAS regularizer (Saphra et al. 2024) — autoregressive variant.
        if self._sas is not None and self._sas.is_active(step):
            sas_term = self._sas.penalty(self.model, step, device=loss.device)
            for k, v in self._sas.diagnostics().items():
                try:
                    self.fabric.log(k, v, step=step)
                except Exception:
                    pass
            loss = loss + sas_term

        # Generative Replay penalty
        if self._gen_replay and self._gen_replay.is_ready:
            gen_loss, gen_stats = self._gen_replay.compute_replay_loss(
                self.model, input_ids, step
            )
            loss = loss + gen_loss
            for k, v in gen_stats.items():
                try:
                    self.fabric.log(k, v, step=step)
                except Exception:
                    pass

        # Similarity Replay penalty
        if self._sim_replay and self._sim_replay.is_ready:
            current_task = self.pipeline_state.data_mix.current_phase_label()
            sim_loss, sim_stats = self._sim_replay.weighted_replay_loss(
                self.model, current_task, step
            )
            loss = loss + sim_loss
            for k, v in sim_stats.items():
                try:
                    self.fabric.log(k, v, step=step)
                except Exception:
                    pass

        # MoE auxiliary losses (load-balance + z-loss stored on model after forward)
        raw_for_moe = getattr(self.model, "module", self.model)
        if hasattr(raw_for_moe, "_last_aux_losses") and raw_for_moe._last_aux_losses:
            for k, v in raw_for_moe._last_aux_losses.items():
                try:
                    self.fabric.log(k, float(v.item()), step=step)
                except Exception:
                    pass
                loss = loss + v

        # Language-ID auxiliary head (mean-pools layer K output, CE vs phase label)
        if self._lang_id is not None:
            _phase_label = self.pipeline_state.data_mix.current_phase_label()
            _label_idx = self._lang_id.label_for_phase(_phase_label)
            lang_id_loss = self._lang_id.compute_loss(_label_idx)
            lang_id_term = self._lang_id.weight * lang_id_loss
            try:
                self.fabric.log("lang_id/loss", lang_id_loss.item(), step=step)
            except Exception:
                pass
            loss = loss + lang_id_term

        return loss

    # ─────────────────────────────────────────────────────────────────────────
    # Training loop
    # ─────────────────────────────────────────────────────────────────────────

    def _install_stop_handlers(self) -> None:
        """
        Install SIGTERM / SIGINT handlers so that an external kill (job
        preemption, Ctrl-C) is converted into a cooperative shutdown: the
        training loop writes one final rank-synced checkpoint and exits
        normally. The outer supervisor (``script/train_with_resume.sh``) then
        relaunches, and ``auto_resume`` picks up from that checkpoint.

        Handlers are installed on every rank. The flag itself is a plain
        Python attribute; the training loop polls it after each optimizer step,
        so every rank reaches the checkpoint call at the same iteration and
        ``_save_step_checkpoint_local`` (which calls ``fabric.save``) stays
        collective.

        Signal handlers must only set the flag — doing anything heavyweight
        from inside a signal context is not safe.
        """
        def _handler(signum, _frame):
            if not self._stop_requested:
                try:
                    self.log(
                        f"[Train] Received signal {signum}; "
                        f"requesting cooperative shutdown at end of iteration."
                    )
                except Exception:
                    pass
                self._stop_requested = True

        try:
            signal.signal(signal.SIGTERM, _handler)
            signal.signal(signal.SIGINT,  _handler)
        except ValueError:
            # Not on the main thread (e.g. called from a worker): fall back to
            # leaving the flag untouched. The default handler behaviour still
            # applies, which for SIGTERM is immediate termination.
            pass

    def _update_heartbeat(self) -> None:
        """Touch ``runs/<run_name>/.heartbeat`` so the orchestrator's
        gpu_watchdog can detect a wedged rank (NCCL hang, dead data loader).

        Called from the training loop after each optimizer step. Throttled
        to once every ``BP_HEARTBEAT_THROTTLE_S`` seconds (default 10s) to
        avoid filesystem thrash. Only rank-0 writes — the watchdog only
        needs one fresh mtime per run, and 8 ranks racing on the same path
        would create false-positive "stale" windows.

        All errors are swallowed silently: a heartbeat write failure must
        never block training. The watchdog can read absent or stale files
        — at worst we lose this safety net for one run, not the run itself.
        """
        try:
            if not getattr(self.fabric, "is_global_zero", True):
                return
            import time as _time
            now = _time.time()
            if (now - self._heartbeat_last_ts) < self._heartbeat_throttle_s:
                return
            runs_dir = self.checkpoint_cfg.get("runs_dir", "./runs")
            hb_dir = os.path.join(runs_dir, self.run_name)
            os.makedirs(hb_dir, exist_ok=True)
            hb_path = os.path.join(hb_dir, ".heartbeat")
            with open(hb_path, "w") as f:
                f.write(f"{int(now)}\n")
            self._heartbeat_last_ts = now
        except Exception:
            # Never propagate a heartbeat write failure into training.
            pass

    def train(self):
        if self.tokenizer_only:
            self.log("Tokenizer-only mode completed. Exiting.")
            return

        torch.set_float32_matmul_precision("high")

        # Graceful-shutdown flag: SIGTERM / SIGINT flips this, the training loop
        # sees it at the end of the iteration, writes a final rank-synced
        # checkpoint, and exits cleanly so the outer supervisor can relaunch and
        # `auto_resume` can pick up from that checkpoint.
        self._stop_requested = False
        self._install_stop_handlers()

        step      = self.initial_batch_step
        max_steps = self.training_cfg.get("max_steps", 1000)
        self.log(
            f"[Train] Starting at step {step} "
            f"| monolingual={self.monolingual_mode} "
            f"| curriculum: {self.pipeline_state.data_mix.curriculum_type} "
            f"| phase: {self.pipeline_state.data_mix.current_phase_label()} "
            f"| EWC={'on' if self._ewc else 'off'} "
            f"| MAML={'on' if self._maml else 'off'} "
            f"| GenReplay={'on' if self._gen_replay else 'off'} "
            f"| SimReplay={'on' if self._sim_replay else 'off'} "
            f"| LearningDynamics={'on' if self._ld_enabled else 'off'}"
        )

        log_every        = self.monitoring_cfg.get("logging", {}).get("log_every_n_steps", 100)
        # Periodic (linear) HF checkpoint pushes have been removed: every push
        # is now gated by ``curriculum_scheduler.should_checkpoint_for_hf`` (log₂
        # schedule + phase boundaries). See should_push_ckpt below.
        per_gpu_batch    = self.dm.batch_size if self.dm is not None else self.training_cfg.get("batch_size", 8)
        num_gpus         = self.fabric.world_size
        target_batch     = self.training_cfg.get("target_effective_batch", 64)
        accumulate_steps = max(1, target_batch // (per_gpu_batch * num_gpus))

        self.log(
            f"[Train] per-GPU batch={per_gpu_batch}, GPUs={num_gpus}, "
            f"effective batch={per_gpu_batch * num_gpus * accumulate_steps}, "
            f"grad_accum={accumulate_steps}"
        )

        # ── Optional formal-language pre-pretraining (stage 1) ──────────────
        # Only runs when the YAML opts in AND we are not resuming from a
        # checkpoint (resume implies stage 1 has already happened).
        if self._pre_pretraining is not None and self.initial_batch_step == 0:
            run_pre_pretraining_stage(self)

        optimizer, lr_scheduler, fabric = self.optimizer, self.lr_scheduler, self.fabric
        optimizer.zero_grad()

        running_loss  = 0.0
        running_steps = 0
        _prev_phase   = self.pipeline_state.data_mix.current_phase_label()

        # Pico-analyze baseline: emit the step-0 pico commit sequence on `main`
        # before the loop starts. Without this, pico-analyze's commit-history
        # scan has no anchor and the first data point is silently dropped.
        # Matches pico-train's trainer.py::train L242.
        if step == 0:
            step0_ckpt = self._save_step_checkpoint_local(0)
            self.async_checkpointer.join()

            # Round-trip smoke test: load the just-saved checkpoint via
            # AutoModelForCausalLM(trust_remote_code=True) and run one forward
            # pass. Catches the failure mode where a model trains but its
            # uploaded HF branch is not loadable (e.g. missing modeling file,
            # HF wrapper / state-dict key mismatch, broken auto_map). Run on
            # rank 0 only — the checkpoint is rank-0-only on disk.
            if fabric.is_global_zero:
                self._smoke_test_hf_load(step0_ckpt)

            if self.checkpoint_cfg.get("save_to_hf", False) and fabric.is_global_zero:
                self.async_checkpointer.save(
                    self._upload_step_checkpoint_to_hf, 0, step0_ckpt
                )

        while step < max_steps:
            try:
                # Drain any batches that ``_collect_fisher_batches`` (or
                # similar introspection passes) consumed from the live
                # training stream. This guarantees those batches are not
                # lost to the L2 token budget.
                if self._pushback_batches:
                    batch = self._pushback_batches.popleft()
                else:
                    batch = next(self.train_iterator)
            except StopIteration:
                self.log(
                    f"[Train] Data exhausted at step {step} "
                    f"({self.pipeline_state.token_counter:,} tokens). Ending."
                )
                break

            input_ids = batch["input_ids"].to(fabric.device)

            # Advance the curriculum scheduler for paths that do not call
            # pipeline_state.get_next_batch() (pretokenized + BeetleStream v2).
            # The streaming path (BilingualDataset) already steps via get_next_batch().
            if self.dm is None:
                self.pipeline_state.step(tokens=input_ids.numel())

            data_mix      = self.pipeline_state.data_mix
            current_phase = data_mix.current_phase_label()
            phase_changed = data_mix.phase_just_changed()

            if phase_changed and current_phase != _prev_phase:
                self._on_phase_change(step, current_phase, prev_phase=_prev_phase)
                _prev_phase = current_phase

            # Tick the working-memory ALiBi scheduler before the forward pass
            # so the slope reflects the current training progress.
            if self._working_memory is not None:
                raw = getattr(self.model, "module", self.model)
                raw.update_progress(step, max_steps)

            loss = self._compute_loss(input_ids, step)

            # ── NaN / Inf guard ──────────────────────────────────────────────
            # An unguarded NaN loss corrupts AdamW's m/v state irreversibly
            # and the corruption survives across checkpoints. Detect BEFORE
            # backward so the optimizer never sees the bad gradient. Exit
            # with a recognizable code so the outer retry layer can decide
            # whether to resume from the last good checkpoint or skip.
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                _bad = "NaN" if torch.isnan(loss).any() else "Inf"
                try:
                    self.log(
                        f"[Train][ABORT] {_bad} loss at step {step} "
                        f"(value={loss.detach().float().item()!r}). "
                        "Skipping backward and exiting to protect optimizer state."
                    )
                except Exception:
                    pass
                # Sentinel for the outer orchestrator: a watchdog-equivalent
                # signal that this yaml is broken under current config.
                try:
                    _sentinel = os.path.join(
                        self.checkpoint_cfg.get("runs_dir", "./runs"),
                        self.run_name, ".divergence",
                    )
                    os.makedirs(os.path.dirname(_sentinel), exist_ok=True)
                    with open(_sentinel, "w") as _f:
                        _f.write(f"step={step}\nkind={_bad}\nvalue={loss.detach().float().item()!r}\n")
                except Exception:
                    pass
                # rc=42 is recognized by run_job as a non-retry divergence kill.
                raise SystemExit(42)

            fabric.backward(loss / accumulate_steps)

            running_loss  += loss.item()
            running_steps += 1

            # ── Heartbeat ────────────────────────────────────────────────────
            # Touch the heartbeat file so the orchestrator's gpu_watchdog can
            # detect NCCL hangs / wedged ranks (memory stays stable when a
            # rank is stuck at a barrier, so this is the only signal). Cheap:
            # we throttle to once every BP_HEARTBEAT_THROTTLE_S seconds.
            self._update_heartbeat()

            if (step + 1) % accumulate_steps == 0:
                # Gaussian Noise Injection (Aoyama et al. 2024 baseline).
                if self._gni is not None and self._gni.target == "gradients":
                    self._gni.apply(self.model, step)
                optimizer.step()
                if self._gni is not None and self._gni.target == "parameters":
                    self._gni.apply(self.model, step)
                if self._gni is not None:
                    for k, v in self._gni.diagnostics().items():
                        try:
                            self.fabric.log(k, v, step=step)
                        except Exception:
                            pass
                optimizer.zero_grad()
                if self._exposure_lr is not None:
                    self._exposure_lr.step(
                        tokens_seen=self.pipeline_state.token_counter,
                        current_phase=current_phase,
                    )
                else:
                    lr_scheduler.step()

            if step % log_every == 0:
                avg_loss = running_loss / max(1, running_steps)
                self.log(
                    f"Step {step:>6} | loss={avg_loss:.4f} "
                    f"| phase={current_phase} | progress={data_mix._progress():.3f}"
                )
                self._log_continual_metrics(step)
                running_loss  = 0.0
                running_steps = 0
                if self._maml:
                    self._maml.reset_histories()

            self.curriculum_scheduler.checkpoint(step=step, phase_changed=phase_changed)

            # Learning dynamics extraction at checkpoint steps.
            # Gated to the curriculum scheduler's log₂ checkpoint set (plus
            # phase boundaries) so LD fires exactly at the steps pico-analyze
            # will consume — not at arbitrary modulo intervals, and not at the
            # step N-1 pair (dropped; pico-analyze Δ-weights use consecutive
            # LD events in the set, not N/N-1 pairs).
            if self._ld_enabled and (
                phase_changed
                or step in self.curriculum_scheduler.checkpoint_steps
            ):
                self._extract_and_save_dynamics(step, input_ids)

            # POLCA artifact extraction.  Same gating as learning dynamics
            # so artifacts land on the steps that pico-analyze /
            # beetle-analyze already consume.  Rank-0 only inside the
            # extractor; other ranks short-circuit immediately.
            if self._polca is not None and (
                phase_changed
                or step in self.curriculum_scheduler.checkpoint_steps
                or self._polca.should_run(step)
            ):
                if not self._polca_holdout_installed:
                    self._polca.set_holdout_batch({"input_ids": input_ids})
                    self._polca_holdout_installed = True
                self._polca.extract(
                    self.model,
                    step=step,
                    device=self.fabric.device,
                    is_global_zero=self.fabric.is_global_zero,
                    rank=self.fabric.global_rank,
                )

            should_push_ckpt = (
                phase_changed
                or self.curriculum_scheduler.should_checkpoint_for_hf(step, phase_changed)
            )
            if should_push_ckpt:
                # 1. Wait for any previous async HF upload to finish BEFORE we
                #    save locally. save_checkpoint() calls
                #    prune_local_checkpoints() which can delete the previous
                #    step's directory; if that directory is still being uploaded
                #    by the async worker, upload_folder() fails with
                #    "...step_N is not a directory" and after 4 backoff retries
                #    raises. Joining first guarantees the worker is idle before
                #    we prune its source. No-op on ranks != 0 (idle worker).
                self.async_checkpointer.join()

                # 2. Rank-synced local save (fabric.save is collective). Every
                #    rank enters this call; returns on every rank.
                checkpoint_path = self._save_step_checkpoint_local(step)

                # 3. Rank-0 only: dispatch the HF upload to a background thread
                #    so the main loop can resume training immediately. Ranks
                #    1..N never entered any @rank_zero_only call, so there is
                #    no divergence and no allreduce race — no barrier needed.
                if self.checkpoint_cfg.get("save_to_hf", False) and fabric.is_global_zero:
                    self.async_checkpointer.save(
                        self._upload_step_checkpoint_to_hf,
                        step,
                        checkpoint_path,
                        phase_changed,
                    )

            step += 1

            # Cooperative shutdown: SIGTERM/SIGINT handler flipped the flag.
            # Write a final rank-synced checkpoint and exit cleanly so the
            # outer supervisor can relaunch into `auto_resume`.
            if self._stop_requested:
                self.log(
                    f"[Train] Stop requested at step {step}. "
                    f"Writing a final local checkpoint and exiting cleanly."
                )
                # Drain pending uploads first so the subsequent save's prune
                # cannot delete a directory that's still being uploaded.
                self.async_checkpointer.join()
                try:
                    self._save_step_checkpoint_local(step)
                except Exception as exc:
                    self.log(
                        f"[Train] Final checkpoint save failed: {exc}",
                        level=logging.ERROR,
                    )
                break

        self.log(f"Training complete at step {step}.")
        # Bounded join: a wedged HF upload should not block forever.
        self.async_checkpointer.join()
        if self.async_checkpointer.upload_failed:
            self.log(
                "[Checkpointing] One or more async checkpoint uploads "
                "did not complete cleanly (timed_out=%s); local checkpoint "
                "will be preserved." % self.async_checkpointer.timed_out,
                level=logging.WARNING,
            )
        else:
            self.log("[Checkpointing] All async writes flushed.")

        fabric.barrier()
        # _push_to_hf already catches its own exceptions; tag a failure on the
        # async checkpointer so the launcher-side sentinel reflects it.
        try:
            self._push_to_hf()
        except Exception as exc:  # defensive — _push_to_hf has its own try
            self.async_checkpointer.upload_failed = True
            self.async_checkpointer.last_error = exc
            self.log(f"[HF] Final push raised: {exc}", level=logging.ERROR)

        # Write the launcher-visible sentinel that gates runs/ cleanup.
        self._write_hf_upload_sentinel()

        if hasattr(self.train_iterator, "close"):
            self.train_iterator.close()

        torch.cuda.empty_cache()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    # ─────────────────────────────────────────────────────────────────────────
    # Logging helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _log_meta_stats(self, stats, step: int) -> None:
        if not stats.did_meta:
            return
        try:
            self.fabric.log("maml/inner_acc_sup",        stats.inner_acc_sup,        step=step)
            self.fabric.log("maml/outer_acc_q",          stats.outer_acc_q,          step=step)
            self.fabric.log("maml/inner_grad_norm_mean", stats.inner_grad_norm_mean, step=step)
        except Exception:
            pass

    def _log_continual_metrics(self, step: int) -> None:
        if self._maml:
            avg_sup = self._maml.avg_inner_acc()
            avg_q   = self._maml.avg_query_acc()
            self.log(
                f"  [MAML] meta_steps={self._maml.step_count_meta} "
                f"| ar_steps={self._maml.step_count_ar} "
                f"| avg_sup_acc={avg_sup:.4f} | avg_q_acc={avg_q:.4f}"
            )
            try:
                self.fabric.log("maml/avg_sup_acc",     avg_sup,                    step=step)
                self.fabric.log("maml/avg_q_acc",       avg_q,                      step=step)
                self.fabric.log("maml/step_count_meta", self._maml.step_count_meta, step=step)
                self.fabric.log("maml/step_count_ar",   self._maml.step_count_ar,   step=step)
            except Exception:
                pass
        if self._ewc:
            for k, v in self._ewc.diagnostics().items():
                try:
                    self.fabric.log(k, v, step=step)
                except Exception:
                    pass
        if self._gen_replay:
            for k, v in self._gen_replay.diagnostics().items():
                try:
                    self.fabric.log(k, v, step=step)
                except Exception:
                    pass
        if self._sim_replay:
            for k, v in self._sim_replay.diagnostics().items():
                try:
                    self.fabric.log(k, v, step=step)
                except Exception:
                    pass
        if self._kl_distill:
            for k, v in self._kl_distill.diagnostics().items():
                try:
                    self.fabric.log(k, v, step=step)
                except Exception:
                    pass
        if self._lang_id:
            for k, v in self._lang_id.diagnostics().items():
                try:
                    self.fabric.log(k, v, step=step)
                except Exception:
                    pass
        if self._lora:
            for k, v in self._lora.diagnostics().items():
                try:
                    self.fabric.log(k, v, step=step)
                except Exception:
                    pass
        if self._layer_growth:
            for k, v in self._layer_growth.diagnostics().items():
                try:
                    self.fabric.log(k, v, step=step)
                except Exception:
                    pass
        if self._alignment_loss:
            for k, v in self._alignment_loss.diagnostics().items():
                try:
                    self.fabric.log(k, v, step=step)
                except Exception:
                    pass

    # ─────────────────────────────────────────────────────────────────────────
    # Learning dynamics (interpretability checkpointing)
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_and_save_dynamics(
        self, step: int, current_batch: torch.Tensor
    ) -> None:
        """Extract activations, weights, gradients and push to HF."""
        from beetlelm.config.checkpointing_config import (
            LearningDynamicsCheckpointingConfig,
        )
        from beetlelm.checkpointing.learning_dynamics import (
            CheckpointStateExtractor,
            save_learning_dynamics_states as ld_save,
        )
        from datasets import Dataset
        from torch.utils.data import DataLoader

        self.log(f"[LearningDynamics] Extracting states at step {step}…")

        ld_config = LearningDynamicsCheckpointingConfig(
            enabled=True,
            batch_size=self._ld_cfg.get("batch_size", 64),
            sequence_idx=self._ld_cfg.get("sequence_idx", -1),
            layer_suffixes=self._ld_cfg.get(
                # Match the literal module names in pico_decoder:
                # attention.{q,k,v,o}_proj, swiglu.w_{1,2,3}, *_norm.
                # pico-analyze's ov_circuit + simple-swiglu metrics need
                # these substrings; shorter aliases like "attn"/"mlp" would
                # only capture the \*_norm modules.
                "layer_suffixes", ["attention", "swiglu", "norm"]
            ),
            compute_gradients=self._ld_cfg.get("compute_gradients", False),
            learning_dynamics_dir=self._ld_cfg.get(
                "learning_dynamics_dir", "learning_dynamics"
            ),
        )

        # Build dataset from current batch
        batch_np = current_batch.cpu().numpy().tolist()
        ld_dataset = Dataset.from_dict({"input_ids": batch_np})

        # Extract states
        extractor = CheckpointStateExtractor(
            ld_config, self.fabric, self.model
        )
        sub_batch_size = max(
            1, ld_config.batch_size // self.fabric.world_size
        )
        loader = DataLoader(
            ld_dataset,
            batch_size=sub_batch_size,
            shuffle=False,
            collate_fn=lambda b: {"input_ids": [e["input_ids"] for e in b]},
            drop_last=False,
        )
        loader = self.fabric.setup_dataloaders(
            loader, use_distributed_sampler=True
        )

        activations, weights, gradients = extractor.extract_states(
            loader, compute_gradients=ld_config.compute_gradients
        )

        learning_dynamics_states = {
            "activations": activations,
            "weights": weights,
            "gradients": gradients,
        }

        # Save locally
        runs_dir = self.checkpoint_cfg.get("runs_dir", "./runs")
        ld_dir = os.path.join(
            runs_dir, self.run_name, "checkpoints",
            f"step_{step}", ld_config.learning_dynamics_dir,
        )
        os.makedirs(ld_dir, exist_ok=True)

        for key, value in learning_dynamics_states.items():
            if value and len(value) > 0:
                torch.save(value, os.path.join(ld_dir, f"train_{key}.pt"))

        # Save training batch (raw input_ids tensor — pico-analyze does not
        # read this, but inspection tools / resume paths may).
        if self._ld_cfg.get("save_training_batch", True):
            torch.save(
                current_batch.cpu(),
                os.path.join(ld_dir, "training_batch.pt"),
            )

        # Save decoded dataset (tokens + detokenised text) as an HF Dataset.
        # pico-analyze loads but never reads this (see pico-analyze/src/utils/
        # data.py:196-198 + zero references under src/metrics and src/
        # components).  Default is now OFF to save HF storage; set
        # ``learning_dynamics.save_train_data: true`` to restore the
        # decoded-batch dump for debugging.
        if self._ld_cfg.get("save_train_data", False):
            ld_dataset_with_text = {"input_ids": batch_np, "text": []}
            for ids in batch_np:
                ld_dataset_with_text["text"].append(
                    self.tokenizer.decode(ids, skip_special_tokens=True)
                )
            Dataset.from_dict(ld_dataset_with_text).save_to_disk(
                os.path.join(ld_dir, "train_data")
            )

        self.log(f"[LearningDynamics] States saved at step {step}")

        # Push learning_dynamics to HF `main` for pico-analyze compatibility.
        # pico-analyze scans commit history for the verbatim message
        # "Saving Learning Dynamics Data (train) -- Step {step}".
        #
        # Dispatched onto the AsyncCheckpointer so rank 0's HF upload doesn't
        # block ranks 1..N at the next allreduce. The dispatch is guarded by
        # is_global_zero so non-rank-0 workers take no thread-level action,
        # mirroring the pattern used for step-checkpoint uploads.
        if self.checkpoint_cfg.get("save_to_hf", False):
            self.async_checkpointer.join()
            if self.fabric.is_global_zero:
                self.async_checkpointer.save(
                    self._push_learning_dynamics_to_main, step=step, ld_dir=ld_dir
                )

        # Clean up old dynamics directories
        if self._ld_cfg.get("cleanup_local", True):
            self._cleanup_old_dynamics(step)

    @rank_zero_only
    @use_backoff(max_retries=4, initial_delay=5, backoff_factor=2)
    def _push_learning_dynamics_to_main(self, step: int, ld_dir: str) -> None:
        """Upload learning_dynamics/ to `main` with pico-compatible commit message."""
        repo_id = f"{self.hf_user}/{self.run_name}"
        api = HfApi()

        # Ensure repo + main exist (safe on re-invocation).
        create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
        existing = {b.name for b in api.list_repo_refs(repo_id).branches}
        if "main" not in existing:
            api.create_branch(repo_id=repo_id, branch="main")

        upload_folder(
            folder_path=ld_dir,
            path_in_repo="learning_dynamics",
            repo_id=repo_id,
            revision="main",
            commit_message=f"Saving Learning Dynamics Data (train) -- Step {step}",
        )
        self.log(f"[HF/main] learning_dynamics uploaded (step {step}) → {repo_id}@main")

    def _cleanup_old_dynamics(self, current_step: int) -> None:
        """Delete learning dynamics data from all but the most recent N."""
        keep_n = self._ld_cfg.get("keep_n_local", 1)
        runs_dir = self.checkpoint_cfg.get("runs_dir", "./runs")
        ckpt_root = os.path.join(
            runs_dir, self.run_name, "checkpoints"
        )
        if not os.path.exists(ckpt_root):
            return

        import shutil

        ld_steps = []
        for d in os.listdir(ckpt_root):
            m = re.match(r"step_(\d+)", d)
            if m:
                ld_path = os.path.join(ckpt_root, d, "learning_dynamics")
                if os.path.exists(ld_path):
                    ld_steps.append((int(m.group(1)), ld_path))

        ld_steps.sort(key=lambda x: x[0])
        for step_num, ld_path in ld_steps[:-keep_n]:
            shutil.rmtree(ld_path, ignore_errors=True)
            self.log(
                f"[LearningDynamics] Cleaned up dynamics from step {step_num}"
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _process_raw_config(raw_configs):
        from omegaconf import OmegaConf

        def to_dict(obj):
            if obj is None:
                return {}
            if isinstance(obj, dict):
                return obj
            if hasattr(obj, "__dict__") and "DictConfig" in str(type(obj)):
                return OmegaConf.to_container(obj, resolve=True)
            if is_dataclass(obj):
                return asdict(obj)
            if hasattr(obj, "dict") and callable(obj.dict):
                return obj.dict()
            if hasattr(obj, "__dict__"):
                return vars(obj)
            try:
                return dict(obj)
            except Exception:
                return {}

        processed = {}
        for k, v in to_dict(raw_configs).items():
            processed[k] = (
                to_dict(v)
                if isinstance(v, (dict, object)) and not isinstance(v, str)
                else v
            )
        return processed

    @rank_zero_only
    def log(self, msg: str, level: int = logging.INFO):
        self.logger.log(level, msg)

    # ─────────────────────────────────────────────────────────────────────────
    # Tokenizer
    # ─────────────────────────────────────────────────────────────────────────

    def _setup_tokenizer(self, cfg: dict):
        # ── Tokenizer-free path: bypass BPE training entirely ────────────────
        # When tokenizer.type ∈ TOKFREE_TOKENIZERS the manager owns its own
        # encoding (UTF-8 bytes or Unicode codepoints) and has no vocab to
        # train. Vocab_size is fixed (256 for bytes); the model config picks
        # it up via len(self.tokenizer) downstream just like the BPE path.
        from beetlelm.bytes import TOKFREE_TOKENIZERS  # noqa: WPS433 - local import keeps top-of-file BPE-only

        tok_type_early = (cfg or {}).get("type", "bpe")
        if tok_type_early in TOKFREE_TOKENIZERS:
            tokfree_cls = TOKFREE_TOKENIZERS[tok_type_early]
            tok_params = dict(cfg.get("params", {}) or {})
            tok_params.setdefault(
                "max_length",
                int(self.configs.get("training", {}).get("seq_len", 2048)),
            )
            tokfree_vocab = cfg.get("vocab_size")
            if tokfree_vocab is not None and tok_type_early != "byte":
                tok_params.setdefault("vocab_size", int(tokfree_vocab))
            tokenizer = tokfree_cls(**tok_params)
            out_dir = cfg.get("output_dir", "./tokenizers")
            os.makedirs(out_dir, exist_ok=True)
            tokenizer.save_pretrained(out_dir)
            self.log(
                f"[Tokenizer] tokenizer-free path: type={tok_type_early} "
                f"vocab={len(tokenizer)} max_length={tok_params['max_length']}"
            )
            return tokenizer

        os.makedirs(cfg.get("output_dir", "./tokenizers"), exist_ok=True)
        manager = MultilingualTokenizerManager(
            output_dir=cfg.get("output_dir", "./tokenizers")
        )
        pretrained_path   = cfg.get("pretrained_tokenizer_path")
        target_vocab_size = (
            self.configs.get("model", {}).get("params", {}).get("vocab_size", 32000)
        )
        if pretrained_path:
            self.log(f"[Tokenizer] Loading pretrained from {pretrained_path}")
            # Two-stage load to avoid one HF Hub HEAD request per job in a
            # 60+ job sweep: first try the local cache only (zero network);
            # fall back to a normal fetch if the cache is cold. This keeps
            # the very first job working as today and drops every
            # subsequent job's tokenizer cold-start to a local file read.
            try:
                tokenizer = PreTrainedTokenizerFast.from_pretrained(
                    pretrained_path, local_files_only=True,
                )
            except (OSError, EnvironmentError, ValueError) as exc:
                self.log(
                    f"[Tokenizer] Local cache miss for {pretrained_path} "
                    f"({type(exc).__name__}); falling back to HF Hub fetch.",
                    level=logging.INFO,
                )
                tokenizer = PreTrainedTokenizerFast.from_pretrained(pretrained_path)
            self._adjust_vocab(tokenizer, target_vocab_size)
            tokenizer.save_pretrained(cfg.get("output_dir", "./tokenizers"))
            return tokenizer

        tokenizer_type    = cfg.get("type", "bpe")
        target_tokens     = cfg.get("target_tokens", 1_000_000)
        overwrite         = cfg.get("overwrite", False)
        push_to_hub       = cfg.get("push_to_hub", True)
        hf_repo_override  = cfg.get("hf_repo")
        hf_user           = self.configs.get("hf_user", HUGGINGFACE_USER)
        vocab_size        = cfg.get("vocab_size") or target_vocab_size
        compression       = cfg.get("compression", "equal")
        lang_weights_cfg  = cfg.get("lang_weights", None)
        lang_datasets_cfg = cfg.get("lang_datasets")
        l1_dataset        = cfg.get("l1_dataset")
        l2_dataset        = cfg.get("l2_dataset")

        if lang_datasets_cfg:
            mode     = "polyglot"
            datasets: Dict[str, str] = dict(lang_datasets_cfg)
        elif l1_dataset and l2_dataset:
            mode     = "bilingual"
            datasets = {_short(l1_dataset): l1_dataset, _short(l2_dataset): l2_dataset}
        elif l1_dataset:
            mode     = "monolingual"
            datasets = {_short(l1_dataset): l1_dataset}
        else:
            raise ValueError(
                "[Tokenizer] YAML must supply 'lang_datasets' (polyglot) "
                "or 'l1_dataset' (bilingual / monolingual)."
            )

        self.log(
            f"[Tokenizer] Mode={mode} | langs={list(datasets.keys())} "
            f"| type={tokenizer_type} | vocab={vocab_size} "
            f"| target_tokens={target_tokens:,}"
        )

        def _load_stream(ds_name: str) -> Iterator[str]:
            ds  = load_dataset(ds_name, split="train")
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

        lang_streams  = {key: _load_stream(hf_name) for key, hf_name in datasets.items()}
        vocab_sharing = cfg.get("vocab_sharing", "shared")   # "shared" | "per_language"
        shared        = (vocab_sharing != "per_language")

        result = manager.get_tokenizer(
            lang_streams=lang_streams,
            tokenizer_type=tokenizer_type,
            vocab_size=vocab_size,
            target_tokens=target_tokens,
            overwrite=overwrite,
            add_lang_tokens=(mode == "polyglot"),
            shared=shared,
            compression=compression,
            lang_weights=lang_weights_cfg,
        )

        if not shared:
            # Per-language: push each tokenizer separately, return the first for training
            self.log(f"[Tokenizer] Per-language tokenizers: {list(result.keys())}")
            dataset_source = cfg.get("dataset_source", "babybabel")
            for lang_key, raw_tok in result.items():
                tok_fast = PreTrainedTokenizerFast(
                    tokenizer_object=raw_tok,
                    unk_token="[UNK]", sep_token="[SEP]",
                    pad_token="[PAD]", cls_token="[CLS]", mask_token="[MASK]",
                )
                self._adjust_vocab(tok_fast, target_vocab_size)
                if push_to_hub:
                    repo_id = f"{hf_user}/{dataset_source}_bpe_perlang_{lang_key}"
                    tok_fast.push_to_hub(repo_id, private=False)
                    self.log(f"[Tokenizer] Per-language pushed: {repo_id}")
            # Training uses the shared tokenizer trained on all langs (standard path)
            # for the actual model; per-language tokenizers are for the investigation.
            self.log("[Tokenizer] Falling back to shared tokenizer for model training.")
            raw_tokenizer = manager.get_tokenizer(
                lang_streams={key: _load_stream(hf_name) for key, hf_name in datasets.items()},
                tokenizer_type=tokenizer_type,
                vocab_size=vocab_size,
                target_tokens=target_tokens,
                overwrite=False,  # should already be cached
                add_lang_tokens=(mode == "polyglot"),
                shared=True,
            )
        else:
            raw_tokenizer = result

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=raw_tokenizer,
            unk_token="[UNK]", sep_token="[SEP]",
            pad_token="[PAD]", cls_token="[CLS]", mask_token="[MASK]",
        )
        self._adjust_vocab(tokenizer, target_vocab_size)
        tokenizer.save_pretrained(cfg.get("output_dir", "./tokenizers"))
        self.log(f"[Tokenizer] Saved at {cfg.get('output_dir', './tokenizers')}")
        if push_to_hub and shared:
            if hf_repo_override:
                repo_id = hf_repo_override
            else:
                tag            = "-".join(sorted(datasets.keys()))
                dataset_source = cfg.get("dataset_source", "babybabel")  # "babybabel" | "fineweb"
                repo_id        = f"{hf_user}/{dataset_source}_bpe_{tag}"
            tokenizer.push_to_hub(repo_id, private=False)
            self.log(f"[Tokenizer] Pushed to HF Hub: {repo_id}")
        return tokenizer

    def _adjust_vocab(self, tokenizer, target_vocab_size: int):
        current = tokenizer.vocab_size
        if current != target_vocab_size:
            self.log(f"[Tokenizer] Adjusting vocab: {current} -> {target_vocab_size}")
            if current < target_vocab_size:
                tokenizer.add_tokens(
                    [f"<unused{i}>" for i in range(target_vocab_size - current)]
                )
            else:
                tokenizer._tokenizer.model.truncate(target_vocab_size)

    # ─────────────────────────────────────────────────────────────────────────
    # Data pipeline
    # ─────────────────────────────────────────────────────────────────────────

    def _setup_data_pipeline(self):
        from itertools import cycle
        from beetlelm.bilingual.data.curriculum import DataMixConfig, ConstantExposure

        # ── BeetleStream v2: Curriculum mode shortcut ─────────────────────
        stream_mode = self.data_cfg.get("stream_mode", "static")
        if stream_mode == "curriculum":
            self._setup_curriculum_pipeline()
            return
        # ── End curriculum mode shortcut ──────────────────────────────────

        hf_cache   = self.data_cfg.get("hf_cache_dir", "./hf_cache")
        world_size = self.fabric.world_size
        rank       = self.fabric.global_rank

        def get_stream(ds_name: str):
            ds = load_dataset(ds_name, split="train", streaming=True, cache_dir=hf_cache)
            ds = ds.shard(num_shards=world_size, index=rank)
            def _stream():
                for entry in ds:
                    text = entry.get("text", "").strip()
                    if text:
                        yield text
            return cycle(_stream())

        pretokenized = self.data_cfg.get("pretokenized", False)

        lang_sources_cfg = self.data_cfg.get("lang_sources", [])
        if isinstance(lang_sources_cfg, dict):
            is_polyglot = True
            if pretokenized:
                lang_streams_raw = {key: [None] for key in lang_sources_cfg}
            else:
                lang_streams_raw = {
                    key: [get_stream(hf_name)]
                    for key, hf_name in lang_sources_cfg.items()
                }
        else:
            l2_names    = self.data_cfg.get("l2_streams", [])
            l1_names    = lang_sources_cfg
            is_polyglot = False
            if pretokenized:
                lang_streams_raw = {"l1": [None for _ in l1_names]}
                if l2_names:
                    lang_streams_raw["l2"] = [None for _ in l2_names]
            else:
                lang_streams_raw = {"l1": [get_stream(n) for n in l1_names]}
                if l2_names:
                    lang_streams_raw["l2"] = [get_stream(n) for n in l2_names]

        self.monolingual_mode = not is_polyglot and "l2" not in lang_streams_raw
        if self.monolingual_mode:
            self.log("[Data] Monolingual mode — L2 pipeline disabled.")

        lang_counts       = {k: len(v) for k, v in lang_streams_raw.items()}
        self.stream_state = DeterministicStreamState(lang_counts=lang_counts)

        total_steps   = self.training_cfg.get("max_steps", 2000)
        data_mix_name = self.data_cfg.get(
            "data_mix", "balanced" if not is_polyglot else "polyglot_equal"
        )

        # Resolve curriculum condition
        curriculum_cfg = self.configs.get("curriculum", {})
        condition = curriculum_cfg.get("condition", data_mix_name)
        condition = CONDITION_ALIASES.get(condition.lower(), condition)

        lang_keys = list(lang_streams_raw.keys())

        if self.monolingual_mode:
            # Only l1 is available — use a single-language constant schedule so
            # next_source() never returns a key missing from the dataset pools.
            condition = "monolingual"
            data_mix = DataMixConfig(
                schedule=ConstantExposure({"l1": 1.0}),
                mode="steps",
                total_steps=total_steps,
                curriculum_type="monolingual",
            )
        elif condition.startswith("T"):
            data_mix = get_condition(
                condition, total_steps=total_steps, lang_keys=lang_keys,
            )
        elif len(lang_keys) > 2 and not condition.startswith("B"):
            # General polyglot fallback: equal weights
            w = 1.0 / len(lang_keys)
            data_mix = DataMixConfig(
                schedule=ConstantExposure({k: w for k in lang_keys}),
                mode="steps", total_steps=total_steps,
                curriculum_type="polyglot_equal",
            )
        else:
            data_mix = get_condition(condition, total_steps=total_steps)

        self.log(f"[Data] Curriculum: condition={condition}, langs={lang_keys}")

        self.pipeline_state = DataPipelineState(
            stream_state=self.stream_state, data_mix=data_mix,
        )

        batch_size = self.training_cfg.get("batch_size", 8)
        seq_len    = self.training_cfg.get("seq_len", 128)
        prefetch   = 32 if world_size <= 4 else 64 if world_size <= 8 else 128

        if self.data_cfg.get("pretokenized", False):
            # ── Pretokenized path (human-scale BabyBabel experiments) ──────────
            run_name  = self.run_name
            cache_dir = self.data_cfg.get(
                "pretokenized_cache_dir", f"./pretok_cache/{run_name}"
            ).replace("{run_name}", run_name)

            # Build lang_key → hf_dataset_name mapping from data config
            if isinstance(lang_sources_cfg, dict):
                lang_datasets = dict(lang_sources_cfg)
            else:
                l1_names = lang_sources_cfg
                l2_names = self.data_cfg.get("l2_streams", [])
                lang_datasets = {}
                for i, n in enumerate(l1_names):
                    lang_datasets[f"l1_{i}" if len(l1_names) > 1 else "l1"] = n
                for i, n in enumerate(l2_names):
                    lang_datasets[f"l2_{i}" if len(l2_names) > 1 else "l2"] = n

            overwrite_pretok = self.data_cfg.get("pretokenized_overwrite", False)

            # Only rank 0 writes to the cache; other ranks wait then read.
            if rank == 0:
                self.log(f"[Data] Loading pretokenized HF Arrow {list(lang_datasets.keys())} → {cache_dir}")
                arrows = load_pretokenized_hf_multilang(
                    lang_datasets=lang_datasets,
                    cache_dir=cache_dir,
                    overwrite=overwrite_pretok,
                )
            self.fabric.barrier()
            if rank != 0:
                arrows = {k: os.path.join(cache_dir, k) for k in lang_datasets}

            from torch.utils.data import DataLoader
            dataset = PretokenizedMultilingualDataset(
                arrows=arrows,
                pipeline_state=self.pipeline_state,
                seq_len=seq_len,
            )
            # No dm attribute for pretokenized path; Fisher collection uses dm if present
            self.dm = None
            base_iterator = iter(DataLoader(dataset, batch_size=batch_size))
        else:
            # ── Streaming path (existing on-the-fly tokenization) ──────────────
            self.dm = BilingualDataModule(
                lang_streams=lang_streams_raw,
                pipeline_state=self.pipeline_state,
                tokenizer=self.tokenizer,
                batch_size=batch_size,
                seq_len=seq_len,
            )
            self.dm.setup()
            base_iterator = iter(self.dm.train_dataloader())

        self.log(f"[Data] Prefetch buffer = {prefetch} batches (GPUs={world_size})")
        self.train_iterator = PrefetchIterator(base_iterator, buffer_size=prefetch)

    # ─────────────────────────────────────────────────────────────────────────
    # BeetleStream v2: Curriculum data pipeline
    # ─────────────────────────────────────────────────────────────────────────

    def _setup_curriculum_pipeline(self):
        """Set up data pipeline for BeetleStream v2 curriculum mode.

        Loads pretokenized Arrow datasets with quality/difficulty/topic_id
        columns and creates a CurriculumDataset that filters based on
        per-condition quality tables.
        """
        from beetlelm.bilingual.data.curriculum_sampler_v2 import (
            CurriculumSamplerV2,
            CurriculumDataset,
        )
        from beetlelm.bilingual.data.quality_tables import get_quality_range
        from beetlelm.bilingual.data.stream_state import DeterministicStreamState
        from beetlelm.bilingual.data.condition_presets import get_condition
        from torch.utils.data import DataLoader

        world_size = self.fabric.world_size
        batch_size = self.training_cfg.get("batch_size", 64)
        seq_len = self.training_cfg.get("seq_len", 512)
        total_steps = self.training_cfg.get("max_steps", 730_000)
        prefetch = 32 if world_size <= 4 else 64 if world_size <= 8 else 128

        # Get indexed data repos
        indexed_repos = self.data_cfg.get("indexed_hf_repos", {})
        if not indexed_repos:
            raise ValueError(
                "curriculum mode requires data.indexed_hf_repos in config"
            )

        # Get condition for quality tables
        curriculum_cfg = self.configs.get("curriculum", {})
        condition = curriculum_cfg.get("condition", "B1")

        # Set up the exposure schedule from condition presets
        lang_keys = list(indexed_repos.keys())
        data_mix = get_condition(
            condition,
            total_steps=total_steps,
            lang_keys=lang_keys if len(lang_keys) >= 3 else None,
        )

        # Stream state for tracking positions
        lang_counts = {k: 1 for k in lang_keys}
        self.stream_state = DeterministicStreamState(lang_counts=lang_counts)

        # Create curriculum sampler (wraps data_mix + quality filtering)
        sampler = CurriculumSamplerV2(
            inner_mix=data_mix,
            condition=condition,
            total_steps=total_steps,
        )

        # Create curriculum dataset
        n_topics = self.data_cfg.get("n_topics", 200)
        topic_aligned = self.data_cfg.get("topic_aligned_batching", True)

        dataset = CurriculumDataset(
            indexed_hf_repos=indexed_repos,
            sampler=sampler,
            tokenizer=self.tokenizer,
            seq_len=seq_len,
            batch_size=batch_size,
            topic_aligned=topic_aligned,
            n_topics=n_topics,
            world_size=world_size,
            rank=self.fabric.global_rank,
        )

        # Create pipeline state
        self.pipeline_state = DataPipelineState(
            stream_state=self.stream_state,
            data_mix=sampler,
        )

        self.dm = None
        self.monolingual_mode = False
        base_iterator = iter(DataLoader(dataset, batch_size=batch_size))

        self.log(f"[Data] BeetleStream v2 curriculum mode: condition={condition}")
        self.log(f"[Data] Languages: {lang_keys}")
        self.log(f"[Data] Topic-aligned batching: {topic_aligned}")
        self.log(f"[Data] Prefetch buffer = {prefetch} batches (GPUs={world_size})")
        self.train_iterator = PrefetchIterator(base_iterator, buffer_size=prefetch)

    # ─────────────────────────────────────────────────────────────────────────
    # Model setup
    # ─────────────────────────────────────────────────────────────────────────

    def _setup_model(self):
        import yaml

        model_cfg = self.configs.get("model")
        if not model_cfg:
            with open(self.config_path, "r") as f:
                model_cfg = yaml.safe_load(f).get("model")
            if not model_cfg:
                raise ValueError(f"'model' key missing in YAML: {self.config_path}")
            self.configs["model"] = model_cfg

        arch   = model_cfg.get("architecture")
        params = model_cfg.get("params", {}) or {}
        params["vocab_size"] = len(self.tokenizer)
        model_config_obj = SimpleNamespace(**params)

        # Architecture registry — populates MODEL_REGISTRY with the built-in
        # architectures (pico, bgpt) and any third-party additions
        # (beetle_moe, beetle_ssm, geo_bilingual). Imported here (not at
        # module top) so the trainer can be unit-tested without requiring
        # every architecture's optional dependencies.
        from beetlelm.training import _arch_imports  # noqa: F401  (registers builtins)
        from beetlelm.training.registry import build_model

        model = build_model(arch, model_config_obj)
        self.log(f"[Model] Created {arch} | vocab_size={model.model_config.vocab_size}")
        return model

    # ─────────────────────────────────────────────────────────────────────────
    # HF push helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _make_hf_model(self):
        from beetlelm.pico_decoder.pico_decoder import PicoDecoderHF, PicoDecoderHFConfig
        raw = getattr(self.model, "module", self.model)
        mc  = raw.model_config
        hf_config = PicoDecoderHFConfig(
            n_layers=mc.n_layers, d_model=mc.d_model, vocab_size=mc.vocab_size,
            attention_n_heads=mc.attention_n_heads,
            attention_n_kv_heads=getattr(mc, "attention_n_kv_heads", 1),
            max_seq_len=mc.max_seq_len, batch_size=mc.batch_size,
            position_emb_theta=mc.position_emb_theta,
            activation_hidden_dim=mc.activation_hidden_dim,
            norm_eps=mc.norm_eps, dropout=mc.dropout,
        )
        hf_model = PicoDecoderHF(hf_config)
        raw_state      = raw.state_dict()
        backbone_state = {k: v for k, v in raw_state.items()
                          if not k.startswith("classifier_head")}
        hf_model.load_state_dict(backbone_state, strict=False)
        hf_model.eval()
        return hf_model

    @rank_zero_only
    def _push_modeling_files(self, repo_id: str, revision: str):
        from huggingface_hub import HfApi, CommitOperationAdd

        pico_src = _PICO_DECODER_SRC
        if not os.path.exists(pico_src):
            pico_src = os.path.join("src", "pico_decoder", "pico_decoder.py")
        if not os.path.exists(pico_src):
            self.log(
                f"[HF] pico_decoder.py not found — skipping for {revision}.",
                level=logging.WARNING,
            )
            return

        raw = getattr(self.model, "module", self.model)
        mc  = raw.model_config
        config = {
            "architectures": ["PicoDecoderHF"], "model_type": "pico_decoder",
            "auto_map": {
                "AutoConfig":           "pico_decoder.PicoDecoderHFConfig",
                "AutoModelForCausalLM": "pico_decoder.PicoDecoderHF",
            },
            "n_layers": mc.n_layers, "d_model": mc.d_model,
            "vocab_size": mc.vocab_size, "attention_n_heads": mc.attention_n_heads,
            "attention_n_kv_heads": getattr(mc, "attention_n_kv_heads", 1),
            "max_seq_len": mc.max_seq_len, "batch_size": mc.batch_size,
            "position_emb_theta": mc.position_emb_theta,
            "activation_hidden_dim": mc.activation_hidden_dim,
            "norm_eps": mc.norm_eps, "dropout": mc.dropout,
            "torch_dtype": "float32", "transformers_version": "4.48.3",
            # Reproducibility: record the training seed used for this checkpoint so
            # multi-seed ablations (e.g. B1/B2/B3 significance sweep) can be paired
            # across variants without ambiguity.
            "training_seed": getattr(self, "seed", 42),
        }
        api = HfApi()
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = os.path.join(tmp, "config.json")
            with open(cfg_path, "w") as f:
                json_mod.dump(config, f, indent=2)
            try:
                api.create_commit(
                    repo_id=repo_id, repo_type="model",
                    operations=[
                        CommitOperationAdd(path_in_repo="pico_decoder.py",
                                           path_or_fileobj=pico_src),
                        CommitOperationAdd(path_in_repo="config.json",
                                           path_or_fileobj=cfg_path),
                    ],
                    commit_message=f"Add pico_decoder.py + config.json (vocab={mc.vocab_size})",
                    revision=revision,
                )
                self.log(f"[HF] pico_decoder.py + config.json → {repo_id}@{revision}")
            except Exception as e:
                if "No files have been modified" not in str(e):
                    self.log(f"[HF] Failed to push modeling files to {revision}: {e}",
                             level=logging.WARNING)

    @rank_zero_only
    def _smoke_test_hf_load(self, checkpoint_path: str) -> None:
        """Load the just-saved checkpoint as a HuggingFace model and run one forward.

        Crashes the run on rank 0 if the checkpoint is not loadable. Intended
        to be invoked once at step 0 so the failure surfaces in <1 minute,
        not after thousands of training steps. Catches:

        - Missing/wrong modeling file in the checkpoint dir.
        - HF config ↔ raw decoder state-dict key mismatch.
        - Broken ``auto_map`` in config.json.
        - Architecture not registered with ``register_for_auto_class``.

        Skipped (with a loud warning, not a hard fail) when the architecture
        has no ``convert_to_hf_model`` method — in that case the checkpoint
        was uploaded as a raw fabric checkpoint with no HF surface, and the
        repo is documented as not loadable via AutoModelForCausalLM.
        """
        from transformers import AutoModelForCausalLM

        raw = getattr(self.model, "module", self.model)
        if not hasattr(raw, "convert_to_hf_model"):
            arch = (self.configs.get("model") or {}).get("architecture", "?")
            self.log(
                f"[HF-smoke] architecture={arch!r} has no convert_to_hf_model() — "
                f"skipping round-trip test. The repo will NOT be loadable via "
                f"AutoModelForCausalLM(trust_remote_code=True).",
                level=logging.WARNING,
            )
            return

        try:
            hf_model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path, trust_remote_code=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"[HF-smoke] Failed to load checkpoint at {checkpoint_path} via "
                f"AutoModelForCausalLM(trust_remote_code=True): "
                f"{type(exc).__name__}: {exc}.\n"
                f"This means the just-saved step-0 branch will not be loadable "
                f"by downstream consumers. Refusing to continue training. Check "
                f"that (a) the modeling .py file is present in the checkpoint dir, "
                f"(b) config.json contains a valid auto_map, and (c) the HF "
                f"wrapper's state_dict keys match the raw decoder's."
            ) from exc

        try:
            vocab_size = int(getattr(hf_model.config, "vocab_size", 256))
            dummy = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
            dummy = dummy.clamp_(0, max(0, vocab_size - 1))
            with torch.no_grad():
                out = hf_model(input_ids=dummy)
            logits = getattr(out, "logits", None)
            if logits is None or logits.ndim < 2 or logits.shape[-1] != vocab_size:
                raise RuntimeError(
                    f"forward returned unexpected logits shape: "
                    f"{tuple(logits.shape) if logits is not None else None} "
                    f"(expected last dim == {vocab_size})"
                )
        except Exception as exc:
            raise RuntimeError(
                f"[HF-smoke] Loaded checkpoint at {checkpoint_path} but forward "
                f"pass failed: {type(exc).__name__}: {exc}. The HF wrapper is "
                f"broken — refusing to continue training."
            ) from exc

        self.log(
            f"[HF-smoke] Round-trip OK: {type(hf_model).__name__} loaded from "
            f"{checkpoint_path} and forward pass succeeded (vocab={vocab_size})."
        )

    def _save_step_checkpoint_local(self, step: int) -> str:
        """
        Write a full HF-compatible checkpoint to the local ``checkpoints/step_{N}/``
        directory.

        This method is **rank-synced** — every rank must enter, because
        ``save_checkpoint`` internally calls ``fabric.save`` which is a collective.
        It is intentionally NOT decorated with ``@rank_zero_only``.

        Returns the absolute path to the saved checkpoint directory on rank 0.
        On other ranks the return value is still correct (same path formula) but
        those ranks should not touch the directory.
        """
        from beetlelm.checkpointing.training import save_checkpoint

        # Schedule-Free AdamW lifecycle: switch the optimizer into eval mode
        # so the persisted parameters are the running average ``x`` (the
        # inference iterate), not the auxiliary ``y`` iterate used during
        # training. After the save, restore train mode. For all other
        # optimizers these calls are no-ops. See AdamWScheduleFree docstring
        # and tests/test_features_differential.py::TestScheduleFreeAdamW.
        self._optimizer_eval_mode()
        try:
            save_checkpoint(
                configs=self.configs,
                checkpoint_step=step,
                fabric=self.fabric,
                model=self.model,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                tokenizer=self.tokenizer,
                pipeline_state=self.pipeline_state,
            )
        finally:
            # Restore Schedule-Free's training-time iterate. No-op for other
            # optimizers. MUST be in a ``finally`` so a save error cannot
            # leave the loop unable to ``step()``.
            self._optimizer_train_mode()

        runs_dir = self.checkpoint_cfg.get("runs_dir", "./runs")
        ckpt_dir = os.path.join(runs_dir, self.run_name, "checkpoints")
        checkpoint_path = os.path.join(ckpt_dir, f"step_{step}")

        # Rank 0 also stages the modeling .py file next to the checkpoint so
        # the HF branch is directly loadable with trust_remote_code=True. The
        # specific file depends on architecture — pico/bgpt/beetle_moe/
        # beetle_ssm/geo_bilingual all have their own. This is local filesystem
        # work only — safe to do here rather than during the async upload.
        if self.fabric.is_global_zero:
            arch = (self.configs.get("model") or {}).get("architecture", "pico")
            resolved = _resolve_modeling_src(arch)
            if resolved is None:
                self.log(
                    f"[HF] No modeling-file mapping for architecture={arch!r} — "
                    f"step-{step} branch will not be loadable with trust_remote_code=True. "
                    f"Add {arch!r} to _ARCH_MODELING_SRC in beetle_trainer.py.",
                    level=logging.ERROR,
                )
            else:
                fname, src_path = resolved
                if src_path is None:
                    self.log(
                        f"[HF] Modeling source file for {arch!r} ({fname}) not found "
                        f"on disk — step-{step} branch will not be loadable with "
                        f"trust_remote_code=True.",
                        level=logging.ERROR,
                    )
                else:
                    os.makedirs(checkpoint_path, exist_ok=True)
                    dst = os.path.join(checkpoint_path, fname)
                    if not os.path.exists(dst):
                        import shutil
                        shutil.copy(src_path, dst)

        return checkpoint_path

    @rank_zero_only
    @use_backoff(max_retries=4, initial_delay=5, backoff_factor=2)
    def _upload_step_checkpoint_to_hf(
        self, step: int, checkpoint_path: str, phase_changed: bool = False
    ) -> None:
        """
        Rank-0 background upload: publish a locally-saved checkpoint as HF branch
        ``step-{N}`` and also replay it onto ``main`` for pico-analyze.

        Runs on :class:`AsyncCheckpointer`'s worker thread — it must NOT call any
        CUDA / NCCL / Fabric API. Wrapped in ``use_backoff`` to survive transient
        network failures; after the final retry the exception propagates and the
        worker thread records it, but it does NOT crash the training loop.
        """
        repo_id = f"{self.hf_user}/{self.run_name}"
        branch  = f"step-{step}"

        # Ensure repo + branch exist.
        api = HfApi()
        create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)

        existing = {b.name for b in api.list_repo_refs(repo_id).branches}
        if branch not in existing:
            api.create_branch(repo_id=repo_id, branch=branch)

        # Upload the full checkpoint directory.
        upload_folder(
            folder_path=checkpoint_path,
            repo_id=repo_id,
            repo_type="model",
            revision=branch,
            commit_message=f"Checkpoint step {step}",
        )
        self.log(f"[HF] Full checkpoint uploaded → {repo_id}@{branch}")

        # Mirror onto `main` in pico-analyze-compatible layout. Only valid for
        # pico-style architectures — beetle_moe / beetle_ssm / geo_bilingual
        # have router/expert/SSM/path-init modules pico-analyze cannot inspect,
        # so we skip the mirror and leave `main` empty for those repos. The
        # primary step-{N} branch is the load surface.
        # Wrapped in try so a pico-mirror failure doesn't mask the primary
        # step-{N} branch upload having succeeded.
        arch = (self.configs.get("model") or {}).get("architecture", "pico")
        if arch in ("pico", "bgpt"):
            try:
                self._push_pico_compatible_to_main(
                    step=step,
                    checkpoint_path=checkpoint_path,
                    phase_changed=phase_changed,
                )
            except Exception as e:
                self.log(f"[HF/main] Pico-mirror push failed for step {step}: {e}",
                         level=logging.WARNING)
        else:
            self.log(
                f"[HF/main] Skipping pico-analyze mirror for arch={arch!r} "
                f"(unsupported by pico-analyze). step-{step} branch is the "
                f"canonical load surface."
            )

    def _build_pico_compatible_training_config(self) -> dict:
        """Return ``self.configs`` normalized to pico-train's ``training_config.yaml`` layout.

        pico-analyze reads ``training_config["model"]["d_model"]``,
        ``training_config["model"]["attention_n_heads"]``,
        ``training_config["model"]["attention_n_kv_heads"]``,
        and ``training_config["checkpointing"]["run_name"]`` directly
        (see ``pico-analyze/src/components/ov_circuit.py:34-37``).

        Beetle's native config nests model hyperparameters under
        ``model.params.*``. Flatten those up to ``model.*`` (without mutating
        ``self.configs``) so pico-analyze components work unchanged.
        """
        cfg = copy.deepcopy(self.configs) if isinstance(self.configs, dict) else {}
        model = cfg.get("model") or {}
        if isinstance(model, dict):
            params = model.pop("params", None)
            if isinstance(params, dict):
                for k, v in params.items():
                    model.setdefault(k, v)
            cfg["model"] = model

        ckpt = cfg.setdefault("checkpointing", {})
        if isinstance(ckpt, dict):
            ckpt.setdefault("run_name", self.run_name)
        return cfg

    @rank_zero_only
    @use_backoff(max_retries=4, initial_delay=5, backoff_factor=2)
    def _push_pico_compatible_to_main(
        self, step: int, checkpoint_path: str, phase_changed: bool = False
    ) -> None:
        """
        Replay a pico-train-compatible commit sequence onto the `main` branch
        so pico-analyze can consume this repo via commit-history resolution.

        Commit messages are verbatim from pico-train and must match the
        regexes in pico-analyze/src/utils/data.py.

        Sequence:
            [step 0 only]
              "Saving Tokenizer -- Step 0"         → tokenizer*.json
              "Saving Training Config -- Step 0"   → training_config.yaml
              "Saving Model Class -- Step 0"       → pico_decoder.py
            [phase boundaries and final step only — NOT every log₂ step]
              "Saving HF Model -- Step {step}"     → config.json, model.safetensors
            [never pushed — fabric_state stays local-only for throughput]
              fabric_state/
        """
        repo_id = f"{self.hf_user}/{self.run_name}"

        # Ensure `main` branch exists. `create_repo` already created `main` as the
        # default branch for new repos; for older repos we make it explicitly.
        api = HfApi()
        existing = {b.name for b in api.list_repo_refs(repo_id).branches}
        if "main" not in existing:
            api.create_branch(repo_id=repo_id, branch="main")

        runs_dir = self.checkpoint_cfg.get("runs_dir", "./runs")
        run_root = os.path.join(runs_dir, self.run_name)

        # ── Step-0 one-time uploads ──
        if step == 0:
            # Tokenizer files
            tok_patterns = ["tokenizer.json", "tokenizer_config.json",
                            "special_tokens_map.json", "vocab.json", "merges.txt"]
            for fname in tok_patterns:
                src = os.path.join(checkpoint_path, fname)
                if os.path.exists(src):
                    upload_file(
                        path_or_fileobj=src,
                        path_in_repo=fname,
                        repo_id=repo_id,
                        revision="main",
                        commit_message=f"Saving Tokenizer -- Step 0",
                    )
            self.log(f"[HF/main] Tokenizer uploaded → {repo_id}@main")

            # training_config.yaml — write a pico-analyze-flavoured copy to a
            # temp path before upload. pico-analyze reads
            # training_config["model"]["d_model"] (and siblings) directly,
            # while beetle's native config nests these under model.params.*.
            # _build_pico_compatible_training_config flattens that path so
            # pico-analyze components (e.g. ov_circuit.py) work unmodified.
            pico_cfg = self._build_pico_compatible_training_config()
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as tf:
                yaml.dump(pico_cfg, tf, sort_keys=False)
                tmp_cfg_path = tf.name
            try:
                upload_file(
                    path_or_fileobj=tmp_cfg_path,
                    path_in_repo="training_config.yaml",
                    repo_id=repo_id,
                    revision="main",
                    commit_message="Saving Training Config -- Step 0",
                )
                self.log(f"[HF/main] training_config.yaml uploaded → {repo_id}@main")
            finally:
                try:
                    os.unlink(tmp_cfg_path)
                except OSError:
                    pass

            # pico_decoder.py (custom model class)
            pico_src = _PICO_DECODER_SRC
            if not os.path.exists(pico_src):
                pico_src = os.path.join("src", "pico_decoder", "pico_decoder.py")
            if os.path.exists(pico_src):
                upload_file(
                    path_or_fileobj=pico_src,
                    path_in_repo="pico_decoder.py",
                    repo_id=repo_id,
                    revision="main",
                    commit_message="Saving Model Class -- Step 0",
                )
                self.log(f"[HF/main] pico_decoder.py uploaded → {repo_id}@main")

        # ── Per-step uploads ──
        # HF model files (config.json + model.safetensors) — upload individually
        # so the commit message can be pico-compatible ("Saving HF Model -- Step N").
        #
        # Restricted to phase boundaries and the final step. Previously pushed
        # at every log₂ checkpoint, which dominated wall-clock (775 MB per
        # event × ~17 events on the human-scale schedule). pico-analyze does
        # not require a per-step model.safetensors on `main` — it reads
        # per-step weights from the learning_dynamics/ tensors, which are
        # still published at every log₂ step below.
        max_steps = (self.configs.get("training") or {}).get("max_steps")
        is_final = isinstance(max_steps, int) and step == max_steps
        if phase_changed or is_final:
            for fname in ("config.json", "model.safetensors"):
                src = os.path.join(checkpoint_path, fname)
                if os.path.exists(src):
                    upload_file(
                        path_or_fileobj=src,
                        path_in_repo=fname,
                        repo_id=repo_id,
                        revision="main",
                        commit_message=f"Saving HF Model -- Step {step}",
                    )
            self.log(f"[HF/main] HF model uploaded (step {step}) → {repo_id}@main")

        # Fabric checkpoint directory is intentionally NOT pushed to HF.
        # The local `fabric_checkpoint/` written by fabric.save() is all
        # `auto_resume` needs; pushing optimizer + RNG state (~1.5–3 GB per
        # event) to the Hub was the dominant upload cost and is not consumed
        # by pico-analyze.

    @rank_zero_only
    def _prune_hf_checkpoints(self, keep_last_n: int = 10):
        if keep_last_n <= 0:
            return
        repo_id = f"{self.hf_user}/{self.run_name}"
        try:
            from huggingface_hub import HfApi
            api           = HfApi()
            branches      = [b.name for b in api.list_repo_refs(repo_id).branches]
            step_branches = sorted(
                [(int(m.group(1)), b) for b in branches
                 for m in [re.match(r"^step-(\d+)$", b)] if m],
                key=lambda x: x[0],
            )
            to_delete = step_branches[:-keep_last_n] if len(step_branches) > keep_last_n else []
            for step_num, branch_name in to_delete:
                api.delete_branch(repo_id=repo_id, branch=branch_name)
                self.log(f"[HF/prune] Deleted old branch {branch_name} (step {step_num})")
        except Exception as exc:
            self.log(f"[HF/prune] Pruning failed: {exc}", level=logging.WARNING)

    @staticmethod
    def _latest_step_from_step_branches(api: HfApi, repo_id: str) -> Optional[int]:
        """Return the max N across ``step-{N}`` branches, or None if unreachable."""
        try:
            refs = use_backoff(max_retries=4, initial_delay=5, backoff_factor=2)(
                api.list_repo_refs
            )(repo_id)
            branches = [b.name for b in refs.branches]
        except Exception:
            return None
        steps = [
            int(m.group(1))
            for b in branches
            for m in [re.match(r"^step-(\d+)$", b)]
            if m
        ]
        return max(steps) if steps else None

    @staticmethod
    def _latest_step_from_main_commits(api: HfApi, repo_id: str) -> Optional[int]:
        """
        Return the max N found in commit messages on the ``main`` branch
        (format ``"... -- Step N"`` as written by :meth:`_push_pico_compatible_to_main`).
        Returns None if the commit history is unreachable or contains no step commits.
        """
        try:
            commits = use_backoff(max_retries=4, initial_delay=5, backoff_factor=2)(
                api.list_repo_commits
            )(repo_id=repo_id, revision="main")
        except Exception:
            return None
        pat = re.compile(r"Step (\d+)")
        steps = [
            int(m.group(1))
            for c in commits
            for m in [pat.search(getattr(c, "title", "") or "")]
            if m
        ]
        return max(steps) if steps else None

    def _resume_from_hf(self):
        """
        Resume from HuggingFace Hub, preferring the newest available checkpoint.

        Resolution order on rank 0:

        1. Scan ``step-{N}`` branches (the canonical beetle-analyze layout).
        2. Scan ``main`` branch commit history for ``"Step N"`` messages
           (the pico-analyze layout, mirrored by
           :meth:`_push_pico_compatible_to_main`).
        3. Pick the newer of (1) and (2). If the ``step-{N}`` branch wins (or
           both point at the same N), download it — it's a single directory.
           If ``main`` is strictly newer, download the pico-layout from main
           and re-arrange it into ``checkpoints/step_{N}/``.

        Ranks 1..N wait at a barrier for rank 0 to finish the download, then
        delegate to :func:`load_checkpoint` in ``src/checkpointing/training.py``.
        """
        if not self.fabric.is_global_zero:
            self.fabric.barrier()
            return load_checkpoint(
                checkpointing_config=self.checkpoint_cfg,
                checkpoint_step="latest",
                fabric=self.fabric,
                model=self.model,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                pipeline_state=self.pipeline_state,
            )

        repo_id  = f"{self.hf_user}/{self.run_name}"
        runs_dir = self.checkpoint_cfg.get("runs_dir", "./runs")
        ckpt_dir = os.path.join(runs_dir, self.run_name, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        api = HfApi()
        branch_step = self._latest_step_from_step_branches(api, repo_id)
        main_step   = self._latest_step_from_main_commits(api, repo_id)

        if branch_step is None and main_step is None:
            self.log(
                f"[Resume/HF] No step-* branches and no 'Step N' commits on main "
                f"in {repo_id} — starting fresh."
            )
            self.fabric.barrier()
            return None

        # Decide source. Prefer step-{N} branches when tied: they ship as a
        # ready-made step_{N}/ directory, no repath needed.
        use_main = (main_step is not None) and (
            branch_step is None or main_step > branch_step
        )
        latest_step = main_step if use_main else branch_step

        try:
            if use_main:
                self.log(
                    f"[Resume/HF] Latest on main is step {latest_step} "
                    f"(step-branches max = {branch_step}). Downloading from main…"
                )
                tmp_root = os.path.join(ckpt_dir, "_hf_main_snapshot")
                os.makedirs(tmp_root, exist_ok=True)
                use_backoff(max_retries=4, initial_delay=5, backoff_factor=2)(
                    snapshot_download
                )(
                    repo_id=repo_id,
                    revision="main",
                    local_dir=tmp_root,
                )
                # Re-lay out into checkpoints/step_{N}/. The pico layout writes
                # config.json, model.safetensors, tokenizer*.json, pico_decoder.py,
                # training_config.yaml at the root and fabric_state/ as a
                # subdirectory — collect them all into step_{N}/. The local
                # checkpointing code expects fabric_checkpoint/, so rename on
                # the way in. Legacy repos still using fabric_checkpoint/ copy
                # through unchanged.
                target = os.path.join(ckpt_dir, f"step_{latest_step}")
                if os.path.exists(target):
                    import shutil
                    shutil.rmtree(target)
                os.makedirs(target, exist_ok=True)
                import shutil
                for entry in os.listdir(tmp_root):
                    src = os.path.join(tmp_root, entry)
                    if entry.startswith(".") or entry == "README.md":
                        continue
                    dst_name = "fabric_checkpoint" if entry == "fabric_state" else entry
                    dst = os.path.join(target, dst_name)
                    if os.path.isdir(src):
                        shutil.copytree(src, dst)
                    else:
                        shutil.copy2(src, dst)
                shutil.rmtree(tmp_root, ignore_errors=True)
                # Point `latest` symlink at step_{N} so load_checkpoint finds it.
                latest_link = os.path.join(ckpt_dir, "latest")
                try:
                    if os.path.islink(latest_link) or os.path.exists(latest_link):
                        os.remove(latest_link)
                    os.symlink(f"step_{latest_step}", latest_link)
                except OSError:
                    # Filesystems that don't support symlinks (rare) — rely on
                    # load_checkpoint's max-step scan instead.
                    pass
                self.log(
                    f"[Resume/HF] Rebuilt step_{latest_step}/ from main → {target}"
                )
            else:
                latest_branch = f"step-{latest_step}"
                self.log(
                    f"[Resume/HF] Downloading step_{latest_step}/ "
                    f"from {repo_id}@{latest_branch} "
                    f"(main commits max = {main_step})…"
                )
                use_backoff(max_retries=4, initial_delay=5, backoff_factor=2)(
                    snapshot_download
                )(
                    repo_id=repo_id,
                    revision=latest_branch,
                    local_dir=ckpt_dir,
                    allow_patterns=[f"step_{latest_step}/**"],
                )
                self.log(f"[Resume/HF] Downloaded step_{latest_step}/ to {ckpt_dir}")

        except Exception as exc:
            self.log(
                f"[Resume/HF] Could not fetch from {repo_id}: {exc}",
                level=logging.WARNING,
            )
            self.fabric.barrier()
            return None

        self.fabric.barrier()
        return load_checkpoint(
            checkpointing_config=self.checkpoint_cfg,
            checkpoint_step="latest",
            fabric=self.fabric,
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            pipeline_state=self.pipeline_state,
        )

    @rank_zero_only
    def _write_hf_upload_sentinel(self) -> None:
        """Write a launcher-visible sentinel signalling whether the run's
        full checkpoint flush to HuggingFace Hub completed cleanly.

        The launcher (`script/launch/_lib.sh::cleanup_runs_if_pushed`) checks
        for ``runs/<run_name>/.hf_upload_done`` before removing
        ``runs/<run_name>/``. If `_push_to_hf` errored OR the async
        checkpointer reports any failed/timed-out background upload, we
        write ``.hf_upload_failed`` instead so the local checkpoint is
        preserved for a safe RESUME.

        Skipped entirely if HF push is disabled in the config — in that
        case the launcher policy can be left as-is (CLEANUP_AFTER_SUCCESS=0
        recommended, but not enforced here).
        """
        if not self.checkpoint_cfg.get("save_to_hf", False):
            return
        runs_dir = self.checkpoint_cfg.get("runs_dir", "./runs")
        run_root = os.path.join(runs_dir, self.run_name)
        try:
            os.makedirs(run_root, exist_ok=True)
        except Exception as exc:
            self.log(
                f"[Sentinel] Could not create {run_root}: {exc}",
                level=logging.WARNING,
            )
            return
        ok_path  = os.path.join(run_root, ".hf_upload_done")
        bad_path = os.path.join(run_root, ".hf_upload_failed")
        if self.async_checkpointer.upload_failed:
            try:
                with open(bad_path, "w") as f:
                    err = self.async_checkpointer.last_error
                    f.write(
                        f"timed_out={self.async_checkpointer.timed_out}\n"
                        f"error={type(err).__name__ if err else 'unknown'}: "
                        f"{err}\n"
                    )
            except Exception:
                pass
            # Make sure no stale "done" sentinel survives from a previous run.
            try:
                if os.path.exists(ok_path):
                    os.remove(ok_path)
            except Exception:
                pass
            self.log(
                f"[Sentinel] Wrote {bad_path} — local checkpoint will be "
                f"preserved by the launcher for RESUME.",
                level=logging.WARNING,
            )
        else:
            try:
                with open(ok_path, "w") as f:
                    f.write("ok\n")
            except Exception as exc:
                self.log(
                    f"[Sentinel] Could not write {ok_path}: {exc}",
                    level=logging.WARNING,
                )

    def _push_to_hf(self):
        if not self.fabric.is_global_zero:
            return
        if not self.checkpoint_cfg.get("save_to_hf", False):
            return
        repo_id = f"{self.hf_user}/{self.run_name}"
        try:
            from huggingface_hub import create_repo
            create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
            hf_model = self._make_hf_model()
            hf_model.push_to_hub(repo_id,
                                  commit_message="End-of-training model weights (PicoDecoderHF)")
            self.tokenizer.push_to_hub(repo_id, commit_message="End-of-training tokenizer")
            self._push_modeling_files(repo_id, revision="main")
            self.log(f"[HF] Final model pushed → https://huggingface.co/{repo_id}")
            self.log(
                f"[HF] Load with:\n"
                f"    from transformers import AutoModelForCausalLM\n"
                f"    model = AutoModelForCausalLM.from_pretrained(\n"
                f"        '{repo_id}', trust_remote_code=True\n"
                f"    )"
            )
        except Exception as exc:
            # Mark the failure on the async checkpointer so the sentinel
            # writer drops `.hf_upload_failed` (preserves runs/<run_name>/).
            self.async_checkpointer.upload_failed = True
            self.async_checkpointer.last_error = exc
            self.log(f"[HF] Final push failed: {exc}", level=logging.ERROR)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BeetleLM Trainer")
    parser.add_argument("--config_path",    type=str, required=True)
    parser.add_argument("--tokenizer_only", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    trainer = Trainer(config_path=args.config_path, tokenizer_only=args.tokenizer_only)
    trainer.train()
