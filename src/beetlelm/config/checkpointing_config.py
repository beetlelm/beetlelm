"""
src/config/checkpointing_config.py
──────────────────────────────────
Dataclass for learning-dynamics checkpointing configuration.

Consumed by CheckpointStateExtractor in src/checkpointing/learning_dynamics.py
and by the beetle trainer for interpretability checkpointing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class LearningDynamicsCheckpointingConfig:
    """Configuration for extracting and saving intermediate activations,
    weights, and gradients at user-defined checkpoint intervals.

    Parameters
    ----------
    enabled : bool
        Master switch.  When False, no dynamics extraction takes place.
    log_every_n_steps : int
        Deprecated and ignored. LD now fires at the curriculum checkpoint
        schedule (log₂-spaced steps + phase boundaries) in
        ``BeetleTrainer``, so any LD-specific interval here has no effect.
        Retained for backward compatibility with existing YAMLs.
    batch_size : int
        Number of samples used for dynamics extraction.
    sequence_idx : int
        Token position from which activations are gathered.
        -1 means the last token.
    layer_suffixes : list[str]
        Module-name substrings used to select which layers to hook.
        The default ``["attention", "swiglu", "norm"]`` matches every
        sub-projection of the pico_decoder block (attention.q_proj /
        .k_proj / .v_proj / .o_proj, swiglu.w_1 / .w_2 / .w_3, plus the
        RMSNorm weights attention_norm / swiglu_norm / output_norm), so
        pico-analyze's ov_circuit and simple-swiglu metrics work
        out-of-the-box. Shorter substrings like ``["attn", "mlp"]`` DO
        NOT match pico_decoder's literal module names and would only
        capture the \\*_norm modules.
    compute_gradients : bool
        Whether to run a backward pass and capture per-layer gradients.
    save_training_batch : bool
        Whether to persist the raw ``input_ids`` tensor of the LD batch
        (``training_batch.pt``).
    save_train_data : bool
        Whether to persist the LD batch as an HF Dataset of ``input_ids`` +
        detokenised ``text`` under ``learning_dynamics/train_data/``.
        pico-analyze does NOT read this (it loads it into
        ``checkpoint_states["dataset"]`` but no metric consumes it); default
        OFF to save HF storage.  Set true only for batch-level debugging.
    push_to_hf : bool
        Upload dynamics artifacts to HuggingFace Hub alongside the
        model checkpoint.
    cleanup_local : bool
        Delete old dynamics directories locally after upload.
    keep_n_local : int
        How many checkpoints' dynamics directories to keep locally.
    learning_dynamics_dir : str
        Subdirectory name under each ``step_{N}/`` checkpoint.
    """

    enabled: bool = False
    log_every_n_steps: int = 1000
    batch_size: int = 64
    sequence_idx: int = -1
    layer_suffixes: List[str] = field(
        default_factory=lambda: ["attention", "swiglu", "norm"]
    )
    compute_gradients: bool = False
    save_training_batch: bool = True
    save_train_data: bool = False
    push_to_hf: bool = True
    cleanup_local: bool = True
    keep_n_local: int = 1
    learning_dynamics_dir: str = "learning_dynamics"
