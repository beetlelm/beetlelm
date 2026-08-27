# src/config/__init__.py
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HFCheckpointConfig:
    """HuggingFace Hub push configuration (pico trainer only)."""
    repo_id: Optional[str] = None
    private: bool = False


@dataclass
class LearningDynamicsConfig:
    """Configuration for learning dynamics logging (pico trainer only)."""
    enabled: bool = False
    log_every_n_steps: int = 100


@dataclass
class POLCAConfig:
    """
    Configuration for POLCA artifact extraction during training.

    POLCA (Phase-wise Loss Contribution Analysis; arXiv 2510.05064) needs
    per-checkpoint gradient SVD bases and Hessian top-k eigenvectors to
    decompose loss changes along the low-rank training subspace.  When
    ``enabled`` is True, the beetle trainer runs a small fp32 forward/back
    pass on a held-out batch every ``every_n_steps`` to compute and persist
    these artifacts to ``{runs_dir}/{run_name}/{artifact_dir}/step_{N}.pt``.

    Defaults are conservative (rank-32 SVD basis, top-32 Hessian
    eigenvectors via Lanczos, holdout batch of 8 sequences) so the extra
    work fits comfortably inside the existing 8×A100 80 GB step budget for
    the 0.2B Beetle pico model — see plan
    ``.claude/plans/binary-bouncing-quokka.md`` for the rationale.

    NB: this dataclass exists for documentation; the beetle trainer reads
    the ``polca:`` YAML block as a plain dict via
    ``self.configs.get("polca", {})``, mirroring how it reads ``ewc`` /
    ``maml`` / ``generative_replay``.  Field names and defaults below are
    the source of truth for ``configs/.../*.yaml`` defaults written by
    ``script/add_polca_block.py``.
    """

    enabled: bool = True
    every_n_steps: int = 500
    k_eigenvectors: int = 32
    basis_rank: int = 32
    holdout_batch_size: int = 8
    holdout_seq_len: int = 512
    artifact_dir: str = "polca"
    push_to_hf: bool = True
    precision: str = "fp32"


@dataclass
class CheckpointingConfig:
    """
    Configuration for checkpoint saving and loading.

    Used directly by the pico trainer (dataclass path).
    The beetle trainer passes an equivalent plain dict, which is
    normalised by _CkptCfg / _resolve() in src/checkpointing/training.py.
    """

    # ── Directory layout ────────────────────────────────────────────────────
    runs_dir: str = "runs"
    run_name: str = "run"
    checkpoints_dir: str = "checkpoints"
    fabric_checkpoint_dir: str = "fabric_checkpoint"
    fabric_checkpoint_filename: str = "checkpoint.pt"
    logs_dir: str = "logs"

    # ── Save behaviour ──────────────────────────────────────────────────────
    save_every_n_steps: int = 1000
    keep_local_checkpoints: int = 1          # 0 = keep all

    # ── HuggingFace Hub ─────────────────────────────────────────────────────
    save_to_hf: bool = False
    hf_checkpoint: HFCheckpointConfig = field(default_factory=HFCheckpointConfig)

    # ── Pico-only extras ────────────────────────────────────────────────────
    learning_dynamics: LearningDynamicsConfig = field(
        default_factory=LearningDynamicsConfig
    )
