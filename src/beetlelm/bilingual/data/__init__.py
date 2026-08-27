# src/bilingual/data/__init__.py

# ── Dataset / DataModule ──────────────────────────────────────────────────────
from .bilingual_dataset import BilingualDataset, initialize_dataset
from .datamodule import BilingualDataModule

# ── Curriculum ───────────────────────────────────────────────────────────────
from .curriculum import DataMixConfig, ExposureSchedule

# ── Condition presets (B1-B6, T1-T4) ─────────────────────────────────────────
from .condition_presets import (
    get_condition,
    get_condition_defaults,
    list_conditions,
    CONDITION_ALIASES,
    BILINGUAL_BASE_CONDITIONS,
    ALL_CONDITION_BUILDERS,
)
from .experiment_builder import Experiment, Sweep

# ── Pipeline state ────────────────────────────────────────────────────────────
from .pipeline_state import DataPipelineState
from .stream_state import DeterministicStreamState

# ── Sampling ──────────────────────────────────────────────────────────────────
from .sampling import CurriculumSampler

# ── BeetleStream v2 (curriculum mode) ─────────────────────────────────────────
from .stream_mode import StreamMode
from .quality_tables import QUALITY_TABLES, get_quality_range
from .indexed_shard_reader import IndexedShardReader
from .topic_aligner import TopicAligner
from .curriculum_sampler_v2 import CurriculumSamplerV2, CurriculumDataset

__all__ = [
    # Dataset / DataModule
    "BilingualDataset",
    "BilingualDataModule",
    "initialize_dataset",
    # Curriculum
    "DataMixConfig",
    "ExposureSchedule",
    # Condition presets (B1-B6, T1-T4)
    "get_condition",
    "get_condition_defaults",
    "list_conditions",
    "CONDITION_ALIASES",
    "BILINGUAL_BASE_CONDITIONS",
    "ALL_CONDITION_BUILDERS",
    # Experiment builder
    "Experiment",
    "Sweep",
    # Pipeline / stream state
    "DataPipelineState",
    "DeterministicStreamState",
    # Sampler
    "CurriculumSampler",
    # BeetleStream v2
    "StreamMode",
    "QUALITY_TABLES",
    "get_quality_range",
    "IndexedShardReader",
    "TopicAligner",
    "CurriculumSamplerV2",
    "CurriculumDataset",
]
