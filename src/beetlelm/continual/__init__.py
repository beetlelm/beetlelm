"""
src/continual
─────────────
Modular continual-learning extensions for BeetleLM.

    from beetlelm.continual import build_ewc, build_maml
    from beetlelm.continual import build_generative_replay, build_similarity_replay

All builders return None if the corresponding YAML block is absent or
disabled, so the default beetle_trainer.py training path is completely
unaffected.
"""

from beetlelm.continual.ewc               import EWCRegularizer,          build_ewc               # noqa: F401
from beetlelm.continual.maml              import SMLMTModule,             build_maml              # noqa: F401
from beetlelm.continual.exposure_lr       import ExposureLRScheduler,    build_exposure_lr       # noqa: F401
from beetlelm.continual.generative_replay import GenerativeReplayModule, build_generative_replay # noqa: F401
from beetlelm.continual.similarity_replay import SimilarityReplayModule, build_similarity_replay # noqa: F401
from beetlelm.continual.tilt              import TILTModule,             build_tilt              # noqa: F401
from beetlelm.continual.sas_regularizer   import SASRegularizer,         build_sas_regularizer   # noqa: F401
from beetlelm.continual.gaussian_noise    import GaussianNoiseInjection, build_gaussian_noise    # noqa: F401
from beetlelm.continual.kl_distillation   import KLDistillationModule,  build_kl_distillation   # noqa: F401
from beetlelm.continual.lang_id_head      import LangIDHead,            build_lang_id_head      # noqa: F401
from beetlelm.continual.lora_adapter      import LoRAModule,            build_lora              # noqa: F401
from beetlelm.continual.layer_growth      import LayerGrowthModule,     build_layer_growth      # noqa: F401
from beetlelm.continual.alignment_loss    import AlignmentLossModule,   build_alignment_loss    # noqa: F401

__all__ = [
    "EWCRegularizer",          "build_ewc",
    "SMLMTModule",             "build_maml",
    "ExposureLRScheduler",     "build_exposure_lr",
    "GenerativeReplayModule",  "build_generative_replay",
    "SimilarityReplayModule",  "build_similarity_replay",
    "TILTModule",              "build_tilt",
    "SASRegularizer",          "build_sas_regularizer",
    "GaussianNoiseInjection",  "build_gaussian_noise",
    "KLDistillationModule",    "build_kl_distillation",
    "LangIDHead",              "build_lang_id_head",
    "LoRAModule",              "build_lora",
    "LayerGrowthModule",       "build_layer_growth",
    "AlignmentLossModule",     "build_alignment_loss",
]
