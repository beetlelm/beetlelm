# beetlelm/src/bilingual/data/experiment_builder.py
"""
Cognitive Scientist Experiment Builder for BeetleLM.

Provides a Python API for composing bilingual/trilingual experiments
without editing YAML. Designed for researchers who want to:
  1. Select a cognitive condition (B1–B5, T1–T4)
  2. Override EWC, MAML, and LR parameters
  3. Generate combinatorial sweeps across parameter grids
  4. Export YAML configs or run experiments directly

Example
-------
    from beetlelm.bilingual.data.experiment_builder import Experiment, Sweep

    # Single experiment with overrides
    exp = (
        Experiment("B2")
        .languages(l1="nld", l2="eng")
        .ewc(lambda_=30)
        .lr(boost=2.5)
        .build()
    )
    exp.to_yaml("configs/my_experiment.yaml")

    # Full combinatorial sweep
    sweep = (
        Sweep(conditions=["B1", "B2", "B3", "B4", "B5"])
        .language_pairs([("nld", "eng"), ("deu", "eng"), ("zho", "eng")])
        .ewc_grid([0, 10, 20, 30, 40])
        .maml_grid([0, 0.2, 0.4])
        .lr_boost_grid([1.0, 2.0, 3.0])
        .generate("configs/conditions")
    )
    print(f"Generated {sweep.n_configs} configs")

Condition Parameter Guide
-------------------------
For cognitive scientists unfamiliar with ML training parameters:

EWC (Elastic Weight Consolidation):
    Controls how strongly the model "remembers" earlier languages.
    lambda_ = 0   → no protection (free to forget L1)
    lambda_ = 10  → mild protection
    lambda_ = 40  → strong protection (resists forgetting)
    Use higher values when L1 retention matters (B2, B4, B5).

MAML (Model-Agnostic Meta-Learning):
    Controls rapid adaptation to new language input.
    hybrid_ratio = 0   → no meta-learning
    hybrid_ratio = 0.3 → moderate meta-learning
    hybrid_ratio = 0.5 → strong meta-learning
    Use when the model should quickly adapt (B1, B3, T1, T3).

LR Boost (Learning Rate Plasticity):
    Controls learning speed increase when new language appears.
    boost = 1.0 → no boost (constant learning rate)
    boost = 2.5 → 2.5× faster learning at phase transitions
    boost = 3.0 → 3× faster learning
    Use higher values for late/sudden language introduction (B5).
"""

from __future__ import annotations

import copy
import itertools
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep-merge override into base (override wins)."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


class Experiment:
    """
    Builder for a single training experiment.

    Fluent interface for cognitive scientists to configure and export
    a bilingual/trilingual training run.

    Parameters
    ----------
    condition : str
        Condition code (B1–B5, T1–T4).
    """

    def __init__(self, condition: str):
        self.condition = condition.upper()
        self._l1: Optional[str] = None
        self._l2: Optional[str] = None
        self._langs: Optional[List[str]] = None
        self._ewc_overrides: Dict[str, Any] = {}
        self._maml_overrides: Dict[str, Any] = {}
        self._lr_overrides: Dict[str, Any] = {}
        self._extra_overrides: Dict[str, Any] = {}

    def languages(
        self,
        l1: Optional[str] = None,
        l2: Optional[str] = None,
        langs: Optional[List[str]] = None,
    ) -> Experiment:
        """Set language pair (bilingual) or triple (trilingual)."""
        if langs:
            self._langs = langs
        if l1:
            self._l1 = l1
        if l2:
            self._l2 = l2
        return self

    def ewc(
        self,
        enabled: Optional[bool] = None,
        lambda_: Optional[float] = None,
        n_fisher_samples: Optional[int] = None,
    ) -> Experiment:
        """
        Configure Elastic Weight Consolidation.

        EWC protects knowledge of earlier languages from being overwritten.
        Higher lambda_ = stronger protection against forgetting.
        """
        if enabled is not None:
            self._ewc_overrides["enabled"] = enabled
        if lambda_ is not None:
            self._ewc_overrides["enabled"] = lambda_ > 0
            self._ewc_overrides["lambda_"] = float(lambda_)
        if n_fisher_samples is not None:
            self._ewc_overrides["n_fisher_samples"] = n_fisher_samples
        return self

    def maml(
        self,
        enabled: Optional[bool] = None,
        hybrid_ratio: Optional[float] = None,
        inner_steps: Optional[int] = None,
        inner_lr: Optional[float] = None,
    ) -> Experiment:
        """
        Configure Model-Agnostic Meta-Learning.

        MAML enables rapid adaptation to new language input, modeling
        the ability to quickly "pick up" a new language.
        """
        if enabled is not None:
            self._maml_overrides["enabled"] = enabled
        if hybrid_ratio is not None:
            self._maml_overrides["enabled"] = hybrid_ratio > 0
            self._maml_overrides["hybrid_ratio"] = float(hybrid_ratio)
        if inner_steps is not None:
            self._maml_overrides["inner_steps"] = inner_steps
        if inner_lr is not None:
            self._maml_overrides["inner_lr"] = float(inner_lr)
        return self

    def lr(
        self,
        base: Optional[float] = None,
        boost: Optional[float] = None,
        scheduler: Optional[str] = None,
    ) -> Experiment:
        """
        Configure learning rate.

        boost: multiplier for learning rate at language transitions.
        scheduler: "exposure_based" or "linear_with_warmup".
        """
        if base is not None:
            self._lr_overrides["lr"] = float(base)
        if scheduler is not None:
            self._lr_overrides["lr_scheduler"] = scheduler
        if boost is not None:
            if "exposure_lr" not in self._lr_overrides:
                self._lr_overrides["exposure_lr"] = {}
            self._lr_overrides["exposure_lr"]["plasticity_boost"] = float(boost)
        return self

    def override(self, **kwargs) -> Experiment:
        """Set arbitrary top-level config overrides."""
        self._extra_overrides.update(kwargs)
        return self

    def build(self) -> ExperimentConfig:
        """Build the final config object."""
        from beetlelm.bilingual.data.condition_presets import CONDITION_DEFAULTS
        from beetlelm.config._human_scale_defaults import (
            HF_USER_MODELS, MODEL_BASE, TRAINING_DEFAULTS,
            CHECKPOINTING_DEFAULTS, MAX_STEPS, VOCAB_SIZE,
            TOKENIZER_TYPE, TOKENIZER_TOKENS,
            babybabel_l1_dataset, babybabel_l2_dataset, babybabel_tri_dataset,
            babybabel_bilingual_tokenizer, babybabel_trilingual_tokenizer,
        )
        from configs.ckpt_schedule import phase_log2_schedule

        defaults = copy.deepcopy(CONDITION_DEFAULTS[self.condition])

        # Apply overrides
        if self._ewc_overrides:
            defaults["ewc"] = _deep_merge(defaults.get("ewc", {}), self._ewc_overrides)
        if self._maml_overrides:
            defaults["maml"] = _deep_merge(defaults.get("maml", {}), self._maml_overrides)
        if self._lr_overrides:
            defaults["optimization"] = _deep_merge(
                defaults.get("optimization", {}), self._lr_overrides
            )

        # Determine languages
        is_trilingual = self.condition.startswith("T")
        if is_trilingual:
            lang_keys = self._langs or ["eng", "nld", "deu"]
        else:
            l1 = self._l1 or "eng"
            l2 = self._l2 or "nld"
            lang_keys = [l1, l2]

        # Build suffix from overrides
        suffix_parts = []
        ewc = defaults.get("ewc", {})
        if self._ewc_overrides and "lambda_" in self._ewc_overrides:
            suffix_parts.append(f"ewc{self._ewc_overrides['lambda_']:.0f}")
        maml = defaults.get("maml", {})
        if self._maml_overrides and "hybrid_ratio" in self._maml_overrides:
            suffix_parts.append(f"maml{self._maml_overrides['hybrid_ratio']}")
        opt = defaults.get("optimization", {})
        if self._lr_overrides.get("exposure_lr", {}).get("plasticity_boost") is not None:
            suffix_parts.append(
                f"lrb{self._lr_overrides['exposure_lr']['plasticity_boost']}"
            )
        suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""

        # Build run name
        if is_trilingual:
            pk = "-".join(lang_keys)
        else:
            pk = f"{lang_keys[0]}_L1-{lang_keys[1]}_L2"
        run_name = f"beetlelm_{self.condition}_{pk}{suffix}"

        # Phase starts for checkpoint scheduling
        phase_starts = [0]
        if self.condition in ("B2", "B5"):
            phase_starts = [0, int(MAX_STEPS * 0.4)]
        elif self.condition == "B3":
            phase_starts = [0, int(MAX_STEPS * 0.35)]
        elif self.condition == "B4":
            phase_starts = [0, int(MAX_STEPS * 0.1)]
        elif self.condition == "T2":
            phase_starts = [0, int(MAX_STEPS * 0.3), int(MAX_STEPS * 0.65)]
        elif self.condition == "T3":
            phase_starts = [0, int(MAX_STEPS * 0.35), int(MAX_STEPS * 0.5)]
        elif self.condition == "T4":
            phase_starts = [0, int(MAX_STEPS * 0.4)]
        ckpt_steps = phase_log2_schedule(phase_starts, max_steps=MAX_STEPS)

        training = copy.deepcopy(TRAINING_DEFAULTS)
        training["max_steps"] = MAX_STEPS
        training["optimization"] = _deep_merge(
            training["optimization"], defaults.get("optimization", {})
        )

        # Datasets
        if is_trilingual:
            l1, l2, l3 = lang_keys
            datasets = {lk: babybabel_tri_dataset(lk, l1, l2, l3) for lk in lang_keys}
            tokenizer_repo = babybabel_trilingual_tokenizer(l1, l2, l3)
            data_block = {
                "data_mix": self.condition,
                "lang_streams": datasets,
                "pretokenized": True,
                "pretokenized_cache_dir": f"./pretok_cache/{run_name}",
            }
            curriculum_block = {
                "type": "smooth",
                "condition": self.condition,
                "lang_keys": lang_keys,
            }
        else:
            l1, l2 = lang_keys
            l1_ds = babybabel_l1_dataset(l1, l2)
            l2_ds = babybabel_l2_dataset(l1, l2)
            tokenizer_repo = babybabel_bilingual_tokenizer(l1, l2)
            data_block = {
                "data_mix": self.condition,
                "lang_sources": [l1_ds],
                "l2_streams": [l2_ds],
                "pretokenized": True,
                "pretokenized_cache_dir": f"./pretok_cache/{run_name}",
            }
            curriculum_block = {
                "type": "smooth",
                "condition": self.condition,
            }

        cfg = {
            "run_name": run_name,
            "hf_user": HF_USER_MODELS,
            "model": {
                "architecture": MODEL_BASE["architecture"],
                "params": dict(MODEL_BASE["params"]),
            },
            "data": data_block,
            "tokenizer": {
                "pretrained_tokenizer_path": tokenizer_repo,
                "dataset_source": "babybabel",
                "vocab_size": VOCAB_SIZE,
                "vocab_sharing": "shared",
                "type": TOKENIZER_TYPE,
                "target_tokens": TOKENIZER_TOKENS,
                "output_dir": f"./tokenizers/{sorted_tag}_shared",
                "overwrite": False,
                "push_to_hub": True,
            },
            "training": training,
            "checkpointing": {**CHECKPOINTING_DEFAULTS, "checkpoint_steps": ckpt_steps},
            "ewc": defaults.get("ewc", {"enabled": False}),
            "maml": defaults.get("maml", {"enabled": False}),
            "curriculum": curriculum_block,
        }

        if self._extra_overrides:
            cfg = _deep_merge(cfg, self._extra_overrides)

        return ExperimentConfig(cfg, self.condition, lang_keys, suffix)


class ExperimentConfig:
    """A built experiment config ready for export or inspection."""

    def __init__(
        self,
        config: dict,
        condition: str,
        lang_keys: List[str],
        suffix: str,
    ):
        self.config = config
        self.condition = condition
        self.lang_keys = lang_keys
        self.suffix = suffix

    @property
    def run_name(self) -> str:
        return self.config["run_name"]

    def to_yaml(self, path: Union[str, Path]) -> Path:
        """Write config to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.config, f, sort_keys=False)
        return path

    def to_dict(self) -> dict:
        """Return the config as a dict."""
        return copy.deepcopy(self.config)

    def summary(self) -> str:
        """Human-readable summary of the experiment."""
        ewc = self.config.get("ewc", {})
        maml = self.config.get("maml", {})
        opt = self.config.get("training", {}).get("optimization", {})

        lines = [
            f"Condition: {self.condition}",
            f"Languages: {self.lang_keys}",
            f"Run name:  {self.run_name}",
            f"EWC:       {'ON (lambda={})'.format(ewc.get('lambda_', '?')) if ewc.get('enabled') else 'OFF'}",
            f"MAML:      {'ON (ratio={})'.format(maml.get('hybrid_ratio', '?')) if maml.get('enabled') else 'OFF'}",
            f"LR:        {opt.get('lr', '?')} ({opt.get('lr_scheduler', '?')})",
        ]
        exp_lr = opt.get("exposure_lr", {})
        if exp_lr.get("plasticity_boost"):
            lines.append(f"LR Boost:  {exp_lr['plasticity_boost']}x at transitions")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"ExperimentConfig({self.run_name})"


class Sweep:
    """
    Combinatorial sweep generator for MECO parameter fitting.

    Generates all combinations of conditions × language pairs × EWC × MAML × LR
    and exports them as YAML configs for distributed training.

    Example
    -------
        sweep = (
            Sweep(conditions=["B2", "B3"])
            .language_pairs([("nld", "eng"), ("deu", "eng")])
            .ewc_grid([0, 10, 20])
            .maml_grid([0, 0.3])
            .lr_boost_grid([1.0, 2.0])
            .generate("configs/sweep")
        )

    This generates 2 conditions × 2 pairs × 3 EWC × 2 MAML × 2 LR = 48 configs.
    """

    def __init__(
        self,
        conditions: Optional[List[str]] = None,
        bilingual: Optional[List[str]] = None,
        trilingual: Optional[List[str]] = None,
    ):
        if conditions:
            self._bilingual = [c for c in conditions if c.upper().startswith("B")]
            self._trilingual = [c for c in conditions if c.upper().startswith("T")]
        else:
            self._bilingual = bilingual or []
            self._trilingual = trilingual or []

        self._bi_pairs: List[Tuple[str, str]] = []
        self._tri_triples: List[List[str]] = []
        self._ewc_values: List[float] = [None]  # None = use condition default
        self._maml_values: List[float] = [None]
        self._lr_boost_values: List[float] = [None]
        self._meco_only: bool = False
        self.n_configs: int = 0
        self._configs: List[ExperimentConfig] = []

    def language_pairs(self, pairs: List[Tuple[str, str]]) -> Sweep:
        """Set bilingual language pairs to sweep."""
        self._bi_pairs = list(pairs)
        return self

    def language_triples(self, triples: List[List[str]]) -> Sweep:
        """Set trilingual language triples to sweep."""
        self._tri_triples = [list(t) for t in triples]
        return self

    def ewc_grid(self, values: List[float]) -> Sweep:
        """Set EWC lambda values to sweep. Only applied to EWC-enabled conditions."""
        self._ewc_values = list(values)
        return self

    def maml_grid(self, values: List[float]) -> Sweep:
        """Set MAML hybrid_ratio values to sweep. Only applied to MAML-enabled conditions."""
        self._maml_values = list(values)
        return self

    def lr_boost_grid(self, values: List[float]) -> Sweep:
        """Set LR plasticity_boost values. Only applied to exposure_based LR conditions."""
        self._lr_boost_values = list(values)
        return self

    def meco_pairs_only(self) -> Sweep:
        """Restrict to MECO-relevant language pairs only."""
        self._bi_pairs = [("nld", "eng"), ("deu", "eng"), ("zho", "eng")]
        self._tri_triples = [
            ["nld", "eng", "deu"],
            ["deu", "eng", "fra"],
            ["zho", "eng", "deu"],
        ]
        self._meco_only = True
        return self

    def _should_sweep_ewc(self, condition: str) -> bool:
        from beetlelm.bilingual.data.condition_presets import CONDITION_DEFAULTS
        return CONDITION_DEFAULTS.get(condition, {}).get("ewc", {}).get("enabled", False)

    def _should_sweep_maml(self, condition: str) -> bool:
        from beetlelm.bilingual.data.condition_presets import CONDITION_DEFAULTS
        return CONDITION_DEFAULTS.get(condition, {}).get("maml", {}).get("enabled", False)

    def _should_sweep_lr(self, condition: str) -> bool:
        from beetlelm.bilingual.data.condition_presets import CONDITION_DEFAULTS
        opt = CONDITION_DEFAULTS.get(condition, {}).get("optimization", {})
        return opt.get("lr_scheduler") == "exposure_based"

    def generate(self, output_dir: Union[str, Path]) -> Sweep:
        """Generate all sweep configs and write to output_dir."""
        output_dir = Path(output_dir)
        self._configs = []

        # Bilingual sweep
        for condition in self._bilingual:
            cond = condition.upper()
            ewc_vals = self._ewc_values if self._should_sweep_ewc(cond) else [None]
            maml_vals = self._maml_values if self._should_sweep_maml(cond) else [None]
            lr_vals = self._lr_boost_values if self._should_sweep_lr(cond) else [None]

            for (l1, l2) in self._bi_pairs:
                for ewc_lam, maml_ratio, lr_boost in itertools.product(
                    ewc_vals, maml_vals, lr_vals
                ):
                    exp = Experiment(cond).languages(l1=l1, l2=l2)
                    if ewc_lam is not None:
                        exp.ewc(lambda_=ewc_lam)
                    if maml_ratio is not None:
                        exp.maml(hybrid_ratio=maml_ratio)
                    if lr_boost is not None:
                        exp.lr(boost=lr_boost)

                    cfg = exp.build()
                    subdir = output_dir / "bilingual" / cond
                    cfg.to_yaml(subdir / f"{cfg.run_name}.yaml")
                    self._configs.append(cfg)

        # Trilingual sweep
        for condition in self._trilingual:
            cond = condition.upper()
            ewc_vals = self._ewc_values if self._should_sweep_ewc(cond) else [None]
            maml_vals = self._maml_values if self._should_sweep_maml(cond) else [None]
            lr_vals = self._lr_boost_values if self._should_sweep_lr(cond) else [None]

            for langs in self._tri_triples:
                for ewc_lam, maml_ratio, lr_boost in itertools.product(
                    ewc_vals, maml_vals, lr_vals
                ):
                    exp = Experiment(cond).languages(langs=langs)
                    if ewc_lam is not None:
                        exp.ewc(lambda_=ewc_lam)
                    if maml_ratio is not None:
                        exp.maml(hybrid_ratio=maml_ratio)
                    if lr_boost is not None:
                        exp.lr(boost=lr_boost)

                    cfg = exp.build()
                    subdir = output_dir / "trilingual" / cond
                    cfg.to_yaml(subdir / f"{cfg.run_name}.yaml")
                    self._configs.append(cfg)

        self.n_configs = len(self._configs)
        return self

    def configs(self) -> List[ExperimentConfig]:
        """Return all generated configs."""
        return list(self._configs)

    def summary(self) -> str:
        """Human-readable sweep summary."""
        lines = [
            f"Sweep Summary",
            f"{'='*50}",
            f"Bilingual conditions:  {self._bilingual}",
            f"Trilingual conditions: {self._trilingual}",
            f"Bilingual pairs:       {len(self._bi_pairs)}",
            f"Trilingual triples:    {len(self._tri_triples)}",
            f"EWC grid:              {self._ewc_values}",
            f"MAML grid:             {self._maml_values}",
            f"LR boost grid:         {self._lr_boost_values}",
            f"Total configs:         {self.n_configs}",
        ]
        return "\n".join(lines)
