# beetlelm/src/bilingual/data/condition_presets.py
"""
Cognitively-inspired bilingual (B1-B6) and trilingual (T1-T4) condition presets.

Each preset returns a DataMixConfig with appropriate:
  - Exposure schedule (sigmoid transitions, bursts, episodic clustering)
  - Default EWC / MAML / LR configuration recommendations
  - Phase thresholds for backward-compatible checkpoint triggers

Usage
-----
    from beetlelm.bilingual.data.condition_presets import get_condition

    mix = get_condition("B2", total_steps=10_000)
    # Returns DataMixConfig

    defaults = get_condition_defaults("B2")
    # Returns dict with recommended ewc, maml, lr settings

Bilingual Conditions
--------------------
B1 - Balanced
    50:50 L1/L2 throughout (from step 0).
    Ablations: B1a (delay L2 to 50%), B1b (+episodic context clustering).

B2 - Simultaneous
    100% L1 until midpoint (50%), then sigmoid to 50:50.
    Ablations: B2_later (65% midpoint), B2_smooth (steepness=8),
    B2_burst (+L2 bursts), B2_ewc (+EWC), B2_lr (+LR spike).

B3 - Sequential
    100% L1 until 50%, then 33:67 L1:L2 (Part-Time, default).
    Variants: B3_full (0:100), B3_25_75 (25:75).

B4 - Classroom
    L1=80M, L2=20M. L2 at 50%, episodic bursts, strong EWC.
    Ablations: B4a (no episodic), B4b (no EWC).

B5 - Late
    L2 introduced at 80% of training steps, target 50:50.
    Ablation: B5_ewc (+EWC).

B6 - Heritage
    90% L1 -> L2 dominance -> L1 re-exposure via bursts.
    EWC + MAML, cyclical LR.

Trilingual Conditions
---------------------
T1 - Simultaneous Trilingual
    ~33/33/33 with strong context clustering.

T2 - Sequential L1 -> L2 -> L3
    Smooth staged introduction via chained sigmoids.

T3 - Heritage + L3 Learner
    Early L1 -> L2 dominance -> later L3, L1 reintroduced episodically.

T4 - Dominant L2 + Weak L1 + Late L3
    L2 dominant, L1 weak but persistent, L3 introduced later.

String Aliases
--------------
    "balanced" -> "B1", "simultaneous" -> "B2", "sequential" -> "B3",
    "classroom" -> "B4", "late" -> "B5", "heritage" -> "B6"
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Any

from beetlelm.bilingual.data.curriculum import (
    DataMixConfig,
    ExposureSchedule,
    ConstantExposure,
    SigmoidRamp,
    LinearRamp,
    BurstSchedule,
    CyclicalSchedule,
    CompositeSchedule,
    EpisodicSampler,
    normalise_weights,
)


# ---------------------------------------------------------------------------
# String aliases
# ---------------------------------------------------------------------------

CONDITION_ALIASES: Dict[str, str] = {
    "balanced": "B1",
    "simultaneous": "B2",
    "sequential": "B3",
    "classroom": "B4",
    "late": "B5",
    "heritage": "B6",
    "part_time": "B3",
}


def _resolve_name(name: str) -> str:
    """Resolve string aliases to condition codes."""
    return CONDITION_ALIASES.get(name.lower(), name)


def _base_condition(name: str) -> str:
    """Extract base condition code: B2_later -> B2, B1a -> B1, T3 -> T3."""
    m = re.match(r'^([BT]\d+)', name)
    return m.group(1) if m else name


# ---------------------------------------------------------------------------
# Bilingual presets -- B1 Balanced
# ---------------------------------------------------------------------------

def B1_balanced(*, total_steps: int, seed: int = 0) -> DataMixConfig:
    """B1 - Balanced: 50:50 L1/L2 throughout (from step 0)."""
    return DataMixConfig(
        schedule=ConstantExposure({"l1": 0.5, "l2": 0.5}),
        mode="steps",
        total_steps=total_steps,
        seed=seed,
        curriculum_type="B1_balanced",
        noise_std=0.02,
    )


def B1a_balanced_delayed(*, total_steps: int, seed: int = 0) -> DataMixConfig:
    """B1a - Delay L2 introduction to 50% of training, then 50:50."""
    return DataMixConfig(
        schedule=SigmoidRamp(
            from_weights={"l1": 1.0, "l2": 0.0},
            to_weights={"l1": 0.5, "l2": 0.5},
            midpoint=0.5,
            steepness=20.0,
        ),
        mode="steps",
        total_steps=total_steps,
        seed=seed,
        curriculum_type="B1a_balanced_delayed",
        phase_thresholds={"l2_intro": 0.5},
        noise_std=0.02,
    )


def B1b_balanced_episodic(*, total_steps: int, seed: int = 0) -> DataMixConfig:
    """B1b - 50:50 overall with episodic context clustering."""
    return DataMixConfig(
        schedule=ConstantExposure({"l1": 0.5, "l2": 0.5}),
        mode="steps",
        total_steps=total_steps,
        seed=seed,
        curriculum_type="B1b_balanced_episodic",
        episodic=EpisodicSampler(
            contexts={
                "home": {"l1": 0.7, "l2": 0.3},
                "social": {"l1": 0.3, "l2": 0.7},
            },
            context_weights={"home": 0.5, "social": 0.5},
            block_size=64,
            seed=seed,
        ),
        noise_std=0.02,
    )


# ---------------------------------------------------------------------------
# Bilingual presets -- B2 Simultaneous
# ---------------------------------------------------------------------------

def _B2_schedule(*, midpoint: float = 0.5, steepness: float = 20.0) -> SigmoidRamp:
    """Base B2 schedule: L1-only -> 50:50 via sigmoid."""
    return SigmoidRamp(
        from_weights={"l1": 1.0, "l2": 0.0},
        to_weights={"l1": 0.5, "l2": 0.5},
        midpoint=midpoint,
        steepness=steepness,
    )


def B2_simultaneous(*, total_steps: int, seed: int = 0) -> DataMixConfig:
    """B2 - Simultaneous: 100% L1 until 50%, then strict sigmoid to 50:50."""
    return DataMixConfig(
        schedule=_B2_schedule(),
        mode="steps",
        total_steps=total_steps,
        seed=seed,
        curriculum_type="B2_simultaneous",
        phase_thresholds={"l2_intro": 0.5},
        noise_std=0.01,
    )


def B2_later(*, total_steps: int, seed: int = 0) -> DataMixConfig:
    """B2 ablation: later L2 introduction (midpoint at 65%)."""
    return DataMixConfig(
        schedule=_B2_schedule(midpoint=0.65),
        mode="steps",
        total_steps=total_steps,
        seed=seed,
        curriculum_type="B2_later",
        phase_thresholds={"l2_intro": 0.65},
        noise_std=0.01,
    )


def B2_smooth(*, total_steps: int, seed: int = 0) -> DataMixConfig:
    """B2 ablation: gradual sigmoid (steepness=8) vs strict transition."""
    return DataMixConfig(
        schedule=_B2_schedule(steepness=8.0),
        mode="steps",
        total_steps=total_steps,
        seed=seed,
        curriculum_type="B2_smooth",
        phase_thresholds={"l2_intro": 0.5},
        noise_std=0.01,
    )


def B2_burst(*, total_steps: int, seed: int = 0) -> DataMixConfig:
    """B2 ablation: strict sigmoid + L2 bursts."""
    return DataMixConfig(
        schedule=BurstSchedule(
            base=_B2_schedule(),
            lang="l2",
            amplitude=0.15,
            period=0.08,
            duty_cycle=0.35,
            onset=0.35,
        ),
        mode="steps",
        total_steps=total_steps,
        seed=seed,
        curriculum_type="B2_burst",
        phase_thresholds={"l2_intro": 0.5},
        noise_std=0.01,
    )


def B2_ewc(*, total_steps: int, seed: int = 0) -> DataMixConfig:
    """B2 ablation: base B2 schedule + EWC (see CONDITION_DEFAULTS)."""
    return DataMixConfig(
        schedule=_B2_schedule(),
        mode="steps",
        total_steps=total_steps,
        seed=seed,
        curriculum_type="B2_ewc",
        phase_thresholds={"l2_intro": 0.5},
        noise_std=0.01,
    )


def B2_lr(*, total_steps: int, seed: int = 0) -> DataMixConfig:
    """B2 ablation: base B2 schedule + LR plasticity boost at transition."""
    return DataMixConfig(
        schedule=_B2_schedule(),
        mode="steps",
        total_steps=total_steps,
        seed=seed,
        curriculum_type="B2_lr",
        phase_thresholds={"l2_intro": 0.5},
        noise_std=0.01,
    )


# ---------------------------------------------------------------------------
# Bilingual presets -- B3 Sequential
# ---------------------------------------------------------------------------

def B3_sequential(*, total_steps: int, seed: int = 0) -> DataMixConfig:
    """B3 - Sequential (Part-Time): 100% L1 until 50%, then 33:67 L1:L2."""
    return DataMixConfig(
        schedule=SigmoidRamp(
            from_weights={"l1": 1.0, "l2": 0.0},
            to_weights={"l1": 0.33, "l2": 0.67},
            midpoint=0.5,
            steepness=20.0,
        ),
        mode="steps",
        total_steps=total_steps,
        seed=seed,
        curriculum_type="B3_sequential",
        phase_thresholds={"l2_intro": 0.5},
        noise_std=0.01,
    )


def B3_full(*, total_steps: int, seed: int = 0) -> DataMixConfig:
    """B3 variant: Full immersion -- 100% L1 -> 100% L2."""
    return DataMixConfig(
        schedule=SigmoidRamp(
            from_weights={"l1": 1.0, "l2": 0.0},
            to_weights={"l1": 0.0, "l2": 1.0},
            midpoint=0.5,
            steepness=20.0,
        ),
        mode="steps",
        total_steps=total_steps,
        seed=seed,
        curriculum_type="B3_full",
        phase_thresholds={"l2_intro": 0.5},
        noise_std=0.01,
    )


def B3_25_75(*, total_steps: int, seed: int = 0) -> DataMixConfig:
    """B3 variant: 100% L1 -> 25:75 L1:L2."""
    return DataMixConfig(
        schedule=SigmoidRamp(
            from_weights={"l1": 1.0, "l2": 0.0},
            to_weights={"l1": 0.25, "l2": 0.75},
            midpoint=0.5,
            steepness=20.0,
        ),
        mode="steps",
        total_steps=total_steps,
        seed=seed,
        curriculum_type="B3_25_75",
        phase_thresholds={"l2_intro": 0.5},
        noise_std=0.01,
    )


# ---------------------------------------------------------------------------
# Bilingual presets -- B4 Classroom
# ---------------------------------------------------------------------------

def _B4_base_schedule() -> BurstSchedule:
    """B4 schedule: sigmoid to 80:20 + L2 episodic bursts."""
    base = SigmoidRamp(
        from_weights={"l1": 1.0, "l2": 0.0},
        to_weights={"l1": 0.8, "l2": 0.2},
        midpoint=0.5,
        steepness=15.0,
    )
    return BurstSchedule(
        base=base,
        lang="l2",
        amplitude=0.10,
        period=0.15,
        duty_cycle=0.2,
        onset=0.4,
    )


def _B4_episodic(seed: int = 0) -> EpisodicSampler:
    """B4 episodic: home (L1) + classroom (L2) contexts."""
    return EpisodicSampler(
        contexts={
            "home": {"l1": 1.0, "l2": 0.0},
            "classroom": {"l1": 0.3, "l2": 0.7},
        },
        context_weights={"home": 0.8, "classroom": 0.2},
        block_size=48,
        seed=seed,
    )


def B4_classroom(*, total_steps: int, seed: int = 0) -> DataMixConfig:
    """
    B4 - Classroom: L1=80M, L2=20M, L2 at 50%, episodic bursts, strong EWC.
    Uses separate 80M L1 + 20M L2 pretokenized datasets.
    """
    return DataMixConfig(
        schedule=_B4_base_schedule(),
        mode="steps",
        total_steps=total_steps,
        seed=seed,
        curriculum_type="B4_classroom",
        episodic=_B4_episodic(seed),
        phase_thresholds={"l2_intro": 0.5},
        noise_std=0.01,
    )


def B4a_no_episodic(*, total_steps: int, seed: int = 0) -> DataMixConfig:
    """B4 ablation: without episodic context clustering."""
    return DataMixConfig(
        schedule=_B4_base_schedule(),
        mode="steps",
        total_steps=total_steps,
        seed=seed,
        curriculum_type="B4a_no_episodic",
        phase_thresholds={"l2_intro": 0.5},
        noise_std=0.01,
    )


def B4b_no_ewc(*, total_steps: int, seed: int = 0) -> DataMixConfig:
    """B4 ablation: with episodic but without EWC (see CONDITION_DEFAULTS)."""
    return DataMixConfig(
        schedule=_B4_base_schedule(),
        mode="steps",
        total_steps=total_steps,
        seed=seed,
        curriculum_type="B4b_no_ewc",
        episodic=_B4_episodic(seed),
        phase_thresholds={"l2_intro": 0.5},
        noise_std=0.01,
    )


# ---------------------------------------------------------------------------
# Bilingual presets -- B5 Late
# ---------------------------------------------------------------------------

def _B5_schedule() -> SigmoidRamp:
    """B5 schedule: L1-only -> 50:50, midpoint at 80%."""
    return SigmoidRamp(
        from_weights={"l1": 1.0, "l2": 0.0},
        to_weights={"l1": 0.5, "l2": 0.5},
        midpoint=0.8,
        steepness=20.0,
    )


def B5_late(*, total_steps: int, seed: int = 0) -> DataMixConfig:
    """B5 - Late: L2 introduced at 80% of training, target 50:50."""
    return DataMixConfig(
        schedule=_B5_schedule(),
        mode="steps",
        total_steps=total_steps,
        seed=seed,
        curriculum_type="B5_late",
        phase_thresholds={"l2_onset": 0.8},
        noise_std=0.01,
    )


def B5_ewc(*, total_steps: int, seed: int = 0) -> DataMixConfig:
    """B5 ablation: base B5 schedule + EWC (see CONDITION_DEFAULTS)."""
    return DataMixConfig(
        schedule=_B5_schedule(),
        mode="steps",
        total_steps=total_steps,
        seed=seed,
        curriculum_type="B5_ewc",
        phase_thresholds={"l2_onset": 0.8},
        noise_std=0.01,
    )


# ---------------------------------------------------------------------------
# Bilingual presets -- B6 Heritage
# ---------------------------------------------------------------------------

def B6_heritage(*, total_steps: int, seed: int = 0) -> DataMixConfig:
    """
    B6 - Heritage Speaker (Attrition + Re-exposure).

    ~90% L1 early -> transition to L2 dominance -> L1 reduced to ~25%
    but reintroduced via periodic bursts (modeling re-exposure).
    EWC + MAML recommended, cyclical LR.
    """
    base_transition = SigmoidRamp(
        from_weights={"l1": 0.90, "l2": 0.10},
        to_weights={"l1": 0.25, "l2": 0.75},
        midpoint=0.35,
        steepness=8.0,
    )
    with_l1_bursts = BurstSchedule(
        base=base_transition,
        lang="l1",
        amplitude=0.20,
        period=0.12,
        duty_cycle=0.25,
        onset=0.5,
    )
    schedule = CyclicalSchedule(
        base=with_l1_bursts,
        lang="l1",
        amplitude=0.08,
        frequency=4.0,
    )
    return DataMixConfig(
        schedule=schedule,
        mode="steps",
        total_steps=total_steps,
        seed=seed,
        curriculum_type="B6_heritage",
        phase_thresholds={"l2_shift": 0.3, "l1_reexposure": 0.5},
        noise_std=0.02,
    )


# ---------------------------------------------------------------------------
# Trilingual presets
# ---------------------------------------------------------------------------

def T1_simultaneous_trilingual(
    *, total_steps: int, lang_keys: List[str], seed: int = 0,
) -> DataMixConfig:
    """T1 - Simultaneous Trilingual: ~33/33/33 with context clustering."""
    assert len(lang_keys) == 3, f"T1 requires 3 lang_keys, got {len(lang_keys)}"
    l1, l2, l3 = lang_keys

    w = 1.0 / 3
    return DataMixConfig(
        schedule=ConstantExposure({l1: w, l2: w, l3: w}),
        mode="steps",
        total_steps=total_steps,
        seed=seed,
        curriculum_type="T1_simultaneous_trilingual",
        episodic=EpisodicSampler(
            contexts={
                "home": {l1: 0.6, l2: 0.2, l3: 0.2},
                "school": {l1: 0.2, l2: 0.6, l3: 0.2},
                "community": {l1: 0.2, l2: 0.2, l3: 0.6},
            },
            context_weights={"home": 0.34, "school": 0.33, "community": 0.33},
            block_size=48,
            seed=seed,
        ),
        noise_std=0.02,
    )


def T2_sequential_trilingual(
    *, total_steps: int, lang_keys: List[str], seed: int = 0,
) -> DataMixConfig:
    """T2 - Sequential L1 -> L2 -> L3 via chained sigmoids."""
    assert len(lang_keys) == 3
    l1, l2, l3 = lang_keys

    l2_ramp = SigmoidRamp(
        from_weights={l1: 1.0, l2: 0.0, l3: 0.0},
        to_weights={l1: 0.6, l2: 0.4, l3: 0.0},
        midpoint=0.3,
        steepness=10.0,
    )
    l3_ramp = SigmoidRamp(
        from_weights={l1: 0.6, l2: 0.4, l3: 0.0},
        to_weights={l1: 0.45, l2: 0.30, l3: 0.25},
        midpoint=0.65,
        steepness=10.0,
    )
    schedule = CompositeSchedule([
        (l2_ramp, 0.0, 0.5),
        (l3_ramp, 0.5, 1.0),
    ])

    return DataMixConfig(
        schedule=schedule,
        mode="steps",
        total_steps=total_steps,
        seed=seed,
        curriculum_type="T2_sequential_trilingual",
        phase_thresholds={"l2_intro": 0.25, "l3_intro": 0.6},
        noise_std=0.01,
    )


def T3_heritage_l3(
    *, total_steps: int, lang_keys: List[str], seed: int = 0,
) -> DataMixConfig:
    """T3 - Heritage + L3: L1 -> L2 dominance -> L3, L1 re-exposure."""
    assert len(lang_keys) == 3
    l1, l2, l3 = lang_keys

    base = SigmoidRamp(
        from_weights={l1: 0.85, l2: 0.15, l3: 0.0},
        to_weights={l1: 0.20, l2: 0.55, l3: 0.25},
        midpoint=0.35,
        steepness=8.0,
    )
    with_bursts = BurstSchedule(
        base=base,
        lang=l1,
        amplitude=0.15,
        period=0.1,
        duty_cycle=0.25,
        onset=0.4,
    )
    schedule = CyclicalSchedule(
        base=with_bursts,
        lang=l1,
        amplitude=0.06,
        frequency=3.0,
    )

    return DataMixConfig(
        schedule=schedule,
        mode="steps",
        total_steps=total_steps,
        seed=seed,
        curriculum_type="T3_heritage_l3",
        phase_thresholds={"l2_shift": 0.3, "l3_intro": 0.5, "l1_reexposure": 0.4},
        noise_std=0.015,
    )


def T4_dominant_l2_late_l3(
    *, total_steps: int, lang_keys: List[str], seed: int = 0,
) -> DataMixConfig:
    """T4 - Dominant L2 + Weak L1 + Late L3."""
    assert len(lang_keys) == 3
    l1, l2, l3 = lang_keys

    return DataMixConfig(
        schedule=SigmoidRamp(
            from_weights={l1: 0.5, l2: 0.5, l3: 0.0},
            to_weights={l1: 0.15, l2: 0.60, l3: 0.25},
            midpoint=0.5,
            steepness=8.0,
        ),
        mode="steps",
        total_steps=total_steps,
        seed=seed,
        curriculum_type="T4_dominant_l2_late_l3",
        phase_thresholds={"l3_intro": 0.4},
        noise_std=0.01,
    )


# ---------------------------------------------------------------------------
# Tokenizer-free (R) presets — multilingual conditions paired with a
# byte/codepoint representation and a tokenizer-free architecture
# (H-Net or MambaByte). Reuse the bilingual / trilingual exposure
# schedules so the curriculum is comparable across BPE and byte runs.
# ---------------------------------------------------------------------------

def _R_balanced_schedule(lang_keys: List[str]) -> ExposureSchedule:
    """Equal-weight constant exposure across all provided languages."""
    w = 1.0 / len(lang_keys)
    return ConstantExposure({k: w for k in lang_keys})


def R1_bytes_hnet(
    *, total_steps: int, lang_keys: Optional[List[str]] = None, seed: int = 0,
) -> DataMixConfig:
    """R1 - Bytes + H-Net, balanced multilingual exposure (T1-style)."""
    keys = list(lang_keys) if lang_keys else ["l1"]
    return DataMixConfig(
        schedule=_R_balanced_schedule(keys),
        mode="steps",
        total_steps=total_steps,
        seed=seed,
        curriculum_type="R1_bytes_hnet",
        noise_std=0.01,
    )


def R2_bytes_mambabyte(
    *, total_steps: int, lang_keys: Optional[List[str]] = None, seed: int = 0,
) -> DataMixConfig:
    """R2 - Bytes + MambaByte (control SSM backbone), balanced exposure."""
    keys = list(lang_keys) if lang_keys else ["l1"]
    return DataMixConfig(
        schedule=_R_balanced_schedule(keys),
        mode="steps",
        total_steps=total_steps,
        seed=seed,
        curriculum_type="R2_bytes_mambabyte",
        noise_std=0.01,
    )


def R3_codepoints_hnet(
    *, total_steps: int, lang_keys: Optional[List[str]] = None, seed: int = 0,
) -> DataMixConfig:
    """R3 - Codepoints + H-Net, balanced exposure.

    Same exposure schedule as R1; the representation knob (codepoints instead
    of bytes) is what differs. Set ``tokenizer.type: codepoint`` in YAML.
    """
    keys = list(lang_keys) if lang_keys else ["l1"]
    return DataMixConfig(
        schedule=_R_balanced_schedule(keys),
        mode="steps",
        total_steps=total_steps,
        seed=seed,
        curriculum_type="R3_codepoints_hnet",
        noise_std=0.01,
    )


def R4_bytes_hnet_sequential(
    *, total_steps: int, lang_keys: Optional[List[str]] = None, seed: int = 0,
) -> DataMixConfig:
    """R4 - Bytes + H-Net with sequential T2-style curriculum.

    Tests the curriculum × architecture interaction: same H-Net, same bytes,
    but L1 -> L2 -> L3 staged introduction instead of constant exposure.
    Falls back to bilingual sequential if only 2 lang_keys are provided.
    """
    keys = list(lang_keys) if lang_keys else ["l1", "l2"]

    if len(keys) == 3:
        return T2_sequential_trilingual(
            total_steps=total_steps, lang_keys=keys, seed=seed,
        )
    if len(keys) == 2:
        l1, l2 = keys
        return DataMixConfig(
            schedule=SigmoidRamp(
                from_weights={l1: 1.0, l2: 0.0},
                to_weights={l1: 0.5, l2: 0.5},
                midpoint=0.5,
                steepness=20.0,
            ),
            mode="steps",
            total_steps=total_steps,
            seed=seed,
            curriculum_type="R4_bytes_hnet_sequential",
            phase_thresholds={"l2_intro": 0.5},
            noise_std=0.01,
        )
    # Monolingual fallback: no schedule, just constant.
    return DataMixConfig(
        schedule=_R_balanced_schedule(keys),
        mode="steps",
        total_steps=total_steps,
        seed=seed,
        curriculum_type="R4_bytes_hnet_sequential",
        noise_std=0.01,
    )


# ---------------------------------------------------------------------------
# Condition defaults (recommended EWC / MAML / LR settings)
# ---------------------------------------------------------------------------

CONDITION_DEFAULTS: Dict[str, Dict[str, Any]] = {
    # --- B1 Balanced ---
    "B1": {
        "description": "Balanced (50:50 from step 0)",
        "ewc": {"enabled": False},
        "maml": {"enabled": False},
        "optimization": {
            "lr": 5e-4,
            "lr_scheduler": "linear_with_warmup",
            "lr_warmup_steps": 500,
        },
    },
    # --- B2 Simultaneous ---
    "B2": {
        "description": "Simultaneous (100% L1 -> 50:50 at midpoint)",
        "ewc": {"enabled": False},
        "maml": {"enabled": False},
        "optimization": {
            "lr": 5e-4,
            "lr_scheduler": "linear_with_warmup",
            "lr_warmup_steps": 500,
        },
    },
    "B2_ewc": {
        "description": "Simultaneous + EWC",
        "ewc": {
            "enabled": True,
            "lambda_": 20.0,
            "n_fisher_samples": 10,
            "mu": 0.0,
            "trigger": "phase_change",
        },
        "maml": {"enabled": False},
        "optimization": {
            "lr": 5e-4,
            "lr_scheduler": "linear_with_warmup",
            "lr_warmup_steps": 500,
        },
    },
    "B2_lr": {
        "description": "Simultaneous + LR plasticity boost",
        "ewc": {"enabled": False},
        "maml": {"enabled": False},
        "optimization": {
            "lr": 5e-4,
            "lr_scheduler": "exposure_based",
            "exposure_lr": {
                "warmup_tokens": 5_000_000,
                "total_tokens": 100_000_000,
                "plasticity_boost": 2.5,
                "min_lr_frac": 0.1,
                "decay_shape": "cosine",
            },
        },
    },
    # --- B3 Sequential ---
    "B3": {
        "description": "Sequential (100% L1 -> 33:67 at midpoint)",
        "ewc": {"enabled": False},
        "maml": {"enabled": False},
        "optimization": {
            "lr": 5e-4,
            "lr_scheduler": "linear_with_warmup",
            "lr_warmup_steps": 500,
        },
    },
    # --- B4 Classroom ---
    "B4": {
        "description": "Classroom (80M L1, 20M L2, episodic, EWC)",
        "ewc": {
            "enabled": True,
            "lambda_": 30.0,
            "n_fisher_samples": 15,
            "mu": 0.0,
            "trigger": "phase_change",
        },
        "maml": {"enabled": False},
        "optimization": {
            "lr": 5e-4,
            "lr_scheduler": "exposure_based",
            "exposure_lr": {
                "warmup_tokens": 5_000_000,
                "total_tokens": 100_000_000,
                "plasticity_boost": 2.0,
                "min_lr_frac": 0.1,
                "decay_shape": "cosine",
            },
        },
    },
    "B4b_no_ewc": {
        "description": "Classroom without EWC",
        "ewc": {"enabled": False},
        "maml": {"enabled": False},
        "optimization": {
            "lr": 5e-4,
            "lr_scheduler": "exposure_based",
            "exposure_lr": {
                "warmup_tokens": 5_000_000,
                "total_tokens": 100_000_000,
                "plasticity_boost": 2.0,
                "min_lr_frac": 0.1,
                "decay_shape": "cosine",
            },
        },
    },
    # --- B5 Late ---
    "B5": {
        "description": "Late (L2 at 80% of training)",
        "ewc": {"enabled": False},
        "maml": {"enabled": False},
        "optimization": {
            "lr": 5e-4,
            "lr_scheduler": "exposure_based",
            "exposure_lr": {
                "warmup_tokens": 5_000_000,
                "total_tokens": 100_000_000,
                "plasticity_boost": 3.0,
                "min_lr_frac": 0.1,
                "decay_shape": "cosine",
            },
        },
    },
    "B5_ewc": {
        "description": "Late + EWC",
        "ewc": {
            "enabled": True,
            "lambda_": 40.0,
            "n_fisher_samples": 15,
            "mu": 0.0,
            "trigger": "phase_change",
        },
        "maml": {"enabled": False},
        "optimization": {
            "lr": 5e-4,
            "lr_scheduler": "exposure_based",
            "exposure_lr": {
                "warmup_tokens": 5_000_000,
                "total_tokens": 100_000_000,
                "plasticity_boost": 3.0,
                "min_lr_frac": 0.1,
                "decay_shape": "cosine",
            },
        },
    },
    # --- B6 Heritage ---
    "B6": {
        "description": "Heritage Speaker (attrition + re-exposure)",
        "ewc": {
            "enabled": True,
            "lambda_": 15.0,
            "n_fisher_samples": 10,
            "mu": 0.0,
            "trigger": "phase_change",
        },
        "maml": {
            "enabled": True,
            "hybrid_ratio": 0.4,
            "n_way": 5,
            "k_shot": 1,
            "inner_steps": 3,
            "inner_lr": 0.1,
            "head": {"num_layers": 2, "hidden_dim": 256, "init_method": "xavier"},
        },
        "optimization": {
            "lr": 5e-4,
            "lr_scheduler": "exposure_based",
            "exposure_lr": {
                "warmup_tokens": 5_000_000,
                "total_tokens": 100_000_000,
                "plasticity_boost": 1.5,
                "min_lr_frac": 0.1,
                "decay_shape": "cosine",
            },
        },
    },
    # --- Trilingual ---
    "T1": {
        "description": "Simultaneous Trilingual (~33/33/33)",
        "ewc": {"enabled": False},
        "maml": {
            "enabled": True,
            "hybrid_ratio": 0.3,
            "n_way": 5,
            "k_shot": 1,
            "inner_steps": 3,
            "inner_lr": 0.1,
            "head": {"num_layers": 2, "hidden_dim": 256, "init_method": "xavier"},
        },
        "optimization": {
            "lr": 5e-4,
            "lr_scheduler": "linear_with_warmup",
            "lr_warmup_steps": 500,
        },
    },
    "T2": {
        "description": "Sequential L1 -> L2 -> L3",
        "ewc": {
            "enabled": True,
            "lambda_": 20.0,
            "n_fisher_samples": 10,
            "mu": 0.0,
            "trigger": "phase_change",
        },
        "maml": {"enabled": False},
        "optimization": {
            "lr": 5e-4,
            "lr_scheduler": "exposure_based",
            "exposure_lr": {
                "warmup_tokens": 3_300_000,
                "total_tokens": 100_000_000,
                "plasticity_boost": 2.5,
                "min_lr_frac": 0.1,
                "decay_shape": "cosine",
                "phase_lr_resets": {
                    "l2_intro": 0.8,
                    "l3_intro": 0.6,
                },
            },
        },
    },
    "T3": {
        "description": "Heritage + L3 Learner",
        "ewc": {
            "enabled": True,
            "lambda_": 15.0,
            "n_fisher_samples": 10,
            "mu": 0.0,
            "trigger": "phase_change",
        },
        "maml": {
            "enabled": True,
            "hybrid_ratio": 0.35,
            "n_way": 5,
            "k_shot": 1,
            "inner_steps": 3,
            "inner_lr": 0.1,
            "head": {"num_layers": 2, "hidden_dim": 256, "init_method": "xavier"},
        },
        "optimization": {
            "lr": 5e-4,
            "lr_scheduler": "exposure_based",
            "exposure_lr": {
                "warmup_tokens": 3_300_000,
                "total_tokens": 100_000_000,
                "plasticity_boost": 1.5,
                "min_lr_frac": 0.1,
                "decay_shape": "cosine",
            },
        },
    },
    "T4": {
        "description": "Dominant L2 + Weak L1 + Late L3",
        "ewc": {
            "enabled": True,
            "lambda_": 10.0,
            "n_fisher_samples": 10,
            "mu": 0.0,
            "trigger": "phase_change",
        },
        "maml": {"enabled": False},
        "optimization": {
            "lr": 5e-4,
            "lr_scheduler": "exposure_based",
            "exposure_lr": {
                "warmup_tokens": 3_300_000,
                "total_tokens": 100_000_000,
                "plasticity_boost": 2.0,
                "min_lr_frac": 0.1,
                "decay_shape": "cosine",
            },
        },
    },
    # --- Tokenizer-free (R) ---
    "R1": {
        "description": "Bytes + H-Net, balanced multilingual exposure",
        "ewc": {"enabled": False},
        "maml": {"enabled": False},
        "optimization": {
            "lr": 1e-3,
            "lr_scheduler": "linear_with_warmup",
            "lr_warmup_steps": 500,
            "lr_multiplier": [3.0, 1.0],
        },
        "hnet": {
            "lb_loss_weight": 0.03,
            "initializer_range": 0.02,
            "target_downsample": [4.5],
        },
    },
    "R2": {
        "description": "Bytes + MambaByte, balanced multilingual exposure",
        "ewc": {"enabled": False},
        "maml": {"enabled": False},
        "optimization": {
            "lr": 1e-3,
            "lr_scheduler": "linear_with_warmup",
            "lr_warmup_steps": 500,
        },
    },
    "R3": {
        "description": "Codepoints + H-Net, balanced multilingual exposure",
        "ewc": {"enabled": False},
        "maml": {"enabled": False},
        "optimization": {
            "lr": 5e-4,
            "lr_scheduler": "linear_with_warmup",
            "lr_warmup_steps": 500,
            "lr_multiplier": [3.0, 1.0],
        },
        "hnet": {
            "lb_loss_weight": 0.03,
            "initializer_range": 0.02,
            "target_downsample": [4.5],
        },
    },
    "R4": {
        "description": "Bytes + H-Net with sequential curriculum (T2-style)",
        "ewc": {
            "enabled": True,
            "lambda_": 15.0,
            "n_fisher_samples": 10,
            "mu": 0.0,
            "trigger": "phase_change",
        },
        "maml": {"enabled": False},
        "optimization": {
            "lr": 1e-3,
            "lr_scheduler": "linear_with_warmup",
            "lr_warmup_steps": 500,
            "lr_multiplier": [3.0, 1.0],
        },
        "hnet": {
            "lb_loss_weight": 0.03,
            "initializer_range": 0.02,
            "target_downsample": [4.5],
        },
    },
}


# ---------------------------------------------------------------------------
# Registry and lookup
# ---------------------------------------------------------------------------

BILINGUAL_CONDITION_BUILDERS = {
    "B1": B1_balanced,
    "B1a": B1a_balanced_delayed,
    "B1b": B1b_balanced_episodic,
    "B2": B2_simultaneous,
    "B2_later": B2_later,
    "B2_smooth": B2_smooth,
    "B2_burst": B2_burst,
    "B2_ewc": B2_ewc,
    "B2_lr": B2_lr,
    "B3": B3_sequential,
    "B3_full": B3_full,
    "B3_25_75": B3_25_75,
    "B4": B4_classroom,
    "B4a": B4a_no_episodic,
    "B4b": B4b_no_ewc,
    "B5": B5_late,
    "B5_ewc": B5_ewc,
    "B6": B6_heritage,
}

BILINGUAL_BASE_CONDITIONS = ["B1", "B2", "B3", "B4", "B5", "B6"]

TRILINGUAL_CONDITION_BUILDERS = {
    "T1": T1_simultaneous_trilingual,
    "T2": T2_sequential_trilingual,
    "T3": T3_heritage_l3,
    "T4": T4_dominant_l2_late_l3,
}

# Tokenizer-free conditions: cross-architecture comparisons over a fixed
# multilingual exposure schedule. Builders accept lang_keys (any arity).
TOKFREE_CONDITION_BUILDERS = {
    "R1": R1_bytes_hnet,
    "R2": R2_bytes_mambabyte,
    "R3": R3_codepoints_hnet,
    "R4": R4_bytes_hnet_sequential,
}

ALL_CONDITION_BUILDERS = {
    **BILINGUAL_CONDITION_BUILDERS,
    **TRILINGUAL_CONDITION_BUILDERS,
    **TOKFREE_CONDITION_BUILDERS,
}


def get_condition(
    name: str,
    *,
    total_steps: int,
    lang_keys: Optional[List[str]] = None,
    seed: int = 0,
) -> DataMixConfig:
    """
    Get a condition preset by name (B1-B6 + ablations, T1-T4).

    Accepts string aliases: "balanced" -> B1, "simultaneous" -> B2, etc.

    Parameters
    ----------
    name : str
        Condition code (e.g. "B1", "B2_ewc", "T2") or string alias.
    total_steps : int
        Total training steps.
    lang_keys : list of str, optional
        Required for trilingual conditions (T1-T4).
    seed : int
        Random seed.

    Returns
    -------
    DataMixConfig
    """
    resolved = _resolve_name(name)
    if resolved not in ALL_CONDITION_BUILDERS:
        raise KeyError(
            f"Unknown condition {name!r} (resolved: {resolved!r}). "
            f"Available: {sorted(ALL_CONDITION_BUILDERS.keys())}"
        )

    builder = ALL_CONDITION_BUILDERS[resolved]
    if resolved.startswith("T"):
        if lang_keys is None:
            raise ValueError(f"Trilingual condition {resolved} requires lang_keys")
        return builder(total_steps=total_steps, lang_keys=lang_keys, seed=seed)
    if resolved.startswith("R"):
        # Tokenizer-free R-conditions accept lang_keys (any arity, optional).
        return builder(
            total_steps=total_steps, lang_keys=lang_keys, seed=seed,
        )
    return builder(total_steps=total_steps, seed=seed)


def get_condition_defaults(name: str) -> Dict[str, Any]:
    """
    Get recommended EWC / MAML / LR defaults for a condition.

    For ablation variants without explicit defaults (e.g. B2_later),
    falls back to the base condition (B2).
    """
    resolved = _resolve_name(name)
    if resolved in CONDITION_DEFAULTS:
        return dict(CONDITION_DEFAULTS[resolved])
    # Fall back to base condition
    base = _base_condition(resolved)
    if base in CONDITION_DEFAULTS:
        return dict(CONDITION_DEFAULTS[base])
    raise KeyError(
        f"Unknown condition {name!r} (resolved: {resolved!r}). "
        f"Available: {sorted(CONDITION_DEFAULTS.keys())}"
    )


def list_conditions() -> Dict[str, str]:
    """Return {name: description} for all conditions with explicit defaults."""
    return {k: v["description"] for k, v in CONDITION_DEFAULTS.items()}
