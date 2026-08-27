"""Unit tests for the tokenizer-free R-condition presets.

Verifies that the new R1-R4 conditions integrate cleanly with the existing
``get_condition`` / ``get_condition_defaults`` / curriculum scheduling
machinery — schedule weights sum to 1 at every sampled progress point,
defaults expose ``optimization``, the registry lists them, etc.
"""

from __future__ import annotations

import pytest

from beetlelm.bilingual.data.condition_presets import (
    ALL_CONDITION_BUILDERS,
    CONDITION_DEFAULTS,
    TOKFREE_CONDITION_BUILDERS,
    get_condition,
    get_condition_defaults,
)


R_KEYS = ["R1", "R2", "R3", "R4"]


def test_tokfree_builders_registered():
    assert set(TOKFREE_CONDITION_BUILDERS) == set(R_KEYS)
    for key in R_KEYS:
        assert key in ALL_CONDITION_BUILDERS


@pytest.mark.parametrize("name", R_KEYS)
def test_get_condition_returns_valid_data_mix(name):
    mix = get_condition(name, total_steps=1000, lang_keys=["eng", "nld", "deu"])
    assert mix is not None
    # R1/R2/R3 carry their own curriculum_type; R4 reuses T2's sequential
    # builder for the 3-lang case (and gets curriculum_type 'T2_...').
    expected_prefixes = (name + "_", "T2_") if name == "R4" else (name + "_",)
    assert mix.curriculum_type.startswith(expected_prefixes), (
        f"{name}: unexpected curriculum_type {mix.curriculum_type!r}"
    )


@pytest.mark.parametrize("name", R_KEYS)
@pytest.mark.parametrize("progress", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_schedule_weights_normalised(name, progress):
    mix = get_condition(name, total_steps=1000, lang_keys=["eng", "nld", "deu"])
    weights = mix.schedule.weights_at(progress)
    total = sum(weights.values())
    assert abs(total - 1.0) < 1e-6, (
        f"{name}@progress={progress}: weights sum to {total}, expected 1.0 (got {weights})"
    )


def test_R4_bilingual_falls_back_to_sigmoid():
    """R4 sequential must still work with only 2 lang_keys."""
    mix = get_condition("R4", total_steps=500, lang_keys=["eng", "nld"])
    w0 = mix.schedule.weights_at(0.0)
    w1 = mix.schedule.weights_at(1.0)
    # At progress 0, L1 dominates; at progress 1, balanced.
    assert w0["eng"] > w0["nld"]
    assert abs(w1["eng"] - w1["nld"]) < 0.05


@pytest.mark.parametrize("name", R_KEYS)
def test_condition_defaults_present(name):
    defaults = get_condition_defaults(name)
    assert "description" in defaults
    assert "optimization" in defaults
    assert isinstance(defaults["optimization"]["lr"], float)


def test_hnet_conditions_carry_hnet_block():
    """R1, R3, R4 use H-Net; defaults must expose lb_loss_weight + target_downsample."""
    for name in ["R1", "R3", "R4"]:
        defaults = CONDITION_DEFAULTS[name]
        assert "hnet" in defaults, f"{name} missing 'hnet' defaults block"
        h = defaults["hnet"]
        assert "lb_loss_weight" in h
        assert "target_downsample" in h
        assert isinstance(h["target_downsample"], list)
        assert all(n > 1.0 for n in h["target_downsample"]), (
            "target_downsample N must be > 1 (load_balancing_loss requires this)"
        )


def test_R2_uses_mambabyte_optim_only():
    """R2 is MambaByte (no H-Net) — must not advertise hnet defaults."""
    defaults = CONDITION_DEFAULTS["R2"]
    assert "hnet" not in defaults


def test_unknown_condition_raises():
    with pytest.raises(KeyError):
        get_condition("R99", total_steps=100, lang_keys=["eng"])
