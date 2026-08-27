"""Unit tests for src/config/_repo_naming.py."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beetlelm.config._repo_naming import (
    CURRICULUM_ALIASES,
    DEFAULT_L2_STEP,
    build_model_repo_id,
    build_model_run_name,
)


_HF_NAME_RE = re.compile(r"^[a-z0-9-]+$")


# ── Base conditions (defaults) ──────────────────────────────────────────────

@pytest.mark.parametrize(
    "dataset,condition,l1,l2,expected",
    [
        ("humanscale", "B1", "nld", "eng",
         "beetle-bilingual-balanced-b1-humanscale-nld-eng"),
        ("humanscale", "B2", "nld", "eng",
         "beetle-bilingual-l2-50-simultaneous-b2-humanscale-nld-eng"),
        ("humanscale", "B3", "nld", "eng",
         "beetle-bilingual-l2-50-sequential-33-67-b3-humanscale-nld-eng"),
        ("humanscale", "B4", "nld", "eng",
         "beetle-bilingual-l2-50-classroom-20-b4-humanscale-nld-eng"),
        ("humanscale", "B5", "nld", "eng",
         "beetle-bilingual-l2-80-late-b5-humanscale-nld-eng"),
        ("humanscale", "B6", "nld", "eng",
         "beetle-bilingual-heritage-b6-humanscale-nld-eng"),
        ("fineweb", "B1", "nld", "eng",
         "beetle-bilingual-balanced-b1-fineweb-nld-eng"),
        ("fineweb", "B2", "zho", "eng",
         "beetle-bilingual-l2-50-simultaneous-b2-fineweb-zho-eng"),
    ],
)
def test_base_conditions(dataset, condition, l1, l2, expected):
    assert build_model_run_name(
        dataset=dataset, condition=condition, l1=l1, l2=l2
    ) == expected


# ── Monolingual ─────────────────────────────────────────────────────────────

def test_monolingual():
    assert build_model_run_name(
        dataset="humanscale", condition="monolingual", l1="nld", l2=None,
    ) == "beetle-monolingual-humanscale-nld"


def test_monolingual_fineweb():
    assert build_model_run_name(
        dataset="fineweb", condition="monolingual", l1="zho", l2=None,
    ) == "beetle-monolingual-fineweb-zho"


# ── B3 ratios ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "variant,ratio",
    [("B3", "33-67"), ("B3_25_75", "25-75"), ("B3_full", "0-100")],
)
def test_b3_ratios(variant, ratio):
    name = build_model_run_name(
        dataset="humanscale", condition=variant, l1="nld", l2="eng",
    )
    assert f"sequential-{ratio}-b3" in name


def test_b3_explicit_ratio_overrides_variant():
    name = build_model_run_name(
        dataset="humanscale", condition="B3", l1="nld", l2="eng",
        b3_ratio="25-75",
    )
    assert name == "beetle-bilingual-l2-50-sequential-25-75-b3-humanscale-nld-eng"


# ── B4 proportion ──────────────────────────────────────────────────────────

def test_b4_default_proportion():
    name = build_model_run_name(
        dataset="humanscale", condition="B4", l1="nld", l2="eng",
    )
    assert "classroom-20-b4" in name


def test_b4_custom_proportion():
    name = build_model_run_name(
        dataset="humanscale", condition="B4", l1="nld", l2="eng",
        b4_proportion_pct=30,
    )
    assert "classroom-30-b4" in name


# ── B5 step override ───────────────────────────────────────────────────────

def test_b5_default_step():
    name = build_model_run_name(
        dataset="humanscale", condition="B5", l1="nld", l2="eng",
    )
    assert "l2-80-late" in name


def test_b5_custom_step():
    name = build_model_run_name(
        dataset="humanscale", condition="B5", l1="nld", l2="eng",
        l2_step_pct=65,
    )
    assert "l2-65-late" in name


def test_b2_later_resolves_to_65():
    name = build_model_run_name(
        dataset="humanscale", condition="B2_later", l1="nld", l2="eng",
    )
    assert name == "beetle-bilingual-l2-65-simultaneous-b2-humanscale-nld-eng"


# ── Feature suffixes ───────────────────────────────────────────────────────

def test_feature_suffix_ewc_only():
    name = build_model_run_name(
        dataset="humanscale", condition="B2", l1="nld", l2="eng",
        ewc_lambda=10,
    )
    assert name.endswith("-ewc-l10")


def test_feature_suffix_maml_only():
    name = build_model_run_name(
        dataset="humanscale", condition="B2", l1="nld", l2="eng",
        maml_ratio=0.2,
    )
    assert name.endswith("-maml-r20")


def test_feature_suffix_all_four():
    name = build_model_run_name(
        dataset="humanscale", condition="B2", l1="nld", l2="eng",
        ewc_lambda=10, maml_ratio=0.4, lr_boost=2.0, lamol_lambda=0.5,
    )
    assert name.endswith(
        "-ewc-l10-maml-r40-lr-b2-lamol-l50"
    )


def test_features_skipped_when_default():
    # lr_boost=1.0 is default (no boost), should not appear
    name = build_model_run_name(
        dataset="humanscale", condition="B2", l1="nld", l2="eng",
        lr_boost=1.0, ewc_lambda=0, maml_ratio=0, lamol_lambda=0,
    )
    assert "lr-b" not in name
    assert "ewc-l" not in name
    assert "maml-r" not in name
    assert "lamol-l" not in name


# ── Vocab sharing ──────────────────────────────────────────────────────────

def test_shared_is_default():
    shared = build_model_run_name(
        dataset="humanscale", condition="B1", l1="nld", l2="eng",
        vocab_sharing="shared",
    )
    assert "perlang" not in shared


def test_perlang_suffix():
    name = build_model_run_name(
        dataset="humanscale", condition="B1", l1="nld", l2="eng",
        vocab_sharing="per_language",
    )
    assert name.endswith("-perlang")


# ── Ablation flags ─────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "condition,flag",
    [
        ("B1a", "delayed"),
        ("B1b", "episodic"),
        ("B2_smooth", "smoothness8"),
        ("B2_burst", "burst"),
        ("B4a", "no-episodic"),
    ],
)
def test_variant_flags(condition, flag):
    name = build_model_run_name(
        dataset="humanscale", condition=condition, l1="nld", l2="eng",
    )
    assert name.endswith(flag) or f"-{flag}" in name


# ── HF safety ──────────────────────────────────────────────────────────────

def test_hf_safety_regex():
    name = build_model_run_name(
        dataset="humanscale", condition="B2", l1="nld", l2="eng",
        ewc_lambda=10, maml_ratio=0.4, lr_boost=2.0, lamol_lambda=0.5,
    )
    assert _HF_NAME_RE.match(name), f"name not HF-safe: {name}"


def test_realistic_2_feature_base_under_96():
    # Base B3/B4 with two features fit comfortably under HF's 96-char limit.
    a = build_model_run_name(
        dataset="humanscale", condition="B3", l1="nld", l2="eng",
        ewc_lambda=30, maml_ratio=0.4,
    )
    b = build_model_run_name(
        dataset="humanscale", condition="B4", l1="nld", l2="eng",
        ewc_lambda=30, lr_boost=2.0,
    )
    assert len(a) <= 96, f"length {len(a)}: {a}"
    assert len(b) <= 96, f"length {len(b)}: {b}"


def test_long_3_feature_sweep_warns_but_does_not_error():
    # User chose verbose labels; some 3-feature sweep combinations on B3 exceed
    # HF's 96-char soft limit but must still generate without raising.
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        name = build_model_run_name(
            dataset="humanscale", condition="B3_25_75", l1="nld", l2="eng",
            ewc_lambda=30, maml_ratio=0.4, lr_boost=3.0,
        )
    assert len(name) <= 120, f"exceeds hard cap: {len(name)}: {name}"


def test_pair_symmetry():
    a = build_model_run_name(
        dataset="humanscale", condition="B2", l1="nld", l2="eng",
    )
    b = build_model_run_name(
        dataset="humanscale", condition="B2", l1="eng", l2="nld",
    )
    assert a != b
    assert a.endswith("-nld-eng")
    assert b.endswith("-eng-nld")


def test_unknown_condition_raises():
    with pytest.raises(ValueError):
        build_model_run_name(
            dataset="humanscale", condition="B7", l1="nld", l2="eng",
        )


# ── build_model_repo_id wrapper ────────────────────────────────────────────

def test_repo_id_prepends_org():
    repo_id = build_model_repo_id(
        "Beetle-HumanScale",
        dataset="humanscale", condition="B1", l1="nld", l2="eng",
    )
    assert repo_id == "Beetle-HumanScale/beetle-bilingual-balanced-b1-humanscale-nld-eng"


# ── Alias table sanity ─────────────────────────────────────────────────────

def test_alias_table_complete():
    assert set(CURRICULUM_ALIASES.keys()) == {"B1", "B2", "B3", "B4", "B5", "B6"}


def test_default_l2_step_table():
    assert DEFAULT_L2_STEP == {"B2": 50, "B3": 50, "B4": 50, "B5": 80}
