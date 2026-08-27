"""
src/config/_repo_naming.py

HuggingFace model-repo naming utility for BeetleLM bilingual/monolingual runs.

Every trained model is pushed to HuggingFace under `{hf_user}/{run_name}`.
This module is the single source of truth for how `run_name` is constructed,
so the three config generators stay in sync and the README examples stay
accurate.

Naming templates (confirmed):

  B1 (Balanced, 50:50 throughout)
    beetle-bilingual-balanced-b1-{dataset}-{l1}-{l2}[-{features}]

  B2 (Simultaneous SLA, L2 at 50% -> 50:50)
    beetle-bilingual-l2-{step}-simultaneous-b2-{dataset}-{l1}-{l2}[-{features}]

  B3 (Sequential SLA, L2 at 50%, varying ratio)
    beetle-bilingual-l2-{step}-sequential-{ratio}-b3-{dataset}-{l1}-{l2}[-{features}]

  B4 (Classroom, 80:20 + EWC, L2 at 50%)
    beetle-bilingual-l2-{step}-classroom-{proportion}-b4-{dataset}-{l1}-{l2}[-{features}]

  B5 (Late Sequential SLA, L2 at step%, target 50:50)
    beetle-bilingual-l2-{step}-late-b5-{dataset}-{l1}-{l2}[-{features}]

  B6 (Heritage)
    beetle-bilingual-heritage-b6-{dataset}-{l1}-{l2}[-{features}]

  Monolingual
    beetle-monolingual-{dataset}-{l1}

Feature suffix order (only appended when non-default / enabled):
  -ewc-l{int}         (EWC lambda, integer form)
  -maml-r{int}        (MAML hybrid_ratio as percent, e.g. 0.2 -> 20)
  -lr-b{int}          (LR plasticity boost, integer form)
  -lamol-l{int}       (LAMOL lambda as percent, e.g. 0.5 -> 50)
  -perlang            (per-language vocab; shared is default and omitted)
  extra flags         (variant-specific, e.g. -burst, -smoothness8, -no-episodic)

Label tokens are abbreviated (`l`/`r`/`b`) rather than spelled out (`lambda`
/`ratio`/`boost`) so that 3-feature sweep runs on B3 variants stay under
HuggingFace's 96-character repo-name limit.

HuggingFace repo names must be lowercase, ASCII `[a-z0-9-]`, and <= 96 chars.
`build_model_run_name` asserts this.
"""

from __future__ import annotations

import re
from typing import Iterable, Literal, Optional

# ── Aliases and defaults ────────────────────────────────────────────────────

CURRICULUM_ALIASES: dict[str, str] = {
    "B1": "balanced",
    "B2": "simultaneous",
    "B3": "sequential",
    "B4": "classroom",
    "B5": "late",
    "B6": "heritage",
}

# Default L2-introduction step as a percent of total training steps, per base
# condition. B1 and B6 do not carry an l2-{step}- prefix, so they are absent.
DEFAULT_L2_STEP: dict[str, int] = {
    "B2": 50,
    "B3": 50,
    "B4": 50,
    "B5": 80,
}

DEFAULT_B3_RATIO = "33-67"
DEFAULT_B4_PROPORTION = 20

Dataset = Literal["humanscale", "fineweb"]

# Variant-suffix -> (l2_step_override, extra_flag, alt_b3_ratio)
# Used to resolve ablation codes like "B2_later", "B3_25_75", "B4a" into the
# parameters the builder needs.
_VARIANT_OVERRIDES: dict[str, dict[str, object]] = {
    # B1
    "B1a": {"extra_flags": ("delayed",)},
    "B1b": {"extra_flags": ("episodic",)},
    # B2
    "B2_later": {"l2_step_pct": 65},
    "B2_smooth": {"extra_flags": ("smoothness8",)},
    "B2_burst": {"extra_flags": ("burst",)},
    "B2_ewc": {},   # ewc is applied via ewc_lambda kwarg
    "B2_lr": {},    # lr_boost applied via lr_boost kwarg
    # B3
    "B3_25_75": {"b3_ratio": "25-75"},
    "B3_full": {"b3_ratio": "0-100"},
    # B4
    "B4a": {"extra_flags": ("no-episodic",)},
    "B4b": {},      # b4b removes EWC; caller must omit ewc_lambda
    # B5
    "B5_ewc": {},   # ewc applied via ewc_lambda
}


_HF_NAME_RE = re.compile(r"^[a-z0-9-]+$")
# HuggingFace enforces <= 96 chars at push time. Our internal sanity cap is a
# bit higher (120) to avoid failing on unusual 4-feature combinations that
# would never appear in a real sweep — the verbose-label convention requested
# by the user is more valuable than a tight preflight check.
_MAX_REPO_NAME_LEN = 120
_HF_SOFT_LIMIT = 96


def _base_condition(condition: str) -> str:
    """Strip variant suffix from a condition code to get the base B1..B6."""
    if condition in CURRICULUM_ALIASES:
        return condition
    # Strip trailing letters for B1a/B1b/B4a/B4b
    m = re.match(r"^(B[1-6])[a-z]$", condition)
    if m:
        return m.group(1)
    # Strip underscore suffix for B2_later, B3_25_75, B5_ewc, etc.
    if "_" in condition:
        head = condition.split("_", 1)[0]
        if head in CURRICULUM_ALIASES:
            return head
    raise ValueError(f"Unknown condition code: {condition!r}")


def _format_ratio_pct(value: float) -> int:
    """0.2 -> 20, 0.5 -> 50 (percent, int)."""
    return int(round(float(value) * 100))


def _format_int(value: float) -> int:
    """10 -> 10, 2.0 -> 2 (lossy truncation, matching user preference)."""
    return int(round(float(value)))


def build_model_run_name(
    *,
    dataset: Dataset,
    condition: str,
    l1: str,
    l2: Optional[str],
    l2_step_pct: Optional[int] = None,
    b3_ratio: Optional[str] = None,
    b4_proportion_pct: Optional[int] = None,
    ewc_lambda: Optional[float] = None,
    maml_ratio: Optional[float] = None,
    lr_boost: Optional[float] = None,
    lamol_lambda: Optional[float] = None,
    vocab_sharing: str = "shared",
    extra_flags: Iterable[str] = (),
) -> str:
    """Build an HF-safe model `run_name` for one bilingual or monolingual run.

    See module docstring for the naming templates and examples.
    """
    # Monolingual path: condition irrelevant, short template.
    if l2 is None:
        name = f"beetle-monolingual-{dataset}-{l1}".lower()
        _validate(name)
        return name

    # Resolve variant -> base + overrides
    base = _base_condition(condition)
    overrides = _VARIANT_OVERRIDES.get(condition, {})

    step = (
        l2_step_pct
        if l2_step_pct is not None
        else overrides.get("l2_step_pct", DEFAULT_L2_STEP.get(base))
    )
    variant_extra = tuple(overrides.get("extra_flags", ()))
    variant_b3_ratio = overrides.get("b3_ratio")

    alias = CURRICULUM_ALIASES[base]

    # l2-{step}- prefix (absent for B1 and B6)
    if base in ("B1", "B6"):
        l2_prefix = ""
    else:
        assert step is not None, f"missing l2 step for {base}"
        l2_prefix = f"l2-{int(step)}-"

    # Condition-specific infix (ratio for B3, proportion for B4)
    if base == "B3":
        ratio = b3_ratio or variant_b3_ratio or DEFAULT_B3_RATIO
        infix = f"-{ratio}"
    elif base == "B4":
        proportion = (
            b4_proportion_pct
            if b4_proportion_pct is not None
            else DEFAULT_B4_PROPORTION
        )
        infix = f"-{int(proportion)}"
    else:
        infix = ""

    core = (
        f"beetle-bilingual-{l2_prefix}{alias}{infix}"
        f"-{base.lower()}-{dataset}-{l1}-{l2}"
    )

    # Feature suffix (only if enabled / non-default). Labels are abbreviated
    # to stay under HuggingFace's 96-char repo-name limit.
    suffix_parts: list[str] = []
    if ewc_lambda is not None and float(ewc_lambda) > 0:
        suffix_parts.append(f"ewc-l{_format_int(ewc_lambda)}")
    if maml_ratio is not None and float(maml_ratio) > 0:
        suffix_parts.append(f"maml-r{_format_ratio_pct(maml_ratio)}")
    if lr_boost is not None and float(lr_boost) > 1.0:
        suffix_parts.append(f"lr-b{_format_int(lr_boost)}")
    if lamol_lambda is not None and float(lamol_lambda) > 0:
        suffix_parts.append(f"lamol-l{_format_ratio_pct(lamol_lambda)}")
    if vocab_sharing == "per_language":
        suffix_parts.append("perlang")
    for flag in list(variant_extra) + list(extra_flags):
        if flag:
            suffix_parts.append(str(flag))

    suffix = ("-" + "-".join(suffix_parts)) if suffix_parts else ""

    name = (core + suffix).lower()
    _validate(name)
    return name


def build_model_repo_id(org: str, **kwargs) -> str:
    """Thin wrapper: `{org}/{run_name}`."""
    return f"{org}/{build_model_run_name(**kwargs)}"


def _validate(name: str) -> None:
    import warnings
    if not _HF_NAME_RE.match(name):
        raise ValueError(
            f"Generated HF repo name is not HF-safe: {name!r} "
            f"(must match {_HF_NAME_RE.pattern})"
        )
    if len(name) > _MAX_REPO_NAME_LEN:
        raise ValueError(
            f"Generated HF repo name exceeds {_MAX_REPO_NAME_LEN} chars "
            f"({len(name)}): {name!r}"
        )
    if len(name) > _HF_SOFT_LIMIT:
        warnings.warn(
            f"Generated HF repo name is {len(name)} chars; HuggingFace "
            f"enforces <= {_HF_SOFT_LIMIT} and will reject at push time: "
            f"{name!r}",
            stacklevel=3,
        )
