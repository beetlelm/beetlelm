# beetlelm/src/bilingual/data/curriculum.py
"""
Continuous curriculum sampler for cognitively-inspired language exposure.

The sole curriculum implementation for BeetleLM training. Provides:
  1. Sigmoid-interpolated exposure transitions (no abrupt phase changes)
  2. Intra-phase stochasticity (burst noise, oscillations)
  3. Episodic context clustering (home vs school sampling blocks)
  4. Performance-adaptive gating (gate L2/L3 on model competence)

The core abstraction is an ExposureSchedule — a callable that maps
training progress t ∈ [0, 1] → Dict[str, float] of language weights.

Multiple schedule primitives can be composed:
  - ConstantExposure: fixed weights
  - SigmoidRamp: smooth S-curve transition between two weight sets
  - BurstSchedule: periodic bursts of a target language on top of a base
  - CompositeSchedule: sum/blend of multiple schedules

DataMixConfig wraps an ExposureSchedule and provides the public interface
(next_source, step, state_dict, etc.) consumed by the trainer.

YAML-driven construction
------------------------
    curriculum:
      type: "smooth"
      schedule:
        - kind: "sigmoid_ramp"
          from_weights: {l1: 1.0, l2: 0.0}
          to_weights: {l1: 0.5, l2: 0.5}
          midpoint: 0.5
          steepness: 12.0
        - kind: "burst"
          lang: "l2"
          amplitude: 0.15
          period: 0.1
          duty_cycle: 0.3
          onset: 0.3
      episodic:
        enabled: true
        block_size: 64
        contexts:
          home: {l1: 1.0, l2: 0.0}
          school: {l1: 0.2, l2: 0.8}
        context_weights: {home: 0.5, school: 0.5}
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Schedule primitives
# ---------------------------------------------------------------------------

class ExposureSchedule(ABC):
    """Maps training progress t ∈ [0, 1] → language weights."""

    @abstractmethod
    def weights_at(self, t: float) -> Dict[str, float]:
        """Return unnormalised language weights at progress t."""
        ...

    def __call__(self, t: float) -> Dict[str, float]:
        return self.weights_at(t)


class ConstantExposure(ExposureSchedule):
    """Fixed language weights throughout training."""

    def __init__(self, weights: Dict[str, float]):
        self._weights = dict(weights)

    def weights_at(self, t: float) -> Dict[str, float]:
        return dict(self._weights)


class SigmoidRamp(ExposureSchedule):
    """
    Smooth sigmoid transition between two weight configurations.

    w(t) = from_weights + (to_weights - from_weights) * σ(steepness * (t - midpoint))

    Parameters
    ----------
    from_weights : dict
        Starting weight configuration.
    to_weights : dict
        Ending weight configuration.
    midpoint : float
        Progress value at which the transition is 50% complete.
    steepness : float
        Controls the sharpness of the sigmoid. Higher = sharper.
        ~6 gives a gentle S-curve, ~20 approaches a step function.
    """

    def __init__(
        self,
        from_weights: Dict[str, float],
        to_weights: Dict[str, float],
        midpoint: float = 0.5,
        steepness: float = 12.0,
    ):
        self.from_weights = dict(from_weights)
        self.to_weights = dict(to_weights)
        self.midpoint = midpoint
        self.steepness = steepness
        # Ensure both dicts have the same keys
        all_keys = set(from_weights) | set(to_weights)
        for k in all_keys:
            self.from_weights.setdefault(k, 0.0)
            self.to_weights.setdefault(k, 0.0)

    def weights_at(self, t: float) -> Dict[str, float]:
        sigma = 1.0 / (1.0 + math.exp(-self.steepness * (t - self.midpoint)))
        return {
            k: self.from_weights[k] + (self.to_weights[k] - self.from_weights[k]) * sigma
            for k in self.from_weights
        }


class LinearRamp(ExposureSchedule):
    """Linear interpolation between two weight configurations over [onset, offset]."""

    def __init__(
        self,
        from_weights: Dict[str, float],
        to_weights: Dict[str, float],
        onset: float = 0.0,
        offset: float = 1.0,
    ):
        self.from_weights = dict(from_weights)
        self.to_weights = dict(to_weights)
        self.onset = onset
        self.offset = offset
        all_keys = set(from_weights) | set(to_weights)
        for k in all_keys:
            self.from_weights.setdefault(k, 0.0)
            self.to_weights.setdefault(k, 0.0)

    def weights_at(self, t: float) -> Dict[str, float]:
        if t <= self.onset:
            return dict(self.from_weights)
        if t >= self.offset:
            return dict(self.to_weights)
        alpha = (t - self.onset) / (self.offset - self.onset)
        return {
            k: self.from_weights[k] + (self.to_weights[k] - self.from_weights[k]) * alpha
            for k in self.from_weights
        }


class BurstSchedule(ExposureSchedule):
    """
    Periodic bursts of a target language overlaid on a base schedule.

    During burst windows, the target language weight is boosted by `amplitude`.
    Other languages are proportionally reduced to maintain normalisation.

    Parameters
    ----------
    base : ExposureSchedule
        Underlying exposure schedule.
    lang : str
        Language key to boost during bursts.
    amplitude : float
        Additional weight for the target language during bursts.
    period : float
        Period of burst cycle as fraction of total training.
    duty_cycle : float
        Fraction of each period that is "on" (burst active).
    onset : float
        Progress value at which bursts begin.
    """

    def __init__(
        self,
        base: ExposureSchedule,
        lang: str,
        amplitude: float = 0.15,
        period: float = 0.1,
        duty_cycle: float = 0.3,
        onset: float = 0.0,
    ):
        self.base = base
        self.lang = lang
        self.amplitude = amplitude
        self.period = period
        self.duty_cycle = duty_cycle
        self.onset = onset

    def _is_burst(self, t: float) -> bool:
        if t < self.onset:
            return False
        cycle_pos = ((t - self.onset) % self.period) / self.period
        return cycle_pos < self.duty_cycle

    def weights_at(self, t: float) -> Dict[str, float]:
        w = self.base.weights_at(t)
        if not self._is_burst(t):
            return w
        # Boost target language, reduce others proportionally
        w.setdefault(self.lang, 0.0)
        boost = self.amplitude
        current_lang = w[self.lang]
        new_lang = min(current_lang + boost, 1.0)
        actual_boost = new_lang - current_lang
        # Reduce others
        others_total = sum(v for k, v in w.items() if k != self.lang)
        if others_total > 0:
            scale = max(0, (others_total - actual_boost)) / others_total
            for k in w:
                if k != self.lang:
                    w[k] *= scale
        w[self.lang] = new_lang
        return w


class CyclicalSchedule(ExposureSchedule):
    """
    Sinusoidal oscillation around a base schedule.

    Adds cos(2π * freq * t) * amplitude to the target language,
    reducing others proportionally.
    """

    def __init__(
        self,
        base: ExposureSchedule,
        lang: str,
        amplitude: float = 0.1,
        frequency: float = 5.0,
    ):
        self.base = base
        self.lang = lang
        self.amplitude = amplitude
        self.frequency = frequency

    def weights_at(self, t: float) -> Dict[str, float]:
        w = self.base.weights_at(t)
        w.setdefault(self.lang, 0.0)
        osc = self.amplitude * 0.5 * (1 + math.cos(2 * math.pi * self.frequency * t))
        current = w[self.lang]
        new_val = min(current + osc, 1.0)
        delta = new_val - current
        others_total = sum(v for k, v in w.items() if k != self.lang)
        if others_total > 0:
            scale = max(0, (others_total - delta)) / others_total
            for k in w:
                if k != self.lang:
                    w[k] *= scale
        w[self.lang] = new_val
        return w


class CompositeSchedule(ExposureSchedule):
    """
    Sequentially blend multiple schedules over different time windows.

    Each (schedule, from_t, to_t) entry is active in [from_t, to_t).
    Overlap regions are averaged. Gaps default to the nearest schedule.
    """

    def __init__(self, segments: List[Tuple[ExposureSchedule, float, float]]):
        self.segments = sorted(segments, key=lambda s: s[1])

    def weights_at(self, t: float) -> Dict[str, float]:
        active = [(s, f, u) for s, f, u in self.segments if f <= t < u]
        if not active:
            # Use the nearest segment
            if t <= self.segments[0][1]:
                return self.segments[0][0].weights_at(t)
            return self.segments[-1][0].weights_at(t)
        if len(active) == 1:
            return active[0][0].weights_at(t)
        # Average active schedules
        all_weights = [s.weights_at(t) for s, _, _ in active]
        all_keys = set()
        for w in all_weights:
            all_keys.update(w.keys())
        return {
            k: sum(w.get(k, 0.0) for w in all_weights) / len(all_weights)
            for k in all_keys
        }


class MultiSigmoidSchedule(ExposureSchedule):
    """
    Chain multiple sigmoid transitions for staged language introduction.

    Each transition is a SigmoidRamp. The transitions are applied sequentially:
    the output of one becomes the "from" state for the next.
    """

    def __init__(self, transitions: List[SigmoidRamp]):
        self.transitions = transitions

    def weights_at(self, t: float) -> Dict[str, float]:
        # Start from the first transition's from_weights
        if not self.transitions:
            return {}
        # Apply each transition cumulatively
        result = dict(self.transitions[0].from_weights)
        for tr in self.transitions:
            w = tr.weights_at(t)
            result = w  # Each sigmoid maps the full range
        # Actually, for multi-stage we need to blend them
        # Use the last active transition
        for tr in reversed(self.transitions):
            if t >= tr.midpoint - 2.0 / tr.steepness:
                return tr.weights_at(t)
        return self.transitions[0].weights_at(t)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalise_weights(w: Dict[str, float]) -> Dict[str, float]:
    """Normalise weights to sum to 1.0, handling edge cases."""
    total = sum(w.values())
    if total <= 0:
        n = len(w)
        return {k: 1.0 / n for k in w} if n > 0 else {}
    out = {k: v / total for k, v in w.items()}
    # Fix floating-point residual
    first = next(iter(out))
    out[first] += 1.0 - sum(out.values())
    return out


# ---------------------------------------------------------------------------
# Episodic context clustering
# ---------------------------------------------------------------------------

class EpisodicSampler:
    """
    Sample in context-clustered blocks rather than IID tokens.

    Models alternating environments (home, school, playground) where
    each context has a characteristic language distribution.

    Parameters
    ----------
    contexts : dict
        {context_name: {lang: weight}} — language distribution per context.
    context_weights : dict
        {context_name: probability} — how often each context is sampled.
    block_size : int
        Number of consecutive samples drawn from the same context.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        contexts: Dict[str, Dict[str, float]],
        context_weights: Dict[str, float],
        block_size: int = 64,
        seed: int = 0,
    ):
        self.contexts = {
            name: normalise_weights(w) for name, w in contexts.items()
        }
        total = sum(context_weights.values())
        self.context_weights = {k: v / total for k, v in context_weights.items()}
        self.block_size = block_size
        self.rng = random.Random(seed)

        self._current_context: Optional[str] = None
        self._remaining_in_block: int = 0

    def sample_context(self) -> str:
        """Sample a context according to context_weights."""
        r = self.rng.random()
        cumulative = 0.0
        for name, prob in self.context_weights.items():
            cumulative += prob
            if r < cumulative:
                return name
        return list(self.context_weights.keys())[-1]

    def next_weights(self) -> Dict[str, float]:
        """Return language weights for the current position in the episode."""
        if self._remaining_in_block <= 0:
            self._current_context = self.sample_context()
            self._remaining_in_block = self.block_size
        self._remaining_in_block -= 1
        return dict(self.contexts[self._current_context])

    def state_dict(self) -> dict:
        return {
            "current_context": self._current_context,
            "remaining_in_block": self._remaining_in_block,
            "rng_state": self.rng.getstate(),
        }

    def load_state_dict(self, state: dict):
        self._current_context = state["current_context"]
        self._remaining_in_block = state["remaining_in_block"]
        self.rng.setstate(state["rng_state"])


# ---------------------------------------------------------------------------
# DataMixConfig — drop-in replacement for DataMixConfig
# ---------------------------------------------------------------------------

class DataMixConfig:
    """
    Continuous curriculum sampler with smooth transitions, bursts, and episodic structure.

    Drop-in replacement for DataMixConfig. Implements the same public interface:
    next_source(), step(), current_phase_label(), phase_just_changed(),
    state_dict(), load_state_dict().

    The key difference: instead of discrete phases with hard boundaries,
    weights are computed continuously from an ExposureSchedule.

    Phase labels are derived from weight thresholds for backward compatibility
    with EWC/checkpoint triggers.
    """

    def __init__(
        self,
        schedule: ExposureSchedule,
        *,
        mode: str = "steps",
        total_steps: Optional[int] = None,
        total_tokens: Optional[int] = None,
        seed: int = 0,
        curriculum_type: str = "smooth",
        episodic: Optional[EpisodicSampler] = None,
        # Phase-change detection thresholds for backward compat
        phase_thresholds: Optional[Dict[str, float]] = None,
        lang_keys: Optional[List[str]] = None,
        noise_std: float = 0.0,
    ):
        assert mode in {"steps", "tokens"}
        if mode == "steps":
            assert total_steps is not None
        if mode == "tokens":
            assert total_tokens is not None

        self.schedule = schedule
        self.mode = mode
        self.total_steps = total_steps
        self.total_tokens = total_tokens
        self.curriculum_type = curriculum_type
        self.episodic = episodic
        self.phase_thresholds = phase_thresholds or {}
        self.lang_keys = lang_keys
        self.noise_std = noise_std

        # Counters
        self.step_count = 0
        self.token_count = 0
        self._lang_token_counts: Dict[str, int] = {}

        # Phase tracking
        self._last_phase_label: str = self._compute_phase_label(0.0)
        self._phase_changed: bool = False

        self.rng = random.Random(seed)

        # Precompute phases for DataMixConfig compatibility
        self.phases = []  # Empty — smooth curriculum has no discrete phases

    def _progress(self) -> float:
        if self.mode == "steps":
            return min(1.0, self.step_count / max(1, self.total_steps))
        return min(1.0, self.token_count / max(1, self.total_tokens))

    def _compute_phase_label(self, t: float) -> str:
        """Derive a phase label from current weights for backward compat."""
        w = normalise_weights(self.schedule.weights_at(t))

        # Check thresholds
        for label, threshold in self.phase_thresholds.items():
            if t >= threshold:
                return label

        # Auto-derive label from dominant language
        if not w:
            return "empty"
        dominant = max(w, key=w.get)
        dominant_frac = w[dominant]
        if dominant_frac > 0.9:
            return f"{dominant}_dominant"
        if dominant_frac > 0.6:
            return f"{dominant}_primary"
        return "mixed"

    def _get_current_weights(self) -> Dict[str, float]:
        """Get normalised language weights at current progress."""
        t = self._progress()
        w = self.schedule.weights_at(t)

        # Apply episodic modulation if enabled
        if self.episodic is not None:
            ep_w = self.episodic.next_weights()
            # Blend: episodic weights modulate the schedule
            # Use the episodic context to bias the schedule weights
            all_keys = set(w) | set(ep_w)
            blended = {}
            for k in all_keys:
                sched_val = w.get(k, 0.0)
                ep_val = ep_w.get(k, 0.0)
                # Geometric mean of schedule and episodic weights
                blended[k] = math.sqrt(max(0, sched_val) * max(0, ep_val)) if sched_val > 0 and ep_val > 0 else max(sched_val * 0.1, ep_val * 0.1)
            w = blended

        # Add noise
        if self.noise_std > 0:
            for k in w:
                w[k] = max(0, w[k] + self.rng.gauss(0, self.noise_std))

        return normalise_weights(w)

    # -- Public interface (DataMixConfig compatible) -------------------------

    def current_phase_label(self) -> str:
        return self._last_phase_label

    def phase_just_changed(self) -> bool:
        return self._phase_changed

    def next_source(self) -> str:
        """Return the language key to sample next using ratio tracking."""
        weights = self._get_current_weights()
        total = max(1, sum(self._lang_token_counts.values()))

        best_lang = None
        best_deficit = -1.0

        for lang, target in weights.items():
            if target <= 0.0:
                continue
            seen = self._lang_token_counts.get(lang, 0)
            current = seen / total
            deficit = target - current
            if deficit > best_deficit:
                best_deficit = deficit
                best_lang = lang

        if best_lang is None:
            best_lang = next(iter(weights))

        self._lang_token_counts[best_lang] = (
            self._lang_token_counts.get(best_lang, 0) + 1
        )
        return best_lang

    def step(self, *, tokens: Optional[int] = None):
        """Advance curriculum state by one training step."""
        if self.mode == "tokens":
            assert tokens is not None
            self.token_count += tokens
        self.step_count += 1

        new_label = self._compute_phase_label(self._progress())
        self._phase_changed = (new_label != self._last_phase_label)
        self._last_phase_label = new_label

    # -- Legacy bilingual convenience properties ----------------------------

    @property
    def l1_token_count(self) -> int:
        return self._lang_token_counts.get("l1", 0)

    @property
    def l2_token_count(self) -> int:
        return self._lang_token_counts.get("l2", 0)

    # -- Boundary helpers ---------------------------------------------------

    def step_boundaries(self) -> List[int]:
        """Return approximate phase boundaries based on thresholds."""
        if self.total_steps is None:
            return [0]
        boundaries = [0]
        for label, threshold in sorted(self.phase_thresholds.items(), key=lambda x: x[1]):
            boundaries.append(int(threshold * self.total_steps))
        return sorted(set(boundaries))

    def token_boundaries(self) -> List[int]:
        if self.total_tokens is None:
            return [0]
        boundaries = [0]
        for label, threshold in sorted(self.phase_thresholds.items(), key=lambda x: x[1]):
            boundaries.append(int(threshold * self.total_tokens))
        return sorted(set(boundaries))

    # -- Checkpoint support -------------------------------------------------

    def state_dict(self) -> Dict:
        state = {
            "step_count": self.step_count,
            "token_count": self.token_count,
            "lang_token_counts": dict(self._lang_token_counts),
            "last_phase_label": self._last_phase_label,
            "curriculum_type": self.curriculum_type,
        }
        if self.episodic is not None:
            state["episodic"] = self.episodic.state_dict()
        return state

    def load_state_dict(self, state: Dict):
        self.step_count = state.get("step_count", 0)
        self.token_count = state.get("token_count", 0)
        if "lang_token_counts" in state:
            self._lang_token_counts = dict(state["lang_token_counts"])
        else:
            self._lang_token_counts = {
                "l1": state.get("l1_token_count", 0),
                "l2": state.get("l2_token_count", 0),
            }
        self._last_phase_label = state.get("last_phase_label", "initial")
        self._phase_changed = False
        if self.episodic is not None and "episodic" in state:
            self.episodic.load_state_dict(state["episodic"])

    def __repr__(self) -> str:
        return (
            f"DataMixConfig(curriculum={self.curriculum_type!r}, "
            f"mode={self.mode!r}, progress={self._progress():.3f})"
        )


# ---------------------------------------------------------------------------
# YAML-driven schedule builder
# ---------------------------------------------------------------------------

def build_schedule_from_config(spec: dict) -> ExposureSchedule:
    """
    Build an ExposureSchedule from a YAML spec dict.

    Supported kinds:
      - constant: {kind: "constant", weights: {l1: 0.5, l2: 0.5}}
      - sigmoid_ramp: {kind: "sigmoid_ramp", from_weights: ..., to_weights: ..., midpoint: 0.5, steepness: 12}
      - linear_ramp: {kind: "linear_ramp", from_weights: ..., to_weights: ..., onset: 0.0, offset: 1.0}
      - burst: {kind: "burst", base: <schedule_spec>, lang: "l2", amplitude: 0.15, period: 0.1, ...}
      - cyclical: {kind: "cyclical", base: <schedule_spec>, lang: "l1", amplitude: 0.1, frequency: 5}
      - composite: {kind: "composite", segments: [{schedule: <spec>, from: 0.0, to: 0.5}, ...]}
    """
    kind = spec.get("kind", "constant")

    if kind == "constant":
        return ConstantExposure(spec["weights"])

    elif kind == "sigmoid_ramp":
        return SigmoidRamp(
            from_weights=spec["from_weights"],
            to_weights=spec["to_weights"],
            midpoint=spec.get("midpoint", 0.5),
            steepness=spec.get("steepness", 12.0),
        )

    elif kind == "linear_ramp":
        return LinearRamp(
            from_weights=spec["from_weights"],
            to_weights=spec["to_weights"],
            onset=spec.get("onset", 0.0),
            offset=spec.get("offset", 1.0),
        )

    elif kind == "burst":
        base = build_schedule_from_config(spec["base"]) if "base" in spec else ConstantExposure(spec.get("base_weights", {}))
        return BurstSchedule(
            base=base,
            lang=spec["lang"],
            amplitude=spec.get("amplitude", 0.15),
            period=spec.get("period", 0.1),
            duty_cycle=spec.get("duty_cycle", 0.3),
            onset=spec.get("onset", 0.0),
        )

    elif kind == "cyclical":
        base = build_schedule_from_config(spec["base"]) if "base" in spec else ConstantExposure(spec.get("base_weights", {}))
        return CyclicalSchedule(
            base=base,
            lang=spec["lang"],
            amplitude=spec.get("amplitude", 0.1),
            frequency=spec.get("frequency", 5.0),
        )

    elif kind == "composite":
        segments = []
        for seg in spec["segments"]:
            sched = build_schedule_from_config(seg["schedule"])
            segments.append((sched, seg["from"], seg["to"]))
        return CompositeSchedule(segments)

    else:
        raise ValueError(f"Unknown schedule kind: {kind!r}")


def build_smooth_curriculum(
    config: dict,
    *,
    total_steps: Optional[int] = None,
    total_tokens: Optional[int] = None,
    mode: str = "steps",
    seed: int = 0,
) -> DataMixConfig:
    """
    Build a DataMixConfig from a curriculum config dict.

    Expected format:
        curriculum:
          type: "smooth"
          curriculum_type: "B2_sequential"
          schedule: [...]  or {kind: ...}
          episodic:
            enabled: true
            block_size: 64
            contexts: {home: {l1: 1.0}, school: {l2: 1.0}}
            context_weights: {home: 0.6, school: 0.4}
          phase_thresholds: {l2_intro: 0.3, l2_dominant: 0.7}
          noise_std: 0.02
    """
    # Build schedule
    schedule_spec = config.get("schedule", {})
    if isinstance(schedule_spec, list):
        # Multiple schedule components — compose them
        if len(schedule_spec) == 1:
            schedule = build_schedule_from_config(schedule_spec[0])
        else:
            # Layer them: first is base, rest are modifiers (bursts, cyclical)
            schedule = build_schedule_from_config(schedule_spec[0])
            for spec in schedule_spec[1:]:
                spec_copy = dict(spec)
                spec_copy["base"] = None  # Will be set below
                kind = spec_copy.get("kind", "")
                if kind == "burst":
                    schedule = BurstSchedule(
                        base=schedule,
                        lang=spec_copy["lang"],
                        amplitude=spec_copy.get("amplitude", 0.15),
                        period=spec_copy.get("period", 0.1),
                        duty_cycle=spec_copy.get("duty_cycle", 0.3),
                        onset=spec_copy.get("onset", 0.0),
                    )
                elif kind == "cyclical":
                    schedule = CyclicalSchedule(
                        base=schedule,
                        lang=spec_copy["lang"],
                        amplitude=spec_copy.get("amplitude", 0.1),
                        frequency=spec_copy.get("frequency", 5.0),
                    )
    else:
        schedule = build_schedule_from_config(schedule_spec)

    # Episodic sampler
    episodic = None
    ep_cfg = config.get("episodic", {})
    if ep_cfg.get("enabled", False):
        episodic = EpisodicSampler(
            contexts=ep_cfg["contexts"],
            context_weights=ep_cfg.get("context_weights", {}),
            block_size=ep_cfg.get("block_size", 64),
            seed=seed,
        )

    return DataMixConfig(
        schedule=schedule,
        mode=mode,
        total_steps=total_steps,
        total_tokens=total_tokens,
        seed=seed,
        curriculum_type=config.get("curriculum_type", "smooth"),
        episodic=episodic,
        phase_thresholds=config.get("phase_thresholds", {}),
        noise_std=config.get("noise_std", 0.0),
    )
