"""Trainer-side registries.

Each registry is a plain ``dict`` keyed by the string used in YAML configs.
The ``register`` decorator and ``build_model`` helper let new architectures /
data sources / regimes / checkpointers / evaluators slot in without touching
the trainer's hot path.

For models, registry entries are ``{"cls": <class>, "clamp_input_ids": bool}``
dicts. ``clamp_input_ids: True`` enables a defensive ``input_ids.clamp(0,
vocab_size - 1)`` wrapper around ``forward`` (preserving the historical
behaviour the trainer applied to the ``pico`` architecture).
"""

from __future__ import annotations

from typing import Any, Callable, Dict


# ─── Registries ────────────────────────────────────────────────────────────
MODEL_REGISTRY:      Dict[str, Dict[str, Any]] = {}
DATA_REGISTRY:       Dict[str, Any]            = {}
REGIME_REGISTRY:     Dict[str, Any]            = {}
CHECKPOINT_REGISTRY: Dict[str, Any]            = {}
EVAL_REGISTRY:       Dict[str, Any]            = {}


def register(registry, name):
    """Decorator: ``@register(MODEL_REGISTRY, "pico")`` registers the class."""
    def wrapper(cls):
        registry[name] = cls
        return cls
    return wrapper


def register_model(name: str, cls: type, *, clamp_input_ids: bool = False) -> None:
    """Imperative registration with metadata. Used by ``_arch_imports.py``."""
    MODEL_REGISTRY[name] = {"cls": cls, "clamp_input_ids": clamp_input_ids}


def _wrap_with_input_clamp(cls: type) -> type:
    """Return ``cls`` with its ``forward`` patched once to clamp ``input_ids``.

    The wrapper is idempotent: applying it twice to the same class is a no-op.
    Mirrors the ad-hoc patch the trainer used to apply only to ``pico``.
    """
    if getattr(cls, "_input_clamp_wrapped", False):
        return cls

    orig_forward: Callable = cls.forward

    def safe_forward(self, input_ids, *args, **kwargs):
        vocab_size = getattr(
            getattr(self, "config", None), "vocab_size",
            getattr(getattr(self, "model_config", None), "vocab_size", 32000),
        )
        input_ids = input_ids.clamp(0, vocab_size - 1)
        return orig_forward(self, input_ids, *args, **kwargs)

    cls.forward = safe_forward
    cls._input_clamp_wrapped = True
    return cls


def build_model(arch: str, model_config) -> Any:
    """Instantiate a model from the registry.

    Parameters
    ----------
    arch:
        Architecture key (e.g. ``"pico"``, ``"bgpt"``).
    model_config:
        Object passed verbatim to ``Cls(model_config=...)``. Typically a
        ``types.SimpleNamespace`` built from the YAML ``model.params`` block.

    Raises
    ------
    ValueError
        If ``arch`` is not registered. The error lists known architectures so
        the failure mode is obvious in CI logs.
    """
    entry = MODEL_REGISTRY.get(arch)
    if entry is None:
        known = sorted(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown architecture: {arch!r}. Registered architectures: {known}. "
            f"If you added a new model, ensure it is registered in "
            f"src/training/_arch_imports.py."
        )

    cls = entry["cls"]
    if entry.get("clamp_input_ids", False):
        cls = _wrap_with_input_clamp(cls)

    # Pass positionally so we accommodate both historical signatures:
    # ``PicoDecoder.__init__(self, model_config)`` and
    # ``GPT2Decoder.__init__(self, config)``. The registry contract is "the
    # first positional argument is the namespace-style config object".
    model = cls(model_config)
    if not hasattr(model, "model_config"):
        model.model_config = model_config
    return model
