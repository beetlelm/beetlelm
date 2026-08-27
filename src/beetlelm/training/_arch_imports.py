"""Side-effect module: registers the built-in model architectures.

Importing this module populates ``beetlelm.training.registry.MODEL_REGISTRY`` with
the architectures shipped with beetlelm. Future architectures extend this file
with additional ``register_model`` calls.

Imperative (vs decorator-based) registration is used here to avoid forcing
every architecture module to import from ``registry`` — keeps the model
files self-contained and prevents circular imports with the HuggingFace
wrappers (``PicoDecoderHF``, ``GPT2HF``).
"""

from __future__ import annotations

from beetlelm.training.registry import register_model

from beetlelm.pico_decoder.pico_decoder import PicoDecoder
from beetlelm.bgpt.bgpt import BGPT

# ``pico`` historically ran with a defensive input_ids clamp; preserve that.
register_model("pico", PicoDecoder, clamp_input_ids=True)

# ``bgpt`` was introduced without the clamp; keep parity.
register_model("bgpt", BGPT, clamp_input_ids=False)

# ── MoE architecture ─────────────────────────────────────────────────────────
from beetlelm.beetle_moe.beetle_moe import BeetleMoEDecoder  # noqa: E402
register_model("beetle_moe", BeetleMoEDecoder, clamp_input_ids=True)

# ── SSM architecture (lazy import inside BeetleSSMDecoder — safe here) ───────
from beetlelm.beetle_ssm.beetle_ssm import BeetleSSMDecoder  # noqa: E402
register_model("beetle_ssm", BeetleSSMDecoder, clamp_input_ids=True)

# ── Geometry-aware bilingual architecture ─────────────────────────────────────
from beetlelm.geo_bilingual.geo_bilingual import GeoBilingualDecoder  # noqa: E402
register_model("geo_bilingual", GeoBilingualDecoder, clamp_input_ids=True)

# ── Tokenizer-free architectures (byte-level) ────────────────────────────────
# H-Net: hierarchical dynamic chunking. Lazy import inside HNetDecoder; safe
# to register here on macOS where mamba_ssm / flash_attn aren't installed.
from beetlelm.hnet_arch.hnet_arch import HNetDecoder  # noqa: E402
register_model("hnet", HNetDecoder, clamp_input_ids=False)

# MambaByte: byte-level Mamba SSM baseline (subclass of beetle_ssm).
from beetlelm.mambabyte.mambabyte import MambaByteDecoder  # noqa: E402
register_model("mambabyte", MambaByteDecoder, clamp_input_ids=True)
