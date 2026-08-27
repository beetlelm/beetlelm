"""
src/continual/generative_replay.py
──────────────────────────────────
LAMOL-style Generative Replay for continual bilingual/multilingual LM training.

Core idea: at the L1→L2 curriculum phase boundary, snapshot the current model
and use it to *generate* pseudo-data that approximates the L1 distribution.
During L2 training, mix this pseudo-data with the real L2 stream so the model
rehearses L1 behaviour without storing any real L1 examples.

    L = E_{x~D_new}[-log p_θ(x)]  +  λ · E_{x~D̃_old}[-log p_θ(x)]

Usage (controlled via YAML):

    generative_replay:
      enabled: true
      lambda_replay: 0.3
      replay_ratio: 0.3
      temperature: 0.9
      top_p: 0.92
      max_gen_len: 128
      prompt_style: "lang_tag"
      buffer_size: 1000
      trigger: phase_change
"""

from __future__ import annotations

import copy
import logging
from collections import deque
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _unwrap(model: nn.Module) -> nn.Module:
    """Strip Lightning/DDP wrappers to reach the raw nn.Module."""
    return getattr(model, "module", model)


def _nucleus_sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 0.92,
) -> torch.Tensor:
    """Sample from logits with temperature scaling and nucleus (top-p) filtering.

    Parameters
    ----------
    logits : (B, V) unnormalised logits for the next token.
    temperature : softmax temperature.
    top_p : cumulative probability threshold for nucleus sampling.

    Returns
    -------
    (B,) sampled token indices.
    """
    logits = logits / max(temperature, 1e-8)

    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Mask tokens beyond the nucleus
    mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
    sorted_logits[mask] = float("-inf")

    # Scatter back to original ordering
    logits = torch.zeros_like(logits).scatter_(-1, sorted_idx, sorted_logits)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────────
# Main module
# ──────────────────────────────────────────────────────────────────────────────

class GenerativeReplayModule:
    """
    LAMOL-style generative replay for catastrophic-forgetting prevention.

    Parameters
    ----------
    lambda_replay : float
        Weight applied to the replay loss term.
    replay_ratio : float
        Fraction of each training batch that should be replay data
        (used for diagnostics; the actual mixing is done per-step).
    temperature : float
        Softmax temperature for pseudo-data generation.
    top_p : float
        Nucleus sampling threshold.
    max_gen_len : int
        Maximum number of tokens to generate per pseudo-sample.
    prompt_style : str
        How to construct generation prompts:
        ``"lang_tag"`` – use ``<lang:xx>`` tokens if available;
        ``"template"`` – short instruction prefix;
        ``"unconditional"`` – generate from BOS/empty context.
    buffer_size : int
        Maximum number of pseudo-samples stored in the CPU replay buffer.
    device : torch.device | None
        Target device for generation.  Falls back to the model's device.
    """

    def __init__(
        self,
        lambda_replay: float = 0.3,
        replay_ratio: float = 0.3,
        temperature: float = 0.9,
        top_p: float = 0.92,
        max_gen_len: int = 128,
        prompt_style: str = "unconditional",
        buffer_size: int = 1000,
        device: Optional[torch.device] = None,
        lang_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
    ):
        if prompt_style not in ("unconditional", "lang_tag", "template"):
            raise ValueError(
                f"[GenReplay] prompt_style={prompt_style!r} is not recognised; "
                "expected 'unconditional', 'lang_tag', or 'template'."
            )
        if prompt_style == "lang_tag" and lang_token_id is None:
            # Fail loudly instead of silently degrading to unconditional — that
            # was the original LAMOL faithfulness bug. Configs that select
            # 'lang_tag' must wire a real tokenizer ID through.
            raise ValueError(
                "[GenReplay] prompt_style='lang_tag' requires lang_token_id "
                "(the tokenizer's <lang:xx> token id for the L1 task). Pass it "
                "via the YAML 'lang_token_id' field or fall back to "
                "prompt_style='unconditional'."
            )
        if prompt_style == "template":
            raise NotImplementedError(
                "[GenReplay] prompt_style='template' is reserved for a future "
                "instruction-tuned LAMOL variant; not implemented in v1. Use "
                "'lang_tag' or 'unconditional'."
            )
        self.lambda_replay = lambda_replay
        self.replay_ratio = replay_ratio
        self.temperature = temperature
        self.top_p = top_p
        self.max_gen_len = max_gen_len
        self.prompt_style = prompt_style
        self.buffer_size = buffer_size
        self._device = device
        self.lang_token_id = (
            int(lang_token_id) if lang_token_id is not None else None
        )
        self.bos_token_id = (
            int(bos_token_id) if bos_token_id is not None else None
        )

        # Frozen model weights (CPU state_dict copy)
        self._frozen_state: Optional[Dict[str, torch.Tensor]] = None

        # Replay buffer: deque of (seq_len,) int64 tensors on CPU
        self._buffer: deque[torch.Tensor] = deque(maxlen=buffer_size)

        self._is_ready = False
        self._replay_loss_ema = 0.0
        self._gen_count = 0

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        """True once a model snapshot has been taken."""
        return self._is_ready

    def snapshot(self, model: nn.Module) -> None:
        """Store a frozen copy of the current model weights.

        Called at the L1→L2 curriculum phase boundary, before any L2 data
        reaches the model.  Only the ``state_dict`` is kept (on CPU) to
        avoid doubling permanent GPU memory.
        """
        raw = _unwrap(model)
        self._frozen_state = {
            k: v.detach().cpu().clone()
            for k, v in raw.state_dict().items()
        }
        self._is_ready = True
        logger.info(
            "[GenReplay] Model snapshot taken — %d tensors stored.",
            len(self._frozen_state),
        )

    @torch.no_grad()
    def generate_replay_batch(
        self,
        model: nn.Module,
        batch_size: int,
        seq_len: int,
        prompts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate a batch of pseudo-data using the frozen model snapshot.

        Parameters
        ----------
        model : nn.Module
            The *current* model architecture (used as a scaffold to load
            the frozen state_dict into).
        batch_size : int
            Number of sequences to generate.
        seq_len : int
            Target sequence length.
        prompts : (B, P) int64 tensor, optional
            Prompt prefix tokens.  If None, generation starts from a
            single BOS/PAD token.

        Returns
        -------
        (batch_size, seq_len) int64 tensor of generated token IDs.
        """
        if not self._is_ready or self._frozen_state is None:
            raise RuntimeError(
                "[GenReplay] Call snapshot() before generate_replay_batch()."
            )

        raw = _unwrap(model)
        dev = self._device or next(raw.parameters()).device

        # Load frozen weights into a temporary copy
        tmp_model = copy.deepcopy(raw)
        tmp_model.load_state_dict(self._frozen_state)
        tmp_model.to(dev)
        tmp_model.eval()

        vocab_size = getattr(
            tmp_model, "model_config", None
        )
        if vocab_size is not None:
            vocab_size = getattr(vocab_size, "vocab_size", 32000)
        else:
            vocab_size = 32000

        # Start from prompts or a single token
        if prompts is not None:
            generated = prompts.to(dev)
        else:
            generated = torch.zeros(
                (batch_size, 1), dtype=torch.long, device=dev
            )

        for _ in range(seq_len - generated.size(1)):
            out = tmp_model(generated)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            next_logits = logits[:, -1, :]  # (B, V)
            next_tokens = _nucleus_sample(
                next_logits, self.temperature, self.top_p
            )
            generated = torch.cat(
                [generated, next_tokens.unsqueeze(1)], dim=1
            )

        del tmp_model
        torch.cuda.empty_cache()

        # Store in buffer
        for seq in generated:
            self._buffer.append(seq.cpu())
        self._gen_count += batch_size

        return generated

    def replay_loss(
        self,
        model: nn.Module,
        replay_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute standard LM cross-entropy on pseudo-data using the
        *current* (training) model.

        Parameters
        ----------
        model : the raw or Fabric-wrapped model.
        replay_ids : (B, T) int64 tensor of generated pseudo-tokens.

        Returns
        -------
        Scalar loss tensor.
        """
        raw = _unwrap(model)
        V = getattr(raw.model_config, "vocab_size", 32000)

        x = replay_ids[:, :-1].clamp(0, V - 1)
        y = replay_ids[:, 1:].clamp(0, V - 1)
        out = raw(x)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        B, T, _ = logits.shape
        return F.cross_entropy(
            logits.reshape(B * T, V), y.contiguous().reshape(-1)
        )

    def compute_replay_loss(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        step: int,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute the replay loss to add to the main training loss.

        If the buffer is empty, generates fresh pseudo-data first.

        Parameters
        ----------
        model : the Fabric-wrapped or raw model.
        input_ids : (B, T) current-task training batch (used to infer
                    batch_size and seq_len).
        step : training step (for diagnostics).

        Returns
        -------
        (replay_loss_term, stats_dict)
            replay_loss_term is already scaled by ``lambda_replay``.
        """
        raw = _unwrap(model)
        dev = next(raw.parameters()).device
        B, T = input_ids.shape

        # Draw from buffer or generate on-the-fly
        replay_batch_size = max(1, int(B * self.replay_ratio))

        if len(self._buffer) >= replay_batch_size:
            # Sample from buffer
            indices = torch.randint(0, len(self._buffer), (replay_batch_size,))
            replay_ids = torch.stack(
                [self._buffer[i] for i in indices]
            ).to(dev)
            # Truncate or pad to match seq_len
            if replay_ids.size(1) > T:
                replay_ids = replay_ids[:, :T]
            elif replay_ids.size(1) < T:
                pad = torch.zeros(
                    replay_batch_size, T - replay_ids.size(1),
                    dtype=torch.long, device=dev,
                )
                replay_ids = torch.cat([replay_ids, pad], dim=1)
        else:
            # Generate fresh pseudo-data — honour prompt_style by constructing
            # a (replay_batch_size, P) prompt prefix per the LAMOL recipe
            # (Sun et al. 2020), instead of the previously hard-coded zeros
            # prefix that ignored prompt_style entirely.
            prompts = self._build_prompts(replay_batch_size, dev)
            replay_ids = self.generate_replay_batch(
                model, replay_batch_size, T, prompts=prompts,
            )

        r_loss = self.replay_loss(model, replay_ids)
        scaled_loss = self.lambda_replay * r_loss

        # EMA tracking for diagnostics
        self._replay_loss_ema = (
            0.95 * self._replay_loss_ema + 0.05 * r_loss.item()
        )

        stats = {
            "gen_replay/loss": r_loss.item(),
            "gen_replay/scaled_loss": scaled_loss.item(),
            "gen_replay/buffer_size": len(self._buffer),
            "gen_replay/gen_count": self._gen_count,
        }
        return scaled_loss, stats

    # ──────────────────────────────────────────────────────────────────────
    # Prompt construction (LAMOL §3.2)
    # ──────────────────────────────────────────────────────────────────────

    def _build_prompts(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Build the per-batch prompt prefix that conditions generation.

        Returns ``None`` for ``prompt_style='unconditional'`` so
        ``generate_replay_batch`` falls through to its zero-prefix branch
        (preserving the legacy unconditional behaviour as one explicit
        option).  For ``'lang_tag'`` returns a ``(B, 1)`` tensor of the
        configured ``lang_token_id`` — this is the LAMOL task-token
        conditioning (Sun et al. 2020), specialised to bilingual L1 rehearsal
        where the task identifier is the L1 language tag.
        """
        if self.prompt_style == "unconditional":
            return None
        if self.prompt_style == "lang_tag":
            assert self.lang_token_id is not None  # guarded in __init__
            return torch.full(
                (batch_size, 1),
                self.lang_token_id,
                dtype=torch.long,
                device=device,
            )
        # 'template' is rejected in __init__; defensive raise here keeps
        # _build_prompts honest if a future code path bypasses the constructor.
        raise NotImplementedError(
            f"[GenReplay] prompt construction for prompt_style={self.prompt_style!r} "
            "is not implemented."
        )

    # ──────────────────────────────────────────────────────────────────────
    # Checkpointing
    # ──────────────────────────────────────────────────────────────────────

    def state_dict(self) -> dict:
        """Serialisable state for checkpointing."""
        return {
            "lambda_replay": self.lambda_replay,
            "replay_ratio": self.replay_ratio,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_gen_len": self.max_gen_len,
            "prompt_style": self.prompt_style,
            "lang_token_id": self.lang_token_id,
            "bos_token_id": self.bos_token_id,
            "buffer_size": self.buffer_size,
            "is_ready": self._is_ready,
            "frozen_state": self._frozen_state,
            "buffer": list(self._buffer),
            "replay_loss_ema": self._replay_loss_ema,
            "gen_count": self._gen_count,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore from a previously saved state_dict."""
        self.lambda_replay = state["lambda_replay"]
        self.replay_ratio = state["replay_ratio"]
        self.temperature = state["temperature"]
        self.top_p = state["top_p"]
        self.max_gen_len = state["max_gen_len"]
        self.prompt_style = state.get("prompt_style", "unconditional")
        self.lang_token_id = state.get("lang_token_id")
        self.bos_token_id = state.get("bos_token_id")
        self.buffer_size = state["buffer_size"]
        self._is_ready = state["is_ready"]
        self._frozen_state = state.get("frozen_state")
        self._buffer = deque(state.get("buffer", []), maxlen=self.buffer_size)
        self._replay_loss_ema = state.get("replay_loss_ema", 0.0)
        self._gen_count = state.get("gen_count", 0)
        logger.info(
            "[GenReplay] State restored (is_ready=%s, buffer=%d).",
            self._is_ready, len(self._buffer),
        )

    # ──────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────────────────────────────────────

    def diagnostics(self) -> Dict[str, float]:
        """Scalar statistics for W&B logging."""
        if not self._is_ready:
            return {"gen_replay/is_ready": 0.0}
        return {
            "gen_replay/is_ready": 1.0,
            "gen_replay/buffer_size": float(len(self._buffer)),
            "gen_replay/replay_loss_ema": self._replay_loss_ema,
            "gen_replay/gen_count": float(self._gen_count),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_generative_replay(
    cfg: Optional[dict],
) -> Optional[GenerativeReplayModule]:
    """
    Build a GenerativeReplayModule from the ``generative_replay:`` YAML block.

    Returns None if the block is absent or ``enabled: false``.

    Example YAML::

        generative_replay:
          enabled: true
          lambda_replay: 0.3
          replay_ratio: 0.3
          temperature: 0.9
          top_p: 0.92
          max_gen_len: 128
          prompt_style: "lang_tag"        # or "unconditional"
          lang_token_id: 5                # tokenizer id for <lang:eng>
          bos_token_id: 1                 # optional, for unconditional prefix
          buffer_size: 1000
          trigger: phase_change

    Notes
    -----
    For the LAMOL-faithful path (``prompt_style='lang_tag'``) you MUST supply
    ``lang_token_id`` — the tokenizer id of the L1 language tag.  Wiring this
    end-to-end avoids the historic silent fallback to unconditional generation
    that was flagged in ``FEATURE_FAITHFULNESS_REPORT.md``.
    """
    if not cfg or not cfg.get("enabled", False):
        return None
    lang_token_id = cfg.get("lang_token_id")
    bos_token_id = cfg.get("bos_token_id")
    return GenerativeReplayModule(
        lambda_replay=float(cfg.get("lambda_replay", 0.3)),
        replay_ratio=float(cfg.get("replay_ratio", 0.3)),
        temperature=float(cfg.get("temperature", 0.9)),
        top_p=float(cfg.get("top_p", 0.92)),
        max_gen_len=int(cfg.get("max_gen_len", 128)),
        prompt_style=str(cfg.get("prompt_style", "unconditional")),
        buffer_size=int(cfg.get("buffer_size", 1000)),
        lang_token_id=int(lang_token_id) if lang_token_id is not None else None,
        bos_token_id=int(bos_token_id) if bos_token_id is not None else None,
    )
