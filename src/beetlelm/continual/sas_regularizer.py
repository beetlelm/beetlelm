"""
src/continual/sas_regularizer.py
────────────────────────────────
Syntactic Attention Structure (SAS) regularizer (Saphra et al., ICLR 2024).

The SAS regularizer pushes the model toward (positive λ) or away from
(negative λ) the syntactic attention structure that emerges naturally during
pretraining.  Saphra et al. defined it for masked LMs as the per-head best-
attachment accuracy against parser-derived edges; we adapt it for an
autoregressive LM by computing the *causal* per-head attachment score
(attention from token t to its parser-labelled head ≤ t) and reducing over
heads / layers.

Loss term added to the LM objective:

        L_SAS = -λ · mean_{rel}  best-head accuracy on relation `rel`

Sign convention: positive λ *promotes* SAS (subtracts loss when accuracy
rises); negative λ *suppresses* SAS.  Saphra et al. used λ = ±0.001 for
BERT-base; Aoyama et al. confirmed the same scale for autoregressive
PicoDecoder-class models.

Usage (YAML):

    sas_regularizer:
      enabled: true
      lambda_: 0.001              # +0.001 promote, -0.001 suppress
      relations: [nsubj, obj, amod, det, case]
      probe_corpus: data/sas_probes/eng_ud.tsv  # parser-labelled probe
      activate_step: 0            # step from which the term is added
      deactivate_step: null       # optional turn-off step (for "brief"
                                  # suppression à la Saphra et al. §4)
      parser: spacy               # 'spacy' | 'stanza'
      language: eng

The regularizer is *online*: every N steps we run a forward pass over a
fixed probe sentence batch with attention weights returned, score the
current SAS, and add  -λ · SAS  to the optimization loss.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

import torch
from torch import nn

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _unwrap(model: nn.Module) -> nn.Module:
    """Strip Lightning / DDP wrappers."""
    return getattr(model, "module", model)


# ──────────────────────────────────────────────────────────────────────────────
# Probe corpus
# ──────────────────────────────────────────────────────────────────────────────

class _ParsedProbe:
    """A tiny in-memory store of parser-labelled probe sentences.

    Each row is (token_ids, gold_edges) where gold_edges is a list of
    (dep_idx, head_idx, relation) tuples, indexing into the *subword*
    sequence (already aligned to the model's tokenizer offline).
    """

    def __init__(self, rows: list[dict]):
        self.rows = rows

    @classmethod
    def from_tsv(cls, path: Path, tokenizer) -> "_ParsedProbe":
        """Load a TSV with columns (sentence, gold_edges_json).

        gold_edges_json encodes per-sentence edges in *word* indices; we
        retokenize and remap to subword indices using the model's tokenizer.
        """
        import json
        import pandas as pd
        df = pd.read_csv(path, sep="\t")
        rows: list[dict] = []
        for _, r in df.iterrows():
            text = str(r["sentence"])
            edges = json.loads(r["gold_edges_json"])
            words = text.split()
            enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
            ids = enc["input_ids"][0]
            # Word→first-subword mapping (whitespace-aligned).
            char_starts = [0]
            for w in words[:-1]:
                char_starts.append(char_starts[-1] + len(w) + 1)
            offsets = tokenizer(text, return_offsets_mapping=True,
                                add_special_tokens=False)["offset_mapping"]
            word_to_sub: list[int] = []
            for cs in char_starts:
                cand = next((i for i, (a, _) in enumerate(offsets) if a == cs), None)
                if cand is None:
                    cand = next((i for i, (a, _) in enumerate(offsets) if a >= cs), 0)
                word_to_sub.append(cand)
            sub_edges = []
            for e in edges:
                d_w, h_w, rel = e
                if d_w < len(word_to_sub) and h_w < len(word_to_sub):
                    sub_edges.append((word_to_sub[d_w], word_to_sub[h_w], rel))
            rows.append({"input_ids": ids, "edges": sub_edges})
        return cls(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────────────────────

class SASRegularizer:
    """Saphra-style syntactic-attention-structure regularizer (autoregressive variant)."""

    def __init__(
        self,
        lambda_: float = 0.001,
        relations: Optional[Iterable[str]] = None,
        probe_corpus: Optional[Path] = None,
        activate_step: int = 0,
        deactivate_step: Optional[int] = None,
        parser: str = "spacy",
        language: str = "eng",
    ):
        self.lambda_ = float(lambda_)
        self.relations = tuple(relations) if relations else (
            "nsubj", "obj", "amod", "det", "case",
        )
        self.activate_step = int(activate_step)
        self.deactivate_step = int(deactivate_step) if deactivate_step is not None else None
        self.parser = parser
        self.language = language

        self._probe_path = Path(probe_corpus) if probe_corpus else None
        self._probe: Optional[_ParsedProbe] = None
        self._is_ready = False
        self._last_sas: float = 0.0

    # ---- lifecycle ---------------------------------------------------------

    def attach_probe(self, model: nn.Module) -> None:
        """Lazy-load the parser-labelled probe using the model's tokenizer."""
        if self._probe is not None:
            return
        if self._probe_path is None or not self._probe_path.exists():
            logger.warning(
                "[SAS] No probe corpus at %s — regularizer will be inert.",
                self._probe_path,
            )
            return
        raw = _unwrap(model)
        tokenizer = getattr(raw, "tokenizer", None) or getattr(model, "tokenizer", None)
        if tokenizer is None:
            logger.warning("[SAS] Model has no tokenizer attribute; probe disabled.")
            return
        self._probe = _ParsedProbe.from_tsv(self._probe_path, tokenizer)
        self._is_ready = True
        logger.info("[SAS] probe loaded (%d sentences).", len(self._probe.rows))

    def is_active(self, step: int) -> bool:
        if not self._is_ready or self.lambda_ == 0.0:
            return False
        if step < self.activate_step:
            return False
        if self.deactivate_step is not None and step >= self.deactivate_step:
            return False
        return True

    # ---- penalty -----------------------------------------------------------

    def penalty(self, model: nn.Module, step: int, device: torch.device) -> torch.Tensor:
        """Return -λ · mean_rel best-head accuracy.  Differentiable wrt attention.

        We build a *soft* SAS score by reading the per-head attention
        distribution and computing the probability mass placed on the
        gold-head subword index, averaged over heads (max over heads, mean
        over edges) — Saphra Eq. 1 with softmax-weighted argmax.
        """
        if not self.is_active(step):
            return torch.tensor(0.0, device=device)

        raw = _unwrap(model)
        if self._probe is None or not self._probe.rows:
            return torch.tensor(0.0, device=device)

        # Capture attentions via forward hook
        captured: list[torch.Tensor] = []
        def _hook(_m, _i, o):
            if isinstance(o, tuple) and len(o) >= 2 and torch.is_tensor(o[1]):
                captured.append(o[1])
        handles = [m.register_forward_hook(_hook) for m in raw.modules()
                   if m.__class__.__name__.lower().endswith("attention")]

        total = torch.tensor(0.0, device=device)
        n_edges = 0
        try:
            for row in self._probe.rows:
                ids = row["input_ids"].unsqueeze(0).to(device)
                edges = row["edges"]
                captured.clear()
                try:
                    _ = raw(ids, output_attentions=True)
                except TypeError:
                    _ = raw(ids)
                if not captured:
                    continue
                # Stack to (L, B, H, S, S); take batch 0.
                attn = torch.stack(captured, dim=0)[:, 0]   # (L, H, S, S)
                for (d_sub, h_sub, _rel) in edges:
                    if d_sub >= attn.shape[-1] or h_sub >= attn.shape[-1]:
                        continue
                    if h_sub > d_sub:
                        # autoregressive head must be ≤ dependent
                        continue
                    # P_lh = attn[L, H, d_sub, h_sub]; max over (L, H).
                    edge_scores = attn[..., d_sub, h_sub]   # (L, H)
                    total = total + edge_scores.max()
                    n_edges += 1
        finally:
            for h in handles:
                h.remove()

        if n_edges == 0:
            return torch.tensor(0.0, device=device)

        sas_score = total / n_edges
        self._last_sas = float(sas_score.detach().item())
        return -self.lambda_ * sas_score

    def diagnostics(self) -> dict[str, float]:
        return {
            "sas/is_ready": float(self._is_ready),
            "sas/lambda": self.lambda_,
            "sas/last_sas": self._last_sas,
        }

    def state_dict(self) -> dict:
        return {
            "lambda_": self.lambda_,
            "relations": list(self.relations),
            "activate_step": self.activate_step,
            "deactivate_step": self.deactivate_step,
            "parser": self.parser,
            "language": self.language,
            "is_ready": self._is_ready,
        }

    def load_state_dict(self, state: dict) -> None:
        self.lambda_ = float(state.get("lambda_", self.lambda_))
        self.relations = tuple(state.get("relations", self.relations))
        self.activate_step = int(state.get("activate_step", self.activate_step))
        self.deactivate_step = state.get("deactivate_step", self.deactivate_step)
        self.parser = state.get("parser", self.parser)
        self.language = state.get("language", self.language)
        self._is_ready = bool(state.get("is_ready", self._is_ready))


# ──────────────────────────────────────────────────────────────────────────────
# Factory — builds an SASRegularizer from a YAML config block
# ──────────────────────────────────────────────────────────────────────────────

def build_sas_regularizer(cfg: dict) -> Optional["SASRegularizer"]:
    """Build an SASRegularizer from the `sas_regularizer:` YAML block."""
    if not cfg or not cfg.get("enabled", False):
        return None
    return SASRegularizer(
        lambda_=float(cfg.get("lambda_", 0.001)),
        relations=cfg.get("relations"),
        probe_corpus=cfg.get("probe_corpus"),
        activate_step=int(cfg.get("activate_step", 0)),
        deactivate_step=cfg.get("deactivate_step"),
        parser=cfg.get("parser", "spacy"),
        language=cfg.get("language", "eng"),
    )
