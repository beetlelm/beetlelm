"""
Minimal CPU training loop for ``beetlelm.toy``.

Trains the real PicoDecoder architecture (RMSNorm + RoPE + SwiGLU + GQA)
end-to-end on a local text corpus with a tokenizer-free byte vocabulary, so a
run needs zero downloads and finishes on a laptop CPU in minutes. Same model,
same loss, same HuggingFace checkpoint format as the full trainer.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict

# Keep experiment tracking silent and offline unless the user opts in.
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

from beetlelm.bytes import TOKFREE_TOKENIZERS
from beetlelm.pico_decoder.pico_decoder import PicoDecoderHF, PicoDecoderHFConfig
from beetlelm.toy.config import Config, load_config
from beetlelm.toy.data import build_streams, load_text, resolve_path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _lr_at(step: int, base_lr: float, warmup: int, total: int) -> float:
    """Linear warmup then cosine decay to 10% of the base learning rate."""
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    if total <= warmup:
        return base_lr
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def build_tokenizer(cfg: Config):
    tok_type = cfg.tokenizer.type
    if tok_type not in TOKFREE_TOKENIZERS:
        raise ValueError(
            f"Unknown tokenizer type '{tok_type}'. Available: {sorted(TOKFREE_TOKENIZERS)}"
        )
    return TOKFREE_TOKENIZERS[tok_type](max_length=cfg.tokenizer.max_length)


def build_model(cfg: Config, vocab_size: int) -> PicoDecoderHF:
    params = asdict(cfg.model)
    params["vocab_size"] = vocab_size
    params["max_seq_len"] = max(cfg.model.max_seq_len, cfg.training.seq_len)
    return PicoDecoderHF(PicoDecoderHFConfig(**params))


@torch.no_grad()
def evaluate(model, stream, batch_size, generator, device, n_batches: int = 4) -> float:
    model.eval()
    losses = []
    for _ in range(n_batches):
        inp, tgt = stream.batch(batch_size, generator)
        inp, tgt = inp.to(device), tgt.to(device)
        logits = model(input_ids=inp).logits
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), tgt.reshape(-1)
        )
        losses.append(float(loss))
    model.train()
    return sum(losses) / len(losses)


def train(config_path: str, overrides: Dict | None = None) -> Dict:
    """Run a full toy training loop and return a results dict.

    Parameters
    ----------
    config_path : str
        Path to a YAML config (see ``configs/tiny_cpu.yaml``).
    overrides : dict, optional
        Nested dict overriding config values, e.g.
        ``{"model": {"d_model": 256}, "training": {"seed": 7}}``.

    Returns
    -------
    dict
        ``{"final_loss", "final_val_loss", "steps", "n_params",
        "checkpoint_dir", "history"}``.
    """
    cfg = load_config(config_path)
    _apply_overrides(cfg, overrides or {})

    set_seed(cfg.training.seed)
    device = _resolve_device(cfg.training.device)

    tokenizer = build_tokenizer(cfg)
    corpus_path = resolve_path(cfg.data.train_file, cfg.config_dir)
    text = load_text(corpus_path)
    train_stream, val_stream = build_streams(
        text, tokenizer, cfg.training.seq_len, cfg.data.val_fraction
    )

    model = build_model(cfg, vocab_size=len(tokenizer)).to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )
    generator = torch.Generator().manual_seed(cfg.training.seed)

    print(
        f"[beetlelm.toy] model={n_params/1e6:.3f}M params | device={device} | "
        f"corpus={corpus_path.name} ({len(text)} chars) | "
        f"vocab={len(tokenizer)} | steps={cfg.training.max_steps}"
    )

    history = []
    final_loss = float("nan")
    for step in range(cfg.training.max_steps):
        lr = _lr_at(step, cfg.training.lr, cfg.training.warmup_steps, cfg.training.max_steps)
        for g in optimizer.param_groups:
            g["lr"] = lr

        inp, tgt = train_stream.batch(cfg.training.batch_size, generator)
        inp, tgt = inp.to(device), tgt.to(device)

        logits = model(input_ids=inp).logits
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), tgt.reshape(-1)
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
        optimizer.step()

        final_loss = loss.detach().item()
        if step % cfg.training.log_every == 0 or step == cfg.training.max_steps - 1:
            print(f"[beetlelm.toy] step {step:4d} | loss {final_loss:.4f} | lr {lr:.2e}")
            history.append({"step": step, "loss": final_loss, "lr": lr})

        if not math.isfinite(final_loss):
            raise RuntimeError(f"Loss became non-finite at step {step}: {final_loss}")

    final_val_loss = evaluate(model, val_stream, cfg.training.batch_size, generator, device)
    print(f"[beetlelm.toy] final train loss {final_loss:.4f} | val loss {final_val_loss:.4f}")

    ckpt_dir = Path(cfg.training.output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    results = {
        "run_name": cfg.run_name,
        "final_loss": round(final_loss, 6),
        "final_val_loss": round(final_val_loss, 6),
        "steps": cfg.training.max_steps,
        "n_params": n_params,
        "vocab_size": len(tokenizer),
        "seed": cfg.training.seed,
        "checkpoint_dir": str(ckpt_dir),
        "history": history,
    }
    (ckpt_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"[beetlelm.toy] checkpoint + metrics written to {ckpt_dir}/")
    return results


def _apply_overrides(cfg: Config, overrides: Dict) -> None:
    for block, values in overrides.items():
        target = getattr(cfg, block, None)
        if target is None or not isinstance(values, dict):
            continue
        for k, v in values.items():
            if hasattr(target, k):
                setattr(target, k, v)
