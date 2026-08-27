#!/usr/bin/env python3
"""
script/verify_features.py
─────────────────────────
One-shot verification that the six BeetleLM features required for the
EMNLP 2026 nld-eng-b3 sweep actually mutate training behaviour:

    1. EWC                    — penalty term added to loss
    2. ALiBi / working memory — attention bias hook installed + logits change
    3. LaMoL generative replay— replay loss is non-zero post-snapshot
    4. Layer growth           — model layer count + param count grows
    5. Pre-pretraining        — formal-language stage 1 mutates weights
    6. Combined-on smoke      — all five together change loss vs baseline

Runs entirely on CPU against a tiny PicoDecoder (≈100k params); takes
seconds. Exits non-zero (and prints WHICH feature looked indistinguishable
from baseline) if any feature appears to be a silent no-op.

Usage:
    python script/verify_features.py
    python script/verify_features.py --json   # machine-readable summary

This is *not* an end-to-end trainer test — that's what
`tests/test_features_differential.py` covers.  This script is meant to be
run before kicking off a multi-day FineWeb / BabyBabelLM sweep so you have
a single PASS/FAIL signal that toggling each ablation flag will produce a
distinguishable run.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beetlelm.continual.ewc import EWCRegularizer
from beetlelm.continual.generative_replay import GenerativeReplayModule
from beetlelm.continual.layer_growth import LayerGrowthModule
from beetlelm.pico_decoder.pico_decoder import PicoDecoder
from beetlelm.pre_pretraining.formal_lang import ShuffleDyckIterableDataset
from beetlelm.pre_pretraining.stage import (
    PrePretrainingConfig,
    run_pre_pretraining_stage,
)
from beetlelm.working_memory.alibi import WorkingMemoryScheduler


# ─────────────────────────────────────────────────────────────────────────────
# Tiny model
# ─────────────────────────────────────────────────────────────────────────────

def _tiny_cfg(vocab: int = 64) -> SimpleNamespace:
    return SimpleNamespace(
        n_layers=2, d_model=32, vocab_size=vocab,
        attention_n_heads=4, attention_n_kv_heads=4,
        max_seq_len=16, batch_size=2,
        position_emb_theta=10000.0,
        activation_hidden_dim=64, norm_eps=1e-5,
    )


def _build_model(cfg) -> PicoDecoder:
    m = PicoDecoder(cfg)
    m.model_config = cfg  # mirror registry.build_model
    return m


def _ce(model, ids: torch.Tensor) -> torch.Tensor:
    V = model.config.vocab_size
    x, y = ids[:, :-1].clamp(0, V - 1), ids[:, 1:].clamp(0, V - 1)
    logits, _ = model(x)
    B, T, _ = logits.shape
    return F.cross_entropy(logits.view(B * T, V), y.contiguous().view(-1))


# ─────────────────────────────────────────────────────────────────────────────
# Per-feature verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_ewc(cfg, ids) -> dict:
    torch.manual_seed(0)
    model = _build_model(cfg)
    base = _ce(model, ids).item()

    ewc = EWCRegularizer(lambda_=20.0, n_fisher_samples=2)
    ewc.snapshot(model)
    ewc.compute_fisher(model, [ids, ids])
    pen_at_star = ewc.penalty(model).item()

    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    pen_drift = ewc.penalty(model).item()
    diag = ewc.diagnostics()

    ok = (
        pen_at_star == 0.0          # zero at θ*
        and pen_drift > 0.0         # positive after drift
        and diag["ewc/fisher_mean"] > 0.0
    )
    return {
        "feature": "ewc",
        "ok": ok,
        "baseline_loss": base,
        "penalty_at_theta_star": pen_at_star,
        "penalty_after_drift": pen_drift,
        "fisher_mean": diag["ewc/fisher_mean"],
        "fisher_nonzero_frac": diag["ewc/fisher_nonzero_frac"],
        "verdict": "EWC penalty changes loss." if ok
                  else "EWC penalty looks indistinguishable from zero.",
    }


def verify_alibi(cfg, ids) -> dict:
    torch.manual_seed(0)
    model = _build_model(cfg)
    model.eval()
    x = ids[:, :-1]

    with torch.no_grad():
        baseline_logits, _ = model(x)

    sched = WorkingMemoryScheduler(
        schedule="linear", num_heads=cfg.attention_n_heads,
        initial_slope=2.0, final_slope=4.0,
    )
    sched.step(0, 100)
    model.set_working_memory(sched)
    with torch.no_grad():
        alibi_logits_start, _ = model(x)
    sched.step(99, 100)
    model.set_working_memory(sched)
    with torch.no_grad():
        alibi_logits_end, _ = model(x)
    model.set_working_memory(None)

    diff_baseline = (baseline_logits - alibi_logits_start).abs().max().item()
    diff_schedule = (alibi_logits_start - alibi_logits_end).abs().max().item()

    # Verify the constructor now refuses inverted linear schedules — this is
    # the regression guard for the previous defaults bug (1.0, 0.0).
    inverted_rejected = False
    try:
        WorkingMemoryScheduler(
            schedule="linear", num_heads=cfg.attention_n_heads,
            initial_slope=1.0, final_slope=0.0,
        )
    except ValueError:
        inverted_rejected = True

    # And factory defaults must produce a developmental curriculum (locality
    # bias decays so working-memory window grows).
    from beetlelm.working_memory.alibi import build_working_memory as _bwm
    factory_sched = _bwm({"enabled": True}, num_heads=cfg.attention_n_heads)
    factory_sched.step(0, 100); m0 = factory_sched.current_slopes.clone()
    factory_sched.step(99, 100); m1 = factory_sched.current_slopes.clone()
    factory_developmental = bool((m1 < m0).all().item())

    ok = (
        diff_baseline > 1e-5
        and diff_schedule > 1e-7
        and inverted_rejected
        and factory_developmental
    )
    return {
        "feature": "alibi_working_memory",
        "ok": ok,
        "baseline_vs_alibi_max_logit_diff": diff_baseline,
        "schedule_start_vs_end_max_logit_diff": diff_schedule,
        "inverted_schedule_rejected": inverted_rejected,
        "factory_developmental_curriculum": factory_developmental,
        "verdict": (
            "ALiBi hook changes logits, schedule progresses, "
            "factory defaults give a developmental curriculum, "
            "and inverted configs are rejected at construction."
            if ok else
            "ALiBi: at least one of {logit-change, schedule-progress, "
            "inverted-rejection, factory-developmental} failed."
        ),
    }


def verify_lamol(cfg, ids) -> dict:
    torch.manual_seed(0)
    model = _build_model(cfg)

    # 1. Default-path probe (unconditional): replay loss is non-zero,
    #    has gradient, snapshot toggles is_ready.
    gr = GenerativeReplayModule(
        lambda_replay=1.0, replay_ratio=0.5, buffer_size=8, max_gen_len=8,
        prompt_style="unconditional",
    )
    pre_ready = gr.is_ready
    gr.snapshot(model)
    post_ready = gr.is_ready
    scaled, stats = gr.compute_replay_loss(model, ids[:, :8], step=0)
    ok_basic = (
        not pre_ready and post_ready
        and stats["gen_replay/loss"] > 0.0
        and scaled.requires_grad
    )

    # 2. LAMOL-faithful path: prompt_style='lang_tag' must (a) be honoured
    #    end-to-end, (b) inject the configured lang_token_id as the leading
    #    token of every generated sequence, (c) yield a different buffer
    #    than the unconditional path. This is the regression guard that the
    #    historic prompt_style-ignored bug doesn't recur.
    torch.manual_seed(0)
    model_b = _build_model(cfg)
    gr_lt = GenerativeReplayModule(
        lambda_replay=1.0, replay_ratio=1.0, buffer_size=8, max_gen_len=8,
        prompt_style="lang_tag", lang_token_id=5,
    )
    gr_lt.snapshot(model_b)
    torch.manual_seed(99)
    gr_lt.compute_replay_loss(model_b, ids[:, :8], step=0)
    first_tokens = [int(s[0].item()) for s in gr_lt._buffer]
    lang_tag_honoured = (
        len(first_tokens) > 0 and all(t == 5 for t in first_tokens)
    )

    # 3. Constructor-level guards must reject misuse loudly.
    try:
        GenerativeReplayModule(prompt_style="lang_tag", lang_token_id=None)
        guards_ok = False
    except ValueError:
        guards_ok = True
    try:
        GenerativeReplayModule(prompt_style="bogus")
        guards_ok &= False
    except ValueError:
        guards_ok &= True

    ok = ok_basic and lang_tag_honoured and guards_ok
    return {
        "feature": "lamol_generative_replay",
        "ok": ok,
        "ready_before_snapshot": pre_ready,
        "ready_after_snapshot": post_ready,
        "replay_loss": stats["gen_replay/loss"],
        "scaled_replay_loss": stats["gen_replay/scaled_loss"],
        "buffer_size": stats["gen_replay/buffer_size"],
        "lang_tag_prefix_honoured": lang_tag_honoured,
        "constructor_guards_ok": guards_ok,
        "verdict": (
            "Generative replay produces non-zero loss; prompt_style='lang_tag' "
            "is honoured end-to-end (LAMOL-faithful task conditioning)."
            if ok else
            "FAIL: one of {basic-loss, lang_tag-honoured, constructor-guards} "
            "did not hold."
        ),
    }


def verify_layer_growth(cfg, ids) -> dict:
    torch.manual_seed(0)
    model = _build_model(cfg)
    model.eval()
    x = ids[:, :-1]

    with torch.no_grad():
        before_logits, _ = model(x)
    n_before = len(model.layers)
    p_before = sum(p.numel() for p in model.parameters())

    LayerGrowthModule(add_n_layers=2, init_mode="identity").apply(model)
    n_after = len(model.layers)
    p_after = sum(p.numel() for p in model.parameters())

    model.eval()
    with torch.no_grad():
        after_logits, _ = model(x)
    identity_drift = (before_logits - after_logits).abs().max().item()

    # Verify `random` mode actually re-randomizes (regression guard for the
    # bug where it silently fell through to a deepcopy of the last layer).
    torch.manual_seed(0)
    m1 = _build_model(cfg)
    torch.manual_seed(123)
    LayerGrowthModule(add_n_layers=1, init_mode="random").apply(m1)
    torch.manual_seed(0)
    m2 = _build_model(cfg)
    torch.manual_seed(123)
    LayerGrowthModule(add_n_layers=1, init_mode="clone_last").apply(m2)
    diffs = []
    for p1, p2 in zip(m1.layers[-1].parameters(), m2.layers[-1].parameters()):
        diffs.append((p1 - p2).abs().max().item())
    random_differs_from_clone = max(diffs) > 1e-6 if diffs else False

    ok = (
        n_after == n_before + 2
        and p_after > p_before
        and identity_drift < 1e-4
        and random_differs_from_clone
    )
    return {
        "feature": "layer_growth",
        "ok": ok,
        "n_layers_before": n_before,
        "n_layers_after": n_after,
        "params_before": p_before,
        "params_after": p_after,
        "identity_init_logit_drift": identity_drift,
        "random_differs_from_clone_last": random_differs_from_clone,
        "verdict": (
            "Layer growth: identity preserves logits, random re-randomizes, "
            "clone_last copies."
            if ok else
            "Growth no-op, identity drifts, OR random==clone_last."
        ),
    }


def verify_pre_pretraining(cfg, tmp_dir) -> dict:
    torch.manual_seed(0)
    model = _build_model(cfg)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}

    opt = torch.optim.SGD(model.parameters(), lr=0.05)

    class _F:
        device = torch.device("cpu")
        def backward(self, loss): loss.backward()
        def setup(self, m, o):    return m, o

    trainer = SimpleNamespace(
        model=model, optimizer=opt, fabric=_F(), seed=0,
        log=lambda *a, **kw: None,
        run_name="verify",
        checkpoint_cfg={"runs_dir": str(tmp_dir)},
        training_cfg={"learning_rate": 0.05},
        _pre_pretraining=PrePretrainingConfig(
            enabled=True, formal_lang="shuffle_dyck",
            steps=5, num_symbols=4, max_depth=4, p_open=0.5,
            # Use a vocab_offset that fits inside our tiny test vocab of 64
            # while still being above the typical special-token span.
            vocab_offset=16, reset_optimizer=False,
            save_stage1_checkpoint=False,
        ),
        _compute_loss=lambda batch, step: _ce(model, batch),
    )
    run_pre_pretraining_stage(trainer)

    n_moved = sum(
        1 for k, v in model.state_dict().items()
        if not torch.allclose(v, before[k])
    )

    # Validate the Dyck dataset on its own to make sure it produces well-
    # formed strings (catches generator regressions). Use a vocab_offset that
    # fits this tiny test vocab.
    ds = ShuffleDyckIterableDataset(
        num_symbols=4, seq_len=32, batch_size=1, vocab_size=64,
        vocab_offset=16,
    )
    dyck_batch = next(iter(ds))
    dyck_within_alphabet = bool(
        16 <= dyck_batch.min().item() and dyck_batch.max().item() < 16 + 8
    )

    # Confirm factory default is now safe (regression guard for the previous
    # vocab_offset=0 bug).
    from beetlelm.pre_pretraining.stage import build_pre_pretraining as _bpp
    pp_default = _bpp({"enabled": True, "num_symbols": 20, "steps": 1})
    factory_safe = pp_default is not None and pp_default.vocab_offset >= 16

    ok = bool(n_moved > 0 and dyck_within_alphabet and factory_safe)
    return {
        "feature": "pre_pretraining",
        "ok": ok,
        "n_params_moved": n_moved,
        "n_params_total": len(before),
        "dyck_within_alphabet": dyck_within_alphabet,
        "factory_default_vocab_offset_safe": factory_safe,
        "factory_default_vocab_offset": pp_default.vocab_offset if pp_default else None,
        "verdict": (
            "Stage-1 mutates weights, Dyck output valid, factory defaults safe."
            if ok else
            "Stage-1 broken or factory defaults still unsafe."
        ),
    }


def verify_combined(cfg, ids) -> dict:
    """All five features active → final loss should differ measurably from
    a baseline run with everything disabled, after a few SGD steps."""
    def _train_n(model, n_steps, ewc=None, gr=None, sched=None) -> float:
        opt = torch.optim.SGD(model.parameters(), lr=0.05)
        if sched is not None:
            model.set_working_memory(sched)
        loss_last = None
        for step in range(n_steps):
            opt.zero_grad()
            loss = _ce(model, ids)
            if ewc is not None and ewc.is_ready:
                loss = loss + ewc.penalty(model)
            if gr is not None and gr.is_ready:
                gr_loss, _ = gr.compute_replay_loss(model, ids[:, :8], step=step)
                loss = loss + gr_loss
            if sched is not None:
                model.update_progress(step, n_steps)
            loss.backward()
            opt.step()
            loss_last = loss.item()
        return loss_last

    # Baseline run.
    torch.manual_seed(0)
    base_model = _build_model(cfg)
    base_final = _train_n(base_model, n_steps=5)

    # Combined run: pre-pretraining → grow → snapshot EWC + replay → train.
    torch.manual_seed(0)
    combo_model = _build_model(cfg)

    opt = torch.optim.SGD(combo_model.parameters(), lr=0.05)
    class _F:
        device = torch.device("cpu")
        def backward(self, loss): loss.backward()
        def setup(self, m, o):    return m, o

    trainer = SimpleNamespace(
        model=combo_model, optimizer=opt, fabric=_F(), seed=0,
        log=lambda *a, **kw: None,
        run_name="combo",
        checkpoint_cfg={"runs_dir": "/tmp"},
        training_cfg={"learning_rate": 0.05},
        _pre_pretraining=PrePretrainingConfig(
            enabled=True, formal_lang="shuffle_dyck",
            steps=2, num_symbols=4, max_depth=3, p_open=0.5,
            vocab_offset=16, reset_optimizer=False,
            save_stage1_checkpoint=False,
        ),
        _compute_loss=lambda batch, step: _ce(combo_model, batch),
    )
    run_pre_pretraining_stage(trainer)
    LayerGrowthModule(add_n_layers=1, init_mode="identity").apply(combo_model)
    ewc = EWCRegularizer(lambda_=10.0, n_fisher_samples=2)
    ewc.snapshot(combo_model); ewc.compute_fisher(combo_model, [ids, ids])
    gr = GenerativeReplayModule(lambda_replay=0.3, replay_ratio=0.5,
                                  buffer_size=8, max_gen_len=8)
    gr.snapshot(combo_model)
    sched = WorkingMemoryScheduler(
        schedule="linear", num_heads=cfg.attention_n_heads,
        initial_slope=2.0, final_slope=4.0,
    )
    combo_final = _train_n(combo_model, n_steps=5, ewc=ewc, gr=gr, sched=sched)

    diff = abs(combo_final - base_final)
    ok = diff > 1e-4
    return {
        "feature": "combined_on",
        "ok": ok,
        "baseline_final_loss": base_final,
        "combined_final_loss": combo_final,
        "abs_diff": diff,
        "verdict": "Combined-on run diverges measurably from baseline."
                  if ok else "Combined-on run is indistinguishable from baseline.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def verify_schedule_free(cfg, ids) -> dict:
    """Schedule-Free AdamW (Defazio et al. NeurIPS 2024).

    Sanity-checks the trainer-facing contract:
      * Selectable via ``optimization.optimizer == 'schedule_free_adamw'``.
      * step() raises if train_mode is False (loud-failure path).
      * train()/eval() round-trip does not corrupt parameters.
      * eval() actually moves parameters (running average x != y once
        z and y have separated, i.e. after >=2 steps).
      * initialize_lr_scheduler returns a no-op (warmup is INTERNAL).
    """
    import torch.nn as nn
    from beetlelm.optim import AdamWScheduleFree
    from beetlelm.training.utils.initialization import (
        initialize_optimizer,
        initialize_lr_scheduler,
        _NoOpLRScheduler,
    )

    # 1. YAML selectability + lr-scheduler bypass.
    torch.manual_seed(0)
    model = nn.Linear(8, 8, bias=False)
    yaml_cfg = {
        "optimization": {
            "optimizer": "schedule_free_adamw",
            "lr": 1e-3,
            "warmup_steps": 4,
            "lr_scheduler": "linear_with_warmup",  # must be ignored
            "lr_warmup_steps": 99,
        },
        "max_steps": 100,
    }
    opt = initialize_optimizer(yaml_cfg, model)
    sched = initialize_lr_scheduler(yaml_cfg, opt)
    yaml_ok = isinstance(opt, AdamWScheduleFree) and isinstance(sched, _NoOpLRScheduler)

    # 2. step() must raise when train_mode is False.
    raise_ok = False
    try:
        x = torch.randn(2, 8)
        loss = model(x).pow(2).sum()
        loss.backward()
        opt.step()
    except RuntimeError as e:
        raise_ok = "train_mode=False" in str(e)
    finally:
        for p in model.parameters():
            p.grad = None

    # 3. train()/eval() observable swap + round-trip after 2 steps.
    torch.manual_seed(0)
    model = nn.Linear(8, 8, bias=False)
    opt = AdamWScheduleFree(model.parameters(), lr=1e-2)
    opt.train()
    for _ in range(2):
        x = torch.randn(4, 8)
        loss = model(x).pow(2).sum()
        loss.backward()
        opt.step()
        for p in model.parameters():
            p.grad = None

    y_state = {k: v.clone() for k, v in model.state_dict().items()}
    opt.eval()
    x_state = {k: v.clone() for k, v in model.state_dict().items()}
    opt.train()
    y_state2 = {k: v.clone() for k, v in model.state_dict().items()}

    swap_changes = any(
        not torch.allclose(y_state[k], x_state[k]) for k in y_state
    )
    roundtrip_ok = all(
        torch.allclose(y_state[k], y_state2[k], atol=1e-6) for k in y_state
    )

    # 4. Reportable max(|y - x|) magnitude.
    swap_delta = max(
        (y_state[k] - x_state[k]).abs().max().item() for k in y_state
    )

    ok = bool(yaml_ok and raise_ok and swap_changes and roundtrip_ok)
    return {
        "feature": "schedule_free_adamw",
        "ok": ok,
        "yaml_selectable": yaml_ok,
        "loud_failure_outside_train_mode": raise_ok,
        "eval_changes_params": swap_changes,
        "train_eval_roundtrip_lossless": roundtrip_ok,
        "max_y_minus_x": swap_delta,
        "verdict": (
            "Schedule-Free AdamW selects, gates, and swaps iterates correctly."
            if ok
            else "Schedule-Free AdamW failed at least one contract — see above."
        ),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true",
                    help="Emit a single JSON blob instead of human-readable text")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    cfg = _tiny_cfg()
    ids = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.max_seq_len + 1))

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        results = []
        for fn in (
            verify_ewc,
            verify_alibi,
            verify_lamol,
            verify_layer_growth,
            lambda c, i: verify_pre_pretraining(c, td),
            verify_combined,
            verify_schedule_free,
        ):
            try:
                results.append(fn(cfg, ids))
            except Exception as exc:
                results.append({
                    "feature": getattr(fn, "__name__", "unknown"),
                    "ok": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                })

    overall_ok = all(r["ok"] for r in results)

    if args.json:
        print(json.dumps({
            "overall_ok": overall_ok,
            "results": results,
        }, indent=2, default=str))
    else:
        print("=" * 72)
        print("BeetleLM feature verification — nld-eng-b3 readiness check")
        print("=" * 72)
        for r in results:
            tag = "PASS" if r["ok"] else "FAIL"
            print(f"\n[{tag}] {r['feature']}")
            print(f"     {r.get('verdict', '')}")
            for k, v in r.items():
                if k in ("feature", "ok", "verdict", "warning", "traceback"):
                    continue
                if isinstance(v, float):
                    print(f"       {k} = {v:.6g}")
                else:
                    print(f"       {k} = {v}")
            if r.get("warning"):
                print(f"     ⚠ WARNING: {r['warning']}")
            if r.get("traceback"):
                print(r["traceback"])
        print("\n" + "=" * 72)
        print("OVERALL:", "PASS" if overall_ok else "FAIL")
        print("=" * 72)

    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
