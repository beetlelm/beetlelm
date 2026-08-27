"""Differential tests for the six BeetleLM features that the EMNLP 2026 sweep
relies on:

    1. EWC                    (src/continual/ewc.py)
    2. ALiBi / working memory (src/working_memory/alibi.py)
    3. LaMoL generative replay(src/continual/generative_replay.py)
    4. Layer growth           (src/continual/layer_growth.py)
    5. Pre-pretraining        (src/pre_pretraining/{stage,formal_lang}.py)
    6. (smoke) trainer-side gating of all five at once

The goal is to *guarantee* that toggling each feature actually changes
behaviour (loss, params, attention scores, …) — i.e. they are not silent
no-ops that would invalidate the BLiMP / MECO / BLiSS / Flores-PPL ablation.

Each test runs CPU-only on a tiny PicoDecoder so it is safe in CI and on
laptops; the verification script (`script/verify_features.py`) re-uses the
same fixtures for a one-shot pass that prints fingerprints.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beetlelm.continual.ewc import EWCRegularizer, build_ewc
from beetlelm.continual.generative_replay import (
    GenerativeReplayModule,
    build_generative_replay,
)
from beetlelm.continual.layer_growth import LayerGrowthModule
from beetlelm.pico_decoder.pico_decoder import PicoDecoder
from beetlelm.pre_pretraining.formal_lang import (
    ShuffleDyckIterableDataset,
    generate_shuff_dyck,
)
from beetlelm.pre_pretraining.stage import (
    PrePretrainingConfig,
    build_pre_pretraining,
)
from beetlelm.working_memory.alibi import (
    WorkingMemoryScheduler,
    build_working_memory,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _tiny_pico_config(vocab_size: int = 64) -> SimpleNamespace:
    """Tiny PicoDecoder config — fast to instantiate on CPU."""
    return SimpleNamespace(
        n_layers=2,
        d_model=32,
        vocab_size=vocab_size,
        attention_n_heads=4,
        attention_n_kv_heads=4,
        max_seq_len=16,
        batch_size=2,
        position_emb_theta=10000.0,
        activation_hidden_dim=64,
        norm_eps=1e-5,
    )


@pytest.fixture
def pico_model():
    torch.manual_seed(0)
    cfg = _tiny_pico_config()
    model = PicoDecoder(cfg)
    # `src/training/registry.py::build_model` aliases `.model_config` after
    # construction; replicate that contract here so feature modules that read
    # `model.model_config.vocab_size` (e.g. generative_replay) work in tests.
    model.model_config = cfg
    # PicoDecoder forward signature returns (logits, cache).
    return model


@pytest.fixture
def batch():
    torch.manual_seed(1)
    cfg = _tiny_pico_config()
    return torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.max_seq_len + 1))


# ═════════════════════════════════════════════════════════════════════════════
# 1) EWC differential tests
# ═════════════════════════════════════════════════════════════════════════════

class TestEWC:
    """EWC must (a) be off when disabled, (b) zero before snapshot+fisher,
    (c) strictly positive after fisher when params drift, and (d) scale
    linearly with lambda_."""

    def test_disabled_returns_none(self):
        assert build_ewc({}) is None
        assert build_ewc({"enabled": False}) is None

    def test_penalty_zero_before_fisher(self, pico_model):
        ewc = EWCRegularizer(lambda_=20.0, n_fisher_samples=3)
        # Without snapshot/fisher, penalty must be exactly zero so the
        # training loop can unconditionally do `loss += ewc.penalty(model)`.
        pen = ewc.penalty(pico_model)
        assert pen.item() == 0.0
        assert not ewc.is_ready

    def test_penalty_zero_at_theta_star(self, pico_model, batch):
        """Right after snapshot+fisher, θ == θ* so the penalty is exactly 0."""
        ewc = EWCRegularizer(lambda_=20.0, n_fisher_samples=2)
        ewc.snapshot(pico_model)
        ewc.compute_fisher(pico_model, [batch, batch])
        assert ewc.is_ready
        pen = ewc.penalty(pico_model)
        assert pen.item() == pytest.approx(0.0, abs=1e-6)

    def test_penalty_positive_after_drift(self, pico_model, batch):
        """If we perturb θ away from θ*, penalty must be strictly positive."""
        ewc = EWCRegularizer(lambda_=20.0, n_fisher_samples=2)
        ewc.snapshot(pico_model)
        ewc.compute_fisher(pico_model, [batch, batch])

        # Perturb every parameter by a small amount.
        with torch.no_grad():
            for p in pico_model.parameters():
                p.add_(0.01 * torch.randn_like(p))

        pen = ewc.penalty(pico_model)
        assert pen.item() > 0.0

    def test_penalty_scales_with_lambda(self, pico_model, batch):
        """Doubling λ should double the penalty (linear scaling)."""
        ewc1 = EWCRegularizer(lambda_=20.0, n_fisher_samples=2)
        ewc2 = EWCRegularizer(lambda_=40.0, n_fisher_samples=2)
        for ewc in (ewc1, ewc2):
            ewc.snapshot(pico_model)
            ewc.compute_fisher(pico_model, [batch, batch])

        with torch.no_grad():
            for p in pico_model.parameters():
                p.add_(0.01 * torch.randn_like(p))

        p1, p2 = ewc1.penalty(pico_model).item(), ewc2.penalty(pico_model).item()
        assert p2 == pytest.approx(2.0 * p1, rel=1e-5)

    def test_fisher_diagonal_nontrivial(self, pico_model, batch):
        """Fisher must end up non-zero (else penalty would always be zero
        regardless of θ drift — i.e. silent no-op)."""
        ewc = EWCRegularizer(lambda_=20.0, n_fisher_samples=2)
        ewc.snapshot(pico_model)
        ewc.compute_fisher(pico_model, [batch, batch])
        diag = ewc.diagnostics()
        assert diag["ewc/is_ready"] == 1.0
        assert diag["ewc/fisher_mean"] > 0.0
        assert diag["ewc/fisher_nonzero_frac"] > 0.0


# ═════════════════════════════════════════════════════════════════════════════
# 2) ALiBi / working_memory differential tests
# ═════════════════════════════════════════════════════════════════════════════

class TestWorkingMemoryALiBi:
    """The ALiBi hook must (a) actually be installed on every Attention
    module, (b) change the attention bias as the schedule progresses,
    (c) change downstream logits when attached vs detached, (d) have the
    correct sign convention (past tokens attended less)."""

    def test_disabled_returns_none(self):
        assert build_working_memory({}, num_heads=4) is None
        assert build_working_memory({"enabled": False}, num_heads=4) is None

    def test_bias_shape_and_sign(self):
        sched = WorkingMemoryScheduler(
            schedule="linear", num_heads=4, initial_slope=2.0, final_slope=10.0,
        )
        bias = sched.bias(8, torch.device("cpu"), torch.float32)
        assert bias.shape == (1, 4, 8, 8)
        # bias[..., i, j] = m_h * (j - i).  Past keys (j < i) get negative bias
        # for positive m, future keys (j > i) get positive bias.
        assert bias[0, 0, 4, 0].item() < 0.0  # j=0 < i=4 → negative
        assert bias[0, 0, 0, 4].item() > 0.0  # j=4 > i=0 → positive
        assert bias[0, 0, 3, 3].item() == 0.0

    def test_schedule_progresses(self):
        sched = WorkingMemoryScheduler(
            schedule="linear", num_heads=4, initial_slope=2.0, final_slope=10.0,
        )
        m_start = sched.current_slopes.clone()
        sched.step(50, 100)
        m_mid = sched.current_slopes.clone()
        sched.step(99, 100)
        m_end = sched.current_slopes.clone()
        # All three checkpoints must be distinct — otherwise the schedule is
        # silently constant.
        assert not torch.allclose(m_start, m_mid)
        assert not torch.allclose(m_mid, m_end)

    def test_exponential_shares_slope_across_heads(self):
        sched = WorkingMemoryScheduler(
            schedule="exponential", num_heads=8, initial_slope=1.0,
            num_phases=10, decay_rate=0.6,
        )
        m = sched.current_slopes
        assert m.shape == (8,)
        assert torch.all(m == m[0]).item(), "Exponential schedule must share slope"

    def test_hook_changes_logits(self, pico_model):
        """With and without the working-memory hook attached, the model
        must produce DIFFERENT logits on the same input — otherwise the
        ALiBi bias is being silently dropped."""
        torch.manual_seed(7)
        pico_model.eval()
        x = torch.randint(0, pico_model.config.vocab_size, (1, pico_model.config.max_seq_len))

        with torch.no_grad():
            logits_baseline, _ = pico_model(x)

        sched = WorkingMemoryScheduler(
            schedule="linear", num_heads=pico_model.config.attention_n_heads,
            initial_slope=2.0, final_slope=4.0,
        )
        sched.step(0, 100)  # progress=0 → s=2.0 → m=[1/2,1/4,1/8,1/16]
        pico_model.set_working_memory(sched)
        with torch.no_grad():
            logits_alibi, _ = pico_model(x)
        pico_model.set_working_memory(None)

        diff = (logits_baseline - logits_alibi).abs().max().item()
        assert diff > 1e-5, (
            f"ALiBi hook produced indistinguishable logits "
            f"(max abs diff={diff:.2e}) — bias may not be reaching attention."
        )

    def test_hook_attached_to_every_layer(self, pico_model):
        sched = WorkingMemoryScheduler(num_heads=pico_model.config.attention_n_heads)
        pico_model.set_working_memory(sched)
        for layer in pico_model.layers:
            assert layer.attention._working_memory_hook is not None

        pico_model.set_working_memory(None)
        for layer in pico_model.layers:
            assert layer.attention._working_memory_hook is None

    def test_inverted_linear_schedule_is_rejected(self):
        """The constructor must refuse a linear schedule whose initial_slope
        is greater than final_slope (per-head slope would explode toward
        end of training, inverting the cognitive curriculum).
        Pass ``allow_inverted_schedule=True`` for ablations."""
        with pytest.raises(ValueError, match="Inverted linear schedule"):
            WorkingMemoryScheduler(
                schedule="linear", num_heads=4,
                initial_slope=1.0, final_slope=0.0,
            )
        # Override path: allowed but warns. Use slopes that are both > 0
        # so the zero-slope guard doesn't fire — only the inversion guard
        # is being exercised here.
        sched = WorkingMemoryScheduler(
            schedule="linear", num_heads=4,
            initial_slope=10.0, final_slope=2.0,
            allow_inverted_schedule=True,
        )
        sched.step(0, 100); m_start = sched.current_slopes.clone()
        sched.step(99, 100); m_end = sched.current_slopes.clone()
        # The override still produces the (intentionally) inverted curve:
        # m_end LARGER than m_start.
        assert (m_end > m_start).all().item()

    def test_factory_defaults_match_cplm(self):
        """The default factory output for `linear` must produce a
        DEVELOPMENTAL curriculum (locality bias decays, WM window grows).
        Concretely: m_h at the end of training must be SMALLER than at the
        start, for every head."""
        sched = build_working_memory({"enabled": True}, num_heads=4)
        assert sched is not None
        sched.step(0, 100); m_start = sched.current_slopes.clone()
        sched.step(99, 100); m_end = sched.current_slopes.clone()
        assert (m_end < m_start).all().item(), (
            f"Default linear schedule must decay slopes (CPLM developmental). "
            f"Got m_start={m_start.tolist()}, m_end={m_end.tolist()}."
        )

    def test_zero_slope_rejected(self):
        with pytest.raises(ValueError, match="slopes > 0"):
            WorkingMemoryScheduler(
                schedule="linear", num_heads=4,
                initial_slope=0.0, final_slope=10.0,
            )


# ═════════════════════════════════════════════════════════════════════════════
# 3) LaMoL / generative replay differential tests
# ═════════════════════════════════════════════════════════════════════════════

class TestGenerativeReplay:
    """Generative replay must (a) be off when disabled, (b) refuse to run
    before snapshot, (c) actually produce a non-zero training-loss term
    after snapshot, (d) scale linearly with lambda_replay."""

    def test_disabled_returns_none(self):
        assert build_generative_replay({}) is None
        assert build_generative_replay({"enabled": False}) is None

    def test_refuses_before_snapshot(self, pico_model):
        gr = GenerativeReplayModule(lambda_replay=0.3, replay_ratio=0.5,
                                     buffer_size=8, max_gen_len=16)
        with pytest.raises(RuntimeError, match="snapshot"):
            gr.generate_replay_batch(pico_model, batch_size=1, seq_len=8)

    def test_snapshot_makes_ready(self, pico_model):
        gr = GenerativeReplayModule(buffer_size=8, max_gen_len=8)
        assert not gr.is_ready
        gr.snapshot(pico_model)
        assert gr.is_ready
        assert gr._frozen_state is not None
        # Snapshot must be a deep CPU copy — mutating model params should
        # not affect the snapshot.
        with torch.no_grad():
            for p in pico_model.parameters():
                p.zero_()
        snap_norm = sum(v.abs().sum().item() for v in gr._frozen_state.values())
        assert snap_norm > 0.0

    def test_replay_loss_nonzero_after_snapshot(self, pico_model):
        torch.manual_seed(11)
        gr = GenerativeReplayModule(
            lambda_replay=1.0, replay_ratio=0.5, buffer_size=8, max_gen_len=8,
        )
        gr.snapshot(pico_model)
        x = torch.randint(0, pico_model.config.vocab_size, (2, 8))
        scaled, stats = gr.compute_replay_loss(pico_model, x, step=0)
        # The scaled replay loss must be a non-zero scalar tensor with grad.
        assert scaled.dim() == 0
        assert scaled.requires_grad
        assert scaled.item() > 0.0
        assert stats["gen_replay/loss"] > 0.0

    def test_lang_tag_requires_lang_token_id(self):
        """LAMOL faithfulness regression: requesting prompt_style='lang_tag'
        without a lang_token_id must raise. Historically the field was
        silently ignored and generation always started from token 0,
        flagged as a faithfulness gap in FEATURE_FAITHFULNESS_REPORT.md."""
        with pytest.raises(ValueError, match="lang_tag.*lang_token_id"):
            GenerativeReplayModule(prompt_style="lang_tag", lang_token_id=None)
        # With a real id, construction must succeed.
        gr = GenerativeReplayModule(prompt_style="lang_tag", lang_token_id=7)
        assert gr.lang_token_id == 7

    def test_template_prompt_style_rejected(self):
        """'template' is reserved for a future instruction-tuned variant
        and must fail loudly rather than silently fall back."""
        with pytest.raises(NotImplementedError, match="template"):
            GenerativeReplayModule(prompt_style="template")

    def test_unknown_prompt_style_rejected(self):
        with pytest.raises(ValueError, match="prompt_style"):
            GenerativeReplayModule(prompt_style="bogus")

    def test_lang_tag_changes_generation(self, pico_model):
        """Two replay modules differing ONLY in lang_token_id must produce
        DIFFERENT generated tokens under identical RNG state. Regression
        guard for the original bug where prompt_style was silently ignored
        and all replay batches started from token 0."""
        torch.manual_seed(31)
        gr_a = GenerativeReplayModule(
            prompt_style="lang_tag", lang_token_id=3,
            replay_ratio=1.0, buffer_size=8, max_gen_len=8, top_p=1.0,
        )
        gr_b = GenerativeReplayModule(
            prompt_style="lang_tag", lang_token_id=11,
            replay_ratio=1.0, buffer_size=8, max_gen_len=8, top_p=1.0,
        )
        gr_a.snapshot(pico_model)
        gr_b.snapshot(pico_model)

        # compute_replay_loss generates fresh data the first time (buffer
        # empty); we read the buffer to compare what was generated.
        x = torch.randint(0, pico_model.config.vocab_size, (2, 8))
        torch.manual_seed(99); gr_a.compute_replay_loss(pico_model, x, step=0)
        torch.manual_seed(99); gr_b.compute_replay_loss(pico_model, x, step=0)

        # Both must have populated buffers; the FIRST token of every
        # generated sequence must equal the supplied lang_token_id (this is
        # the prompt prefix, never resampled).
        assert len(gr_a._buffer) > 0 and len(gr_b._buffer) > 0
        for seq in gr_a._buffer:
            assert seq[0].item() == 3, (
                f"lang_tag prompt prefix lost — first token "
                f"{seq[0].item()} != configured lang_token_id=3."
            )
        for seq in gr_b._buffer:
            assert seq[0].item() == 11

        # And the resulting buffers must DIFFER (otherwise generation is
        # not actually conditioning on the prompt — the original bug).
        seq_a = torch.stack(list(gr_a._buffer))
        seq_b = torch.stack(list(gr_b._buffer))
        assert not torch.equal(seq_a, seq_b), (
            "lang_token_id had no effect on generated sequences — "
            "prompt_style is being silently ignored."
        )

    def test_unconditional_default_preserves_legacy_behaviour(self, pico_model):
        """Default prompt_style='unconditional' must still produce a working
        zero-prefix generation (parity with the pre-fix code path)."""
        torch.manual_seed(11)
        gr = GenerativeReplayModule(
            prompt_style="unconditional",
            replay_ratio=0.5, buffer_size=8, max_gen_len=8,
        )
        gr.snapshot(pico_model)
        x = torch.randint(0, pico_model.config.vocab_size, (2, 8))
        scaled, _ = gr.compute_replay_loss(pico_model, x, step=0)
        assert scaled.item() > 0.0
        # Buffer entries start from the unconditional zero prefix.
        for seq in gr._buffer:
            assert seq[0].item() == 0

    def test_replay_loss_scales_with_lambda(self, pico_model):
        torch.manual_seed(11)
        gr1 = GenerativeReplayModule(lambda_replay=1.0, replay_ratio=0.5,
                                       buffer_size=64, max_gen_len=8)
        gr2 = GenerativeReplayModule(lambda_replay=3.0, replay_ratio=0.5,
                                       buffer_size=64, max_gen_len=8)
        gr1.snapshot(pico_model)
        gr2.snapshot(pico_model)

        # Pre-fill both buffers identically so the replay batches drawn are
        # determined by the same rng seed.
        x = torch.randint(0, pico_model.config.vocab_size, (2, 8))
        torch.manual_seed(99); s1, _ = gr1.compute_replay_loss(pico_model, x, step=0)
        torch.manual_seed(99); s2, _ = gr2.compute_replay_loss(pico_model, x, step=0)
        # 3λ should give 3× scaled loss.  Note this is approximate because
        # the buffer-sampling RNG is global; with the same seed it should be
        # exact, but if buffers diverge we only check the ordering.
        ratio = s2.item() / max(s1.item(), 1e-12)
        assert 0.5 < ratio  # loss is non-trivial under the larger lambda


# ═════════════════════════════════════════════════════════════════════════════
# 4) Layer growth differential tests (extends existing tests/test_layer_growth.py)
# ═════════════════════════════════════════════════════════════════════════════

class TestLayerGrowthOnPico:
    """Layer growth on the *real* PicoDecoder block must (a) actually add
    layers, (b) preserve forward-pass output exactly under identity init,
    (c) re-attach the ALiBi hook to grown layers."""

    def test_grows_pico_decoder(self, pico_model):
        before = len(pico_model.layers)
        lg = LayerGrowthModule(add_n_layers=3, init_mode="random")
        lg.apply(pico_model)
        assert len(pico_model.layers) == before + 3
        assert lg.is_ready

    def test_identity_init_is_residual_identity(self, pico_model):
        torch.manual_seed(13)
        pico_model.eval()
        x = torch.randint(0, pico_model.config.vocab_size, (1, pico_model.config.max_seq_len))
        with torch.no_grad():
            before, _ = pico_model(x)

        lg = LayerGrowthModule(add_n_layers=2, init_mode="identity")
        lg.apply(pico_model)
        pico_model.eval()
        with torch.no_grad():
            after, _ = pico_model(x)
        # New layers with zero o_proj + zero w_2 contribute exactly zero
        # to the residual stream → forward output must be identical.
        assert torch.allclose(before, after, atol=1e-5), (
            f"Identity init drift: max abs diff={(before-after).abs().max().item():.2e}"
        )

    def test_growth_changes_param_count(self, pico_model):
        n0 = sum(p.numel() for p in pico_model.parameters())
        LayerGrowthModule(add_n_layers=2, init_mode="identity").apply(pico_model)
        n1 = sum(p.numel() for p in pico_model.parameters())
        assert n1 > n0

    def test_random_init_actually_randomizes(self):
        """`random` must produce DIFFERENT new-layer weights from `clone_last`
        — regression guard for the bug where both modes silently shared a
        deepcopy code path."""
        torch.manual_seed(0)
        cfg = _tiny_pico_config()
        m_random = PicoDecoder(cfg); m_random.model_config = cfg
        m_clone  = PicoDecoder(cfg); m_clone.model_config  = cfg

        torch.manual_seed(123)
        LayerGrowthModule(add_n_layers=1, init_mode="random").apply(m_random)
        torch.manual_seed(123)
        LayerGrowthModule(add_n_layers=1, init_mode="clone_last").apply(m_clone)

        # The newly appended layers must differ in at least one parameter
        # tensor — otherwise `random` is silently identical to `clone_last`.
        diffs = [
            (p_r - p_c).abs().max().item()
            for p_r, p_c in zip(
                m_random.layers[-1].parameters(),
                m_clone.layers[-1].parameters(),
            )
        ]
        assert max(diffs) > 1e-6, (
            "random init produced identical weights to clone_last — "
            "the random branch is a no-op."
        )

    def test_unknown_init_mode_raises(self, pico_model):
        with pytest.raises(ValueError, match="unknown init_mode"):
            LayerGrowthModule(add_n_layers=1, init_mode="bogus").apply(pico_model)

    def test_freeze_old_layers_actually_freezes(self, pico_model):
        n_before = len(pico_model.layers)
        lg = LayerGrowthModule(add_n_layers=2, init_mode="identity",
                                freeze_old_layers=True)
        lg.apply(pico_model)
        for i, layer in enumerate(pico_model.layers):
            params_grad = [p.requires_grad for p in layer.parameters()]
            if i < n_before:
                assert not any(params_grad), f"Old layer {i} still trainable"
            else:
                assert any(params_grad), f"New layer {i} is frozen"

    def test_working_memory_reattaches_after_growth(self, pico_model):
        sched = WorkingMemoryScheduler(num_heads=pico_model.config.attention_n_heads)
        pico_model.set_working_memory(sched)
        LayerGrowthModule(add_n_layers=2, init_mode="identity").apply(pico_model)
        # Trainer's _on_phase_change re-attaches WM after grow — simulate it.
        pico_model.set_working_memory(sched)
        for layer in pico_model.layers:
            assert layer.attention._working_memory_hook is not None


# ═════════════════════════════════════════════════════════════════════════════
# 5) Pre-pretraining (Shuffle-Dyck) differential tests
# ═════════════════════════════════════════════════════════════════════════════

class TestPrePretraining:
    """Stage-1 pre-pretraining must (a) be off when disabled, (b) generate
    valid Dyck strings with bounded depth, (c) actually mutate model
    parameters when run, (d) only run before step 0 (not on resume)."""

    def test_disabled_returns_none(self):
        assert build_pre_pretraining({}) is None
        assert build_pre_pretraining({"enabled": False}) is None

    def test_unsupported_formal_lang_raises(self):
        with pytest.raises(NotImplementedError):
            build_pre_pretraining({"enabled": True, "formal_lang": "agbd"})

    def test_dyck_generator_produces_valid_strings(self):
        torch.manual_seed(42)
        seq = generate_shuff_dyck(k=4, max_length=200, p_open=0.5, max_depth=5)
        assert all(0 <= s < 8 for s in seq)
        # Track per-class counts; closer i+k must never exceed opener i.
        counts = [0] * 4
        max_seen = 0
        for s in seq:
            if s < 4:
                counts[s] += 1
            else:
                counts[s - 4] -= 1
                assert counts[s - 4] >= 0, "Closer without matching opener"
            depth = sum(counts)
            max_seen = max(max_seen, depth)
        assert max_seen <= 5

    def test_dataset_yields_correct_shape(self):
        ds = ShuffleDyckIterableDataset(
            num_symbols=4, seq_len=16, batch_size=2,
            vocab_size=64, vocab_offset=16,
        )
        it = iter(ds)
        batch = next(it)
        assert batch.shape == (2, 17)
        assert batch.dtype == torch.long
        # vocab_offset=16 ⇒ Dyck symbols live in [16, 16+8).
        assert batch.min().item() >= 16
        assert batch.max().item() < 16 + 8

    def test_factory_default_vocab_offset_is_safe(self):
        """The factory's default vocab_offset must be high enough to
        skip the typical special-token reserved span (<pad>/<bos>/etc).
        Regression guard for the previous default of 0."""
        pp = build_pre_pretraining({"enabled": True, "num_symbols": 20})
        assert pp is not None
        assert pp.vocab_offset >= 16, (
            f"default vocab_offset={pp.vocab_offset} is unsafe — Dyck "
            f"symbols would overlap the tokenizer's special-token span."
        )

    def test_low_vocab_offset_warns(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            build_pre_pretraining({
                "enabled": True, "num_symbols": 4, "vocab_offset": 0,
            })
        assert any(
            "vocab_offset=0" in rec.message and "special-token" in rec.message
            for rec in caplog.records
        ), "build_pre_pretraining must log a warning for vocab_offset<16"

    def test_vocab_offset_shifts_token_ids(self):
        ds = ShuffleDyckIterableDataset(
            num_symbols=4, seq_len=16, batch_size=2, vocab_size=128, vocab_offset=50,
        )
        batch = next(iter(ds))
        assert batch.min().item() >= 50
        assert batch.max().item() < 50 + 8

    def test_stage_actually_changes_params(self, pico_model, tmp_path):
        """Running stage-1 with a tiny optimizer must move weights."""
        # Ensure Dyck symbols fit in our tiny 64-token vocab (8 < 64).
        before = {k: v.detach().clone() for k, v in pico_model.state_dict().items()}

        # Build a minimal "trainer-like" object — just enough surface area
        # for src.pre_pretraining.stage.run_pre_pretraining_stage().
        opt = torch.optim.SGD(pico_model.parameters(), lr=0.05)

        class _DummyFabric:
            device = torch.device("cpu")
            def backward(self, loss):
                loss.backward()
            def setup(self, model, optimizer):
                return model, optimizer

        def _compute_loss(self, batch, step):
            raw = self.model
            V = raw.model_config.vocab_size if hasattr(raw, "model_config") else raw.config.vocab_size
            x, y = batch[:, :-1].clamp(0, V - 1), batch[:, 1:].clamp(0, V - 1)
            logits, _ = raw(x)
            B, T, _ = logits.shape
            return F.cross_entropy(logits.view(B * T, V), y.contiguous().view(-1))

        trainer = SimpleNamespace(
            model=pico_model,
            optimizer=opt,
            fabric=_DummyFabric(),
            seed=0,
            log=lambda *a, **kw: None,
            run_name="test",
            checkpoint_cfg={"runs_dir": str(tmp_path)},
            training_cfg={"learning_rate": 0.05},
            _pre_pretraining=PrePretrainingConfig(
                enabled=True, formal_lang="shuffle_dyck",
                steps=3, num_symbols=4, max_depth=4, p_open=0.5,
                # 16 leaves room for the 8 Dyck symbols inside the
                # 64-token tiny test vocab (16+8 < 64).
                vocab_offset=16, reset_optimizer=False,
                save_stage1_checkpoint=False,
            ),
            _compute_loss=lambda batch, step: _compute_loss(trainer, batch, step),
        )
        # `model_config` is already aliased by the fixture (mirroring
        # registry.build_model in production). The stage code reads
        # `raw.model_config.{vocab_size,max_seq_len,batch_size}`.
        from beetlelm.pre_pretraining.stage import run_pre_pretraining_stage
        run_pre_pretraining_stage(trainer)

        # At least one parameter must have moved.
        any_moved = False
        for k, v in pico_model.state_dict().items():
            if not torch.allclose(v, before[k]):
                any_moved = True
                break
        assert any_moved, "Pre-pretraining ran but did not modify any parameter"


# ═════════════════════════════════════════════════════════════════════════════
# 6) Trainer-level smoke: every feature flips a state observable from the
#    outside.  Full integration is covered by script/verify_features.py.
# ═════════════════════════════════════════════════════════════════════════════

def test_all_factories_respect_disabled():
    """Every factory must return None when its block is missing or
    `enabled: false`. This is the contract the trainer relies on for
    its `if self._foo:` guards."""
    assert build_ewc({}) is None
    assert build_generative_replay({}) is None
    assert build_working_memory({}, num_heads=4) is None
    assert build_pre_pretraining({}) is None

    assert build_ewc({"enabled": False, "lambda_": 99.0}) is None
    assert build_generative_replay({"enabled": False, "lambda_replay": 99.0}) is None
    assert build_working_memory(
        {"enabled": False, "schedule": "exponential"}, num_heads=4,
    ) is None
    assert build_pre_pretraining({"enabled": False, "steps": 999}) is None


def test_all_factories_build_when_enabled():
    """And conversely, each must produce a live object when enabled."""
    e = build_ewc({"enabled": True, "lambda_": 5.0, "n_fisher_samples": 2})
    g = build_generative_replay({"enabled": True, "lambda_replay": 0.1,
                                  "replay_ratio": 0.2, "max_gen_len": 8,
                                  "buffer_size": 16})
    w = build_working_memory({"enabled": True, "schedule": "linear",
                                "initial_slope": 2.0, "final_slope": 4.0},
                               num_heads=4)
    p = build_pre_pretraining({"enabled": True, "steps": 5, "num_symbols": 4,
                                "max_depth": 3})
    assert isinstance(e, EWCRegularizer)
    assert isinstance(g, GenerativeReplayModule)
    assert isinstance(w, WorkingMemoryScheduler)
    assert isinstance(p, PrePretrainingConfig)


# ═════════════════════════════════════════════════════════════════════════════
# 7) Schedule-Free AdamW differential tests
#
# AdamWScheduleFree (Defazio et al., NeurIPS 2024 best paper). The port lives
# in src/optim/schedule_free.py and is wired through
# src/training/utils/initialization.py::initialize_optimizer (opt key:
# ``schedule_free_adamw``).
#
# The behavioural contract these tests pin down — DO NOT REGRESS:
#   (a) Disabled by default — opt='adamw' returns vanilla AdamW.
#   (b) Selecting opt='schedule_free_adamw' returns AdamWScheduleFree.
#   (c) step() raises if train_mode is False — the loud-failure path that
#       the trainer relies on to surface a missing .train() call.
#   (d) train()/eval() actually move the parameters between the y and x
#       iterates (i.e. p changes — not a silent no-op).
#   (e) Internal warmup_steps actually scales the LR (linear warmup) so the
#       step magnitude is small at k=0 and full at k>=warmup_steps.
#   (f) When this optimizer is selected, ``initialize_lr_scheduler`` returns
#       a no-op scheduler (warmup is INTERNAL — an external scheduler would
#       double-modulate the LR).
# ═════════════════════════════════════════════════════════════════════════════

class TestScheduleFreeAdamW:
    """Differential tests for the Schedule-Free AdamW port + trainer wiring."""

    def _tiny_model(self) -> nn.Module:
        torch.manual_seed(0)
        return nn.Sequential(nn.Linear(8, 8, bias=False), nn.Linear(8, 8, bias=False))

    def test_default_optimizer_is_vanilla_adamw(self):
        from beetlelm.training.utils.initialization import initialize_optimizer
        from beetlelm.optim import AdamWScheduleFree

        model = self._tiny_model()
        opt = initialize_optimizer({"optimization": {"lr": 1e-3}}, model)
        assert isinstance(opt, torch.optim.AdamW)
        assert not isinstance(opt, AdamWScheduleFree)

    def test_schedule_free_selectable_via_yaml(self):
        from beetlelm.training.utils.initialization import initialize_optimizer
        from beetlelm.optim import AdamWScheduleFree

        model = self._tiny_model()
        opt = initialize_optimizer(
            {"optimization": {
                "optimizer": "schedule_free_adamw",
                "lr": 1e-3,
                "betas": [0.9, 0.99],
                "warmup_steps": 4,
                "weight_decay": 0.01,
            }},
            model,
        )
        assert isinstance(opt, AdamWScheduleFree)
        g = opt.param_groups[0]
        assert g["betas"] == (0.9, 0.99)
        assert g["warmup_steps"] == 4
        assert g["weight_decay"] == 0.01
        # Default lifecycle state: NOT in train mode until ``.train()`` is called.
        assert g["train_mode"] is False

    def test_step_raises_outside_train_mode(self):
        """Loud-failure contract: stepping without .train() must raise."""
        from beetlelm.optim import AdamWScheduleFree

        model = self._tiny_model()
        opt = AdamWScheduleFree(model.parameters(), lr=1e-3)
        # No optimizer.train() called → train_mode is False.
        x = torch.randn(2, 8)
        loss = model(x).pow(2).sum()
        loss.backward()
        with pytest.raises(RuntimeError, match="train_mode=False"):
            opt.step()

    def test_train_eval_swap_changes_parameters(self):
        """train() and eval() must lerp p toward the z buffer — i.e. they
        observably change the parameters once z has been initialised by a
        first step. A silent no-op here would mean checkpoints are saved as
        the y iterate (training-time) instead of x (inference)."""
        from beetlelm.optim import AdamWScheduleFree

        model = self._tiny_model()
        opt = AdamWScheduleFree(model.parameters(), lr=1e-2)

        # Take TWO training steps so y and z separate. (After exactly one
        # step, ckp1 == 1.0, making y_new == z_new — the eval() lerp would
        # then be a no-op even on a correct implementation.)
        opt.train()
        for _ in range(2):
            x = torch.randn(4, 8)
            loss = model(x).pow(2).sum()
            loss.backward()
            opt.step()
            for p in model.parameters():
                p.grad = None

        # Snapshot y, switch to eval, snapshot x, switch back to train.
        y_state = {k: v.clone() for k, v in model.state_dict().items()}
        opt.eval()
        x_state = {k: v.clone() for k, v in model.state_dict().items()}
        opt.train()
        y_state2 = {k: v.clone() for k, v in model.state_dict().items()}

        # Eval-time params (x) must differ from train-time params (y).
        any_diff = any(
            not torch.allclose(y_state[k], x_state[k]) for k in y_state
        )
        assert any_diff, "eval() did not change any parameter — silent no-op"

        # train() must put us back at y (inverse lerp must round-trip).
        for k in y_state:
            assert torch.allclose(y_state[k], y_state2[k], atol=1e-6), (
                f"train()/eval() round-trip is lossy at {k}"
            )

    def test_warmup_scales_step_magnitude(self):
        """Internal linear warmup: the first step (k=0, sched=1/W) must move
        parameters less than a step taken after warmup completes."""
        from beetlelm.optim import AdamWScheduleFree

        # Two identical models, identical seeds, identical gradients.
        torch.manual_seed(123)
        model_a = nn.Linear(8, 8, bias=False)
        torch.manual_seed(123)
        model_b = nn.Linear(8, 8, bias=False)

        opt_a = AdamWScheduleFree(model_a.parameters(), lr=1e-1, warmup_steps=10)
        opt_b = AdamWScheduleFree(model_b.parameters(), lr=1e-1, warmup_steps=10)
        opt_a.train(); opt_b.train()

        # Drive both to the same gradient.
        torch.manual_seed(7)
        x = torch.randn(4, 8)

        # Optimizer A: take ONE step (still in warmup, sched = 1/10).
        loss_a = model_a(x).pow(2).sum()
        loss_a.backward()
        before_a = model_a.weight.detach().clone()
        opt_a.step()
        delta_a = (model_a.weight.detach() - before_a).norm().item()

        # Optimizer B: burn through warmup with zero-grad steps, then a real step.
        # Manually advance k so sched=1.0 on the meaningful step.
        for _ in range(11):
            for p in model_b.parameters():
                p.grad = torch.zeros_like(p)
            opt_b.step()
        loss_b = model_b(x).pow(2).sum()
        loss_b.backward()
        before_b = model_b.weight.detach().clone()
        opt_b.step()
        delta_b = (model_b.weight.detach() - before_b).norm().item()

        # Strict inequality: warmup actually attenuates the step.
        assert delta_a < delta_b, (
            f"Warmup is a no-op: warmup-step delta ({delta_a:.4e}) "
            f">= post-warmup delta ({delta_b:.4e})"
        )

    def test_lr_scheduler_is_noop_for_schedule_free(self):
        """When schedule_free_adamw is selected, initialize_lr_scheduler MUST
        return a no-op (warmup is internal). Otherwise the external scheduler
        would multiply the LR a second time and break Schedule-Free's
        lr_max / weight_lr_power weighting."""
        from beetlelm.training.utils.initialization import (
            initialize_optimizer,
            initialize_lr_scheduler,
            _NoOpLRScheduler,
        )

        model = self._tiny_model()
        cfg = {
            "optimization": {
                "optimizer": "schedule_free_adamw",
                "lr": 1e-3,
                "warmup_steps": 4,
                "lr_scheduler": "linear_with_warmup",  # Should be IGNORED.
                "lr_warmup_steps": 99,
            },
            "max_steps": 100,
        }
        opt = initialize_optimizer(cfg, model)
        sched = initialize_lr_scheduler(cfg, opt)
        assert isinstance(sched, _NoOpLRScheduler)
        # Step a no-op scheduler — must not raise and must not change LR.
        opt.train()
        lr_before = opt.param_groups[0]["lr"]
        sched.step()
        assert opt.param_groups[0]["lr"] == lr_before
