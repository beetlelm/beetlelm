# beetlelm

A framework for transparently training bilingual (and monolingual / polyglot)
language models and storing comprehensive checkpoints for learning-dynamics
research. It ships a family of architectures, a bilingual data and tokenizer
pipeline, continual-learning extensions, and a HuggingFace-compatible checkpoint
format — plus a **download-free CPU quickstart** so a newcomer can train a real
model in a couple of minutes without a GPU or any downloads.

There are two entry points:

- **`beetlelm-toy`** — the newcomer path. Trains the real PicoDecoder on a tiny
  in-repo byte corpus on CPU, no downloads, light dependencies. Great for
  kicking the tires and for CI.
- **`beetlelm-train`** — the full research trainer (Lightning, HuggingFace
  datasets, the architecture zoo, continual-learning modules). Built for
  multi-GPU clusters and the configs under `configs/`.

Checkpoints from both are standard HuggingFace and reload with `from_pretrained`.

## Install

Full stack (pulls Lightning, datasets, wandb, HF Hub, ...):

```bash
git clone https://github.com/BeetleLM/beetlelm.git
cd beetlelm
pip install -e .
```

Light install for the CPU quickstart only (no heavy training stack):

```bash
pip install "torch>=2.5.1" transformers numpy pyyaml click
pip install -e . --no-deps
```

Convenience script:

```bash
./install.sh          # full install + smoke train
./install.sh --toy    # light (toy) install + smoke train
```

## Quickstart (CPU, zero downloads)

```bash
beetlelm-toy --config_path configs/tiny_cpu.yaml
```

or from Python:

```python
from beetlelm.toy import train
results = train("configs/tiny_cpu.yaml")
print(results["final_loss"], results["checkpoint_dir"])
```

## Reproduce the headline toy result

`configs/tiny_cpu.yaml` trains a **0.72M-parameter** PicoDecoder for 200 steps
on the in-repo bilingual (English + Spanish) byte corpus. On a laptop CPU it
finishes in about **15 seconds** and produces:

| metric           | expected value |
|------------------|----------------|
| final train loss | ~0.20 |
| final val loss   | ~0.37 |
| parameters       | 722,048 |

Random loss over the 256-way byte vocabulary is `ln(256) ≈ 5.55`, so the drop
to ~0.2 shows the model has learned the toy corpus. The expected numbers are
committed in [`baselines/tiny_cpu_baseline.json`](baselines/tiny_cpu_baseline.json);
diff your run against them (allow ~0.05 tolerance across hardware / PyTorch
versions). Runs are deterministic on CPU given the seed. These are toy-scale
sanity numbers, not research results.

The run writes a HuggingFace checkpoint plus `metrics.json` to `runs/tiny_cpu/`:

```python
from beetlelm.pico_decoder.pico_decoder import PicoDecoderHF
from beetlelm.bytes import ByteTokenizerManager
model = PicoDecoderHF.from_pretrained("runs/tiny_cpu")
tok   = ByteTokenizerManager.from_pretrained("runs/tiny_cpu")
```

## Change one variable

Everything is driven by the YAML config or by `overrides` / CLI flags:

```bash
beetlelm-toy --config_path configs/tiny_cpu.yaml --seed 7 --max_steps 400
beetlelm-toy --config_path configs/tiny_cpu.yaml --d_model 256
```

```python
from beetlelm.toy import train
train("configs/tiny_cpu.yaml", overrides={
    "model":    {"d_model": 256, "n_layers": 6},
    "training": {"seed": 7, "max_steps": 400},
})
```

Point `data.train_file` at any UTF-8 text file to train on your own corpus.

## Using the full framework

The full trainer targets larger runs on real datasets and the architecture zoo
(pico, bgpt, MoE, SSM, geo-bilingual, H-Net, MambaByte), with continual-learning
extensions (EWC, MAML/SMLMT, replay, LoRA, layer growth, ...). Configs live under
`configs/` and launchers under `script/`:

```bash
beetlelm-train --config_path configs/pico_default.yaml
beetlelm-train --config_path configs/quickstart_fineweb.yaml --validate_only
```

Some paths need extra, non-PyPI dependencies (e.g. `architecture: hnet` needs
`mamba_ssm` / `flash_attn` / the upstream `hnet` package; DeepSpeed is
Linux-only) — these import lazily, so the base install stays clean. See the
notes at the bottom of `pyproject.toml`.

## Computational requirements

| run | config | params | steps | device | time | RAM |
|-----|--------|--------|-------|--------|------|-----|
| smoke    | `configs/smoke.yaml`    | 0.12M | 3   | CPU | ~3 s   | < 1 GB |
| tiny_cpu | `configs/tiny_cpu.yaml` | 0.72M | 200 | CPU | ~15 s  | < 1 GB |

The headline reproducible result runs on CPU. To scale up, raise `n_layers`,
`d_model`, `max_steps`, enlarge the corpus, and set `training.device: cuda`. The
full research configs assume multi-GPU nodes.

## Project layout

```
beetlelm/
├── src/beetlelm/
│   ├── pico_decoder/     # PicoDecoder architecture (+ HF wrapper)
│   ├── bgpt/ beetle_moe/ beetle_ssm/ geo_bilingual/ hnet_arch/ mambabyte/
│   │                     # the architecture zoo
│   ├── bilingual/        # bilingual data pipeline + tokenizer manager
│   ├── continual/        # EWC, MAML/SMLMT, replay, LoRA, layer growth, ...
│   ├── working_memory/ pre_pretraining/ checkpointing/ optim/ config/
│   ├── bytes/            # download-free byte / codepoint tokenizers
│   ├── training/         # the full research trainer
│   ├── toy/              # download-free CPU quickstart (beetlelm-toy)
│   └── cli/train.py      # `beetlelm-train` console entry point
├── configs/              # research configs + tiny_cpu.yaml / smoke.yaml
├── script/               # cluster launchers and helpers
├── data/toy_corpus.txt   # ~18 KB synthetic bilingual corpus
├── baselines/            # expected toy-run metrics
├── notebooks/quickstart.ipynb
├── tests/                # full test suite + test_toy_smoke.py (fast, CPU)
└── ci/ci.yml             # GitHub Actions workflow (see "Continuous integration")
```

## Testing

```bash
# fast, CPU, no downloads:
pytest -q tests/test_toy_smoke.py
```

The toy smoke suite trains the real model for a few steps on CPU and asserts the
loss is finite, that the model learns, that runs are deterministic, and that a
loadable checkpoint is written. The remaining `tests/` exercise the full stack
and need its heavy dependencies (and sometimes a GPU).

## Continuous integration

The GitHub Actions workflow lives at [`ci/ci.yml`](ci/ci.yml) (it validates the
toy quickstart on Python 3.10–3.12). It ships outside `.github/workflows/`
because the initial push token lacked the `workflow` OAuth scope. To activate
it, the maintainer runs once:

```bash
gh auth refresh -h github.com -s workflow
mkdir -p .github/workflows && git mv ci/ci.yml .github/workflows/ci.yml
git commit -m "Enable CI workflow" && git push
```

## Experiment tracking

Weights & Biases is **forced offline by default** (`WANDB_MODE=offline`,
`WANDB_DISABLED=true`, set on `import beetlelm`), so nothing phones home unless
you explicitly opt in by setting those environment variables yourself.

## Citation / DOI

If you use this software, please cite it (see [`CITATION.cff`](CITATION.cff)):

```
Salhan, Suchir. beetlelm: a framework for transparently training bilingual
language models. Version 0.1.0, 2026. https://github.com/BeetleLM/beetlelm
```

A `.zenodo.json` is included so the maintainer can connect the repository to
[Zenodo](https://zenodo.org) and mint a DOI on release: enable the repo in the
Zenodo GitHub integration, then publish a GitHub release — Zenodo archives the
tagged version and issues a DOI, which can then be added here as a badge. (The
DOI cannot be minted without the maintainer's Zenodo authorization.)

## License

[Apache-2.0](LICENSE).
