# 🪲 **BeetleLM**

BeetleLM is a lightweight framework for transparently training **bilingual language models**—from human-scale (~100M training tokens) all the way up to large multilingual budgets (24B tokens)—with built-in rich checkpointing that captures activations, gradients, circuits, and the training batch itself, enabling detailed **learning-dynamics** research into second-language acquisition, curricula, and continual learning.

Our **suite of pre-trained models** is already publicly available on our [Hugging Face organization](https://huggingface.co/Beetle-HumanScale), and a family of dedicated companion libraries—[**beetle-data**](https://github.com/beetlelm/beetle-data) for corpus preparation, [**beetle-analyze**](https://github.com/beetlelm/beetle-analyze) for checkpoint studies, and [**beetle-explorer**](https://github.com/beetlelm/beetle-explorer) for interactive browsing—is fully released for deeper investigation.

> For a **detailed run-through**, check out the **full tutorial** on our website at [beetlelm.github.io/docs.html#beetlelm](https://beetlelm.github.io/docs.html#beetlelm).

---

## **Key Features**

1. **PicoDecoder: LLaMA-style Transformer Architecture**
   - A 125M-parameter decoder with **RMSNorm**, **RoPE**, **grouped-query attention** (12 query heads / 1 KV head), and **SwiGLU** activations, trained in BF16 at sequence length 512
   - **Width is the only scale knob**: `d_model ∈ {96, 384, 768, 1536}` gives the `tiny`/`small`/`medium`/`large` configs—the 125M "medium" (14 layers, `d_model` 768) is used for every reported run
   - Ships with an **architecture zoo** alongside the decoder: **BGPT** (GPT-2-style), **BeetleMoE** (top-k mixture-of-experts with ST-MoE auxiliary losses), and **BeetleSSM** (Mamba-style state-space model)

2. **Bilingual-First Tokenisation**
   - The **MultilingualTokenizerManager** trains **BPE**, **Unigram**, and **SuperBPE** tokenisers on top of Hugging Face `tokenizers`
   - Adds **per-language control tokens** (e.g. `<lang:en>`), scales the vocabulary with the number of languages, and uses **equal-compression sampling** so no language dominates the merges
   - Vocabulary size and target compression are set from YAML; the shared bilingual vocabulary is **50,304** (≈50K) tokens

3. **Curriculum Control (`DataMixConfig`)**
   - Expresses a **continuous, time-varying mixture** `w(t)` over the language simplex, with an equivalent discrete-phase representation
   - Five built-in curricula: **B1 Balanced** (50:50 throughout), **B2 Simultaneous** (L1-only for the first 30%, sigmoid onset, ~25% L2), **B3 Sequential** (L1 for the first third, sigmoid to L2-dominant, ~33.5% L2), **B4 Episodic** (L2 throughout but clustered in bursts, 80:20, 20% L2), and **B5 Late** (L1-only for the first 50%, sigmoid over 50–80%, ~10% L2)
   - Custom curricula are defined directly in YAML

4. **Comprehensive Checkpoints**
   - A **log2 schedule** (restarting at each exposure phase) yields ~**30 checkpoints per model**
   - Each checkpoint saves **gradients**, **activations**, **circuits**, **the training batch**, and **evaluation metrics**, in both **PyTorch** and **Hugging Face** formats, auto-uploaded to the Hub

5. **Continual-Learning Toolkit**
   - **15 strategies across 5 families**: parameter regularisation (EWC, SI, MAS), replay (LAMOL, InsCL, episodic), meta-learning (FO-MAML, SMLMT), distillation (KD, self-distillation, FitNets), and architectural/auxiliary methods (TILT, exposure-LR plasticity, ALiBi-scheduler, Shuffle-Dyck, EWC+MAML)
   - Reported runs use **EWC**, **LAMOL**, and **exposure-LR plasticity**

6. **Research Ready**
   - Minimal, well-documented, **YAML-driven** code suitable for **forking and tailoring**
   - GPU-trained on **8×A100 (64GB)** with `DDP`, logging perplexity and evaluation metrics throughout training
   - Works seamlessly with [beetle-analyze](https://github.com/beetlelm/beetle-analyze) for learning dynamics analysis

---

## **Training Philosophy**

All models in the Beetle suite (both pre-trained and user-trained):

- Employ **identical architectures** and **optimizer settings**
- **Release their data in the order it was seen**, so runs are reproducible token-for-token
- Automatically log **rich checkpoint data** (gradients, activations, circuits, the training batch)
- Facilitate **direct cross-curriculum and cross-language comparisons**

This uniformity means you can isolate the **language curriculum** (or the tokeniser, or the continual-learning strategy) as the primary variable, giving you clearer insights into **how bilingual models acquire a second language over the course of training**.

---

## **Resources**

- **Pre-trained Models**, publicly hosted on [Hugging Face](https://huggingface.co/Beetle-HumanScale): **285 bilingual + 45 monolingual** models spanning **21 L1 languages**
- **Datasets**: human-scale **BabyBabelLM** (100M tokens), plus **FineWeb-2** (L1) and **FineWeb-Edu** (English) for larger budgets, with **13-gram decontamination** and data released in the order seen
- **Extensive Checkpoints** logging gradient, activation, and circuit snapshots at ~30 points per run
- **Evaluation Metrics** (perplexity and more) tracked at each checkpoint

Reported training runs (Table 4):

| Setting | Tokens | Model | Layers | `d_model` | Heads (Q/KV) | FFN | Seq | Vocab | Optim | LR | WD | Warmup | Micro-batch | Eff. batch | Tokens/step | Max steps | Precision | Parallelism | GPUs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100M | 100M | PicoDecoder 125M | 14 | 768 | 12 / 1 | 3072 | 512 | 50K | AdamW | 5e-4 | 0.01 | 500 | 64 | 64 | 32,768 | 10,000 | BF16 | DDP | 8×A100 |
| 2B | ~2B | PicoDecoder 125M | 14 | 768 | 12 / 1 | 3072 | 512 | 50K | AdamW | 5e-4 | 0.01 | 381 | 64 | 512 | 262,144 | 7,629 | BF16 | DDP | 8×A100 |
| 24B | ~24B | PicoDecoder 125M | 14 | 768 | 12 / 1 | 3072 | 512 | 50K | AdamW | 5e-4 | 0.01 | 625 | 64 | 512 | 262,144 | 91,552 | BF16 | DDP | 8×A100 |

> **A note on scale.** Only `Beetle-FineWeb-24B-4` and `-24B-5` reached the full 24B budget; earlier "24B" runs are closer to ~3B, and earlier "2B" runs closer to ~300M. The 100M and 2B 125M-parameter models each train in under an hour on a single A100; a full 24B run takes ~5 hours on 8×A100.

---

## **Core Components**

- **PicoDecoder Model & Architecture Zoo**
  - LLaMA-style auto-regressive transformer: RMSNorm, RoPE, grouped-query attention, SwiGLU
  - Alternative backbones: **BGPT**, **BeetleMoE**, **BeetleSSM**—all config-selectable

- **Bilingual Tokenisation**
  - `MultilingualTokenizerManager` for BPE / Unigram / SuperBPE
  - Per-language control tokens, vocabulary scaling, and equal-compression sampling

- **Curriculum Engine (`DataMixConfig`)**
  - Continuous `w(t)` language mixtures over the simplex, with the five built-in B1–B5 curricula and arbitrary YAML-defined schedules

- **Training & Checkpointing**
  - Automatic storage of model and optimizer states on a log2 schedule
  - Periodic hooks for saving **learning dynamics** (gradients, activations, circuits, training batch)
  - Logging to Weights & Biases, with auto-upload of checkpoints to Hugging Face

- **Continual Learning**
  - 15 strategies across parameter-regularisation, replay, meta-learning, distillation, and architectural/auxiliary families

- **Config-Driven Setup**
  - Specify architecture, tokeniser, curriculum, optimizer, continual-learning strategy, and logging settings in YAML
  - Straightforward to extend or modify

---

## **Quick Start**

1. **Install**

   The quickest route is pip:
   ```bash
   # once released to PyPI
   pip install beetlelm

   # always works (installs straight from GitHub)
   pip install git+https://github.com/beetlelm/beetlelm.git
   ```

   Or from source, for development:
   ```bash
   git clone https://github.com/beetlelm/beetlelm
   cd beetlelm
   pip install -e .
   ```
   This installs two console scripts: `beetlelm-train` (the full research trainer) and `beetlelm-toy` (a small, download-free example you can run to check the install).

2. **Configure Environment**

   Create a `.env` file at the root with your Hugging Face and Weights & Biases tokens:
   ```bash
   export HF_TOKEN=your_huggingface_token
   export WANDB_API_KEY=your_wandb_key
   ```

3. **Train Your Model Suite**

   - Edit (or pick) a config file under `configs/` to specify your architecture, tokeniser, and curriculum.
   - Then run:
     ```bash
     beetlelm-train --config_path configs/examples/demo.yaml
     ```
   - This launches training, automatically checkpointing states and saving learning-dynamics data.
   - To sanity-check the install first, run the local example:
     ```bash
     beetlelm-toy
     ```

4. **Explore Checkpoints**
   - By default, checkpoints are stored under your configured run directory.
   - Each checkpoint contains:
     - **Model state** (PyTorch + Hugging Face formats)
     - **Optimizer state**
     - **Gradients, activations, and circuits** for interpretability
     - **The training batch** and **evaluation logs** (e.g. perplexity)

---

## **Repository Structure**

- **`src/beetlelm/pico_decoder/`**
  - Core LLaMA-style decoder (GQA, RMSNorm, RoPE, SwiGLU)

- **`src/beetlelm/bgpt/`**, **`src/beetlelm/beetle_moe/`**, **`src/beetlelm/beetle_ssm/`**
  - The alternative architecture zoo: GPT-2-style, mixture-of-experts, and state-space backbones

- **`src/beetlelm/bilingual/`**
  - Bilingual tokenisation (`MultilingualTokenizerManager`), data mixing, and curriculum logic

- **`src/beetlelm/continual/`**
  - The 15 continual-learning strategies across 5 families

- **`src/beetlelm/checkpointing/`**
  - Logic for saving model states, gradients, activations, and circuits, plus Hugging Face upload

- **`src/beetlelm/config/`**
  - Dataclass-based config system (model, tokeniser, curriculum, training, checkpointing, logging)

- **`src/beetlelm/training/`**
  - Main training loop, distributed/multi-node orchestration, metrics, and checkpoint saving

- **`src/beetlelm/toy/`**
  - The download-free `beetlelm-toy` example for checking the install

- **`configs/`**
  - Ready-to-run YAML configs (curricula B1–B5, per-language and shared-vocabulary sweeps, condition sweeps)

- **`script/`**
  - Data-preparation, evaluation, and launch scripts

- **`tests/`**
  - Unit tests

---

## **Part of the Beetle ecosystem**

BeetleLM is the training half of a wider toolkit for transparent bilingual LM research:

- [**beetle-data**](https://github.com/beetlelm/beetle-data) — corpus preparation, mixing, and decontamination
- [**beetle-analyze**](https://github.com/beetlelm/beetle-analyze) — checkpoint analysis of gradients, activations, and circuits
- [**beetle-explorer**](https://github.com/beetlelm/beetle-explorer) — interactive browsing of models, curricula, and results
- **Website**: [beetlelm.github.io](https://beetlelm.github.io)
- **Models**: [huggingface.co/Beetle-HumanScale](https://huggingface.co/Beetle-HumanScale)
- **Paper**: [arxiv.org/abs/2609.22633](https://arxiv.org/abs/2609.22633)

---

## **License**

BeetleLM is open-source under the [Apache License 2.0](LICENSE).

---

## **Citation**

If you use **BeetleLM** in your research, please cite our paper:

```bibtex
@misc{beetlelm,
    title  = {BeetleLM: Transparently Training Bilingual Language Models for Learning-Dynamics Research},
    author = {Salhan, Suchir and others},
    year   = {2026},
    eprint = {2609.22633},
    archivePrefix = {arXiv},
    primaryClass  = {cs.CL},
    url    = {https://arxiv.org/abs/2609.22633}
}
```

**Happy Training!** For more information and tutorials, visit our website at [beetlelm.github.io](https://beetlelm.github.io).
