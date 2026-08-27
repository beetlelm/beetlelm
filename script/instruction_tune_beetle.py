# ============================================================
# Instruction-tune any BeetleLM Pico variant on learner data.
#
# Designed for Google Colab (A100 / L4) but runs anywhere with a
# CUDA GPU + bitsandbytes 4-bit support. Compatible with all beetle
# variants that share the Pico backbone (PicoDecoderHF).
#
# Required pins (run once at the top of the notebook):
#   !pip -q install -U "transformers>=4.44" "datasets" "accelerate" \
#       "peft>=0.10" "trl>=0.20" "bitsandbytes" sentencepiece
# ============================================================

import os
import types
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from transformers.modeling_outputs import CausalLMOutput, CausalLMOutputWithPast
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


# ---------------- CONFIG ----------------
MODEL_ID    = "Beetle-FineWeb2-24B/beetle-bilingual-l2-80-late-b5-fineweb-deu-eng"
REVISION    = "main"
DATASET_ID  = "lilygoulder/learner_tuning_GER"
OUTPUT_DIR  = "/content/beetle_lora_instruct"

MAX_SEQ_LEN   = 1024
TRAIN_SAMPLES = 2000
EPOCHS        = 1
BATCH         = 1
GRAD_ACC      = 8
LR            = 2e-4


# ---------------- TOKENIZER ----------------
# BeetleLM tokenizer ships [PAD]/[UNK]/[SEP]/[CLS]/[MASK] but no BOS/EOS.
# We promote [SEP] -> eos and [CLS] -> bos so SFTTrainer can append a stop
# token and generation can terminate naturally. The pad != eos invariant
# is required so the data collator masks pads to -100 without masking real
# terminators.
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, revision=REVISION, trust_remote_code=True
)
tokenizer.padding_side = "right"

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
if tokenizer.eos_token is None:
    tokenizer.eos_token = tokenizer.sep_token or tokenizer.unk_token
if tokenizer.bos_token is None and tokenizer.cls_token is not None:
    tokenizer.bos_token = tokenizer.cls_token

assert tokenizer.pad_token_id is not None, "pad_token_id required"
assert tokenizer.eos_token_id is not None, "eos_token_id required for SFT termination"
assert tokenizer.pad_token_id != tokenizer.eos_token_id, (
    "pad must differ from eos so DataCollatorForLanguageModeling can mask "
    "padding to -100 without masking real EOS targets"
)

print(f"[tok] pad={tokenizer.pad_token!r}({tokenizer.pad_token_id}) "
      f"eos={tokenizer.eos_token!r}({tokenizer.eos_token_id}) "
      f"bos={tokenizer.bos_token!r}({tokenizer.bos_token_id})")


# ---------------- MODEL (real 4-bit) ----------------
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    revision=REVISION,
    trust_remote_code=True,
    quantization_config=bnb_cfg,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.config.use_cache = False
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.bos_token_id = tokenizer.bos_token_id
# HF generate() reads num_hidden_layers / hidden_size / num_attention_heads
# for cache setup; Pico names them n_layers / d_model / attention_n_heads.
# Mirror them so generate() works without forking the config class.
if not hasattr(model.config, "num_hidden_layers"):
    model.config.num_hidden_layers = model.config.n_layers
if not hasattr(model.config, "hidden_size"):
    model.config.hidden_size = model.config.d_model
if not hasattr(model.config, "num_attention_heads"):
    model.config.num_attention_heads = model.config.attention_n_heads
if not hasattr(model.config, "num_key_value_heads"):
    model.config.num_key_value_heads = getattr(
        model.config, "attention_n_kv_heads", model.config.attention_n_heads
    )
if getattr(model, "generation_config", None) is not None:
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.bos_token_id = tokenizer.bos_token_id

print(f"[model] vocab_size={model.config.vocab_size} "
      f"n_layers={model.config.n_layers} d_model={model.config.d_model} "
      f"max_seq_len={getattr(model.config, 'max_seq_len', 'n/a')}")

# Pico does not implement gradient checkpointing — keep it OFF everywhere.
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()


# ---------------- LOSS PATCH ----------------
# Upstream PicoDecoderHF.forward computes loss via clamp(0, vocab_size-1)
# instead of ignore_index=-100, which means TRL's -100 padding mask is
# converted to token-id 0 and pad positions contribute full loss.
# We bind a corrected forward onto this model instance only.
# IMPORTANT: must run BEFORE get_peft_model wraps the model.
def _patch_pico_loss(target):
    if not (hasattr(target, "embedding_proj") and hasattr(target, "de_embedding_proj")):
        return target  # not a pico variant — leave it alone

    def forward(self, input_ids=None, past_key_values=None,
                use_cache=False, labels=None, **kwargs):
        # Newer transformers passes a DynamicCache object instead of the
        # legacy tuple-of-tuples format Pico expects. Convert at the
        # boundary; treat empty caches as None.
        if past_key_values is not None and not isinstance(past_key_values, tuple):
            if hasattr(past_key_values, "to_legacy_cache"):
                past_key_values = past_key_values.to_legacy_cache()
            if past_key_values is not None and len(past_key_values) == 0:
                past_key_values = None
        seq_len = input_ids.shape[-1]
        h = self.embedding_proj(input_ids)
        start_pos = 0 if past_key_values is None else past_key_values[0][0].shape[1]
        mask = None
        if seq_len > 1:
            mask = torch.full((seq_len, seq_len), float("-inf"), device=h.device)
            mask = torch.triu(mask, diagonal=1)
            if past_key_values is not None:
                mask = torch.hstack(
                    [torch.zeros((seq_len, start_pos), device=h.device), mask]
                )
        cached = () if use_cache else None
        for idx, layer in enumerate(self.layers):
            lp = past_key_values[idx] if past_key_values is not None else None
            h, lc = layer(h, mask=mask, past_key_values=lp, use_cache=use_cache)
            if use_cache:
                cached += (lc,)
        logits = self.de_embedding_proj(self.output_norm(h)).float()
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        if use_cache:
            return CausalLMOutputWithPast(
                loss=loss, logits=logits, past_key_values=cached
            )
        return CausalLMOutput(loss=loss, logits=logits)

    target.forward = types.MethodType(forward, target)
    return target


_patch_pico_loss(model)


# ---------------- DATASET ----------------
ds = load_dataset(DATASET_ID)
split_name = list(ds.keys())[0]
train_ds = ds[split_name].select(range(min(TRAIN_SAMPLES, len(ds[split_name]))))
print("Columns:", train_ds.column_names)

PROMPT_KEYS = [
    "instruction", "prompt", "input", "question", "source",
    "german", "prompt_de", "prompt_en", "instruction_de", "instruction_en",
]
ANSWER_KEYS = [
    "output", "response", "answer", "target",
    "english", "response_en", "response_de", "answer_en", "answer_de",
]


def row_to_text(example):
    lower = {k.lower(): v for k, v in example.items()}
    prompt = next((str(lower[k]) for k in PROMPT_KEYS if lower.get(k) is not None), None)
    answer = next((str(lower[k]) for k in ANSWER_KEYS if lower.get(k) is not None), None)
    if prompt is None:
        prompt = str(example[train_ds.column_names[0]])
    if answer is None:
        answer = str(example[train_ds.column_names[1]]) if len(train_ds.column_names) > 1 else ""
    text = f"### Instruction:\n{prompt}\n\n### Response:\n{answer}{tokenizer.eos_token}"
    return {"text": text}


train_ds = train_ds.map(row_to_text, remove_columns=train_ds.column_names)
print("Example:", train_ds[0]["text"][:300], "...")


# ---------------- LoRA ----------------
# Pico module names (defined in src/pico_decoder/pico_decoder.py):
#   Attention: q_proj / k_proj / v_proj / o_proj
#   SwiGLU MLP: w_0 / w_1 / w_2  (NOT gate_proj/up_proj/down_proj)
# These names are stable across all beetle variants on the pico backbone.
# We do NOT target de_embedding_proj (LM head): its .float() output would
# clash with bf16 LoRA deltas under bnb-4bit, and it dominates param count.
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "w_0", "w_1", "w_2",
]

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=TARGET_MODULES,
)


# ---------------- TRAIN ----------------
# Clamp max_length against the model's RoPE-precomputed max_seq_len so we
# never emit positions beyond what RoPE knows about.
effective_max_len = min(MAX_SEQ_LEN, getattr(model.config, "max_seq_len", MAX_SEQ_LEN))

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH,
    gradient_accumulation_steps=GRAD_ACC,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    logging_steps=10,
    save_strategy="epoch",
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    report_to="none",
    max_length=effective_max_len,         # TRL >= 0.20 (renamed from max_seq_length)
    dataset_text_field="text",
    packing=False,
    gradient_checkpointing=False,         # Pico has no gc hooks
    optim="paged_adamw_8bit",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_ds,
    args=training_args,
    processing_class=tokenizer,           # TRL >= 0.20 (renamed from tokenizer=)
    peft_config=peft_config,
)
trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)


# ---------------- GENERATE ----------------
# IMPORTANT: KV-cache is disabled for generation.
#
# Why: PicoDecoderHF.prepare_inputs_for_generation does NOT slice input_ids
# to the last token when a cache is present, so cached incremental decoding
# is incorrect (each step would re-encode the full prefix and concatenate
# its keys onto the existing cache, producing duplicate-length key tensors).
# Until upstream pico_decoder.py is fixed, cacheless generation is the only
# correct path. It is slower per step but correct, and for small Pico
# variants on short prompts the wall-clock cost is negligible.
trainer.model.config.use_cache = False
if getattr(trainer.model, "generation_config", None) is not None:
    trainer.model.generation_config.use_cache = False
trainer.model.eval()

prompts = [
    "Translate into English: Ich habe gestern meine Hausaufgaben gemacht.",
    "Correct this learner English: She go to school yesterday.",
    "Write a short advanced English paragraph about climate change.",
]

device = next(trainer.model.parameters()).device

for p in prompts:
    full_prompt = f"### Instruction:\n{p}\n\n### Response:\n"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

    # PicoDecoderHF.prepare_inputs_for_generation only forwards
    # input_ids / past_key_values / use_cache. Newer transformers versions
    # validate generate() kwargs strictly and reject unused ones, so we do
    # NOT pass attention_mask here — Pico does not consume it.
    with torch.no_grad():
        out_ids = trainer.model.generate(
            input_ids=inputs.input_ids,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            use_cache=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    out = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    print("\n" + "=" * 80)
    print("PROMPT:", p)
    print(out)
