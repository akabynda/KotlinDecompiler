from __future__ import annotations

import gc
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset, DatasetDict
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

from global_config import GLOBAL_SEED
from model_train import config
from model_train.config import DATASET
from model_train.config import WARMUP
from utils.make_example import make_example
from utils.clear_hf_cache import clear_hf_cache

MODEL: str = config.MODEL
GLOBAL_SEED: int = GLOBAL_SEED
RUN_DIR: Path = Path(config.RUNS_DIR) / "full_finetune"

raw_ds = load_dataset(DATASET)

raw_ds = DatasetDict({
    split: raw_ds[split].map(make_example)
    for split in raw_ds
})

TRAIN_EPOCHS = 4
GRAD_ACC = 32
SEQ_LEN_PERCENTILE = 95
LORA_CFG = dict(r=32, lora_alpha=128, lora_dropout=0.07, bias="lora_only", target_modules="all-linear")
CLIP_NORM = 0.75
WEIGHT_DECAY = 0.03
LEARNING_RATE = 1e-4

set_seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

print(f"Loading tokenizer for {MODEL} ...")
tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print("Tokenizing ...")
tok_ds = raw_ds.map(
    lambda b: tok(b["text"], truncation=True),
    remove_columns=["text", "kt_path"]
)

print("Computing target sequence length percentile ...")
lengths = [len(rec["input_ids"]) for rec in tok_ds["train"]]
SEQ_LEN = int(np.percentile(lengths, SEQ_LEN_PERCENTILE))
print(f"95‑th percentile: {SEQ_LEN} tokens")

print("Truncating sequences ...")
train_ds = tok_ds["train"].map(
    lambda ex: {
        "input_ids": ex["input_ids"][:SEQ_LEN],
        "attention_mask": ex["attention_mask"][:SEQ_LEN],
    },
    remove_columns=["input_ids", "attention_mask"],
)

print("Loading model", MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    quantization_config=bnb_cfg,
    trust_remote_code=True,
    device_map="auto",
    use_cache=False,
)

model.enable_input_require_grads()
model.gradient_checkpointing_enable()

print("Applying LoRA adapters ...")
model = get_peft_model(model, LoraConfig(**LORA_CFG))

RUN_DIR.mkdir(parents=True, exist_ok=True)

training_args = TrainingArguments(
    output_dir=str(RUN_DIR),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=GRAD_ACC,
    num_train_epochs=TRAIN_EPOCHS,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="linear",
    warmup_ratio=WARMUP,
    max_grad_norm=CLIP_NORM,
    weight_decay=WEIGHT_DECAY,
    logging_steps=100,
    save_strategy="epoch",
    seed=GLOBAL_SEED,
    optim="adamw_torch_fused",
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

print("Starting training ...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    data_collator=data_collator,
)

trainer.train()

print("Saving model & tokenizer ...")
(model_save := RUN_DIR / "model").mkdir(parents=True, exist_ok=True)
model.save_pretrained(model_save)

tok.save_pretrained(RUN_DIR / "tokenizer")

print("Fine‑tuning finished! Artifacts saved to", RUN_DIR)

gc.collect()
torch.cuda.empty_cache()
clear_hf_cache()
