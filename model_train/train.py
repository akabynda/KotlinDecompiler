from __future__ import annotations

import gc
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset, DatasetDict
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed, BitsAndBytesConfig,
)

from global_config import GLOBAL_SEED
from model_train import config
from model_train.config import DATASET, LORA_CFG, SEQ_LEN_PERCENTILE, GRAD_ACC, TRAIN_EPOCHS, LEARNING_RATE, CLIP_NORM, \
    WEIGHT_DECAY
from model_train.config import WARMUP
from utils.clear_hf_cache import clear_hf_cache
from utils.make_example import make_example, wrap_as_row

MODEL: str = config.MODEL
GLOBAL_SEED: int = GLOBAL_SEED
RUN_DIR: Path = Path(config.RUNS_DIR) / "full_finetune"

raw_ds = load_dataset(DATASET)

raw_ds = DatasetDict({
    split: raw_ds[split].map(lambda ex: make_example(wrap_as_row(ex)))
    for split in raw_ds
})

set_seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

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
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        ),
        trust_remote_code=True,
        device_map={"": torch.cuda.current_device()},
        use_cache=False,
    )

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LoraConfig(**LORA_CFG))

model.print_trainable_parameters()

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
model.save_pretrained(str(model_save), save_adapter=True)

tok.save_pretrained(RUN_DIR / "tokenizer")

print("Fine‑tuning finished! Artifacts saved to", RUN_DIR)

gc.collect()
torch.cuda.empty_cache()
clear_hf_cache()
