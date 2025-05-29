import gc
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset, DatasetDict, Dataset
from peft import PeftModel
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

from global_config import GLOBAL_SEED
from model_train import config
from model_train.make_model import make_model
from utils.clear_hf_cache import clear_hf_cache
from utils.make_prompt import make_prompt, wrap_as_row


class FullFineTuner:
    """
    Fine-tunes a language model using the full dataset with LoRA and 4-bit quantization.
    """

    def __init__(self) -> None:
        self.model_name: str = config.MODEL
        self.global_seed: int = GLOBAL_SEED
        self.run_dir: Path = Path(config.RUNS_DIR) / "full_finetune"

        self._set_seed()
        self.tokenizer = self._load_tokenizer()
        self.dataset: DatasetDict = self._load_and_prepare_dataset()
        self.train_dataset: Dataset = self._prepare_train_dataset()
        self.model: Any = self._load_and_prepare_model()

    def _set_seed(self) -> None:
        """
        Set seeds for reproducibility.
        """
        set_seed(self.global_seed)
        random.seed(self.global_seed)

    def _load_tokenizer(self) -> AutoTokenizer:
        """
        Load and prepare the tokenizer.
        """
        print(f"Loading tokenizer for {self.model_name} ...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _tokenize(self, ex: dict[str, Any]) -> dict[str, Any]:
        return self.tokenizer(ex["text"], truncation=True)

    def _load_and_prepare_dataset(self) -> DatasetDict:
        """
        Load the dataset and wrap Kotlin code as prompts.
        """
        raw_ds = load_dataset(config.DATASET)
        wrapped_ds = DatasetDict({
            split: raw_ds[split].map(lambda ex: make_prompt(wrap_as_row(ex)))
            for split in raw_ds
        })

        print("Tokenizing dataset ...")
        tokenized_ds = wrapped_ds.map(
            self._tokenize,
            remove_columns=["text", "kt_path"]
        )
        return tokenized_ds

    def _prepare_train_dataset(self) -> Dataset:
        """
        Prepare the training dataset: truncate to percentile length.
        """
        print("Computing target sequence length percentile ...")
        lengths = [len(rec["input_ids"]) for rec in self.dataset["train"]]
        seq_len = int(np.percentile(lengths, config.SEQ_LEN_PERCENTILE))
        print(f"95‑th percentile: {seq_len} tokens")

        print("Truncating sequences ...")
        train_ds = self.dataset["train"].map(
            lambda ex: {
                "input_ids": ex["input_ids"][:seq_len],
                "attention_mask": ex["attention_mask"][:seq_len],
            },
            remove_columns=["input_ids", "attention_mask"]
        )
        self.seq_len: int = seq_len
        return train_ds

    def _load_and_prepare_model(self) -> PeftModel:
        """
        Load and prepare the quantized model with LoRA.
        """
        print("Loading model ...")
        model = make_model(
            r=config.LORA_CFG["r"],
            alpha=config.LORA_CFG["lora_alpha"],
            dropout=config.LORA_CFG["lora_dropout"]
        )
        model.print_trainable_parameters()
        return model

    def train(self) -> None:
        """
        Fine-tune the model.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(self.run_dir),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=config.GRAD_ACC,
            num_train_epochs=config.TRAIN_EPOCHS,
            learning_rate=config.LEARNING_RATE,
            lr_scheduler_type="linear",
            warmup_ratio=config.WARMUP,
            max_grad_norm=config.CLIP_NORM,
            weight_decay=config.WEIGHT_DECAY,
            logging_steps=100,
            save_strategy="epoch",
            seed=self.global_seed,
            optim="adamw_torch_fused",
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        print("Starting training ...")
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            data_collator=data_collator
        )
        trainer.train()

        self._save_artifacts()

    def _save_artifacts(self) -> None:
        """
        Save the model and tokenizer to the output directory.
        """
        print("Saving model & tokenizer ...")
        (model_save := self.run_dir / "model").mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(model_save), save_adapter=True)
        self.tokenizer.save_pretrained(self.run_dir / "tokenizer")
        print("Fine‑tuning finished! Artifacts saved to", self.run_dir)

        gc.collect()
        torch.cuda.empty_cache()
        clear_hf_cache()


def main() -> None:
    """
    Entry point for full fine-tuning.
    """
    finetuner = FullFineTuner()
    finetuner.train()


if __name__ == "__main__":
    main()
