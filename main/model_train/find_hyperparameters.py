import csv
import gc
import json
import random
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import optuna
import torch
from collect.metrics.common import structural, lm_metrics, load_lm, entropy_metrics
from datasets import load_dataset, DatasetDict, tqdm
from transformers import (
    AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, set_seed
)

from main.collect.process_models.shared import Row
from global_config import GLOBAL_SEED, FEATURES
from main.model_train.make_model import make_model
from train.config import (
    VAL_SUBSET_SIZE, RUNS_DIR, DB_URI, STUDY_NAME, MODEL,
    TRAIN_SUBSET_SIZE, METRIC_TIMEOUT, DATASET
)
from main.utils.clear_hf_cache import clear_hf_cache
from main.utils.extract_kotlin import extract_kotlin
from main.utils.gen_len_stats import get_max_new
from main.utils.make_prompt import to_bytecode, make_prompt, wrap_as_row


class HyperparameterTuner:
    """
    Class to handle hyperparameter search for Kotlin decompilation task
    using Optuna, Transformers, PEFT (LoRA) and custom metrics.
    """

    def __init__(self) -> None:
        self.avg_sec_len = None
        self.base_train = None
        self.tokenized_ds = None
        set_seed(GLOBAL_SEED)
        random.seed(GLOBAL_SEED)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.raw_dataset = load_dataset(DATASET)
        self.rows_for_stats = [Row(r["kt_path"], r["kt_source"], to_bytecode(r)) for r in self.raw_dataset["train"]]
        self.p_uni, self.p_bi, self.p_left = load_lm()
        self.prepare_datasets()

    def prepare_datasets(self) -> None:
        """
        Prepares the dataset: wraps Kotlin code, tokenizes, and selects training subset.
        """
        self.raw_dataset = DatasetDict({
            split: self.raw_dataset[split].map(lambda ex: make_prompt(wrap_as_row(ex)))
            for split in self.raw_dataset
        })

        self.tokenized_ds = self.raw_dataset.map(
            lambda b: self.tokenizer(b["text"], truncation=True),
            remove_columns=["text", "kt_path"]
        )

        self.base_train = self.tokenized_ds["train"].shuffle(seed=GLOBAL_SEED).select(range(TRAIN_SUBSET_SIZE))
        self.avg_sec_len = int(np.percentile(
            [len(r["input_ids"]) for r in self.tokenized_ds["train"]],
            95
        ))

    def generate_test_cases(self, model: Any, test_records: List[Dict[str, Any]],
                            out_path: Path, max_new_tokens: int) -> None:
        """
        Generates Kotlin bytecode for test cases and saves as JSONL.
        """
        batch_size = 4
        out_path.parent.mkdir(parents=True, exist_ok=True)

        for i in tqdm(range(0, len(test_records), batch_size), desc="Generating"):
            batch = test_records[i:i + batch_size]
            prompts = [rec["text"].split("<|im_start|>assistant")[0] for rec in batch]

            try:
                inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                with out_path.open("a", encoding="utf-8") as f:
                    for rec, out_ids in zip(batch, outputs):
                        f.write(json.dumps({
                            "kt_path": rec["kt_path"],
                            "our": extract_kotlin(self.tokenizer.decode(out_ids, skip_special_tokens=False)),
                            "kt_source": extract_kotlin(rec["text"].split("<|im_end|>\n")[-2])
                        }) + "\n")
            except RuntimeError as e:
                if 'out of memory' in str(e).lower() and batch_size > 1:
                    batch_size = max(1, batch_size // 2)
                    torch.cuda.empty_cache()
                    continue
                raise
            finally:
                torch.cuda.empty_cache()
                gc.collect()

    def compute_metrics(self, gen_jsonl: Path, csv_path: Path, metric_list: List[str]) -> None:
        """
        Computes metrics in parallel and saves them to CSV.
        """
        tasks: List[Tuple[str, str, str, str, List[str]]] = []
        with gen_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                for field in ["our", "kt_source"]:
                    tasks.append((rec["kt_path"], field, rec[field], rec["kt_source"], metric_list))

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["kt_path", "model"] + metric_list)
            with ThreadPoolExecutor(max_workers=cpu_count() - 1) as ex:
                futures = [ex.submit(self.compute_row, t) for t in tasks]
                for fut in tqdm(futures):
                    try:
                        row = fut.result(timeout=METRIC_TIMEOUT)
                        if row:
                            writer.writerow(row)
                    except Exception:
                        continue

    def compute_row(self, args: Tuple[str, str, str, str, List[str]]) -> Optional[List[Any]]:
        """
        Compute metrics for a single code snippet.
        """
        kt_path, field, code, orig_code, metric_list = args
        try:
            s = structural(code)
            lm = lm_metrics(self.p_uni, self.p_bi, self.p_left, code)
            ent = entropy_metrics(orig_code, code)
            return [kt_path, field] + [s.get(m, lm.get(m, ent.get(m, 0.0))) for m in metric_list]
        except Exception:
            return None

    def evaluate_metrics(self, csv_path: Path) -> float:
        """
        Evaluate final distance metric for the run.
        """
        rows = list(csv.reader(csv_path.open()))
        header = rows[0]
        idx = {name: i for i, name in enumerate(header)}
        vals = {m: {f: [] for f in FEATURES} for m in ["kt_source", "our"]}

        for row in rows[1:]:
            model = row[1]
            if model not in vals:
                continue
            for f in FEATURES:
                vals[model][f].append(float(row[idx[f]]))

        med_kt = [np.median(vals["kt_source"][f]) for f in FEATURES]
        med_our = [np.median(vals["our"][f]) for f in FEATURES]
        dist = max(abs(a - b) for a, b in zip(med_kt, med_our))
        coverage = len(vals["our"][FEATURES[0]]) / (len(rows) - 1)
        return float('inf') if coverage == 0 or np.isnan(dist) else dist / coverage

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna study.
        """
        r = trial.suggest_categorical("r", [8, 16])
        alpha = trial.suggest_categorical("lora_alpha", [16, 32])
        dropout = trial.suggest_float("lora_dropout", 0.0, 0.1)
        grad_acc = trial.suggest_categorical("grad_acc", [32, 64])
        clip = trial.suggest_float("clip", 0.1, 1.0)
        wd = trial.suggest_float("weight_decay", 0.0, 0.1)

        max_new = get_max_new(self.rows_for_stats, self.tokenizer)
        run_dir = RUNS_DIR / f"trial_{trial.number}"

        model = make_model(r, alpha, dropout)
        train_ds = self.base_train.map(lambda ex: {
            "input_ids": ex["input_ids"][:self.avg_sec_len],
            "attention_mask": ex["attention_mask"][:self.avg_sec_len]
        }, remove_columns=self.base_train.column_names)

        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=str(run_dir),
                per_device_train_batch_size=1,
                gradient_accumulation_steps=grad_acc,
                num_train_epochs=4,
                learning_rate=1e-4,
                lr_scheduler_type="linear",
                warmup_ratio=0.1,
                max_grad_norm=clip,
                weight_decay=wd,
                logging_steps=50,
                save_strategy="no",
                seed=GLOBAL_SEED,
                optim="adamw_torch_fused"
            ),
            train_dataset=train_ds,
            data_collator=DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        )

        trainer.train()
        model.save_pretrained(run_dir / "model")

        test_subset = sorted(list(self.raw_dataset["test"]), key=lambda r: len(r["text"]))[:VAL_SUBSET_SIZE]
        gen_path = run_dir / "test_gen.jsonl"
        self.generate_test_cases(model, test_subset, gen_path, max_new)

        with gen_path.open("r", encoding="utf-8") as f:
            first_line = json.loads(f.readline())
        self.compute_metrics(gen_path, run_dir / "metrics.csv", list(
            set(structural(first_line["our"])) |
            set(lm_metrics(self.p_uni, self.p_bi, self.p_left, first_line["our"])) |
            set(entropy_metrics(first_line["kt_source"], first_line["our"]))
        ))

        score = self.evaluate_metrics(run_dir / "metrics.csv")
        trial.report(score, step=0)

        del model
        torch.cuda.empty_cache()
        gc.collect()
        clear_hf_cache()

        return score

    def run_study(self, n_trials: int = 30) -> None:
        """
        Runs the Optuna study for hyperparameter search.
        """
        optuna.logging.set_verbosity(optuna.logging.INFO)
        study = optuna.create_study(
            study_name=STUDY_NAME,
            direction="minimize",
            storage=DB_URI,
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=GLOBAL_SEED, multivariate=True, group=True),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=0, n_min_trials=5)
        )
        study.optimize(self.objective, n_trials=n_trials)

        Path(STUDY_NAME).mkdir(exist_ok=True)
        study.trials_dataframe().to_csv(Path(STUDY_NAME) / "optuna_trials.csv")
        print("Best hyperparameters:", study.best_params, "score:", study.best_value)
