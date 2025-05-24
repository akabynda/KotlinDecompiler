import csv
import gc
import json
import random
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path

import datasets
import numpy as np
import optuna
import torch
from datasets import tqdm, DatasetDict
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments, Trainer, set_seed,
    DataCollatorForLanguageModeling
)

from collect.metrics.common import structural, lm_metrics, load_lm, entropy_metrics
from collect.process_models.process_model import to_bytecode
from collect.process_models.shared import Row
from global_config import GLOBAL_SEED
from train.config import (
    VAL_SUBSET_SIZE, RUNS_DIR, DB_URI, STUDY_NAME, MODEL,
    TRAIN_SUBSET_SIZE, METRIC_TIMEOUT, DATASET
)
from utils.clear_hf_cache import clear_hf_cache
from utils.extract_kotlin import extract_kotlin
from utils.gen_len_stats import get_max_new
from utils.make_example import make_example, wrap_as_row

set_seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
tok.pad_token = tok.eos_token
raw_ds = datasets.load_dataset(DATASET)

rows_for_stats = [Row(r["kt_path"], r["kt_source"], to_bytecode(r)) for r in raw_ds["train"]]

raw_ds = DatasetDict({
    split: raw_ds[split].map(lambda ex: make_example(wrap_as_row(ex)))
    for split in raw_ds
})

tok_ds = raw_ds.map(
    lambda b: tok(b["text"], truncation=True),
    remove_columns=["text", "kt_path"]
)

BASE_TRAIN = tok_ds["train"].shuffle(seed=GLOBAL_SEED).select(range(TRAIN_SUBSET_SIZE))
AVG_SEC_LEN = int(np.percentile([len(r["input_ids"]) for r in tok_ds["train"]], 95))

p_uni, p_bi, p_left = load_lm()


def make_model(r, alpha, dropout):
    base = AutoModelForCausalLM.from_pretrained(
        MODEL,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        ),
        trust_remote_code=True,
        device_map="auto",
        use_cache=False,
    )
    base = prepare_model_for_kbit_training(base)

    return get_peft_model(base, LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        bias="none", target_modules='all-linear', init_lora_weights="gaussian"
    ))


def generate_test_cases(model, tokenizer, test_records, out_path, max_new_tokens):
    batch_size = 4
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(0, len(test_records), batch_size), desc="Generating"):
        batch = test_records[i:i + batch_size]
        prompts = [rec["text"].split("<|im_start|>assistant")[0] for rec in batch]

        try:
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    top_k=None,
                    temperature=None,
                    top_p=None,
                )
            with out_path.open("a", encoding="utf-8") as f:
                for rec, out_ids in zip(batch, outputs):
                    f.write(json.dumps({
                        "kt_path": rec["kt_path"],
                        "our": extract_kotlin(tok.decode(out_ids, skip_special_tokens=False)),
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


def extract_metric_names(first_line):
    code = first_line.get("our")
    orig_code = first_line.get("kt_source")
    s = structural(code)
    lm = lm_metrics(p_uni, p_bi, p_left, code)
    ent = entropy_metrics(orig_code, code)
    return list(set(s) | set(lm) | set(ent))


def compute_metrics(gen_jsonl, csv_path, metric_list):
    tasks = []
    with gen_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            for field in ["our", "kt_source"]:
                tasks.append((rec["kt_path"], field, rec[field], rec["kt_source"], metric_list))

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["kt_path", "model"] + metric_list)
        with ThreadPoolExecutor(max_workers=cpu_count() - 1) as ex:
            futures = [ex.submit(compute_row, t) for t in tasks]
            for fut in tqdm(futures):
                try:
                    row = fut.result(timeout=METRIC_TIMEOUT)
                    if row:
                        writer.writerow(row)
                except Exception:
                    continue


def compute_row(args):
    kt_path, field, code, orig_code, metric_list = args
    try:
        s = structural(code)
        lm = lm_metrics(p_uni, p_bi, p_left, code)
        ent = entropy_metrics(orig_code, code)
        return [kt_path, field] + [s.get(m, lm.get(m, ent.get(m, 0.0))) for m in metric_list]
    except Exception:
        return None


def evaluate_metrics(csv_path):
    rows = list(csv.reader(csv_path.open()))
    header = rows[0]
    idx = {name: i for i, name in enumerate(header)}
    feats = ['CondE', 'Conditional Complexity', 'Halstead Distinct Operators', 'JSD', 'KL', 'LM_CE', 'LM_CondE',
             'LM_KL']
    vals = {m: {f: [] for f in feats} for m in ["kt_source", "our"]}

    for row in rows[1:]:
        model = row[1]
        if model not in vals:
            continue
        for f in feats:
            vals[model][f].append(float(row[idx[f]]))

    med_kt, med_our = [np.median(vals["kt_source"][f]) for f in feats], [np.median(vals["our"][f]) for f in feats]
    dist = max(abs(a - b) for a, b in zip(med_kt, med_our))
    coverage = len(vals["our"][feats[0]]) / (len(rows) - 1)
    return float('inf') if coverage == 0 or np.isnan(dist) else dist / coverage


def objective(trial):
    r = trial.suggest_categorical("r", [8, 16])
    alpha = trial.suggest_categorical("lora_alpha", [16, 32])
    dropout = trial.suggest_float("lora_dropout", 0.0, 0.1)
    grad_acc = trial.suggest_categorical("grad_acc", [32, 64])
    clip = trial.suggest_float("clip", 0.1, 1.0)
    wd = trial.suggest_float("weight_decay", 0.0, 0.1)
    epochs, lr, warmup = 4, 1e-4, 0.1

    max_new = get_max_new(rows_for_stats, tok)
    seq_len = AVG_SEC_LEN
    run_dir = RUNS_DIR / f"trial_{trial.number}"

    model = make_model(r, alpha, dropout)

    train_ds = BASE_TRAIN.map(lambda ex: {
        "input_ids": ex["input_ids"][:seq_len],
        "attention_mask": ex["attention_mask"][:seq_len]
    }, remove_columns=BASE_TRAIN.column_names)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(run_dir),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=grad_acc,
            num_train_epochs=epochs,
            learning_rate=lr,
            lr_scheduler_type="linear",
            warmup_ratio=warmup,
            max_grad_norm=clip,
            weight_decay=wd,
            logging_steps=50,
            save_strategy="no",
            seed=GLOBAL_SEED,
            optim="adamw_torch_fused"
        ),
        train_dataset=train_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
    )

    trainer.train()
    model.save_pretrained(run_dir / "model")

    test_subset = sorted(list(raw_ds["test"]), key=lambda r: len(r["text"]))[:VAL_SUBSET_SIZE]
    gen_path = run_dir / "test_gen.jsonl"
    generate_test_cases(model, tok, test_subset, gen_path, max_new)

    with gen_path.open("r", encoding="utf-8") as f:
        first_line = json.loads(f.readline())

    compute_metrics(gen_path, run_dir / "metrics.csv", extract_metric_names(first_line))
    score = evaluate_metrics(run_dir / "metrics.csv")

    trial.report(score, step=0)

    del model
    torch.cuda.empty_cache()
    gc.collect()
    clear_hf_cache()

    return score


if __name__ == '__main__':
    optuna.logging.set_verbosity(optuna.logging.INFO)
    study = optuna.create_study(
        study_name=STUDY_NAME, direction="minimize",
        storage=DB_URI, load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=GLOBAL_SEED, multivariate=True, group=True),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=0, n_min_trials=5)
    )
    study.optimize(objective, n_trials=30)
    Path(STUDY_NAME).mkdir(exist_ok=True)
    study.trials_dataframe().to_csv(Path(STUDY_NAME) / "optuna_trials.csv")
    print("Best hypers:", study.best_params, "score:", study.best_value)
