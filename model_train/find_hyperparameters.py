import csv
import gc
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

import datasets
import numpy as np
import optuna
import torch
from peft import LoraConfig, get_peft_model
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                          TrainingArguments, Trainer, set_seed)

from collect.metrics.common import structural, lm_metrics, load_lm, entropy_metrics
from model_train.config import GLOBAL_SEED, MODEL_PATH, METRIC_TIMEOUT, RAW_DS_PATH, VAL_SPLIT, TRAIN_SUBSET_SIZE, \
    TEST_SAMPLE, VAL_SUBSET_SIZE, RUNS_DIR, DB_URI, STUDY_NAME

set_seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

tok = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side='left')
tok.pad_token = tok.eos_token
raw_ds = datasets.load_from_disk(RAW_DS_PATH)

tok_ds = raw_ds.map(
    lambda b: tok(b["text"], truncation=True, max_length=5120),
    remove_columns=["text", "kt_path"]
)

split_ds = tok_ds["train"].train_test_split(test_size=VAL_SPLIT, seed=GLOBAL_SEED)
BASE_TRAIN = split_ds["train"].shuffle(seed=GLOBAL_SEED).select(range(TRAIN_SUBSET_SIZE))
BASE_VAL = split_ds["test"].shuffle(seed=GLOBAL_SEED).select(range(VAL_SUBSET_SIZE))

p_uni, p_bi, p_left = load_lm()


def make_collate(seq_len):
    def collate(features):
        batch = tok.pad(features, return_tensors="pt")
        batch["input_ids"] = batch["input_ids"][:, :seq_len]
        batch["attention_mask"] = batch["attention_mask"][:, :seq_len]
        labels = batch["input_ids"].clone()
        labels[labels == tok.pad_token_id] = -100
        batch["labels"] = labels
        return batch

    return collate


def extract_kotlin(text: str) -> str:
    for pat in (
            r"```[^\n]*kotlin[^\n]*\n([\s\S]*?)(?:```|\Z)",
            r"```[^\n]*\n([\s\S]*?)(?:```|\Z)",
            r"### Kotlin\n([\s\S]*?)(?:\n###|\Z)",
    ):
        m = re.search(pat, text, re.I | re.M)
        if m:
            return m.group(1).strip()
    return ""


def make_collate(seq_len):
    def collate(features):
        batch = tok.pad(features, return_tensors="pt")
        batch["input_ids"] = batch["input_ids"][:, :seq_len]
        batch["attention_mask"] = batch["attention_mask"][:, :seq_len]
        labels = batch["input_ids"].clone()
        labels[labels == tok.pad_token_id] = -100
        batch["labels"] = labels
        return batch

    return collate


def compute_row(args):
    kt_path, field, code, orig_code, metric_list = args
    try:
        s = structural(str(code))
        lm = lm_metrics(src=str(code), p_uni=P_UNI, p_bi=P_BI, p_left=P_LEFT)
        ent = entropy_metrics(str(orig_code), str(code))
        return [kt_path, field] + [s.get(m, lm.get(m, ent.get(m, 0.0))) for m in metric_list]
    except Exception:
        return None


def init_worker(u, b, l):
    global P_UNI, P_BI, P_LEFT
    P_UNI, P_BI, P_LEFT = u, b, l


def objective(trial):
    r = trial.suggest_categorical("r", [8, 16, 32])
    seq_len = trial.suggest_categorical("seq_len", [2048, 3072, 5120])
    lr = trial.suggest_categorical("lr", [1e-4, 1e-5, 1e-6])
    grad_acc = trial.suggest_categorical("grad_acc", [4, 8, 16])
    epochs = trial.suggest_categorical("epochs", [1, 2, 3, 4, 6])
    clip = trial.suggest_categorical("clip", [0.1, 0.3, 0.5, 1.0])

    train_ds = BASE_TRAIN.map(lambda ex: {"input_ids": ex["input_ids"][:seq_len],
                                          "attention_mask": ex["attention_mask"][:seq_len]},
                              remove_columns=BASE_TRAIN.column_names)
    val_ds = BASE_VAL.map(lambda ex: {"input_ids": ex["input_ids"][:seq_len],
                                      "attention_mask": ex["attention_mask"][:seq_len]},
                          remove_columns=BASE_VAL.column_names)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_cfg,
        trust_remote_code=True,
        device_map="auto",
    )

    print(model)

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, LoraConfig(
        r=r, lora_alpha=2 * r, lora_dropout=0.05,
        bias="lora_only", target_modules='all-linear'))

    run_dir = RUNS_DIR / f"trial_{trial.number}"
    args = TrainingArguments(
        output_dir=str(run_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=grad_acc,
        num_train_epochs=epochs,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_grad_norm=clip,
        weight_decay=0.0,
        logging_steps=50,
        save_strategy="no",
        seed=GLOBAL_SEED,
        optim="adamw_torch_fused",
    )

    data_collator = make_collate(seq_len)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator
    )
    trainer.train()
    del trainer
    torch.cuda.empty_cache()
    gc.collect()
    del train_ds, val_ds
    torch.cuda.empty_cache()
    gc.collect()

    out_dir = run_dir / "model"
    print("Saving model…", flush=True)
    model.save_pretrained(out_dir)
    print("Model saved", flush=True)

    test_subset = random.sample(list(raw_ds["test"]), min(TEST_SAMPLE, len(raw_ds["test"])))
    gen_jsonl = out_dir / "test_gen.jsonl"

    batch_size = 4
    print("Generating test cases…", flush=True)
    for i in range(0, len(test_subset), batch_size):
        batch = test_subset[i:i + batch_size]
        prompts = [rec["text"][:rec["text"].find("<|im_start|>assistant")] for rec in batch]
        inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

        try:
            with torch.inference_mode(), torch.amp.autocast("cuda"):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=seq_len,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                    eos_token_id=tok.eos_token_id,
                )

            with gen_jsonl.open("a", encoding="utf-8") as f:
                for rec, out_ids in zip(batch, outputs):
                    gen_code = tok.decode(out_ids, skip_special_tokens=False)
                    f.write(json.dumps({
                        "kt_path": rec["kt_path"],
                        "our": extract_kotlin(gen_code),
                        "kt_source": rec["text"].split("<|im_end|>\n")[-2]
                    }) + "\n")
                    f.flush()

        except RuntimeError as e:
            if 'out of memory' in str(e).lower() and batch_size > 1:
                print(f"OOM with batch size {batch_size}, reducing...")
                torch.cuda.empty_cache()
                batch_size = max(1, batch_size // 2)
                continue
            raise

        del inputs, outputs
        torch.cuda.empty_cache()
        gc.collect()

    print("Generation done", flush=True)

    with gen_jsonl.open("r", encoding="utf-8") as infile:
        first_line = json.loads(infile.readline())
    metric_list = sorted({n for n in structural(str(first_line["our"])) if not n.startswith("detekt_")})

    tasks = []
    with gen_jsonl.open("r", encoding="utf-8") as infile:
        for line in infile:
            rec = json.loads(line)
            kt_path = rec["kt_path"]
            for field in ["our", "kt_source"]:
                tasks.append((kt_path, field, rec[field], rec["kt_source"], metric_list))

    csv_path = out_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["kt_path", "model"] + metric_list)
        with ThreadPoolExecutor(max_workers=cpu_count() - 1) as ex:
            futures = [ex.submit(compute_row, t) for t in tasks]
            for fut in futures:
                try:
                    row = fut.result(timeout=METRIC_TIMEOUT)
                    if row:
                        writer.writerow(row)
                except Exception:
                    continue
            del futures, tasks

    rows = list(csv.reader(csv_path.open()))
    header = rows[0]
    idx = {n: i for i, n in enumerate(header)}
    feats = ['CE', 'CondE', 'Conditional Complexity', 'Halstead Distinct Operators',
             'Halstead Vocabulary', 'JSD', 'KL', 'LM_CondE', 'LM_JSD']

    vals_by_model = {model: {m: [] for m in feats} for model in ["kt_source", "our"]}

    for row in rows[1:]:
        kt_path, model_id = row[0], row[1]
        if model_id not in vals_by_model:
            continue
        for m in feats:
            vals_by_model[model_id][m].append(float(row[idx[m]]))

    med_kt_source = [np.median(vals_by_model["kt_source"][m]) for m in feats]
    med_our = [np.median(vals_by_model["our"][m]) for m in feats]

    chebyshev_dist = max(abs(a - b) for a, b in zip(med_kt_source, med_our))

    total_cases = len(test_subset)
    covered_cases = len(vals_by_model["our"][feats[0]])
    coverage = covered_cases / total_cases if total_cases else 1.0
    adjusted_dist = chebyshev_dist / max(coverage, 1e-3)

    trial.report(adjusted_dist, step=0)

    del model, test_subset, vals_by_model, med_kt_source, med_our
    torch.cuda.empty_cache()
    gc.collect()

    return adjusted_dist


if __name__ == '__main__':
    optuna.logging.set_verbosity(optuna.logging.INFO)
    study = optuna.create_study(study_name=STUDY_NAME, direction="minimize",
                                storage=DB_URI, load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=GLOBAL_SEED, multivariate=True, group=True),
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=0, n_min_trials=5))
    study.optimize(objective,
                   n_trials=20,
                   # timeout=12 * 3600
                   )
    print("Лучшие гиперы:", study.best_params, "score:", study.best_value)
