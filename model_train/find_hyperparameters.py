import csv
import json
import random
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path

import datasets
import numpy as np
import optuna
import torch
from datasets import tqdm
from peft import LoraConfig, get_peft_model
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                          TrainingArguments, Trainer, set_seed)

from collect.metrics.common import structural, lm_metrics, load_lm, entropy_metrics

MODEL_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct"
STUDY_NAME = "KExercises+KStack-clean_Qwen2.5-Coder-3B-Instruct_search"
RUNS_DIR = Path(STUDY_NAME) / "runs"
DB_URI = f"sqlite:///{STUDY_NAME}.db"
RAW_DS_PATH = "KExercises+KStack-clean"
TEST_SAMPLE = 20
TRAIN_SUBSET_SIZE = 500
VAL_SUBSET_SIZE = TRAIN_SUBSET_SIZE // 10
VAL_SPLIT = 0.05
GLOBAL_SEED = 228
METRIC_TIMEOUT = 30

set_seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
tok.pad_token = tok.eos_token
raw_ds = datasets.load_from_disk(RAW_DS_PATH)

tok_ds = raw_ds.map(
    lambda b: tok(b["text"], truncation=True, max_length=6144),
    batched=True,
    remove_columns=["text", "kt_path"]
)

split_ds = tok_ds["train"].train_test_split(test_size=VAL_SPLIT, seed=GLOBAL_SEED)
BASE_TRAIN = split_ds["train"]
BASE_VAL = split_ds["test"]

BASE_TRAIN = BASE_TRAIN.select(random.sample(range(len(BASE_TRAIN)), min(len(BASE_TRAIN), TRAIN_SUBSET_SIZE)))
BASE_VAL = BASE_VAL.select(random.sample(range(len(BASE_VAL)), min(len(BASE_VAL), VAL_SUBSET_SIZE)))

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
    seq_len = trial.suggest_categorical("seq_len", [2048, 3072, 6144])
    lr = trial.suggest_categorical("lr", [1e-5, 2e-5, 3e-5])
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
        MODEL_NAME,
        quantization_config=bnb_cfg,
        trust_remote_code=True,
        device_map="auto",
    )
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, LoraConfig(
        r=r, lora_alpha=2 * r, lora_dropout=0.05,
        bias="none", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    run_dir = RUNS_DIR / f"trial_{trial.number}"
    args = TrainingArguments(
        output_dir=str(run_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=grad_acc,
        num_train_epochs=epochs,
        fp16=True,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_grad_norm=clip,
        weight_decay=0.0,
        logging_steps=50,
        save_strategy="no",
        seed=GLOBAL_SEED,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=make_collate(seq_len),
    )
    trainer.train()

    out_dir = run_dir / "model"
    print("Saving model…", flush=True)
    model.save_pretrained(out_dir)
    print("Model saved", flush=True)

    test_subset = random.sample(list(raw_ds["test"]), min(TEST_SAMPLE, len(raw_ds["test"])))
    gen_jsonl = out_dir / "test_gen.jsonl"
    with gen_jsonl.open("w", encoding="utf-8") as f:
        print("Generating test cases…", flush=True)
        for rec in tqdm(test_subset, desc="gen"):
            cut = rec["text"].find("<|im_start|>assistant")
            prompt = rec["text"][:cut]
            out_ids = model.generate(**tok(prompt, return_tensors="pt").to(model.device),
                                     max_new_tokens=seq_len)
            gen_code = tok.decode(out_ids[0], skip_special_tokens=False)
            f.write(json.dumps({"kt_path": rec["kt_path"],
                                "our": gen_code,
                                "kt_source": rec["text"].split("<|im_end|>\n")[-2]}) + "\n")
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
        with ProcessPoolExecutor(max_workers=cpu_count() - 2,
                                 initializer=init_worker,
                                 initargs=(p_uni, p_bi, p_left)) as ex:
            futures = [ex.submit(compute_row, t) for t in tasks]
            for fut in futures:
                try:
                    row = fut.result(timeout=METRIC_TIMEOUT)
                    if row:
                        writer.writerow(row)
                except Exception:
                    continue

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
    return adjusted_dist


if __name__ == '__main__':
    optuna.logging.set_verbosity(optuna.logging.INFO)
    study = optuna.create_study(study_name=STUDY_NAME, direction="minimize",
                                storage=DB_URI, load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=GLOBAL_SEED))
    study.optimize(objective, n_trials=20, timeout=12 * 3600)
    print("Лучшие гиперы:", study.best_params, "score:", study.best_value)
