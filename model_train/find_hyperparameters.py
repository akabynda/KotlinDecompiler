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
from datasets import tqdm
from peft import LoraConfig, get_peft_model
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                          TrainingArguments, Trainer, set_seed, DataCollatorForLanguageModeling)

from collect.metrics.common import structural, lm_metrics, load_lm, entropy_metrics
from train.config import GLOBAL_SEED, RAW_DS_PATH, VAL_SUBSET_SIZE, RUNS_DIR, DB_URI, STUDY_NAME, MODEL, \
    TRAIN_SUBSET_SIZE, METRIC_TIMEOUT
from utils.clear_hf_cache import clear_hf_cache
from utils.extract_kotlin import extract_kotlin

set_seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

tok = AutoTokenizer.from_pretrained(MODEL, padding_side='left')
tok.pad_token = tok.eos_token
raw_ds = datasets.load_from_disk(RAW_DS_PATH)

tok_ds = raw_ds.map(
    lambda b: tok(b["text"], truncation=True),
    remove_columns=["text", "kt_path"]
)

BASE_TRAIN = tok_ds["train"].shuffle(seed=GLOBAL_SEED).select(range(TRAIN_SUBSET_SIZE))

lengths = [len(rec["input_ids"]) for rec in tok_ds["train"]]
AVG_SEC_LEN = int(np.percentile(lengths, 95))
print("Average length:", AVG_SEC_LEN)

p_uni, p_bi, p_left = load_lm()


def make_collate(seq_len: int):
    def collate(features):
        features = [
            {
                "input_ids": f["input_ids"][:seq_len],
                "attention_mask": f["attention_mask"][:seq_len],
            }
            for f in features
        ]

        batch = tok.pad(
            features,
            padding=True,
            return_tensors="pt"
        )

        # Маска потерь: только где attention_mask == 0
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels

        return batch

    return collate


def compute_row(args):
    kt_path, field, code, orig_code, metric_list = args
    try:
        s = structural(str(code))
        lm = lm_metrics(src=str(code), p_uni=p_uni, p_bi=p_bi, p_left=p_left)
        ent = entropy_metrics(str(orig_code), str(code))
        return [kt_path, field] + [s.get(m, lm.get(m, ent.get(m, 0.0))) for m in metric_list]
    except Exception:
        return None


def objective(trial):
    r = trial.suggest_categorical("r", [8, 16, 32, 64])
    lora_alpha = trial.suggest_categorical("lora_alpha", [16, 32, 64, 128])
    seq_len = AVG_SEC_LEN
    grad_acc = trial.suggest_categorical("grad_acc", [32, 64, 128, 256])
    epochs = 4
    lr = 1e-4
    lora_dropout = trial.suggest_float("lora_dropout", 0.0, 0.1)
    clip = trial.suggest_float("clip", 0.1, 1.0)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
    warmup_ratio = 0.1

    train_ds = BASE_TRAIN.map(lambda ex: {"input_ids": ex["input_ids"][:seq_len],
                                          "attention_mask": ex["attention_mask"][:seq_len]},
                              remove_columns=BASE_TRAIN.column_names)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        quantization_config=bnb_cfg,
        trust_remote_code=True,
        device_map="auto",
        use_cache=False,
    )

    # print(model)

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, LoraConfig(
        r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        bias="lora_only", target_modules='all-linear'))

    run_dir = RUNS_DIR / f"trial_{trial.number}"
    args = TrainingArguments(
        output_dir=str(run_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=grad_acc,
        num_train_epochs=epochs,
        learning_rate=lr,
        lr_scheduler_type="linear",
        warmup_ratio=warmup_ratio,
        max_grad_norm=clip,
        weight_decay=weight_decay,
        logging_steps=50,
        save_strategy="no",
        seed=GLOBAL_SEED,
        optim="adamw_torch_fused",
    )

    # data_collator = make_collate(seq_len)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tok,
        mlm=False
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=data_collator
    )
    trainer.train()

    out_dir = run_dir / "model"
    print("Saving model...")
    model.save_pretrained(out_dir)
    print("Model saved")
    print("Model's device:", model.device)

    test_records = list(raw_ds["test"])
    test_records.sort(key=lambda r: len(r["text"]))
    test_subset = test_records[:min(VAL_SUBSET_SIZE, len(test_records))]

    gen_jsonl = out_dir / "test_gen.jsonl"

    batch_size = 4
    print("Batch size:", batch_size)
    print("Generating test cases...")

    for i in tqdm(range(0, len(test_subset), batch_size), desc="Generating"):
        batch = test_subset[i:i + batch_size]
        prompts = [rec["text"][:rec["text"].find("<|im_start|>assistant")] for rec in batch]
        inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        targets = [rec["text"][rec["text"].find("<|im_start|>assistant"):] for rec in batch]
        expected = tok(targets, return_tensors="pt", padding=True, truncation=True)
        exp_lens = expected["attention_mask"].sum(dim=1)
        max_new_tokens = int(exp_lens.max().item() * 1.4)

        try:
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                    eos_token_id=tok.eos_token_id,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    num_beams=1
                )

            with gen_jsonl.open("a", encoding="utf-8") as f:
                for rec, out_ids in zip(batch, outputs):
                    gen_code = tok.decode(out_ids, skip_special_tokens=False)
                    f.write(json.dumps({
                        "kt_path": rec["kt_path"],
                        "our": extract_kotlin(gen_code),
                        "kt_source": extract_kotlin(rec["text"].split("<|im_end|>\n")[-2])
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

    print("Generation done")

    with gen_jsonl.open("r", encoding="utf-8") as infile:
        first_line = json.loads(infile.readline())

    metric_names = set()
    for field, code in first_line.items():
        if field == 'kt_path' or code is None or code == "":
            continue
        orig_code = first_line.get('kt_source')
        s = structural(str(code))
        lm = lm_metrics(src=str(code), p_uni=p_uni, p_bi=p_bi, p_left=p_left)
        ent = entropy_metrics(str(orig_code), str(code)) if orig_code else {}
        for name in list(s) + list(lm) + list(ent):
            metric_names.add(name)

    metric_list = list(metric_names)

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
            for fut in tqdm(futures):
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
    coverage = covered_cases / total_cases if total_cases else 0
    if coverage == 0 or np.isnan(chebyshev_dist):
        adjusted_dist = float('inf')
    else:
        adjusted_dist = chebyshev_dist / coverage

    trial.report(adjusted_dist, step=0)

    del model, test_subset, vals_by_model, med_kt_source, med_our
    torch.cuda.empty_cache()
    gc.collect()

    clear_hf_cache()

    return adjusted_dist


if __name__ == '__main__':
    optuna.logging.set_verbosity(optuna.logging.INFO)
    study = optuna.create_study(study_name=STUDY_NAME, direction="minimize",
                                storage=DB_URI, load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=GLOBAL_SEED, multivariate=True, group=True),
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=0, n_min_trials=5))
    study.optimize(objective,
                   n_trials=30,
                   )
    df = study.trials_dataframe()
    df.to_csv(Path(f"{STUDY_NAME}/optuna_trials.csv"))
    print("Best hypers:", study.best_params, "score:", study.best_value)
