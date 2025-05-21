import gc
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from collect.process_models.process_model import to_bytecode
from collect.process_models.shared import Row
from model_train.config import RUNS_DIR, DATASET
from utils.gen_len_stats import get_max_new
from utils.make_example import make_example

MODEL_DIR = Path(RUNS_DIR) / "full_finetune" / "model"
TOKENIZER_DIR = Path(RUNS_DIR) / "full_finetune" / "tokenizer"
NUM_EXAMPLES = 100
BATCH_SIZE = 4
OUT_PATH = Path("generated_test_results_1.jsonl")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(DEVICE).eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

ds = load_dataset(DATASET)["test"][:NUM_EXAMPLES]
rows = [Row(r["kt_path"], r["kt_source"], to_bytecode(r)) for r in ds]
max_new_tokens = get_max_new(rows, tokenizer)
ds = ds.map(make_example)


def get_prompt(rec):
    return rec["text"].split("<|im_start|>assistant")[0]


def get_expected_length(rec):
    start = rec["text"].find("<|im_start|>assistant")
    end = rec["text"].rfind("<|im_end|>")
    return len(tokenizer(rec["text"][start:end]).input_ids)


prompts = [get_prompt(r) for r in ds]
kt_paths = [r["kt_path"] for r in ds]

with OUT_PATH.open("w", encoding="utf-8") as out_file:
    i = 0
    while i < NUM_EXAMPLES:
        batch_prompts = prompts[i:i + BATCH_SIZE]
        batch_paths = kt_paths[i:i + BATCH_SIZE]

        try:
            inputs = tokenizer(batch_prompts, return_tensors="pt",
                               padding=True, truncation=True).to(DEVICE)
            input_len = inputs.input_ids.shape[1]

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            for kt_path, gen_ids in zip(batch_paths, outputs):
                gen_text = tokenizer.decode(gen_ids[input_len:], skip_special_tokens=True)
                gen_text = gen_text.split("<|im_end|>")[0].strip()

                out_file.write(json.dumps({
                    "kt_path": kt_path,
                    "model_name": gen_text
                }, ensure_ascii=False) + "\n")
                out_file.flush()

            i += BATCH_SIZE

        except RuntimeError as e:
            if "out of memory" in str(e).lower() and BATCH_SIZE > 1:
                print(f"OOM: уменьшаю batch_size -> {BATCH_SIZE // 2}")
                BATCH_SIZE = max(1, BATCH_SIZE // 2)
                torch.cuda.empty_cache()
                continue
            raise

        finally:
            del inputs, outputs
            torch.cuda.empty_cache()
            gc.collect()

print(f"Результаты сохранены в {OUT_PATH.resolve()}")
