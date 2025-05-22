import gc
import json
from pathlib import Path

import torch
from datasets import load_dataset, Dataset, tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from collect.process_models.process_model import to_bytecode
from collect.process_models.shared import Row
from model_train.config import RUNS_DIR, DATASET, STUDY_NAME
from utils.gen_len_stats import get_max_new
from utils.make_example import make_example

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

MODEL_DIR = Path(RUNS_DIR) / "full_finetune"
OUT_PATH = Path(f"{STUDY_NAME}.jsonl")
NUM_EXAMPLES = 100
INITIAL_BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR / "tokenizer", padding_side="left")

model = AutoModelForCausalLM.from_pretrained(MODEL_DIR / "merged_model", device_map="auto", trust_remote_code=True)

tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

raw_rows = load_dataset(DATASET, split="test")
rows = sorted(
    (Row(r["kt_path"], r["kt_source"], to_bytecode(r)) for r in raw_rows),
    key=lambda r: len(r.bytecode)
)[:NUM_EXAMPLES]

examples = [make_example(r) for r in rows]
ds = Dataset.from_list(examples)
max_new_tokens = get_max_new(rows, tokenizer)


def extract_prompt(text: str) -> str:
    return text.split("<|im_start|>assistant")[0]


def generate_batch(prompts, paths):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    input_len = inputs.input_ids.shape[1]

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

    results = []
    for path, output in zip(paths, outputs):
        text = tokenizer.decode(output[input_len:], skip_special_tokens=True)
        model_name = text.split("<|im_end|>")[0].strip()
        results.append({"kt_path": path, "model": model_name})
    return results


prompts = [extract_prompt(r["text"]) for r in ds]
kt_paths = [r["kt_path"] for r in ds]

with OUT_PATH.open("w", encoding="utf-8") as out_file:
    i = 0
    batch_size = INITIAL_BATCH_SIZE

    with tqdm(total=NUM_EXAMPLES, desc="Generating") as pbar:
        while i < NUM_EXAMPLES:
            batch_prompts = prompts[i:i + batch_size]
            batch_paths = kt_paths[i:i + batch_size]

            try:
                results = generate_batch(batch_prompts, batch_paths)
                for res in results:
                    out_file.write(json.dumps(res, ensure_ascii=False) + "\n")
                i += batch_size
                pbar.update(len(batch_prompts))

            except RuntimeError as e:
                if "out of memory" in str(e).lower() and batch_size > 1:
                    batch_size = max(1, batch_size // 2)
                    print(f"OOM: уменьшаю batch_size -> {batch_size}")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise
            finally:
                torch.cuda.empty_cache()
                gc.collect()

print(f"Результаты сохранены в {OUT_PATH.resolve()}")
