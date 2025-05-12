import gc
import json
import re
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM, BitsAndBytesConfig,
)

DATASET_NAME = "akabynda/KExercises-bytecode"
SPLIT = "train"
MODEL_NAMES = [
    "JetBrains/deepseek-coder-1.3B-kexer",
    "JetBrains/deepseek-coder-6.7B-kexer",
    "JetBrains/CodeLlama-7B-Kexer",
    "JetBrains/CodeLlama-7B-KStack-clean",
    "JetBrains/CodeLlama-7B-KStack",
    "JetBrains/Mellum-4b-base",
    "Qwen/Qwen2.5-Coder-32B-Instruct"
]
OUT_DIR = Path(f"{DATASET_NAME.split("/")[-1]}_with_models")
OUT_DIR.mkdir(exist_ok=True)
TEMPERATURE = 0.2
TOP_P = 0.9
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
MAX_RATIO = 0.3
NUM_VARIANTS = 1
MAX_NEW_TOKENS_LIMIT = 4056
quant_config = BitsAndBytesConfig(load_in_4bit=True)


def extract_kotlin(text: str) -> str:
    m = re.search(r"```[^\n]*kotlin[^\n]*\n([\s\S]*?)(?:```|\Z)", text, re.I)
    if m:
        return m.group(1).strip()

    m = re.compile(r"```[^\n]*\n([\s\S]*?)(?:```|\Z)", re.M).search(text)
    if m:
        return m.group(1).strip()

    m = re.search(r"### Kotlin\n([\s\S]*?)(?:\n###|\Z)", text, re.M)
    return m.group(1).strip() if m else ""


def make_prompt(name: str, code: str) -> str:
    kotlin_task = (
        "Convert the following JVM byte‑code into **Kotlin source**.\n"
        "Output **Kotlin code ONLY**"
    )
    if name.startswith("Qwen/"):
        messages = [
            {"role": "user",
             "content": kotlin_task + "\n\n### Byte‑code\n" + code + "\n\n### Kotlin"}
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        return (
                "### Task\n" + kotlin_task +
                "\n\n### Byte‑code\n" + code + "\n\n### Kotlin\n"
        )


dataset = load_dataset(DATASET_NAME, split=SPLIT, streaming=True)

for model_name in MODEL_NAMES:
    col = model_name.split("/")[-1]
    out_file = OUT_DIR / f"{col}.jsonl"

    done_ids = set()
    if out_file.exists():
        with out_file.open(encoding="utf-8") as f:
            done_ids = {json.loads(line)["kt_path"] for line in f}
        print(f"{col}: {len(done_ids)} lines already present -> will skip")

    print(f"\n===== {col}: loading model …")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=DTYPE,
            quantization_config=quant_config
        ).eval()
    )

    with out_file.open("a", buffering=1, encoding="utf-8") as fout:

        for row in dataset:
            key = row["kt_path"]
            if key in done_ids:
                continue

            bytecode = "\n".join(cls["javap"] for cls in row["classes"])
            prompt = make_prompt(model_name, bytecode)

            enc = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            inp, attn = enc.input_ids.to(DEVICE), enc.attention_mask.to(DEVICE)
            max_new = min(
                int(attn.sum().item() * MAX_RATIO), MAX_NEW_TOKENS_LIMIT
            )

            with torch.inference_mode(), torch.autocast(device_type=DEVICE, dtype=DTYPE):
                out = model.generate(
                    input_ids=inp,
                    attention_mask=attn,
                    max_new_tokens=max_new,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            if DEVICE == "cuda":
                out = out.to("cpu")
                torch.cuda.empty_cache()

            prompt_len = inp.ne(tokenizer.pad_token_id).sum().item()
            txt = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
            kotlin = extract_kotlin(txt)

            result = {
                "kt_path": row["kt_path"],
                col: kotlin
            }
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    print(f"{col}: done, results in {out_file}")

print("\nВсе модели обработаны построчно. JSONL‑файлы теперь можно объединить.")
