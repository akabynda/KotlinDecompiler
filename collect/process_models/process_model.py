import gc
import json
import re
import sys
from statistics import median
from typing import Iterable, List

import torch
from datasets import load_dataset
from numpy import percentile
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from shared import Config, Row

CFG = Config()


def to_bytecode(row) -> str:
    return "\n".join(cls["javap"] for cls in row["classes"])


def load_rows() -> list[Row]:
    ds = load_dataset(CFG.dataset_name, split=CFG.split, streaming=False)
    return [Row(r["kt_path"], r["kt_source"], to_bytecode(r)) for r in ds]


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


def build_prompt(model_name: str, bytecode: str, tokenizer) -> str:
    head = "Convert the following JVM byte‑code into **Kotlin source**.\nOutput **Kotlin code ONLY**"
    if model_name.startswith("Qwen/"):
        tmpl = [{"role": "user", "content": f"{head}\n\n### Byte‑code\n{bytecode}\n\n### Kotlin"}]
        return tokenizer.apply_chat_template(tmpl, tokenize=False, add_generation_prompt=True)
    return f"### Task\n{head}\n\n### Byte‑code\n{bytecode}\n\n### Kotlin\n"


def model_batch_size(model: torch.nn.Module, scale: float) -> int:
    props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    vram = props.total_memory if props else 8 << 30
    size = sum(p.numel() * p.element_size() for p in model.parameters())
    return max(1, int(vram / size * scale))


def gen_stats(rows: Iterable[Row], tokenizer) -> tuple[int, float]:
    kt_lens, ratios = [], []
    for r in rows:
        kt = len(tokenizer(r.kt_source).input_ids)
        bc = len(tokenizer(r.bytecode).input_ids)
        kt_lens.append(kt)
        ratios.append(kt / bc if bc else 0)

    return min(2048, int(percentile(kt_lens, 90))), round(min(0.5, median(ratios)), 3)


def _hf_generate(
        model: torch.nn.Module,
        tokenizer,
        prompts: List[str],
        *,
        max_new: int,
        temperature: float,
        top_p: float,
        variants: int,
) -> List[str]:
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    input_len = enc.input_ids.shape[1]
    max_length = input_len + max_new

    with torch.inference_mode(), torch.amp.autocast("cuda"):
        out = model.generate(
            **enc,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=variants,
            early_stopping=True,
            use_cache=True,
        )
    res = tokenizer.batch_decode(out[:, input_len:], skip_special_tokens=True)

    del enc, out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return res


def load_model(name):
    try:
        return AutoModelForCausalLM.from_pretrained(
            name,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
            trust_remote_code=True,
            quantization_config=CFG.quant,
        )
    except ValueError as e:
        print(f"4-bit quant failed for {name}: {e}\n")
        return AutoModelForCausalLM.from_pretrained(
            name,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )


def unload_model(model, tok):
    gc.collect()
    del model
    del tok

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def process_model_hf(name: str, rows: List[Row]) -> None:
    col = name.split("/")[-1]
    outfile = CFG.out_dir / f"{col}.jsonl"

    done = set()
    if outfile.exists():
        with outfile.open() as file:
            done = {json.loads(line)["kt_path"] for line in file}

    if len(done) >= CFG.dataset_size:
        return

    print(f"[HF] loading {name}")
    tokenizer = AutoTokenizer.from_pretrained(name, padding_side='left')
    model = load_model(name).eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    """try:
        model = torch.compile(model, mode="max-autotune", fullgraph=True)
    except Exception as e:
        print("torch.compile failed:", e)"""

    batch_size = model_batch_size(model, CFG.est_scale)
    print("batch size:", batch_size)
    max_new, _ratio = gen_stats(rows, tokenizer)

    print("max_new:", max_new)

    buf: list[dict] = []
    prompts, payload = [], []
    with outfile.open("a", encoding="utf-8") as f_out:
        for row in tqdm(rows, desc=col):
            if row.kt_path in done:
                continue
            prompts.append(build_prompt(name, row.bytecode, tokenizer=tokenizer))
            payload.append(row)
            if len(prompts) >= batch_size:
                answers = _hf_generate(
                    model,
                    tokenizer,
                    prompts,
                    max_new=max_new,
                    temperature=CFG.temperature,
                    top_p=CFG.top_p,
                    variants=CFG.num_variants,
                )
                for r, ans in zip(payload, answers):
                    buf.append({"kt_path": r.kt_path, col: extract_kotlin(ans)})
                prompts.clear()
                payload.clear()
            if len(buf) >= CFG.flush_every:
                for item in buf:
                    f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                f_out.flush()
                buf.clear()
        # tail
        if prompts:
            answers = _hf_generate(
                model,
                tokenizer,
                prompts,
                max_new=max_new,
                temperature=CFG.temperature,
                top_p=CFG.top_p,
                variants=CFG.num_variants,
            )
            for r, ans in zip(payload, answers):
                buf.append({"kt_path": r.kt_path, col: extract_kotlin(ans)})
        for item in buf:
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

    unload_model(model, tokenizer)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_model.py <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]

    print(f"Processing model: {model_name}")
    rows = load_rows()
    rows.sort(key=lambda r: len(r.bytecode))
    rows = rows[:CFG.dataset_size]
    process_model_hf(model_name, rows)
