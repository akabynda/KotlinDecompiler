import gc
import json
import re
from collections import namedtuple
from pathlib import Path
from statistics import median
from typing import Iterable, List

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

Row = namedtuple("Row", ("kt_path", "kt_source", "bytecode"))


class Config:
    dataset_name: str = "akabynda/KExercises-bytecode"
    split: str = "train"
    model_names: tuple[str, ...] = (
        "JetBrains/deepseek-coder-1.3B-kexer",
        "JetBrains/deepseek-coder-6.7B-kexer",
        "JetBrains/CodeLlama-7B-Kexer",
        "JetBrains/CodeLlama-7B-KStack-clean",
        "JetBrains/CodeLlama-7B-KStack",
        "JetBrains/Mellum-4b-base",
        "Qwen/Qwen2.5-Coder-32B-Instruct",
    )
    temperature: float = 0.2
    top_p: float = 0.9
    flush_every: int = 100
    num_variants: int = 1
    est_scale: float = 1
    out_dir: Path = Path(f"{dataset_name.split("/")[-1]}_with_models")
    quant: BitsAndBytesConfig | None = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        if torch.cuda.is_available()
        else None
    )


CFG = Config()
CFG.out_dir.mkdir(exist_ok=True)


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
    return min(512, int(sum(kt_lens) / len(kt_lens) * 2)), round(min(0.5, median(ratios)), 3)


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
    return tokenizer.batch_decode(out[:, input_len:], skip_special_tokens=True)


def process_model_hf(name: str, rows: List[Row]) -> None:
    col = name.split("/")[-1]
    outfile = CFG.out_dir / f"{col}.jsonl"

    done = set()
    if outfile.exists():
        with outfile.open() as file:
            done = {json.loads(line)["kt_path"] for line in file}

    print(f"[HF] loading {name}")
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else "auto",
        trust_remote_code=True,
        quantization_config=CFG.quant,
    ).eval()

    try:
        model = torch.compile(model, mode="max-autotune", fullgraph=True)
    except Exception as e:
        print("torch.compile failed:", e)

    # warm‑up
    try:
        _ = _hf_generate(
            model,
            tokenizer,
            prompts=[build_prompt(name, rows[0].bytecode, tokenizer=tokenizer)],
            max_new=32,
            temperature=0.0,
            top_p=0.0,
            variants=1,
        )
    except Exception:
        pass

    batch_size = model_batch_size(model, CFG.est_scale)
    print("batch size:", batch_size)
    max_new, _ratio = gen_stats(rows, tokenizer)

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

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def main() -> None:
    print("Loading dataset stream…")
    rows = load_rows()
    print(f"Total rows: {len(rows):,}")
    rows.sort(key=lambda r: len(r.bytecode))
    rows = rows[:100]

    for name in CFG.model_names:
        process_model_hf(name, rows)

    print("Finished")


if __name__ == "__main__":
    main()
