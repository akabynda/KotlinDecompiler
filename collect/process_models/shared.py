from collections import namedtuple
from pathlib import Path

import torch
from transformers import BitsAndBytesConfig

Row = namedtuple("Row", ("kt_path", "kt_source", "bytecode"))


class Config:
    dataset_name: str = "akabynda/KExercises-KStack-clean-bytecode"
    split: str = "test"
    model_names: tuple[str, ...] = (
        "Qwen/Qwen2.5-Coder-0.5B",
        "Qwen/Qwen2.5-Coder-0.5B-Instruct",

        "Qwen/Qwen2.5-Coder-1.5B",
        "Qwen/Qwen2.5-Coder-1.5B-Instruct",

        "JetBrains/deepseek-coder-1.3B-kexer",
        "deepseek-ai/deepseek-coder-1.3b-base",
        "deepseek-ai/deepseek-coder-1.3b-instruct",

        "Qwen/Qwen2.5-Coder-3B-Instruct",
        "Qwen/Qwen2.5-Coder-3B",

        "JetBrains/Mellum-4b-base",

        "deepseek-ai/deepseek-coder-6.7b-instruct",
        "deepseek-ai/deepseek-coder-6.7b-base",
        "JetBrains/deepseek-coder-6.7B-kexer",
        "deepseek-ai/deepseek-coder-7b-base-v1.5",
        "deepseek-ai/deepseek-coder-7b-instruct-v1.5",

        "Qwen/Qwen2.5-Coder-7B",
        "Qwen/Qwen2.5-Coder-7B-Instruct",

        "JetBrains/CodeLlama-7B-Kexer",
        "JetBrains/CodeLlama-7B-KStack-clean",
        "JetBrains/CodeLlama-7B-KStack",
        "codellama/CodeLlama-7b-hf",
        "codellama/CodeLlama-7b-Instruct-hf",
    )

    # temperature: float = 0.2
    # top_p: float = 0.9
    flush_every: int = 100
    num_variants: int = 1
    est_scale: float = 1
    dataset_size: int = 100
    out_dir: Path = Path(f"{dataset_name.split("/")[-1]}_with_models")

    quant_4bit: BitsAndBytesConfig | None = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=False,
        )
        if torch.cuda.is_available()
        else None
    )

    quant_8bit: BitsAndBytesConfig | None = (
        BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=False,
        )
        if torch.cuda.is_available()
        else None
    )
