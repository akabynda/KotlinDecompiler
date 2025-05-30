import gc
import json
from pathlib import Path
from typing import Any, List

import torch
from datasets import load_dataset, Dataset, tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from main.collect.process_models.shared import Row
from main.model_train.config import RUNS_DIR, DATASET, STUDY_NAME
from main.utils.extract_kotlin import extract_kotlin
from main.utils.gen_len_stats import get_max_new
from main.utils.make_prompt import to_bytecode, make_prompt


class KotlinBytecodeGenerator:
    """
    Generates Kotlin bytecode from prompts using a fine-tuned model.
    """

    def __init__(self) -> None:
        self.model_dir: Path = Path(RUNS_DIR) / "full_finetune"
        self.out_path: Path = Path(f"{STUDY_NAME}.jsonl")
        self.num_examples: int = 100
        self.initial_batch_size: int = 4
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
            self.model_dir / "tokenizer", padding_side="left"
        )
        self.model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
            self.model_dir / "merged_model",
            device_map="auto",
            trust_remote_code=True
        )

        self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token

        raw_rows = load_dataset(DATASET, split="test")
        self.rows: List[Row] = sorted(
            (Row(r["kt_path"], r["kt_source"], to_bytecode(r)) for r in raw_rows),
            key=lambda r: len(r.bytecode)
        )[:self.num_examples]

        self.examples: List[dict[str, Any]] = [make_prompt(r) for r in self.rows]
        self.dataset: Dataset = Dataset.from_list(self.examples)
        self.max_new_tokens: int = get_max_new(self.rows, self.tokenizer)

    @staticmethod
    def extract_prompt(text: str) -> str:
        """
        Extract the prompt part from a text with <|im_start|> marker.
        """
        return text.split("<|im_start|>assistant")[0]

    def generate_batch(self, prompts: List[str], paths: List[str]) -> List[dict[str, str]]:
        """
        Generate bytecode predictions for a batch of prompts.
        """
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        input_len = inputs.input_ids.shape[1]

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                top_k=None,
                temperature=None,
                top_p=None,
            )

        results: List[dict[str, str]] = []
        for path, output in zip(paths, outputs):
            text = self.tokenizer.decode(output[input_len:], skip_special_tokens=True)
            result = text.split("<|im_end|>")[0].strip()
            results.append({
                "kt_path": path,
                f"{STUDY_NAME}": extract_kotlin(result)
            })

        return results

    def run_generation(self) -> None:
        """
        Generate Kotlin bytecode for the dataset and save results to a JSONL file.
        """
        prompts = [self.extract_prompt(r["text"]) for r in self.dataset]
        kt_paths = [r["kt_path"] for r in self.dataset]

        with self.out_path.open("w", encoding="utf-8") as out_file:
            i = 0
            batch_size = self.initial_batch_size

            with tqdm(total=self.num_examples, desc="Generating") as pbar:
                while i < self.num_examples:
                    batch_prompts = prompts[i:i + batch_size]
                    batch_paths = kt_paths[i:i + batch_size]

                    try:
                        results = self.generate_batch(batch_prompts, batch_paths)
                        for res in results:
                            out_file.write(json.dumps(res, ensure_ascii=False) + "\n")
                        i += batch_size
                        pbar.update(len(batch_prompts))

                    except RuntimeError as e:
                        if "out of memory" in str(e).lower() and batch_size > 1:
                            batch_size = max(1, batch_size // 2)
                            print(f"Out of memory: reducing batch_size -> {batch_size}")
                            torch.cuda.empty_cache()
                            continue
                        else:
                            raise
                    finally:
                        torch.cuda.empty_cache()
                        gc.collect()

        print(f"Results saved to {self.out_path.resolve()}")


def main() -> None:
    """
    Entry point for bytecode generation.
    """
    generator = KotlinBytecodeGenerator()
    generator.run_generation()


if __name__ == "__main__":
    main()
