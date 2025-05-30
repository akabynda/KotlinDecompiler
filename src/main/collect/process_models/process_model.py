import gc
import json
import sys
from pathlib import Path
from typing import List, Optional

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from src.main.collect.process_models.shared import Config, Row
from src.main.utils.extract_kotlin import extract_kotlin
from src.main.utils.gen_len_stats import gen_len_stats
from src.main.utils.make_prompt import to_bytecode
from src.main.utils.model_batch_size import model_batch_size


class ModelProcessor:
    """
    Processes a model by generating Kotlin source code from bytecode and saving the results.
    """

    def __init__(self, model_name: str) -> None:
        self.cfg: Config = Config()
        self.model_name: str = model_name
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None
        self.output_file: Path = self.cfg.out_dir / f"{self.model_name.split('/')[-1]}.jsonl"

    def load_rows(self) -> List[Row]:
        """
        Load dataset rows.
        """
        ds = load_dataset(self.cfg.dataset_name, split=self.cfg.split, streaming=False)
        return [Row(r["kt_path"], r["kt_source"], to_bytecode(r)) for r in ds]

    @staticmethod
    def build_prompt(model_name: str, bytecode: str, tokenizer: PreTrainedTokenizerBase) -> str:
        """
        Build a prompt for the model based on the given bytecode.
        """
        head = "Convert the following JVM byte‑code into **Kotlin source**.\nOutput **Kotlin code ONLY**"
        if model_name.startswith("Qwen/"):
            tmpl = [{"role": "user", "content": f"{head}\n\n### Byte‑code\n{bytecode}\n\n### Kotlin"}]
            return tokenizer.apply_chat_template(tmpl, tokenize=False, add_generation_prompt=True)
        return f"### Task\n{head}\n\n### Byte‑code\n{bytecode}\n\n### Kotlin\n"

    @staticmethod
    def _hf_generate(
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizerBase,
            prompts: List[str],
            max_new: int,
            do_sample: bool = False,
            temperature: Optional[float] = None,
            top_p: Optional[float] = None,
            top_k: Optional[float] = None,
    ) -> List[str]:
        """
        Generate predictions from the model.
        """
        encodings = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        input_len = encodings.input_ids.shape[1]
        max_length = input_len + max_new

        with torch.inference_mode(), torch.amp.autocast("cuda"):
            outputs = model.generate(
                **encodings,
                max_length=max_length,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=1,
            )

        results = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
        del encodings, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return results

    def load_model(self) -> PreTrainedModel:
        """
        Load the model with appropriate quantization if available.
        """
        try:
            return AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else None,
                trust_remote_code=True,
                quantization_config=self.cfg.quant_4bit,
            )
        except ValueError as e:
            print(f"4-bit quant failed for {self.model_name}: {e}")
            try:
                return AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else None,
                    trust_remote_code=True,
                    quantization_config=self.cfg.quant_8bit,
                )
            except ValueError as e:
                print(f"8-bit quant failed for {self.model_name}: {e}")
                return AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else None,
                    trust_remote_code=True,
                )

    @staticmethod
    def unload_model(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> None:
        """
        Release GPU memory and clean up.
        """
        gc.collect()
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def process(self) -> None:
        """
        Process the dataset with the model and save the results.
        """
        done = set()
        if self.output_file.exists():
            with self.output_file.open() as f:
                done = {json.loads(line)["kt_path"] for line in f}

        if len(done) >= self.cfg.dataset_size:
            return

        print(f"[HF] loading {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        self.model = self.load_model().eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        batch_size = model_batch_size(self.model, self.cfg.est_scale)
        print("Batch size:", batch_size)
        rows = self.load_rows()
        rows.sort(key=lambda r: len(r.bytecode))
        rows = rows[:self.cfg.dataset_size]
        max_new, _ = gen_len_stats(rows, self.tokenizer)
        print("max_new:", max_new)

        buffer: List[dict] = []
        prompts: List[str] = []
        payload: List[Row] = []

        with self.output_file.open("a", encoding="utf-8") as f_out:
            for row in tqdm(rows, desc=self.model_name.split("/")[-1]):
                if row.kt_path in done:
                    continue
                prompts.append(self.build_prompt(self.model_name, row.bytecode, self.tokenizer))
                payload.append(row)

                if len(prompts) >= batch_size:
                    answers = self._hf_generate(self.model, self.tokenizer, prompts, max_new=max_new)
                    for r, ans in zip(payload, answers):
                        buffer.append({"kt_path": r.kt_path, self.model_name.split("/")[-1]: extract_kotlin(ans)})
                    prompts.clear()
                    payload.clear()

                if len(buffer) >= self.cfg.flush_every:
                    for item in buffer:
                        f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                    f_out.flush()
                    buffer.clear()

            # Final flush
            if prompts:
                answers = self._hf_generate(self.model, self.tokenizer, prompts, max_new=max_new)
                for r, ans in zip(payload, answers):
                    buffer.append({"kt_path": r.kt_path, self.model_name.split("/")[-1]: extract_kotlin(ans)})
            for item in buffer:
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

        self.unload_model(self.model, self.tokenizer)
        print("Processing completed!")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python process_model.py <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]
    print(f"Processing model: {model_name}")

    processor = ModelProcessor(model_name)
    processor.process()


if __name__ == "__main__":
    main()
