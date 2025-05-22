import json
from pathlib import Path

from datasets import load_dataset, concatenate_datasets

from global_config import GLOBAL_SEED

output_dir = Path("KExercises-KStack-clean-bytecode")
output_dir.mkdir(parents=True, exist_ok=True)

dataset1 = load_dataset("akabynda/KExercises-bytecode", split="train")
dataset2 = load_dataset("akabynda/KStack-clean-bytecode", split="train")

combined_dataset = concatenate_datasets([dataset1, dataset2])

combined_dataset = combined_dataset.shuffle(seed=GLOBAL_SEED)

split_dataset = combined_dataset.train_test_split(test_size=0.1)

with open(f"{output_dir}/train.json", "w", encoding="utf-8") as f:
    json.dump(split_dataset['train'].to_list(), f, ensure_ascii=False, indent=2)

with open(f"{output_dir}/test.json", "w", encoding="utf-8") as f:
    json.dump(split_dataset['test'].to_list(), f, ensure_ascii=False, indent=2)

print("Объединённый датасет сохранён с разделением на train и test.")
