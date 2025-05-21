from pathlib import Path

from datasets import load_dataset, Dataset

from global_config import GLOBAL_SEED
from model_train.config import RAW_DS_PATH


def make_example(rec):
    bc = "\n\n".join(f"// {c['class_path']}\n{c['javap'].rstrip()}"
                     for c in rec["classes"])
    prompt = (
        "<|im_start|>system\n"
        "Convert the following JVM byte-code into **Kotlin source code**.\n"
        "Output Kotlin code only. Do not add any explanations.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{bc}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    target = f"```kotlin\n{rec['kt_source'].rstrip()}\n```\n<|im_end|>\n"
    return {"text": prompt + target, "kt_path": rec["kt_path"]}


all_recs = []
for name in ["akabynda/KExercises-KStack-clean-bytecode"]:
    for r in load_dataset(name, split="train", streaming=True):
        all_recs.append(make_example(r))

ds = Dataset.from_list(all_recs)

ds = ds.shuffle(seed=GLOBAL_SEED)
ds_split = ds.train_test_split(test_size=0.1, seed=GLOBAL_SEED)

output_path = Path(RAW_DS_PATH)
ds_split.save_to_disk(str(output_path))