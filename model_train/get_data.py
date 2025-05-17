import random

from datasets import load_dataset, DatasetDict, Dataset


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
for name in ["akabynda/KExercises-bytecode", "akabynda/KStack-clean-bytecode"]:
    for r in load_dataset(name, split="train", streaming=True):
        all_recs.append(make_example(r))

random.shuffle(all_recs)
cut = int(len(all_recs) * 0.9)

ds = DatasetDict({
    "train": Dataset.from_list(all_recs[:cut]),
    "test": Dataset.from_list(all_recs[cut:]),
})
ds.save_to_disk("KExercises-KStack-clean-bytecode")
