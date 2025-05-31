from src.main.collect.process_models.shared import Row


def make_prompt(rec: Row):
    prompt = (
        "<|im_start|>system\n"
        "Convert the following JVM byte-code into **Kotlin source code**.\n"
        "Output Kotlin code only. Do not add any explanations.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{rec.bytecode}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    target = f"```kotlin\n{rec.kt_source.rstrip()}\n```\n<|im_end|>\n"
    return {
        "text": prompt + target,
        "kt_path": rec.kt_path,
        "bytecode": rec.bytecode,
        "kt_source": rec.kt_source,
    }


def wrap_as_row(example):
    return Row(
        kt_path=example["kt_path"],
        kt_source=example["kt_source"],
        bytecode=to_bytecode(example),
    )


def to_bytecode(row) -> str:
    return "\n".join(cls["javap"] for cls in row["classes"])
