def make_example(rec):
    bc = "\n\n".join(f"// {c['class_path']}\n{c['javap'].rstrip()}" for c in rec["classes"])
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
    return {
        "text": prompt + target,
        "kt_path": rec["kt_path"],
        "bytecode": bc,
        "kt_source": rec["kt_source"]
    }
