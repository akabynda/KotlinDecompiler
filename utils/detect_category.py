from pathlib import Path

_DECOMPILERS = {"Bytecode", "CFR", "Fernflower", "JDGUI"}
_CONVERTERS = {"ChatGPT", "J2K"}


def detect_category(base_test_dir: Path, file_path: Path) -> str | None:
    rel_parts = file_path.relative_to(base_test_dir).parts
    if len(rel_parts) == 2:
        return "Original"

    decompiler = rel_parts[1]
    if decompiler not in _DECOMPILERS:
        return None

    if decompiler == "Bytecode":
        return "BytecodeChatGPT"

    converter = None
    for part in rel_parts[2:]:
        for conv in _CONVERTERS:
            if conv.lower() in part.lower():
                converter = conv
                break
        if converter:
            break

    return f"{decompiler}{converter}" if converter else None
