import re


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
