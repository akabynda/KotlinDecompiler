import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from entropy import Entropy
from metrics import registry
from utils.kotlin_parser import parse

language = "kotlin"
entr = Entropy(language)
# language_model_name = "kstack"
language_model_name = "kstack-clean+kexercises"

LANG_MODEL_DIR = Path(f"lang_models/{language}/{language_model_name}")
with open(LANG_MODEL_DIR / "unigram.json", encoding="utf8") as f:
    P_uni = json.load(f)
with open(LANG_MODEL_DIR / "bigram.json", encoding="utf8") as f:
    P_bi = json.load(f)
with open(LANG_MODEL_DIR / "left.json", encoding="utf8") as f:
    P_left = json.load(f)

test_root = Path(input("Путь к папке с тестами: ").strip()).expanduser()
out_csv = Path("full_metrics.csv")

decompilers = {"Bytecode", "CFR", "JDGUI", "Fernflower"}
converters = {"ChatGPT", "J2K"}


def read_kt(path: Path) -> str:
    if path.is_file():
        return path.read_text("utf8")
    return "\n".join(p.read_text("utf8") for p in path.rglob("*.kt"))


def read_kt_flat(dir_or_file: Path) -> str:
    if dir_or_file.is_file():
        return dir_or_file.read_text("utf8")
    return "\n".join(p.read_text("utf8") for p in dir_or_file.glob("*.kt"))


def structural(src: str) -> dict[str, float]:
    tree = parse(src)
    return {n: fn(tree) for n, fn in registry.items()}


def entropy_metrics(orig: str, dec: str) -> dict[str, float]:
    return {
        "CE": entr.cross_entropy(orig, dec),
        "KL": entr.kl_div(orig, dec),
        "PPL": entr.perplexity(orig, dec),
        "JSD": entr.jensen_shannon_distance(orig, dec),
        "CondE": entr.conditional_entropy(orig, dec),
    }


def lm_metrics(src: str) -> dict[str, float]:
    return {
        "LM_CE": entr.cross_entropy_lang(P_uni, src),
        "LM_KL": entr.kl_div_lang(P_uni, src),
        "LM_PPL": entr.perplexity_lang(P_uni, src),
        "LM_CondE": entr.conditional_entropy_lang(P_bi, P_left, src),
    }


tests: dict[str, dict] = {}
for td in test_root.iterdir():
    if not td.is_dir():
        continue
    orig_code = read_kt_flat(td)
    if not orig_code:
        continue
    tests[td.name] = {"orig": orig_code, "decs": {}}
    for dec in decompilers:
        root = td / dec
        if not root.is_dir():
            continue
        buckets: dict[str, set[Path]] = defaultdict(set)
        for f in root.glob("*.kt"):
            if "CodeConvert" in f.stem:
                continue
            cv = next((c for c in converters if c in f.stem), None)
            if cv:
                buckets[f"{dec}{cv}"].add(f)
        for sub in root.iterdir():
            if not sub.is_dir() or "CodeConvert" in sub.name:
                continue
            cv = next((c for c in converters if c in sub.name), None)
            if cv:
                buckets[f"{dec}{cv}"].add(sub)
        for cat, paths in buckets.items():
            code = "\n".join(read_kt(p) for p in sorted(paths))
            tests[td.name]["decs"][cat] = code

pairs: list[tuple[str, str, str, str]] = []
for test, data in tests.items():
    pairs.append((test, "Original", data["orig"], data["orig"]))
    for cat, code in data["decs"].items():
        pairs.append((test, cat, code, data["orig"]))
if not pairs:
    raise SystemExit("Не найдено ни одной пары original/decompiled")

issue_re = re.compile(r"^(?P<issue>\w+)\s+-.*\s+at\s+(?P<path>/.*?):\d+:\d+")
report_path = Path(input("Путь к detekt_report.txt: ").strip()).expanduser()
detekt: dict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
with report_path.open(encoding="utf8") as f:
    for line in f:
        m = issue_re.match(line)
        if not m:
            continue
        issue = m["issue"]
        try:
            rel = Path(m["path"]).relative_to(test_root)
        except ValueError:
            continue
        parts = rel.parts
        decomp = parts[1] if len(parts) > 1 and parts[1] in decompilers else "Original"
        conv = "Original"
        if decomp != "Original" and len(parts) > 2:
            conv = next((c for c in converters if c in parts[2]), conv)
        if decomp != "Original" and conv == "Original":
            continue
        cat = "Original" if decomp == conv == "Original" else f"{decomp}{conv}"
        detekt[cat][issue] += 1

detekt_df = pd.DataFrame(detekt).T.fillna(0).astype(int)


def build_rows() -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for test, cat, dec_code, orig_code in pairs:
        row: dict[str, float] = {"Test": test, "Category": cat}
        row.update(structural(dec_code))
        row.update(entropy_metrics(orig_code, dec_code))
        row.update(lm_metrics(dec_code))
        if cat in detekt_df.index:
            for issue, val in detekt_df.loc[cat].items():
                row[f"detekt_{issue}"] = val
        rows.append(row)
    return rows


df = pd.DataFrame(build_rows()).sort_values(["Test", "Category"]).reset_index(drop=True)
df.to_csv(out_csv, index=False)
print(f"{out_csv} (строк: {len(df)})")
