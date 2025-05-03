from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from entropy import Entropy
from metrics import registry
from utils.kotlin_parser import parse

language = "kotlin"
entr = Entropy(language)

TEST_ROOT = Path(input("Путь к папке с тестами: ").strip()).expanduser()
OUT_CSV = Path("full_metrics.csv")

DECOMPILERS = {"Bytecode", "CFR", "JDGUI", "Fernflower"}
CONVERTERS = {"ChatGPT", "J2K"}


def read_kt(p: Path) -> str:
    if p.is_file():
        return p.read_text("utf8")
    return "\n".join(f.read_text("utf8") for f in p.rglob("*.kt"))


def read_kt_flat(dir_or_file: Path) -> str:
    if dir_or_file.is_file():
        return dir_or_file.read_text("utf8")
    return "\n".join(p.read_text("utf8") for p in dir_or_file.glob("*.kt"))


def structural(src: str) -> dict[str, float]:
    tree = parse(src)
    return {n: fn(tree) for n, fn in registry.items()}


def entropy(orig: str, dec: str) -> dict[str, float]:
    return dict(
        CE=entr.cross_entropy(orig, dec),
        KL=entr.kl_div(orig, dec),
        PPL=entr.perplexity(orig, dec),
        JSD=entr.jensen_shannon_distance(orig, dec),
        CondE=entr.conditional_entropy(orig, dec),
    )


tests: dict[str, dict] = {}

for td in TEST_ROOT.iterdir():
    if not td.is_dir():
        continue

    orig_code = read_kt_flat(td)
    if not orig_code:
        continue

    tests[td.name] = {"orig": orig_code, "decs": {}}

    for dec in DECOMPILERS:
        root = td / dec
        if not root.is_dir():
            continue

        buckets: dict[str, set[Path]] = defaultdict(set)

        # одиночные *.kt прямо в папке <dec>
        for f in root.glob("*.kt"):
            if "CodeConvert" in f.stem:
                continue
            cv = next((c for c in CONVERTERS if c in f.stem), None)
            if cv:
                buckets[f"{dec}{cv}"].add(f)

        # подпапки PersonChatGPT / PersonJ2K и пр.
        for sub in root.iterdir():
            if not sub.is_dir() or "CodeConvert" in sub.name:
                continue
            cv = next((c for c in CONVERTERS if c in sub.name), None)
            if cv:
                buckets[f"{dec}{cv}"].add(sub)

        # склеиваем код каждой категории
        for cat, paths in buckets.items():
            code = "\n".join(read_kt(p) for p in sorted(paths))
            tests[td.name]["decs"][cat] = code

print(len(tests))

pairs: list[tuple[str, str, str, str]] = []
for test, data in tests.items():
    pairs.append((test, "Original", data["orig"], data["orig"]))
    for cat, code in data["decs"].items():
        pairs.append((test, cat, code, data["orig"]))

if not pairs:
    raise SystemExit("Не найдено ни одной пары original/decompiled")

print(f"Всего пар для анализа: {len(pairs)}")

for t, d in tests.items():
    print(f"{t:20s}  ->  {len(d['decs']):2d} категорий: {list(d['decs'])}")

ISSUE_RE = re.compile(r'^(?P<issue>\w+)\s+-.*\s+at\s+(?P<path>/.*?):\d+:\d+')
DET_REPORT = Path(input("Путь к detekt_report.txt: ").strip()).expanduser()
detekt: dict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))

with DET_REPORT.open(encoding="utf8") as f:
    for line in f:
        m = ISSUE_RE.match(line)
        if not m:
            continue
        issue = m["issue"]
        try:
            rel = Path(m["path"]).relative_to(TEST_ROOT)
        except ValueError:
            continue

        parts = rel.parts
        decomp = parts[1] if len(parts) > 1 and parts[1] in DECOMPILERS else "Original"
        conv = "Original"
        if decomp != "Original" and len(parts) > 2:
            conv = next((c for c in CONVERTERS if c in parts[2]), conv)

        if decomp != "Original" and conv == "Original":
            continue

        cat = "Original" if decomp == conv == "Original" else f"{decomp}{conv}"
        detekt[cat][issue] += 1

detekt_df = pd.DataFrame(detekt).T.fillna(0).astype(int)
detekt_df.index.name = "Category"

# 4. финальная таблица
rows: list[dict[str, float]] = []
for test, cat, dec_code, orig_code in pairs:
    row = {"Test": test, "Category": cat}
    row |= structural(dec_code)
    row |= entropy(orig_code, dec_code)

    # detekt-метрики
    if cat in detekt_df.index:
        for issue, val in detekt_df.loc[cat].items():
            row[f"detekt_{issue}"] = val

    rows.append(row)

df = pd.DataFrame(rows).sort_values(["Test", "Category"]).reset_index(drop=True)
df.to_csv(OUT_CSV, index=False)

print(f"{OUT_CSV} (строк: {len(df)})")
