from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from entropy.common import cross_entropy_ngram, _tokens
from entropy.conditional_entropy import compute as Hcond
from entropy.cross_entropy import compute as CE
from entropy.kl_divergence import compute as KL
from entropy.nid import compute as NID
from entropy.perplexity import compute as PPL
from metrics import registry
from utils.kotlin_parser import parse

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


def entropy(orig: str, dec: str, n_max: int) -> dict[str, float]:
    m = dict(CE=CE(orig, dec), KL=KL(orig, dec),
             PPL=PPL(orig, dec), NID=NID(orig, dec), Hcond=Hcond(orig, dec))
    for n in range(1, n_max + 1):
        m[f"CE_{n}gram"] = cross_entropy_ngram(orig, dec, n)
    return m


# 1. читаем тесты
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

# превращаем в список пар (Test, Category, dec_code, orig_code)
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

# 2. ищем best_n
dec_pairs = [p for p in pairs if p[1] != "Original"]
avg_len = int(np.mean([len(_tokens(o)) for *_, o in dec_pairs]))
hard_cap = min(200, max(3, int(avg_len * 0.25)))


def ce_curve(o, d):
    return [cross_entropy_ngram(o, d, n) for n in range(1, hard_cap + 1)]


df_ce = pd.DataFrame([ce_curve(o, c) for *_, c, o in dec_pairs],
                     columns=[f"CE_{n}" for n in range(1, hard_cap + 1)])

rel = df_ce.pct_change(axis=1).abs().iloc[:, 1:]
q1, q3 = rel.stack().quantile([.25, .75])
REL_THRESH = 0.5 * (q3 - q1) or 0.01

plateau_len = 3
best_n = hard_cap
for i in range(len(rel.columns) - plateau_len + 1):
    if (rel.iloc[:, i:i + plateau_len].mean() < REL_THRESH).all():
        best_n = i + 2  # +2, т.к. rel начинается с CE_2
        break

Path("ngram_best.txt").write_text(str(best_n))
print(f"★ best_n = {best_n}, REL_THRESH = {REL_THRESH:.4f} (плато {plateau_len})")

# 3. detekt
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
    row |= entropy(orig_code, dec_code, best_n)

    # detekt-метрики
    if cat in detekt_df.index:
        for issue, val in detekt_df.loc[cat].items():
            row[f"detekt_{issue}"] = val

    rows.append(row)

df = pd.DataFrame(rows).sort_values(["Test", "Category"]).reset_index(drop=True)
df.to_csv(OUT_CSV, index=False)

print(f"{OUT_CSV} (строк: {len(df)})")
print("ngram_best.txt создано")
